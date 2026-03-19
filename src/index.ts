import "dotenv/config";
import { stdin as input, stdout as output } from "node:process";
import { createInterface } from "node:readline/promises";

// --- LangChain imports ---
import { GithubRepoLoader } from "@langchain/community/document_loaders/web/github";
import { RecursiveCharacterTextSplitter, SupportedTextSplitterLanguage } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

// =============================================================================
// RAG — Retrieval-Augmented Generation
//
//  ┌─────────────────────────────────────────────────────────────────────────┐
//  │  [R] RETRIEVAL          [A] AUGMENTED           [G] GENERATION          │
//  │                                                                         │
//  │  Source data                Prompt enrichment       LLM response        │
//  │  ──────────                 ────────────────        ────────────        │
//  │  LOAD raw files             context = retrieved     model.invoke()      │
//  │    ↓                        chunks injected         StringOutputParser  │
//  │  SPLIT into chunks          into prompt template                        │
//  │    ↓                              ↑                                     │
//  │  EMBED → vectors            RETRIEVE top-K                              │
//  │    ↓                        similar chunks                              │
//  │  STORE in vector DB ────────────────────                                │
//  └─────────────────────────────────────────────────────────────────────────┘
// =============================================================================

// ---------------------------------------------------------------------------
// STEP 0: Configuration
// ---------------------------------------------------------------------------

const REPO_URL = "https://github.com/Shagon1k/AHurynovich-CV";

// Map file extensions to LangChain language modes so the splitter uses
// language-aware separators (e.g. splits TS on function/class boundaries).
const EXTENSION_TO_LANGUAGE: Record<string, SupportedTextSplitterLanguage> = {
    ts: "js",
    tsx: "js",
    js: "js",
    jsx: "js",
    scss: "markdown", // no SCSS mode; markdown splits on blank lines ≈ rule blocks
    css: "markdown",
    html: "html",
    json: "js",
    md: "markdown",
};

// Prompt template fed to the LLM. {context} = retrieved code chunks, {query} = user question.
const PROMPT_TEMPLATE = `You are an experienced knowledge keeper of a GitHub project.

<context>
{context}
</context>

Answer the following question using only the context above:
{query}`;

// ---------------------------------------------------------------------------
// [R] STEP 1: Load — fetch all source files from the GitHub repo
// ---------------------------------------------------------------------------

async function loadDocs(): Promise<Document[]> {
    const loader = new GithubRepoLoader(REPO_URL, {
        recursive: false,
        ignorePaths: ["*.md", "package-lock.json"],
    });
    return loader.load();
}

// ---------------------------------------------------------------------------
// [R] STEP 2: Split — break large files into smaller overlapping chunks.
// Smaller chunks = more precise retrieval; overlap = no lost context at edges.
// ---------------------------------------------------------------------------

async function splitDocs(docs: Document[]): Promise<Document[]> {
    const splitDocs: Document[] = [];

    for (const doc of docs) {
        const ext = doc.metadata.source?.split(".").pop()?.toLowerCase() ?? "";
        const language = EXTENSION_TO_LANGUAGE[ext];

        const splitter = language
            ? RecursiveCharacterTextSplitter.fromLanguage(language, { chunkSize: 200, chunkOverlap: 20 })
            : new RecursiveCharacterTextSplitter({ chunkSize: 200, chunkOverlap: 20 });

        splitDocs.push(...await splitter.splitDocuments([doc]));
    }

    return splitDocs;
}

// ---------------------------------------------------------------------------
// [R] STEP 3: Embed & Store — convert chunks to vectors and save in-memory.
// Vectors capture semantic meaning so we can find "similar" chunks later.
// ---------------------------------------------------------------------------

function createVectorStore(): MemoryVectorStore {
    const embeddings = new OpenAIEmbeddings({
        model: "text-embedding-3-small",
        dimensions: 1024,
    });
    return new MemoryVectorStore(embeddings);
}

// Helper: serialize retrieved Document objects into an XML-like string
// that the LLM can easily parse inside the prompt.
function formatDocsAsContext(documents: Document[]): string {
    return documents
        .map((doc) => `<doc>\n${doc.pageContent}\n</doc>`)
        .join("\n");
}

// ---------------------------------------------------------------------------
// [R] STEP 4a: Retrieve — find top-K chunks most similar to the query
// [A] STEP 4b: Augment  — inject retrieved chunks into the prompt as context
// [G] STEP 4c: Generate — send augmented prompt to LLM and parse the answer
// ---------------------------------------------------------------------------

function buildRagChain(vectorStore: MemoryVectorStore) {
    // [R] retriever: query string → top-K similar Document chunks
    const retriever = vectorStore.asRetriever();

    // [R] Sub-chain: extract query → retrieve docs → format into a context string
    const retrievalChain = RunnableSequence.from([
        (input: { query: string }) => input.query,
        retriever,
        formatDocsAsContext,
    ]);

    // [A] Prompt template: merges retrieved context + original query into one prompt
    const prompt = ChatPromptTemplate.fromTemplate(PROMPT_TEMPLATE);

    // [G] LLM that will generate the final answer
    const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo-1106" });

    return RunnableSequence.from([
        {
            context: retrievalChain,                          // [R+A] retrieve & inject context
            query: (input: { query: string }) => input.query, // [A]   pass query into prompt
        },
        prompt,                    // [A] augmented prompt = context + query
        model,                     // [G] generate answer
        new StringOutputParser(),  // [G] parse LLM message → plain string
    ]);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
    // [R] Load, split, embed, store
    console.log("Loading repo...");
    const docs = await loadDocs();
    console.log(`Loaded ${docs.length} documents`);

    console.log("Splitting into chunks...");
    const chunks = await splitDocs(docs);
    console.log(`Split into ${chunks.length} chunks`);

    console.log("Embedding and storing...");
    const vectorStore = createVectorStore();
    await vectorStore.addDocuments(chunks);
    console.log("Vector store ready");

    // [R+A+G] Build and run the full RAG chain
    const ragChain = buildRagChain(vectorStore);

    const rl = createInterface({ input, output });
    console.log(`Ready for questions about the project: ${REPO_URL}. \n Type /exit to quit.`);

    try {
        while (true) {
            const query = (await rl.question("\nQuestion: ")).trim();

            if (query === "/exit") {
                console.log("Exiting...");
                break;
            }

            if (!query) {
                console.log("Please enter a question or type /exit.");
                continue;
            }

            console.log("Querying...");
            const answer = await ragChain.invoke({ query });
            console.log("\nAnswer:\n", answer);
        }
    } finally {
        rl.close();
    }
}

main().catch(console.error);
