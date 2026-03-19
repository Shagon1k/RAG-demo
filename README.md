# RAG simple Demo

This project is a minimal Retrieval-Augmented Generation (RAG) demo built with LangChain and OpenAI.
It loads code from a GitHub repository, builds an in-memory vector index, and answers questions in an interactive terminal.

## Project Purpose

The goal is to show the full RAG pipeline end-to-end in a small, readable TypeScript script:

1. Retrieve source files from GitHub.
2. Split files into semantic chunks.
3. Embed chunks into vectors.
4. Retrieve top relevant chunks for each user question.
5. Generate an answer strictly from retrieved context.

## Requirements

- Node.js 18+ (recommended)
- OpenAI API key

## Setup

1. Install dependencies:

```bash
npm install
```

2. Add environment variables in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
```

## Run

Development mode (TypeScript directly):

```bash
npm run dev
```

Build and run compiled output:

```bash
npm run prod
```

## Interactive Usage

After startup, the app will prompt:

```text
Question:
```

Examples:
- `What engines are used in this project?`
- `How is project data loaded?`
- `/exit` to terminate the app.

## Notes

- Vector storage is in-memory, so index data is rebuilt every run.
- The prompt is designed to answer only from retrieved context.
