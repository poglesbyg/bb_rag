# bb_rag

A bare-bones RAG (retrieval-augmented generation) CLI in Rust. No vector DB,
no framework — just chunking, OpenAI embeddings, cosine similarity, and a
chat completion call.

## Setup

```
export OPENAI_API_KEY=sk-...
```

## Usage

```
# Ingest one file or a whole directory (.txt / .md files)
cargo run -- ingest docs/

# Ask a question against the ingested index
cargo run -- query "What is bb_rag?"
```

Ingesting builds/appends to `index.json` in the working directory — the
whole vector store is just chunk text + source path + embedding, serialized
as JSON. Querying embeds the question, ranks chunks by cosine similarity,
takes the top 4, and asks `gpt-4o-mini` to answer using only that context.

## How it works

- **Chunking**: fixed-size (800 chars) sliding window with 100-char overlap.
- **Embeddings**: `text-embedding-3-small` via the OpenAI REST API.
- **Retrieval**: linear scan + cosine similarity (fine up to a few thousand
  chunks; swap in a real vector index if you outgrow this).
- **Generation**: `gpt-4o-mini`, temperature 0, system prompt instructs it to
  answer only from the provided context.
