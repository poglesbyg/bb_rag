# bb_rag

A bare-bones RAG (retrieval-augmented generation) CLI in Rust. No vector DB,
no framework, no embedding API — just chunking, local TF-IDF retrieval, and
a Claude call for generation.

## Setup

```
export ANTHROPIC_API_KEY=sk-ant-...
```

Only `query` needs the API key; `ingest` runs fully offline.

## Usage

```
# Ingest one file or a whole directory (.txt / .md files)
cargo run -- ingest docs/

# Ask a question against the ingested index
cargo run -- query "What is bb_rag?"
```

Ingesting builds/appends to `index.json` in the working directory — the
whole store is just chunk text + source path, serialized as JSON. Querying
tokenizes every chunk, computes TF-IDF vectors, ranks chunks against the
question by cosine similarity, takes the top 4, and asks Claude to answer
using only that context.

## How it works

- **Chunking**: fixed-size (800 chars) sliding window with 100-char overlap.
- **Retrieval**: TF-IDF computed in-process at query time (no embedding
  model, no API call) + cosine similarity linear scan. Fine up to a few
  thousand chunks; swap in real embeddings or a vector index if you outgrow
  this.
- **Generation**: `claude-sonnet-5` via the Anthropic Messages API, system
  prompt instructs it to answer only from the provided context.
