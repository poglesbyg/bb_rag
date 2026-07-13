# bb_rag

A bare-bones RAG (retrieval-augmented generation) CLI in Rust. No vector DB,
no framework — just chunking, pluggable embeddings/generation, and cosine
similarity.

Three providers, chosen per-command with `--provider`:

| Provider      | Embeddings                  | Generation                        | Needs |
|---------------|------------------------------|------------------------------------|-------|
| `claude` (default) | none — falls back to local TF-IDF | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `ollama`      | real, via a local Ollama server | real, via a local Ollama server | `ollama serve` running |
| `huggingface` | real, via HF Inference API   | real, via HF Inference API         | `HF_API_TOKEN` |

## Setup

Copy `.env.example` to `.env` and fill in what you need — it's loaded
automatically at startup, so no manual `export`ing:

```
cp .env.example .env
```

```
# .env
ANTHROPIC_API_KEY=sk-ant-...   # claude (default)
HF_API_TOKEN=hf_...            # huggingface
```

For Ollama there's no token, just a local server:

```
ollama pull nomic-embed-text   # embedding model
ollama pull llama3.2           # generation model
ollama serve                   # if not already running
```

`ingest` only hits the network for `ollama`/`huggingface` (to compute
embeddings); with `--provider claude` (the default) it runs fully offline.
`query` always needs the network, since generation always calls out.

## Usage

```
# Ingest one file or a whole directory (.txt / .md files)
cargo run -- ingest docs/                        # claude/TF-IDF, offline
cargo run -- ingest docs/ --provider ollama       # real embeddings via ollama
cargo run -- ingest docs/ --provider huggingface  # real embeddings via HF

# Ask a question against the ingested index (provider picks the generator)
cargo run -- query "What is bb_rag?"
cargo run -- query "What is bb_rag?" --provider ollama
```

Ingesting builds/appends to `index.json` in the working directory. Querying
picks its retrieval strategy from what's stored there: if chunks have
embeddings, it re-embeds the question with the same provider and ranks by
cosine similarity; otherwise it falls back to in-process TF-IDF. You can't
mix two different real-embedding providers in one index (e.g. `ollama` then
`huggingface`) — delete `index.json` and re-ingest to switch.

Model names are configurable via env vars, defaulting to:

- `OLLAMA_HOST` — `http://localhost:11434`
- `OLLAMA_EMBED_MODEL` — `nomic-embed-text`
- `OLLAMA_GEN_MODEL` — `llama3.2`
- `HF_EMBED_MODEL` — `sentence-transformers/all-MiniLM-L6-v2`
- `HF_GEN_MODEL` — `HuggingFaceH4/zephyr-7b-beta`

HF's free Inference API only serves a subset of models at any given time —
if the default 404s, pick another model from the HF Hub and override
`HF_GEN_MODEL`/`HF_EMBED_MODEL`.

## How it works

- **Chunking**: fixed-size (800 chars) sliding window with 100-char overlap.
- **Retrieval**: dense cosine similarity over real embeddings (ollama/HF), or
  in-process TF-IDF cosine similarity (claude) — a linear scan either way,
  fine up to a few thousand chunks.
- **Generation**: one HTTP call per query to whichever provider you pick,
  with a system prompt instructing it to answer only from the retrieved
  context.
