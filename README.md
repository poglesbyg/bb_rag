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

# Interactive, streaming, multi-turn chat (ollama only, for now)
cargo run -- chat --provider ollama
```

`chat` opens a REPL: each turn re-runs retrieval, always prints the sources
it found (score + path) before answering, streams the model's reply token by
token, and keeps the whole conversation in memory so follow-up questions have
context from earlier turns. Nothing is persisted to disk — history resets
when you exit (`exit`/`quit`/Ctrl-D). `query` and `ingest` still work with
any provider; `chat` currently requires `--provider ollama` since streaming
and multi-turn history aren't wired up for claude/huggingface yet.

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

- **Chunking**: sentence-aware packing up to an ~800-char budget with
  ~100-char overlap — sentences are never split mid-way; only a single
  "sentence" bigger than the whole budget (e.g. unpunctuated text) falls
  back to a hard character window.
- **Retrieval**: dense cosine similarity over real embeddings (ollama/HF), or
  in-process TF-IDF cosine similarity (claude) — a linear scan either way,
  fine up to a few thousand chunks.
- **Generation**: `query` makes one buffered HTTP call per question to
  whichever provider you pick. `chat` streams the response from Ollama's
  `/api/chat` (NDJSON chunks printed as they arrive) and replays the growing
  message history on every turn so the model has multi-turn context.
