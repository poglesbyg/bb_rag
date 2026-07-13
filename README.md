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

## Install

```
cargo install --path .
```

Puts a `bb_rag` binary on your `PATH` (via `~/.cargo/bin`), so every command
below can drop the `cargo run --` prefix — just `bb_rag ingest docs/`,
`bb_rag chat --provider ollama`, etc. Run it from whatever directory holds
the `docs/`, `.env`, and `index.bin` you want it to use (it reads/writes
relative to your current directory, same as `cargo run` does). Re-run
`cargo install --path .` after pulling changes to update the installed copy.

If you'd rather not install it globally, every command also works as
`cargo run -- <args>` from inside this directory.

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
# Ingest one file or a whole directory (any flat text file, plus .pdf / .docx)
bb_rag ingest docs/                        # claude/TF-IDF, offline
bb_rag ingest docs/ --provider ollama       # real embeddings via ollama
bb_rag ingest docs/ --provider huggingface  # real embeddings via HF

# Ask a question against the ingested index (provider picks the generator)
bb_rag query "What is bb_rag?"
bb_rag query "What is bb_rag?" --provider ollama

# Interactive, streaming, multi-turn chat (ollama only, for now)
bb_rag chat --provider ollama
```

`chat` opens a REPL: each turn re-runs retrieval, always prints the sources
it found (score + path) before answering, streams the model's reply token by
token, and keeps the whole conversation in memory so follow-up questions have
context from earlier turns. Nothing is persisted to disk — history resets
when you exit (`exit`/`quit`/Ctrl-D). `query` and `ingest` still work with
any provider; `chat` currently requires `--provider ollama` since streaming
and multi-turn history aren't wired up for claude/huggingface yet.

Ingesting builds/appends to `index.bin` in the working directory. Querying
picks its retrieval strategy from what's stored there: if chunks have
embeddings, it re-embeds the question with the same provider and ranks by
cosine similarity; otherwise it falls back to in-process TF-IDF. You can't
mix two different real-embedding providers in one index (e.g. `ollama` then
`huggingface`) — delete `index.bin` and re-ingest to switch.

Model names (and API base URLs, mainly useful for pointing at a proxy or a
test double) are configurable via env vars, defaulting to:

- `OLLAMA_HOST` — `http://localhost:11434`
- `OLLAMA_EMBED_MODEL` — `nomic-embed-text`
- `OLLAMA_GEN_MODEL` — `llama3.2`
- `HF_API_BASE` — `https://api-inference.huggingface.co`
- `HF_EMBED_MODEL` — `sentence-transformers/all-MiniLM-L6-v2`
- `HF_GEN_MODEL` — `HuggingFaceH4/zephyr-7b-beta`
- `ANTHROPIC_API_BASE` — `https://api.anthropic.com`

HF's free Inference API only serves a subset of models at any given time —
if the default 404s, pick another model from the HF Hub and override
`HF_GEN_MODEL`/`HF_EMBED_MODEL`.

## How it works

The whole program is one file, [`src/main.rs`](src/main.rs), with a module
doc comment at the top giving the same overview as this section plus links
to every function it mentions. Every non-trivial function has a `///` doc
comment on its contract/gotchas (not just what the name already says); run
`cargo doc --open --document-private-items` to browse it as generated docs.

- **Input formats**: there's no extension allowlist — `ingest` picks up
  every file under the given path. `.pdf` is run through
  [pdf-extract](https://docs.rs/pdf-extract) to pull out its text layer;
  `.docx` is unzipped and its `word/document.xml` text runs are extracted
  (paragraph breaks preserved, formatting/tables/images ignored). Everything
  else is read as plain UTF-8 text — any flat text file (`.md`, `.csv`,
  `.json`, `.log`, source code, no extension at all, ...) works as-is. Files
  that fail (binary junk, scanned/image-only PDFs with no text layer,
  invalid UTF-8) are skipped with a warning printed to stderr rather than
  aborting the whole ingest; the final summary line reports how many were
  skipped. Because there's no extension filter, pointing this at a directory
  full of binary files (images, `.git`, compiled artifacts, ...) is safe but
  noisy — expect a warning per file it can't read as text.
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
- **Storage**: `index.bin` is [bincode](https://docs.rs/bincode)-encoded, not
  JSON — smaller and faster to (de)serialize than text, especially for the
  embedding vectors, at the cost of not being human-readable. It's not a
  format you should expect to stay compatible across bb_rag versions; if
  loading ever fails after an update, delete it and re-ingest.

## Testing

`cargo test` runs everything, including the HTTP-calling functions
(`ollama_embed`/`ollama_generate`/`ollama_chat_stream`, `claude_generate`,
`hf_embed`/`hf_generate`) — those are exercised against a local
[wiremock](https://docs.rs/wiremock) server rather than the real APIs, by
pointing `OLLAMA_HOST`/`ANTHROPIC_API_BASE`/`HF_API_BASE` at it for the
duration of each test. Those tests are `#[serial]` (via
[serial_test](https://docs.rs/serial_test)) since env vars are
process-global — they'd otherwise race with each other under `cargo test`'s
default parallelism. No network access or real API keys are needed to run
the suite.
