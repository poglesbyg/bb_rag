# bb_rag

bb_rag is a bare-bones retrieval-augmented generation CLI written in Rust.

It works in two steps:

1. `ingest` splits text files into overlapping chunks, embeds them with
   OpenAI's `text-embedding-3-small` model, and stores the chunks and their
   embeddings in a local `index.json` file.
2. `query` embeds your question, finds the most similar chunks in the index
   using cosine similarity, and sends them as context to `gpt-4o-mini` to
   produce a grounded answer.

There is no vector database, no server, and no framework. The whole index is
a JSON file and the whole retrieval step is a linear scan with cosine
similarity, which is plenty fast for a few thousand chunks.
