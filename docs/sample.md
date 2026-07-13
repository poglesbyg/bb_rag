# bb_rag

bb_rag is a bare-bones retrieval-augmented generation CLI written in Rust.

It works in two steps:

1. `ingest` splits text files into overlapping chunks and stores the chunk
   text and source path in a local `index.json` file. This step runs fully
   offline — no API calls, no embeddings.
2. `query` tokenizes every chunk and the question, ranks chunks by TF-IDF
   cosine similarity, and sends the top matches as context to Claude to
   produce a grounded answer.

There is no vector database, no server, and no embedding model. The whole
index is a JSON file, and the whole retrieval step is a linear scan with
TF-IDF cosine similarity, which is plenty fast for a few thousand chunks.
