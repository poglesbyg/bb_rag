use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const EMBED_MODEL: &str = "text-embedding-3-small";
const CHAT_MODEL: &str = "gpt-4o-mini";
const INDEX_FILE: &str = "index.json";
const CHUNK_SIZE: usize = 800;
const CHUNK_OVERLAP: usize = 100;
const TOP_K: usize = 4;

#[derive(Serialize, Deserialize, Clone)]
struct Chunk {
    source: String,
    text: String,
    embedding: Vec<f32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.get(1).is_none() {
        println!("usage:\n  bb_rag ingest <file-or-dir>\n  bb_rag query <question>");
        return Ok(());
    }

    let api_key = env::var("OPENAI_API_KEY").context("set OPENAI_API_KEY")?;
    let client = reqwest::Client::new();

    match args.get(1).map(String::as_str) {
        Some("ingest") => {
            let path = args.get(2).context("usage: bb_rag ingest <file-or-dir>")?;
            ingest(&client, &api_key, Path::new(path)).await?;
        }
        Some("query") => {
            let question = args[2..].join(" ");
            if question.is_empty() {
                bail!("usage: bb_rag query <question>");
            }
            query(&client, &api_key, &question).await?;
        }
        _ => {
            println!("usage:\n  bb_rag ingest <file-or-dir>\n  bb_rag query <question>");
        }
    }
    Ok(())
}

// ---------- ingest ----------

async fn ingest(client: &reqwest::Client, api_key: &str, path: &Path) -> Result<()> {
    let files = collect_files(path)?;
    if files.is_empty() {
        bail!("no .txt or .md files found at {}", path.display());
    }

    let mut texts = Vec::new();
    let mut sources = Vec::new();
    for file in &files {
        let content = fs::read_to_string(file)
            .with_context(|| format!("reading {}", file.display()))?;
        for chunk in chunk_text(&content, CHUNK_SIZE, CHUNK_OVERLAP) {
            sources.push(file.display().to_string());
            texts.push(chunk);
        }
    }
    println!("chunked {} file(s) into {} chunks", files.len(), texts.len());

    let embeddings = embed(client, api_key, &texts).await?;

    let mut index = load_index().unwrap_or_default();
    for ((text, source), embedding) in texts.into_iter().zip(sources).zip(embeddings) {
        index.push(Chunk { source, text, embedding });
    }
    save_index(&index)?;
    println!("index now has {} chunk(s) -> {}", index.len(), INDEX_FILE);
    Ok(())
}

fn collect_files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if path.is_file() {
        out.push(path.to_path_buf());
    } else if path.is_dir() {
        walk_dir(path, &mut out)?;
    } else {
        bail!("path not found: {}", path.display());
    }
    out.retain(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("txt") | Some("md")));
    Ok(out)
}

fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            walk_dir(&p, out)?;
        } else {
            out.push(p);
        }
    }
    Ok(())
}

fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + size).min(chars.len());
        let piece: String = chars[start..end].iter().collect();
        let trimmed = piece.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
        if end == chars.len() {
            break;
        }
        start += size - overlap;
    }
    chunks
}

// ---------- query ----------

async fn query(client: &reqwest::Client, api_key: &str, question: &str) -> Result<()> {
    let index = load_index().context("no index found; run `bb_rag ingest` first")?;
    if index.is_empty() {
        bail!("index is empty; run `bb_rag ingest` first");
    }

    let question_embedding = embed(client, api_key, &[question.to_string()])
        .await?
        .remove(0);

    let mut scored: Vec<(f32, &Chunk)> = index
        .iter()
        .map(|c| (cosine_similarity(&question_embedding, &c.embedding), c))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top = &scored[..TOP_K.min(scored.len())];

    println!("--- retrieved context ---");
    for (score, chunk) in top {
        println!("[{:.3}] {}", score, chunk.source);
    }

    let context = top
        .iter()
        .map(|(_, c)| format!("Source: {}\n{}", c.source, c.text))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let answer = chat(client, api_key, question, &context).await?;
    println!("\n--- answer ---\n{}", answer);
    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ---------- OpenAI API ----------

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

async fn embed(client: &reqwest::Client, api_key: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let body = EmbedRequest { model: EMBED_MODEL, input: texts };
    let resp = client
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        bail!("embeddings request failed: {} - {}", resp.status(), resp.text().await?);
    }
    let parsed: EmbedResponse = resp.json().await?;
    Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageOut,
}

#[derive(Deserialize)]
struct ChatMessageOut {
    content: String,
}

async fn chat(client: &reqwest::Client, api_key: &str, question: &str, context: &str) -> Result<String> {
    let system = "Answer the user's question using only the provided context. \
        If the context doesn't contain the answer, say you don't know.";
    let user = format!("Context:\n{}\n\nQuestion: {}", context, question);

    let body = ChatRequest {
        model: CHAT_MODEL.to_string(),
        messages: vec![
            ChatMessage { role: "system".to_string(), content: system.to_string() },
            ChatMessage { role: "user".to_string(), content: user },
        ],
        temperature: 0.0,
    };

    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        bail!("chat request failed: {} - {}", resp.status(), resp.text().await?);
    }
    let parsed: ChatResponse = resp.json().await?;
    Ok(parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .unwrap_or_default())
}

// ---------- index persistence ----------

fn load_index() -> Result<Vec<Chunk>> {
    let data = fs::read_to_string(INDEX_FILE)?;
    Ok(serde_json::from_str(&data)?)
}

fn save_index(index: &[Chunk]) -> Result<()> {
    let data = serde_json::to_string(index)?;
    fs::write(INDEX_FILE, data)?;
    Ok(())
}
