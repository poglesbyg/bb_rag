use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const CHAT_MODEL: &str = "claude-sonnet-5";
const INDEX_FILE: &str = "index.json";
const CHUNK_SIZE: usize = 800;
const CHUNK_OVERLAP: usize = 100;
const TOP_K: usize = 4;

#[derive(Serialize, Deserialize, Clone)]
struct Chunk {
    source: String,
    text: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("ingest") => {
            let path = args.get(2).context("usage: bb_rag ingest <file-or-dir>")?;
            ingest(Path::new(path))?;
        }
        Some("query") => {
            let question = args[2..].join(" ");
            if question.is_empty() {
                bail!("usage: bb_rag query <question>");
            }
            let api_key = env::var("ANTHROPIC_API_KEY").context("set ANTHROPIC_API_KEY")?;
            let client = reqwest::Client::new();
            query(&client, &api_key, &question).await?;
        }
        _ => {
            println!("usage:\n  bb_rag ingest <file-or-dir>\n  bb_rag query <question>");
        }
    }
    Ok(())
}

// ---------- ingest ----------

fn ingest(path: &Path) -> Result<()> {
    let files = collect_files(path)?;
    if files.is_empty() {
        bail!("no .txt or .md files found at {}", path.display());
    }

    let mut index = load_index().unwrap_or_default();
    let mut added = 0;
    for file in &files {
        let content = fs::read_to_string(file)
            .with_context(|| format!("reading {}", file.display()))?;
        for chunk in chunk_text(&content, CHUNK_SIZE, CHUNK_OVERLAP) {
            index.push(Chunk { source: file.display().to_string(), text: chunk });
            added += 1;
        }
    }
    save_index(&index)?;
    println!(
        "chunked {} file(s) into {} chunks; index now has {} chunk(s) -> {}",
        files.len(),
        added,
        index.len(),
        INDEX_FILE
    );
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

// ---------- retrieval (local TF-IDF, no network) ----------

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(|s| s.to_string())
        .collect()
}

fn term_freq(tokens: &[String]) -> HashMap<String, f32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    for t in tokens {
        *counts.entry(t.clone()).or_insert(0) += 1;
    }
    let total = tokens.len() as f32;
    counts.into_iter().map(|(k, v)| (k, v as f32 / total)).collect()
}

fn compute_idf(doc_tokens: &[Vec<String>]) -> HashMap<String, f32> {
    let n = doc_tokens.len() as f32;
    let mut df: HashMap<String, u32> = HashMap::new();
    for tokens in doc_tokens {
        let unique: HashSet<&String> = tokens.iter().collect();
        for t in unique {
            *df.entry(t.clone()).or_insert(0) += 1;
        }
    }
    df.into_iter()
        .map(|(t, d)| (t, (n / (1.0 + d as f32)).ln() + 1.0))
        .collect()
}

fn tfidf_vector(tokens: &[String], idf: &HashMap<String, f32>) -> HashMap<String, f32> {
    term_freq(tokens)
        .into_iter()
        .map(|(t, f)| {
            let w = f * idf.get(&t).copied().unwrap_or(0.0);
            (t, w)
        })
        .collect()
}

fn sparse_cosine(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> f32 {
    let (small, big) = if a.len() < b.len() { (a, b) } else { (b, a) };
    let dot: f32 = small.iter().map(|(k, v)| v * big.get(k).copied().unwrap_or(0.0)).sum();
    let na: f32 = a.values().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.values().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ---------- query ----------

async fn query(client: &reqwest::Client, api_key: &str, question: &str) -> Result<()> {
    let index = load_index().context("no index found; run `bb_rag ingest` first")?;
    if index.is_empty() {
        bail!("index is empty; run `bb_rag ingest` first");
    }

    let doc_tokens: Vec<Vec<String>> = index.iter().map(|c| tokenize(&c.text)).collect();
    let idf = compute_idf(&doc_tokens);
    let doc_vectors: Vec<HashMap<String, f32>> =
        doc_tokens.iter().map(|t| tfidf_vector(t, &idf)).collect();
    let query_vector = tfidf_vector(&tokenize(question), &idf);

    let mut scored: Vec<(f32, &Chunk)> = index
        .iter()
        .zip(&doc_vectors)
        .map(|(c, v)| (sparse_cosine(&query_vector, v), c))
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

// ---------- Anthropic API ----------

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<AnthropicMessage>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

async fn chat(client: &reqwest::Client, api_key: &str, question: &str, context: &str) -> Result<String> {
    let system = "Answer the user's question using only the provided context. \
        If the context doesn't contain the answer, say you don't know.";
    let user = format!("Context:\n{}\n\nQuestion: {}", context, question);

    let body = AnthropicRequest {
        model: CHAT_MODEL.to_string(),
        max_tokens: 1024,
        system: system.to_string(),
        messages: vec![AnthropicMessage { role: "user".to_string(), content: user }],
    };

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        bail!("chat request failed: {} - {}", resp.status(), resp.text().await?);
    }
    let parsed: AnthropicResponse = resp.json().await?;
    Ok(parsed
        .content
        .into_iter()
        .find(|b| b.block_type == "text")
        .map(|b| b.text)
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
