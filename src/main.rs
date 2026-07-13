use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const CLAUDE_MODEL: &str = "claude-sonnet-5";
const INDEX_FILE: &str = "index.json";
const CHUNK_SIZE: usize = 800;
const CHUNK_OVERLAP: usize = 100;
const TOP_K: usize = 4;
const SYSTEM_PROMPT: &str = "Answer the user's question using only the provided context. \
    If the context doesn't contain the answer, say you don't know.";

#[derive(Clone, Copy, PartialEq)]
enum Provider {
    Claude,
    Ollama,
    HuggingFace,
}

impl Provider {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "claude" => Ok(Self::Claude),
            "ollama" => Ok(Self::Ollama),
            "huggingface" | "hf" => Ok(Self::HuggingFace),
            other => bail!("unknown provider '{other}', expected claude|ollama|huggingface"),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Ollama => "ollama",
            Self::HuggingFace => "huggingface",
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct Chunk {
    source: String,
    text: String,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

#[derive(Serialize, Deserialize, Default)]
struct Index {
    #[serde(default)]
    embedding_provider: Option<String>,
    #[serde(default)]
    chunks: Vec<Chunk>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let provider_flag = extract_flag(&mut args, "--provider");
    let provider = Provider::parse(provider_flag.as_deref().unwrap_or("claude"))?;
    let client = reqwest::Client::new();

    match args.first().map(String::as_str) {
        Some("ingest") => {
            let path = args
                .get(1)
                .context("usage: bb_rag ingest <file-or-dir> [--provider claude|ollama|huggingface]")?;
            ingest(&client, Path::new(path), provider).await?;
        }
        Some("query") => {
            let question = args[1..].join(" ");
            if question.is_empty() {
                bail!("usage: bb_rag query <question> [--provider claude|ollama|huggingface]");
            }
            query(&client, &question, provider).await?;
        }
        _ => {
            println!(
                "usage:\n  \
                bb_rag ingest <file-or-dir> [--provider claude|ollama|huggingface]\n  \
                bb_rag query <question> [--provider claude|ollama|huggingface]\n\n\
                claude has no embeddings API, so it always falls back to local TF-IDF retrieval.\n\
                ollama and huggingface compute real embeddings at ingest time and reuse them at query time."
            );
        }
    }
    Ok(())
}

fn extract_flag(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let pos = args.iter().position(|a| a == flag)?;
    if pos + 1 >= args.len() {
        return None;
    }
    args.remove(pos);
    Some(args.remove(pos))
}

// ---------- ingest ----------

async fn ingest(client: &reqwest::Client, path: &Path, provider: Provider) -> Result<()> {
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

    let embeddings: Option<Vec<Vec<f32>>> = match provider {
        Provider::Claude => None,
        Provider::Ollama | Provider::HuggingFace => Some(embed_texts(client, provider, &texts).await?),
    };

    let mut index = load_index().unwrap_or_default();
    if let Some(existing) = &index.embedding_provider {
        if embeddings.is_some() && existing != provider.as_str() {
            bail!(
                "index was built with '{}' embeddings; re-ingesting with '{}' would produce \
                incomparable vectors. Delete {} and re-ingest from scratch to switch providers.",
                existing,
                provider.as_str(),
                INDEX_FILE
            );
        }
    }
    if embeddings.is_some() {
        index.embedding_provider = Some(provider.as_str().to_string());
    }

    let added = texts.len();
    for (i, (text, source)) in texts.into_iter().zip(sources).enumerate() {
        let embedding = embeddings.as_ref().map(|v| v[i].clone());
        index.chunks.push(Chunk { source, text, embedding });
    }
    save_index(&index)?;
    println!(
        "chunked {} file(s) into {} chunks (embeddings: {}); index now has {} chunk(s) -> {}",
        files.len(),
        added,
        if embeddings.is_some() { provider.as_str() } else { "none, TF-IDF at query time" },
        index.chunks.len(),
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

// ---------- TF-IDF retrieval (fallback when the index has no embeddings) ----------

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

fn dense_cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ---------- query ----------

async fn query(client: &reqwest::Client, question: &str, provider: Provider) -> Result<()> {
    let index = load_index().context("no index found; run `bb_rag ingest` first")?;
    if index.chunks.is_empty() {
        bail!("index is empty; run `bb_rag ingest` first");
    }

    let top: Vec<(f32, &Chunk)> = match &index.embedding_provider {
        Some(embed_provider_str) => {
            let embed_provider = Provider::parse(embed_provider_str)?;
            let embedded: Vec<&Chunk> = index.chunks.iter().filter(|c| c.embedding.is_some()).collect();
            if embedded.is_empty() {
                bail!("index metadata says '{}' embeddings but no chunk has one; re-ingest", embed_provider_str);
            }
            let query_vec = embed_texts(client, embed_provider, &[question.to_string()])
                .await?
                .remove(0);
            let mut scored: Vec<(f32, &Chunk)> = embedded
                .into_iter()
                .map(|c| (dense_cosine(&query_vec, c.embedding.as_ref().unwrap()), c))
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            scored.into_iter().take(TOP_K).collect()
        }
        None => {
            let doc_tokens: Vec<Vec<String>> = index.chunks.iter().map(|c| tokenize(&c.text)).collect();
            let idf = compute_idf(&doc_tokens);
            let doc_vectors: Vec<HashMap<String, f32>> =
                doc_tokens.iter().map(|t| tfidf_vector(t, &idf)).collect();
            let query_vector = tfidf_vector(&tokenize(question), &idf);
            let mut scored: Vec<(f32, &Chunk)> = index
                .chunks
                .iter()
                .zip(&doc_vectors)
                .map(|(c, v)| (sparse_cosine(&query_vector, v), c))
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            scored.into_iter().take(TOP_K).collect()
        }
    };

    println!("--- retrieved context ---");
    for (score, chunk) in &top {
        println!("[{:.3}] {}", score, chunk.source);
    }

    let context = top
        .iter()
        .map(|(_, c)| format!("Source: {}\n{}", c.source, c.text))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let answer = generate(client, provider, question, &context).await?;
    println!("\n--- answer ---\n{}", answer);
    Ok(())
}

// ---------- provider dispatch ----------

async fn embed_texts(client: &reqwest::Client, provider: Provider, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    match provider {
        Provider::Ollama => ollama_embed(client, texts).await,
        Provider::HuggingFace => hf_embed(client, texts).await,
        Provider::Claude => bail!("claude has no embeddings API"),
    }
}

async fn generate(client: &reqwest::Client, provider: Provider, question: &str, context: &str) -> Result<String> {
    match provider {
        Provider::Claude => claude_generate(client, question, context).await,
        Provider::Ollama => ollama_generate(client, question, context).await,
        Provider::HuggingFace => hf_generate(client, question, context).await,
    }
}

// ---------- Anthropic (Claude) ----------

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

async fn claude_generate(client: &reqwest::Client, question: &str, context: &str) -> Result<String> {
    let api_key = env::var("ANTHROPIC_API_KEY").context("set ANTHROPIC_API_KEY")?;
    let user = format!("Context:\n{}\n\nQuestion: {}", context, question);

    let body = AnthropicRequest {
        model: CLAUDE_MODEL.to_string(),
        max_tokens: 1024,
        system: SYSTEM_PROMPT.to_string(),
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
        bail!("claude request failed: {} - {}", resp.status(), resp.text().await?);
    }
    let parsed: AnthropicResponse = resp.json().await?;
    Ok(parsed
        .content
        .into_iter()
        .find(|b| b.block_type == "text")
        .map(|b| b.text)
        .unwrap_or_default())
}

// ---------- Ollama ----------

fn ollama_host() -> String {
    env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn ollama_embed_model() -> String {
    env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string())
}

fn ollama_gen_model() -> String {
    env::var("OLLAMA_GEN_MODEL").unwrap_or_else(|_| "llama3.2".to_string())
}

#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: String,
    input: &'a [String],
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

async fn ollama_embed(client: &reqwest::Client, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let body = OllamaEmbedRequest { model: ollama_embed_model(), input: texts };
    let resp = client
        .post(format!("{}/api/embed", ollama_host()))
        .json(&body)
        .send()
        .await
        .context("connecting to ollama; is `ollama serve` running?")?;
    if !resp.status().is_success() {
        bail!("ollama embed request failed: {} - {}", resp.status(), resp.text().await?);
    }
    let parsed: OllamaEmbedResponse = resp.json().await?;
    Ok(parsed.embeddings)
}

#[derive(Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    system: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

async fn ollama_generate(client: &reqwest::Client, question: &str, context: &str) -> Result<String> {
    let body = OllamaGenerateRequest {
        model: ollama_gen_model(),
        prompt: format!("Context:\n{}\n\nQuestion: {}", context, question),
        system: SYSTEM_PROMPT.to_string(),
        stream: false,
    };
    let resp = client
        .post(format!("{}/api/generate", ollama_host()))
        .json(&body)
        .send()
        .await
        .context("connecting to ollama; is `ollama serve` running?")?;
    if !resp.status().is_success() {
        bail!("ollama generate request failed: {} - {}", resp.status(), resp.text().await?);
    }
    let parsed: OllamaGenerateResponse = resp.json().await?;
    Ok(parsed.response)
}

// ---------- Hugging Face ----------

fn hf_embed_model() -> String {
    env::var("HF_EMBED_MODEL").unwrap_or_else(|_| "sentence-transformers/all-MiniLM-L6-v2".to_string())
}

fn hf_gen_model() -> String {
    env::var("HF_GEN_MODEL").unwrap_or_else(|_| "HuggingFaceH4/zephyr-7b-beta".to_string())
}

fn hf_token() -> Result<String> {
    env::var("HF_API_TOKEN").context("set HF_API_TOKEN")
}

async fn hf_embed(client: &reqwest::Client, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let token = hf_token()?;
    let url = format!("https://api-inference.huggingface.co/models/{}", hf_embed_model());
    let resp = client
        .post(&url)
        .bearer_auth(token)
        .json(&serde_json::json!({ "inputs": texts }))
        .send()
        .await?;
    if !resp.status().is_success() {
        bail!("huggingface embed request failed: {} - {}", resp.status(), resp.text().await?);
    }
    let parsed: Vec<Vec<f32>> = resp
        .json()
        .await
        .context("unexpected huggingface embedding response shape; the model must return one pooled vector per input")?;
    Ok(parsed)
}

#[derive(Deserialize)]
struct HfGenerationItem {
    generated_text: String,
}

async fn hf_generate(client: &reqwest::Client, question: &str, context: &str) -> Result<String> {
    let token = hf_token()?;
    let url = format!("https://api-inference.huggingface.co/models/{}", hf_gen_model());
    let prompt = format!(
        "{}\n\nContext:\n{}\n\nQuestion: {}\n\nAnswer:",
        SYSTEM_PROMPT, context, question
    );
    let body = serde_json::json!({
        "inputs": prompt,
        "parameters": { "max_new_tokens": 512, "return_full_text": false }
    });
    let resp = client.post(&url).bearer_auth(token).json(&body).send().await?;
    if !resp.status().is_success() {
        bail!("huggingface generate request failed: {} - {}", resp.status(), resp.text().await?);
    }
    let parsed: Vec<HfGenerationItem> = resp.json().await?;
    Ok(parsed.into_iter().next().map(|x| x.generated_text).unwrap_or_default())
}

// ---------- index persistence ----------

fn load_index() -> Result<Index> {
    let data = fs::read_to_string(INDEX_FILE)?;
    Ok(serde_json::from_str(&data)?)
}

fn save_index(index: &Index) -> Result<()> {
    let data = serde_json::to_string(index)?;
    fs::write(INDEX_FILE, data)?;
    Ok(())
}
