use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

const CLAUDE_MODEL: &str = "claude-sonnet-5";
const INDEX_FILE: &str = "index.bin";
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
    dotenvy::dotenv().ok();

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
        Some("chat") => {
            if provider != Provider::Ollama {
                bail!("chat currently only supports --provider ollama");
            }
            chat_repl(&client).await?;
        }
        _ => {
            println!(
                "usage:\n  \
                bb_rag ingest <file-or-dir> [--provider claude|ollama|huggingface]\n  \
                bb_rag query <question> [--provider claude|ollama|huggingface]\n  \
                bb_rag chat [--provider ollama]\n\n\
                claude has no embeddings API, so it always falls back to local TF-IDF retrieval.\n\
                ollama and huggingface compute real embeddings at ingest time and reuse them at query time.\n\
                chat is a streaming, multi-turn REPL; currently ollama-only."
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
        bail!("no .txt, .md, or .pdf files found at {}", path.display());
    }

    let mut texts = Vec::new();
    let mut sources = Vec::new();
    for file in &files {
        let content = read_file_text(file)?;
        for chunk in chunk_text(&content, CHUNK_SIZE, CHUNK_OVERLAP) {
            sources.push(file.display().to_string());
            texts.push(chunk);
        }
    }

    let embeddings: Option<Vec<Vec<f32>>> = match provider {
        Provider::Claude => None,
        Provider::Ollama | Provider::HuggingFace => Some(embed_texts(client, provider, &texts).await?),
    };

    let mut index = load_index(Path::new(INDEX_FILE)).unwrap_or_default();
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
    save_index(&index, Path::new(INDEX_FILE))?;
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
    out.retain(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("txt") | Some("md") | Some("pdf")));
    Ok(out)
}

fn read_file_text(path: &Path) -> Result<String> {
    if path.extension().and_then(|e| e.to_str()) == Some("pdf") {
        pdf_extract::extract_text(path).with_context(|| {
            format!(
                "extracting text from {}; scanned/image-only PDFs have no text layer to extract",
                path.display()
            )
        })
    } else {
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))
    }
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

fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences = Vec::new();
    let mut current = String::new();
    for (i, &c) in chars.iter().enumerate() {
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            let at_boundary = chars.get(i + 1).is_none_or(char::is_ascii_whitespace);
            if at_boundary {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string());
                }
                current.clear();
            }
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }
    sentences
}

/// Keeps whole trailing sentences (not partial ones) totaling up to `budget`
/// chars, always keeping at least the last sentence so every chunk after the
/// first has some continuity with the one before it.
fn take_overlap(sentences: &[String], budget: usize) -> (Vec<String>, usize) {
    if budget == 0 || sentences.is_empty() {
        return (Vec::new(), 0);
    }
    let mut carried = Vec::new();
    let mut len = 0;
    for sentence in sentences.iter().rev() {
        let sentence_len = sentence.chars().count();
        if !carried.is_empty() && len + 1 + sentence_len > budget {
            break;
        }
        len += if carried.is_empty() { sentence_len } else { sentence_len + 1 };
        carried.insert(0, sentence.clone());
    }
    (carried, len)
}

/// Fixed-size character window, only used when a single "sentence" (e.g. text
/// with no sentence-ending punctuation) is bigger than the whole chunk budget.
fn hard_split(text: &str, size: usize, overlap: usize) -> Vec<String> {
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
        start += size.saturating_sub(overlap).max(1);
    }
    chunks
}

fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_len = 0;

    for sentence in sentences {
        let sentence_len = sentence.chars().count();

        if sentence_len > size {
            if !current.is_empty() {
                chunks.push(current.join(" "));
                current.clear();
                current_len = 0;
            }
            chunks.extend(hard_split(&sentence, size, overlap));
            continue;
        }

        if !current.is_empty() && current_len + 1 + sentence_len > size {
            chunks.push(current.join(" "));
            let (carried, carried_len) = take_overlap(&current, overlap);
            current = carried;
            current_len = carried_len;
        }

        current_len += if current.is_empty() { sentence_len } else { sentence_len + 1 };
        current.push(sentence);
    }

    if !current.is_empty() {
        chunks.push(current.join(" "));
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

// ---------- retrieval (shared by query and chat) ----------

async fn retrieve<'a>(client: &reqwest::Client, index: &'a Index, question: &str) -> Result<Vec<(f32, &'a Chunk)>> {
    let top = match &index.embedding_provider {
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
    Ok(top)
}

fn print_sources(top: &[(f32, &Chunk)]) {
    println!("--- sources ---");
    for (score, chunk) in top {
        println!("[{:.3}] {}", score, chunk.source);
    }
}

fn build_context(top: &[(f32, &Chunk)]) -> String {
    top.iter()
        .map(|(_, c)| format!("Source: {}\n{}", c.source, c.text))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n")
}

// ---------- query ----------

async fn query(client: &reqwest::Client, question: &str, provider: Provider) -> Result<()> {
    let index = load_index(Path::new(INDEX_FILE)).context("no index found; run `bb_rag ingest` first")?;
    if index.chunks.is_empty() {
        bail!("index is empty; run `bb_rag ingest` first");
    }

    let top = retrieve(client, &index, question).await?;
    print_sources(&top);
    let context = build_context(&top);

    let answer = generate(client, provider, question, &context).await?;
    println!("\n--- answer ---\n{}", answer);
    Ok(())
}

// ---------- chat (streaming, multi-turn REPL; ollama only for now) ----------

async fn chat_repl(client: &reqwest::Client) -> Result<()> {
    let index = load_index(Path::new(INDEX_FILE)).context("no index found; run `bb_rag ingest` first")?;
    if index.chunks.is_empty() {
        bail!("index is empty; run `bb_rag ingest` first");
    }

    println!("bb_rag chat (ollama) — type 'exit' or Ctrl-D to leave.\n");

    let mut history: Vec<OllamaChatMessage> = Vec::new();
    let stdin = io::stdin();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let question = line.trim();
        if question.is_empty() {
            continue;
        }
        if question == "exit" || question == "quit" {
            break;
        }

        let top = retrieve(client, &index, question).await?;
        print_sources(&top);
        let context = build_context(&top);

        print!("\n< ");
        io::stdout().flush()?;
        let user_turn = format!("Context:\n{}\n\nQuestion: {}", context, question);
        let answer = ollama_chat_stream(client, &history, &user_turn).await?;
        println!("\n");

        history.push(OllamaChatMessage { role: "user".to_string(), content: user_turn });
        history.push(OllamaChatMessage { role: "assistant".to_string(), content: answer });
    }

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

#[derive(Serialize, Clone)]
struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: String,
    messages: Vec<&'a OllamaChatMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaChatStreamLine {
    #[serde(default)]
    message: Option<OllamaChatStreamMessage>,
    #[serde(default)]
    done: bool,
}

#[derive(Deserialize)]
struct OllamaChatStreamMessage {
    #[serde(default)]
    content: String,
}

/// Streams the assistant's reply token-by-token to stdout and returns the
/// full text once done, so callers can append it to conversation history.
async fn ollama_chat_stream(client: &reqwest::Client, history: &[OllamaChatMessage], user_turn: &str) -> Result<String> {
    let system = OllamaChatMessage { role: "system".to_string(), content: SYSTEM_PROMPT.to_string() };
    let latest = OllamaChatMessage { role: "user".to_string(), content: user_turn.to_string() };
    let mut messages: Vec<&OllamaChatMessage> = vec![&system];
    messages.extend(history.iter());
    messages.push(&latest);

    let body = OllamaChatRequest { model: ollama_gen_model(), messages, stream: true };

    let resp = client
        .post(format!("{}/api/chat", ollama_host()))
        .json(&body)
        .send()
        .await
        .context("connecting to ollama; is `ollama serve` running?")?;
    if !resp.status().is_success() {
        bail!("ollama chat request failed: {} - {}", resp.status(), resp.text().await?);
    }

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();
    let mut answer = String::new();
    let mut stdout = io::stdout();

    'outer: while let Some(chunk) = stream.next().await {
        buffer.push_str(&String::from_utf8_lossy(&chunk?));
        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer.drain(..=pos);
            if line.is_empty() {
                continue;
            }
            let parsed: OllamaChatStreamLine =
                serde_json::from_str(&line).with_context(|| format!("parsing ollama stream line: {line}"))?;
            if let Some(msg) = parsed.message {
                if !msg.content.is_empty() {
                    print!("{}", msg.content);
                    stdout.flush().ok();
                    answer.push_str(&msg.content);
                }
            }
            if parsed.done {
                break 'outer;
            }
        }
    }

    Ok(answer)
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

fn load_index(path: &Path) -> Result<Index> {
    let data = fs::read(path)?;
    Ok(bincode::deserialize(&data)?)
}

fn save_index(index: &Index, path: &Path) -> Result<()> {
    let data = bincode::serialize(index)?;
    fs::write(path, data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal single-page PDF containing "Hello PDF world", with
    /// correctly computed xref byte offsets (pdf-extract doesn't recover from
    /// a bogus xref table, so these have to be real).
    fn build_minimal_pdf() -> Vec<u8> {
        let stream: &[u8] = b"BT /F1 24 Tf 10 100 Td (Hello PDF world) Tj ET";
        let objs: Vec<Vec<u8>> = vec![
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".to_vec(),
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n".to_vec(),
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 200 200] /Contents 5 0 R >>\nendobj\n".to_vec(),
            b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".to_vec(),
            {
                let mut o = format!("5 0 obj\n<< /Length {} >>\nstream\n", stream.len()).into_bytes();
                o.extend_from_slice(stream);
                o.extend_from_slice(b"\nendstream\nendobj\n");
                o
            },
        ];

        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for o in &objs {
            offsets.push(out.len());
            out.extend_from_slice(o);
        }

        let xref_start = out.len();
        let n = objs.len() + 1;
        let mut xref = format!("xref\n0 {n}\n0000000000 65535 f \n").into_bytes();
        for off in &offsets {
            xref.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(&xref);
        out.extend_from_slice(
            format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n").as_bytes(),
        );
        out
    }

    // ---------- chunk_text ----------

    #[test]
    fn chunk_text_empty_input() {
        assert_eq!(chunk_text("", 10, 2), Vec::<String>::new());
    }

    #[test]
    fn chunk_text_shorter_than_size_is_one_chunk() {
        assert_eq!(chunk_text("  hello world  ", 100, 10), vec!["hello world"]);
    }

    #[test]
    fn chunk_text_drops_whitespace_only_pieces() {
        let chunks = chunk_text("a   ", 1, 0);
        assert!(chunks.iter().all(|c| !c.trim().is_empty()));
    }

    #[test]
    fn chunk_text_never_splits_mid_sentence_when_sentences_fit() {
        let text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india.";
        let chunks = chunk_text(text, 42, 0);
        assert_eq!(
            chunks,
            vec!["Alpha bravo charlie. Delta echo foxtrot.", "Golf hotel india."]
        );
        for chunk in &chunks {
            assert!(chunk.ends_with(['.', '!', '?']), "chunk cut mid-sentence: {chunk:?}");
        }
    }

    #[test]
    fn chunk_text_overlap_repeats_a_whole_trailing_sentence() {
        let text = "One two three. Four five six. Seven eight nine.";
        let chunks = chunk_text(text, 30, 15);
        assert!(chunks.len() >= 2);
        assert!(chunks[1].starts_with("Four five six."));
    }

    #[test]
    fn chunk_text_falls_back_to_hard_split_for_unpunctuated_run_on_text() {
        // no sentence-ending punctuation anywhere, so it can't be packed by sentence
        let text = "0123456789abcdefghij";
        let chunks = chunk_text(text, 10, 3);
        assert_eq!(chunks, vec!["0123456789", "789abcdefg", "efghij"]);
    }

    // ---------- split_sentences ----------

    #[test]
    fn split_sentences_splits_on_terminal_punctuation() {
        assert_eq!(
            split_sentences("Hello world. How are you? Fine!"),
            vec!["Hello world.", "How are you?", "Fine!"]
        );
    }

    #[test]
    fn split_sentences_keeps_text_without_trailing_punctuation() {
        assert_eq!(split_sentences("just one clause"), vec!["just one clause"]);
    }

    #[test]
    fn split_sentences_empty_input_is_empty() {
        assert_eq!(split_sentences(""), Vec::<String>::new());
    }

    // ---------- take_overlap ----------

    #[test]
    fn take_overlap_carries_trailing_sentences_within_budget() {
        let sentences = vec![
            "Short one.".to_string(),
            "A bit longer sentence.".to_string(),
            "Tiny.".to_string(),
        ];
        let (carried, len) = take_overlap(&sentences, 10);
        assert_eq!(carried, vec!["Tiny.".to_string()]);
        assert_eq!(len, 5);
    }

    #[test]
    fn take_overlap_always_keeps_at_least_the_last_sentence() {
        let sentences = vec!["This one is definitely longer than the overlap budget.".to_string()];
        let (carried, _) = take_overlap(&sentences, 5);
        assert_eq!(carried, sentences);
    }

    #[test]
    fn take_overlap_zero_budget_carries_nothing() {
        assert_eq!(take_overlap(&["a.".to_string()], 0), (Vec::new(), 0));
    }

    // ---------- hard_split ----------

    #[test]
    fn hard_split_windows_with_overlap() {
        let chunks = hard_split("0123456789abcdefghij", 10, 3);
        assert_eq!(chunks, vec!["0123456789", "789abcdefg", "efghij"]);
    }

    #[test]
    fn hard_split_overlap_ge_size_does_not_hang() {
        let chunks = hard_split("abcdef", 2, 5);
        assert!(!chunks.is_empty());
    }

    // ---------- tokenize ----------

    #[test]
    fn tokenize_lowercases_and_splits_punctuation() {
        assert_eq!(tokenize("Hello, World!"), vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_drops_single_char_tokens() {
        assert_eq!(tokenize("a bb c dd"), vec!["bb", "dd"]);
    }

    // ---------- term_freq / idf / tfidf ----------

    #[test]
    fn term_freq_normalizes_by_length() {
        let tf = term_freq(&["a".into(), "a".into(), "b".into(), "b".into()]);
        assert_eq!(tf.get("a").copied(), Some(0.5));
        assert_eq!(tf.get("b").copied(), Some(0.5));
    }

    #[test]
    fn compute_idf_rare_term_scores_higher_than_common_term() {
        let docs = vec![
            vec!["common".to_string(), "rare".to_string()],
            vec!["common".to_string()],
            vec!["common".to_string()],
        ];
        let idf = compute_idf(&docs);
        assert!(idf["rare"] > idf["common"]);
    }

    #[test]
    fn tfidf_vector_combines_tf_and_idf() {
        let mut idf = HashMap::new();
        idf.insert("x".to_string(), 2.0);
        let v = tfidf_vector(&["x".to_string()], &idf);
        assert_eq!(v.get("x").copied(), Some(2.0));
    }

    // ---------- cosine similarity ----------

    #[test]
    fn sparse_cosine_identical_vectors_is_one() {
        let mut a = HashMap::new();
        a.insert("x".to_string(), 1.0);
        a.insert("y".to_string(), 2.0);
        assert!((sparse_cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sparse_cosine_disjoint_keys_is_zero() {
        let mut a = HashMap::new();
        a.insert("x".to_string(), 1.0);
        let mut b = HashMap::new();
        b.insert("y".to_string(), 1.0);
        assert_eq!(sparse_cosine(&a, &b), 0.0);
    }

    #[test]
    fn sparse_cosine_empty_is_zero() {
        assert_eq!(sparse_cosine(&HashMap::new(), &HashMap::new()), 0.0);
    }

    #[test]
    fn dense_cosine_identical_vectors_is_one() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((dense_cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dense_cosine_orthogonal_vectors_is_zero() {
        assert_eq!(dense_cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn dense_cosine_zero_vector_is_zero() {
        assert_eq!(dense_cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    // ---------- extract_flag ----------

    #[test]
    fn extract_flag_removes_flag_and_value() {
        let mut args = vec!["query".to_string(), "--provider".to_string(), "ollama".to_string(), "hi".to_string()];
        let value = extract_flag(&mut args, "--provider");
        assert_eq!(value.as_deref(), Some("ollama"));
        assert_eq!(args, vec!["query".to_string(), "hi".to_string()]);
    }

    #[test]
    fn extract_flag_missing_returns_none() {
        let mut args = vec!["query".to_string(), "hi".to_string()];
        let original = args.clone();
        assert_eq!(extract_flag(&mut args, "--provider"), None);
        assert_eq!(args, original);
    }

    #[test]
    fn extract_flag_at_end_with_no_value_returns_none() {
        let mut args = vec!["query".to_string(), "--provider".to_string()];
        let original = args.clone();
        assert_eq!(extract_flag(&mut args, "--provider"), None);
        assert_eq!(args, original);
    }

    // ---------- Provider ----------

    #[test]
    fn provider_parse_known_names() {
        assert!(Provider::parse("claude").unwrap() == Provider::Claude);
        assert!(Provider::parse("ollama").unwrap() == Provider::Ollama);
        assert!(Provider::parse("huggingface").unwrap() == Provider::HuggingFace);
        assert!(Provider::parse("hf").unwrap() == Provider::HuggingFace);
    }

    #[test]
    fn provider_parse_unknown_name_errs() {
        assert!(Provider::parse("bogus").is_err());
    }

    #[test]
    fn provider_as_str_round_trips_through_parse() {
        for p in [Provider::Claude, Provider::Ollama, Provider::HuggingFace] {
            assert!(Provider::parse(p.as_str()).unwrap() == p);
        }
    }

    // ---------- collect_files ----------

    #[test]
    fn collect_files_filters_extensions_recursively() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "a").unwrap();
        fs::write(dir.path().join("b.md"), "b").unwrap();
        fs::write(dir.path().join("c.png"), "not text").unwrap();
        fs::write(dir.path().join("e.pdf"), b"not a real pdf, just checking the extension filter").unwrap();
        let nested = dir.path().join("nested");
        fs::create_dir(&nested).unwrap();
        fs::write(nested.join("d.txt"), "d").unwrap();

        let files = collect_files(dir.path()).unwrap();

        let mut names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec!["a.txt", "b.md", "d.txt", "e.pdf"]);
    }

    #[test]
    fn collect_files_single_file_with_unsupported_extension_is_filtered_out() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.json");
        fs::write(&file, "{}").unwrap();
        assert_eq!(collect_files(&file).unwrap(), Vec::<PathBuf>::new());
    }

    #[test]
    fn collect_files_missing_path_errs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(collect_files(&dir.path().join("does-not-exist")).is_err());
    }

    // ---------- read_file_text ----------

    #[test]
    fn read_file_text_plain_text_reads_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        fs::write(&file, "hello there").unwrap();
        assert_eq!(read_file_text(&file).unwrap(), "hello there");
    }

    #[test]
    fn read_file_text_extracts_pdf_text() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("doc.pdf");
        fs::write(&file, build_minimal_pdf()).unwrap();
        let text = read_file_text(&file).unwrap();
        assert!(text.contains("Hello PDF world"), "got: {text:?}");
    }

    #[test]
    fn read_file_text_missing_file_errs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_file_text(&dir.path().join("nope.txt")).is_err());
    }

    // ---------- index persistence ----------

    #[test]
    fn save_and_load_index_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.bin");

        let mut index = Index::default();
        index.embedding_provider = Some("ollama".to_string());
        index.chunks.push(Chunk {
            source: "doc.txt".to_string(),
            text: "hello".to_string(),
            embedding: Some(vec![0.1, 0.2, 0.3]),
        });

        save_index(&index, &path).unwrap();
        let loaded = load_index(&path).unwrap();

        assert_eq!(loaded.embedding_provider.as_deref(), Some("ollama"));
        assert_eq!(loaded.chunks.len(), 1);
        assert_eq!(loaded.chunks[0].source, "doc.txt");
        assert_eq!(loaded.chunks[0].embedding, Some(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn load_index_missing_file_errs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_index(&dir.path().join("nope.bin")).is_err());
    }
}
