//! Semantic search with EmbeddingGemma 2, on-device.
//!
//! Loads a local embedding bundle once, embeds a small help-centre corpus
//! (English, French, Spanish, German, Japanese and two code snippets), then
//! ranks each query's best documents by cosine similarity. Queries in one
//! language find answers written in another.
//!
//! Run with the built-in corpus and queries:
//!   ./integration-tests/download.sh embeddinggemma-2
//!   cargo run --release -p xybrid-sdk --example semantic_search
//!
//! Run with your own documents (one per line) and queries:
//!   cargo run --release -p xybrid-sdk --example semantic_search -- \
//!     --docs my_docs.txt "first question" "second question"
//!
//! Options:
//!   --model-dir DIR   embedding bundle (default: the embeddinggemma-2 fixture)
//!   --docs FILE       documents, one per line (default: built-in sample)
//!   --top K           results per query (default: 3)
//!   --no-prompts      skip EmbeddingGemma's prompts, for other embedding models
//!
//! EmbeddingGemma expects a prompt before the text. Documents get
//! `title: none | text: ` and queries `task: search result | query: `,
//! unless the text already starts with `title:` or `task:`, so a code query
//! can use `task: code retrieval | query: …` instead.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::model::{ModelLoader, XybridModel};

const DOCUMENT_PROMPT: &str = "title: none | text: ";
const QUERY_PROMPT: &str = "task: search result | query: ";

const SAMPLE_DOCUMENTS: &[&str] = &[
    "To reset your password, open Settings, choose Account, then tap Reset password. We email you a link that stays valid for 24 hours.",
    "Refunds go back to the original payment method within 5 to 10 business days after we receive the returned item.",
    "Our support team answers by chat and email, Monday to Friday, 9:00 to 18:00 CET.",
    "Pour exporter vos données, ouvrez Paramètres > Confidentialité et choisissez « Exporter mes données ». Vous recevrez un fichier ZIP.",
    "Para cancelar tu suscripción, ve a Configuración > Plan y pulsa «Cancelar». Mantienes el acceso hasta el final del periodo.",
    "Die Zwei-Faktor-Authentifizierung schützt Ihr Konto: Nach dem Passwort geben Sie einen Code aus Ihrer Authentifizierungs-App ein.",
    "オフラインモードでは、ダウンロード済みのモデルだけでアプリを使えます。ネットワーク接続は必要ありません。",
    "Conversation history stays on the device and is never uploaded unless you turn on cloud sync.",
    "Battery saver lowers the screen refresh rate and pauses background sync when the charge drops below 20%.",
    "fn reverse(s: &str) -> String { s.chars().rev().collect() }",
    "def is_palindrome(text):\n    cleaned = ''.join(c.lower() for c in text if c.isalnum())\n    return cleaned == cleaned[::-1]",
    "Bananas are rich in potassium and are a popular breakfast fruit.",
];

const SAMPLE_QUERIES: &[&str] = &[
    "I forgot my password",
    "Comment obtenir un remboursement ?",
    "How do I download a copy of my data?",
    "¿La app funciona sin internet?",
    "how do I secure my account with an extra code",
    "task: code retrieval | query: check whether a string reads the same forwards and backwards",
];

struct Options {
    model_dir: PathBuf,
    documents: Vec<String>,
    queries: Vec<String>,
    top: usize,
    prompts: bool,
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
    let mut options = Options {
        model_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../integration-tests/fixtures/models/embeddinggemma-2"),
        documents: SAMPLE_DOCUMENTS.iter().map(|doc| doc.to_string()).collect(),
        queries: Vec::new(),
        top: 3,
        prompts: true,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => {
                options.model_dir = args.next().ok_or("--model-dir needs a directory")?.into();
            }
            "--docs" => {
                let path = args.next().ok_or("--docs needs a file")?;
                let contents = std::fs::read_to_string(&path)
                    .map_err(|e| format!("cannot read {path}: {e}"))?;
                options.documents = contents
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_string)
                    .collect();
            }
            "--top" => {
                options.top = args.next().ok_or("--top needs a number")?.parse()?;
            }
            "--no-prompts" => options.prompts = false,
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}").into()),
            _ => options.queries.push(arg),
        }
    }
    if options.queries.is_empty() {
        options.queries = SAMPLE_QUERIES
            .iter()
            .map(|query| query.to_string())
            .collect();
    }
    if options.documents.is_empty() {
        return Err("no documents to search".into());
    }
    Ok(options)
}

/// Adds `prompt` unless prompts are off or the text already carries one.
fn with_prompt(text: &str, prompt: &str, prompts: bool) -> String {
    if !prompts || text.starts_with("title:") || text.starts_with("task:") {
        text.to_string()
    } else {
        format!("{prompt}{text}")
    }
}

fn embed(model: &XybridModel, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let input = Envelope::new(EnvelopeKind::Text(text.to_string()));
    let result = model.run(&input, None)?;
    let embedding = result
        .embedding()
        .ok_or("the model returned no embedding")?;
    Ok(embedding.to_vec())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

/// The first line of `text`, cut to `width` characters.
fn preview(text: &str, width: usize) -> String {
    let line = text.lines().next().unwrap_or_default();
    if line.chars().count() > width {
        format!("{}…", line.chars().take(width).collect::<String>())
    } else {
        line.to_string()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_options()?;

    let start = Instant::now();
    let model = ModelLoader::from_directory(&options.model_dir)?.load()?;
    // The first run builds the ONNX session and parses the tokenizer; time it
    // separately so the per-text numbers below are steady-state.
    embed(
        &model,
        &with_prompt("warm-up", QUERY_PROMPT, options.prompts),
    )?;
    let model_dir = options
        .model_dir
        .canonicalize()
        .unwrap_or_else(|_| options.model_dir.clone());
    println!(
        "Loaded {} in {:.1} s",
        model_dir.display(),
        start.elapsed().as_secs_f32()
    );

    let start = Instant::now();
    let documents = options
        .documents
        .iter()
        .map(|doc| embed(&model, &with_prompt(doc, DOCUMENT_PROMPT, options.prompts)))
        .collect::<Result<Vec<_>, _>>()?;
    let elapsed = start.elapsed();
    println!(
        "Embedded {} documents ({} dimensions) in {:.0} ms, {:.0} ms each\n",
        documents.len(),
        documents[0].len(),
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / documents.len() as f64
    );

    for query in &options.queries {
        let start = Instant::now();
        let query_embedding = embed(&model, &with_prompt(query, QUERY_PROMPT, options.prompts))?;
        let query_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut ranked: Vec<(f32, &str)> = documents
            .iter()
            .zip(&options.documents)
            .map(|(embedding, doc)| (cosine(&query_embedding, embedding), doc.as_str()))
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

        println!("Query: {query}  ({query_ms:.0} ms)");
        for (score, doc) in ranked.iter().take(options.top) {
            println!("  {score:.3}  {}", preview(doc, 90));
        }
        println!();
    }
    Ok(())
}
