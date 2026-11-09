//! Choice Scoring Example
//!
//! Scores two form actions against one page context with CUA-S1-FORMS, a
//! choice scorer. The scorer reads the context and each candidate's text,
//! appends its fixed choices (`check`, `click`, `skip`) and returns one score
//! per candidate. Scores rank the offered candidates against each other; they
//! are not calibrated probabilities.
//!
//! The model is regenerated from its pinned checkpoint rather than downloaded:
//!
//!   ./integration-tests/download.sh cua-s1-forms
//!
//! Run with:
//!   cargo run -p xybrid-sdk --example choice_scoring [-- <model-dir>]

use std::path::PathBuf;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::{Choice, ModelLoader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args().nth(1).map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../integration-tests/fixtures/models/cua-s1-forms")
        },
        PathBuf::from,
    );
    let model = ModelLoader::from_directory(&model_dir)?.load()?;
    // Loads the ONNX model and checks it against the scorer spec.
    model.warmup()?;

    let context = "TASK fill the form from the document, then submit\n\
                   FORM Intake\n\
                   ELEMENT Edit \"Phone\" value=\"\"";
    let request = Envelope::choice_request(
        Envelope::new(EnvelopeKind::Text(context.to_string())),
        vec![
            Choice::new("tel", "fill Tel: (503) 555-0142"),
            Choice::new("name", "fill Name: Jane Doe"),
        ],
    );
    let result = model.run(&request, None)?;
    let scores = result
        .choice_scores()
        .ok_or("a choice scorer returns choice scores")?;

    let best = scores.best();
    for (index, entry) in scores.entries.iter().enumerate() {
        println!(
            "{} {:<5} score {:.3}  logit {:>7.3}{}",
            if best == Some(index) { "*" } else { " " },
            entry.id,
            entry.score,
            entry.logit,
            if entry.fixed { "  (fixed)" } else { "" },
        );
    }
    if let Some(margin) = scores.margin() {
        println!("lead over the runner-up: {margin:.3}");
    }
    println!("{} ms", result.latency_ms());
    Ok(())
}
