//! Simple example demonstrating the SDK's run_pipeline function.

use xybrid_sdk::run_pipeline;

fn main() {
    println!("🚀 Xybrid SDK Example: Running Pipeline\n");

    match run_pipeline("xybrid-cli/examples/hiiipe.yaml") {
        Ok(result) => {
            println!("✅ Pipeline completed successfully!");
            println!();

            if let Some(name) = &result.name {
                println!("Pipeline: {}", name);
            }

            println!("Total latency: {}ms", result.total_latency_ms);
            println!("Final output: {}", result.final_output);
            println!();

            println!("Stage timings:");
            for (i, stage) in result.stages.iter().enumerate() {
                println!(
                    "  {}. {}: {}ms ({})",
                    i + 1,
                    stage.name,
                    stage.latency_ms,
                    stage.target
                );
            }
        }
        Err(e) => {
            eprintln!("❌ Pipeline failed: {}", e);
            std::process::exit(1);
        }
    }
}
