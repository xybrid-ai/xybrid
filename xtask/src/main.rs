mod setup_env;

use clap::{Parser, Subcommand};
use anyhow::Result;

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Development tasks for Xybrid", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Setup the integration test environment (download models etc)
    SetupTestEnv {
        /// Registry URL to download models from
        #[arg(long)]
        registry: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::SetupTestEnv { registry } => {
            setup_env::run(registry)?;
        }
    }
    
    Ok(())
}
