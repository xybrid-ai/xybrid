//! `xybrid cache` command handler.

use anyhow::{Context, Result};

use super::types::CacheCommand;
use super::utils::format_size;
use crate::ui;

/// Handle `xybrid cache` subcommands.
pub(crate) fn handle_cache_command(command: CacheCommand) -> Result<()> {
    let mut client = xybrid_sdk::registry_client::RegistryClient::from_env()
        .context("Failed to initialize registry client")?;

    match command {
        CacheCommand::List => list_cache(&client),
        CacheCommand::Status => show_cache_status(&client),
        CacheCommand::Clear { model_id } => clear_cache(&mut client, model_id),
    }
}

fn list_cache(client: &xybrid_sdk::registry_client::RegistryClient) -> Result<()> {
    ui::header("Model Cache");

    let entries = client
        .cache_entries()
        .context("Failed to list cache entries")?;
    let total_size: u64 = entries.iter().map(|entry| entry.size_bytes).sum();

    ui::kv("Root", &client.cache_root().display().to_string());
    println!();

    if entries.is_empty() {
        ui::hint("Cache is empty.");
        ui::hint("Use 'xybrid fetch --model <id>' to download models.");
        return Ok(());
    }

    let mut table = ui::Table::new(vec!["Model", "Location", "Size"]);
    for entry in &entries {
        let size = format_size(entry.size_bytes);
        table.row(vec![
            entry.model_id.as_str(),
            entry.location.as_str(),
            size.as_str(),
        ]);
    }
    table.print();

    ui::footer(&format!(
        "{} entries · {}",
        entries.len(),
        xybrid_sdk::registry_client::CacheStats::format_size(total_size)
    ));

    Ok(())
}

fn show_cache_status(client: &xybrid_sdk::registry_client::RegistryClient) -> Result<()> {
    ui::header("Cache Status");

    let stats = client.cache_stats().context("Failed to get cache stats")?;

    ui::panel(&[
        format!(
            "{}  {}",
            ui::dim("Models"),
            ui::value(&stats.model_count.to_string())
        ),
        format!(
            "{}    {}",
            ui::dim("Size"),
            ui::value(&stats.total_size_human())
        ),
        format!(
            "{}    {}",
            ui::dim("Path"),
            ui::dim(&stats.cache_root().display().to_string())
        ),
    ]);

    if !stats.cache_root().exists() {
        println!();
        ui::hint("Cache root does not exist yet.");
        ui::hint("It will be created when you download your first model.");
    }

    println!();

    Ok(())
}

fn clear_cache(
    client: &mut xybrid_sdk::registry_client::RegistryClient,
    model_id: Option<String>,
) -> Result<()> {
    if let Some(id) = model_id {
        ui::header(&format!("Clear Cache · {}", id));

        let removed = client
            .clear_cache(&id)
            .context(format!("Failed to clear cache for '{}'", id))?;

        if removed == 0 {
            ui::warning(&format!("No cached data found for model '{}'", id));
        } else {
            ui::ok(&format!("Cache cleared for model '{}'", id));
        }
    } else {
        ui::header("Clear All Cache");
        println!();
        ui::warning("This will delete ALL cached models.");
        ui::hint("Press Enter to continue or Ctrl+C to cancel...");

        let mut input = String::new();
        // EOF / closed stdin (piped input, /dev/null, CI) must not count as
        // confirmation — a destructive clear requires an explicit keypress.
        if std::io::stdin().read_line(&mut input).unwrap_or(0) == 0 {
            ui::warning("Aborted: no interactive confirmation.");
            return Ok(());
        }

        let removed = client.clear_all_cache().context("Failed to clear cache")?;

        if removed == 0 {
            ui::warning("No cached models found");
        } else {
            ui::ok("All cached models cleared");
        }
    }

    println!();

    Ok(())
}
