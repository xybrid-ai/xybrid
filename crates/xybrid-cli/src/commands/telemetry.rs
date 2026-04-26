//! `xybrid telemetry` command handlers.
//!
//! Reports the runtime status of registry telemetry — whether the
//! `X-Xybrid-Client` header is being sent on outbound registry calls or
//! suppressed by the `XYBRID_TELEMETRY_OPTOUT` env var.

use anyhow::Result;
use clap::Subcommand;

/// Subcommands for `xybrid telemetry`.
#[derive(Subcommand)]
pub(crate) enum TelemetryCommand {
    /// Report whether registry telemetry is enabled or opted out.
    ///
    /// Reads the cached `XYBRID_TELEMETRY_OPTOUT` env var (truthy values:
    /// `1`, `true`, `yes`, case-insensitive) and prints a single status line.
    /// See repos/xybrid/docs/telemetry/registry.md for what the
    /// `X-Xybrid-Client` header carries.
    Status,
}

/// Handle `xybrid telemetry` subcommands.
pub(crate) fn handle_telemetry_command(command: TelemetryCommand) -> Result<()> {
    match command {
        TelemetryCommand::Status => status(),
    }
}

fn status() -> Result<()> {
    if xybrid_sdk::is_telemetry_opted_out() {
        println!("registry telemetry: disabled (XYBRID_TELEMETRY_OPTOUT=1)");
    } else {
        println!("registry telemetry: enabled");
    }
    Ok(())
}
