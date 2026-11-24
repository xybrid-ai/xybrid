//! Registry and OpenTelemetry status without printing credentials or content.
use anyhow::Result;
use clap::Subcommand;
use serde_json::{json, Value};
use xybrid_core::device::ResourceTelemetryMode;
use xybrid_sdk::observability::{ContentPolicy, OtlpConfig, SignalEndpoint};

/// Resource sampling the CLI configures for its observability session.
pub(crate) const CLI_RESOURCE_TELEMETRY: ResourceTelemetryMode = ResourceTelemetryMode::Off;

#[derive(Subcommand)]
pub(crate) enum TelemetryCommand {
    /// Report export and registry telemetry (XYBRID_TELEMETRY_OPTOUT).
    Status {
        /// Print machine-readable status
        #[arg(long)]
        json: bool,
    },
}

fn endpoint(signal: &Option<SignalEndpoint>) -> Value {
    signal.as_ref().map_or(Value::Null, |signal| {
        let origin = url::Url::parse(&signal.endpoint).ok().map(|url| url.origin().ascii_serialization());
        json!({"origin": origin, "protocol": signal.protocol.as_str(), "header_policy": signal.header_policy.as_str()})
    })
}

pub(crate) fn handle_telemetry_command(
    command: TelemetryCommand,
    config: Option<OtlpConfig>,
    explicit: bool,
) -> Result<()> {
    match command {
        TelemetryCommand::Status { json: as_json } => {
            let opted_out = xybrid_sdk::is_telemetry_opted_out();
            let sdk_disabled = crate::otel_sdk_disabled();
            // The same policy the CLI's session uses (environment-resolved).
            let content = ContentPolicy::from_env();
            let signals = config.as_ref().map_or(Value::Null, |config| json!({"traces": endpoint(&config.traces), "metrics": endpoint(&config.metrics), "logs": endpoint(&config.logs)}));
            let status = json!({
                "registry_telemetry": !opted_out,
                "export_configured": config.is_some(), "sdk_disabled": sdk_disabled,
                "export_source": if explicit {"flags"} else if config.is_some() {"OTEL environment"} else {"none"},
                "preset": if explicit {"Xybrid traces only (metrics/logs require generic OTLP)"} else {"generic OTLP"},
                "signals": signals, "content_capture": format!("{:?}", content.capture()),
                "redaction_rules": content.rules().len(), "resource_telemetry": CLI_RESOURCE_TELEMETRY.label(),
                "device_id_set": config.as_ref().is_some_and(|c| c.resource.attributes.iter().any(|kv| kv.key.as_str() == "device.id")),
            });
            if as_json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!(
                    "registry telemetry: {}",
                    if opted_out {
                        "disabled (XYBRID_TELEMETRY_OPTOUT=1)"
                    } else {
                        "enabled"
                    }
                );
                println!("observability: {}", serde_json::to_string_pretty(&status)?);
            }
            Ok(())
        }
    }
}
