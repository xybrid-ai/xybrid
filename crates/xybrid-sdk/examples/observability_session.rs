//! SDK sessions and manual operations, without downloading or running a model.
//! Set an OTEL_EXPORTER_OTLP_* endpoint to export; otherwise stays local.
use std::time::Duration;
use xybrid_sdk::observability::{
    self, ContentPolicy, InferenceRequest, Location, ObservabilityConfig, ObservabilitySession,
    ObservationOptions, Observer, Usage,
};
use xybrid_sdk::SdkError;

fn main() -> Result<(), SdkError> {
    // Keep the provider in the application: borrowed sessions never flush or
    // shut it down. With no exporter this still creates useful offline refs.
    let local_provider = opentelemetry_sdk::trace::SdkTracerProvider::builder().build();
    let session = match ObservabilityConfig::from_env() {
        Some(config) => ObservabilitySession::new(config.content(ContentPolicy::metadata_only()))?,
        None => ObservabilitySession::borrowed(
            Observer::builder().tracer_provider(&local_provider).build(),
        ),
    };
    let controls = ObservationOptions::default().with_session(&session);
    let workflow = observability::start_workflow("application-workflow", &controls);
    // Explicit parents are safe across async/thread/FFI hops; never hold enter()
    // guards across await. with_session pins the original session independently
    // of changes to the process default.
    let child = match workflow.context() {
        Some(parent) => controls.clone().with_parent(parent),
        None => controls,
    };
    let inference = observability::start_inference(
        InferenceRequest::new("generate_content", "application-runtime")
            .with_model("example-model")
            .with_location(Location::Local),
        &child,
    );
    // A future runtime adapter records actual measurements, never estimates.
    inference.record_usage(&Usage::tokens(Some(12), None));
    inference.end();
    workflow.end();
    println!("inference reference: {:?}", inference.observation_ref());
    println!("delivery counters: {:?}", session.stats());
    session.shutdown(Duration::from_secs(5))?;
    local_provider.shutdown().map_err(|error| {
        SdkError::ConfigError(format!("application provider shutdown failed: {error}"))
    })
}
