//! Decoded-wire regressions (CONTRACT.md "Catalog types on the wire",
//! R-WIRE-1..6) and the header rules at the final transport (R-CFG-1/2),
//! over real HTTP in both OTLP encodings, through the packaged pipeline and
//! through the paired exporters in application-owned providers.
#![cfg(feature = "otlp")]

mod support;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use opentelemetry::logs::{AnyValue, LogRecord as _, Logger as _, LoggerProvider as _};
use opentelemetry::metrics::MeterProvider as _;
use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
use opentelemetry::InstrumentationScope;
use opentelemetry_proto::tonic::common::v1::any_value::Value as Wire;
use opentelemetry_proto::tonic::common::v1::KeyValue as WireKeyValue;
use opentelemetry_proto::tonic::metrics::v1::metric::Data;
use opentelemetry_proto::tonic::metrics::v1::number_data_point::Value as Number;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::logs::{InMemoryLogExporter, LogProcessor, SdkLogRecord, SdkLoggerProvider};
use opentelemetry_sdk::trace::SdkTracerProvider;
use serde_json::{json, Value};
use support::{Received, Server};
use xybrid_observability::otlp::{
    exporters, EnvironmentSource, ObservabilityPipeline, OtlpConfig, Protocol,
};
use xybrid_observability::semconv::{attr, metric, DURATION_BUCKETS, SCOPE_NAME, TOKEN_BUCKETS};
use xybrid_observability::wire::{attribute_type, CatalogType};
use xybrid_observability::{
    ContentCapture, ContentPolicy, Evaluation, InferenceRequest, KeyValue, Location,
    ObservationRef, Observer, ResourceSample, StartOptions, ToolCallInfo, Usage,
};

const PROTOCOLS: [Protocol; 2] = [Protocol::HttpProtobuf, Protocol::HttpJson];

/// An inference whose typed inputs are the R-WIRE-1 values: temperature 0,
/// top_p 1 (integral input), TTFC exactly 1 s, cpu 1, battery 1,
/// time_per_output_token 2, plus an integral double for an `int` key.
fn instrument(observer: &Observer) -> Option<ObservationRef> {
    let t0 = SystemTime::now() - Duration::from_secs(10);
    let op = observer.start_inference(
        InferenceRequest {
            temperature: Some(0.0),
            max_tokens: Some(64),
            server_address: Some("api.example".into()),
            server_port: Some(443),
            ..InferenceRequest::new("chat", "fake")
                .with_model("m")
                .with_location(Location::Remote)
                .streaming(true)
        },
        StartOptions::default().with_start_time(t0),
    );
    op.set_attribute(KeyValue::new(attr::REQUEST_TOP_P, 1_i64));
    op.set_attribute(KeyValue::new(attr::REQUEST_PRESENCE_PENALTY, 0_i64));
    op.set_attribute(KeyValue::new(attr::REQUEST_SEED, 7.0_f64));
    op.record_chunk_at(t0 + Duration::from_secs(1));
    op.record_first_token_at(t0 + Duration::from_secs(1));
    op.record_token_timing(2.0);
    op.record_resource(&ResourceSample {
        cpu_utilization: Some(1.0),
        battery_level: Some(1.0),
        ..ResourceSample::default()
    });
    op.record_usage(&Usage::tokens(Some(2), Some(3)));
    let target = op.observation_ref();
    op.end_at(t0 + Duration::from_secs(3));
    target
}

fn wire(kv: &WireKeyValue) -> Option<&Wire> {
    kv.value.as_ref().and_then(|v| v.value.as_ref())
}

fn find<'a>(attributes: &'a [WireKeyValue], key: &str) -> Option<&'a Wire> {
    attributes.iter().find(|kv| kv.key == key).and_then(wire)
}

/// Every attribute with a catalog `int`/`double` type has that wire type.
fn assert_catalog_types(attributes: &[WireKeyValue], context: &str) -> usize {
    let mut checked = 0;
    for kv in attributes {
        match (attribute_type(&kv.key), wire(kv)) {
            (Some(CatalogType::Double), value) => {
                assert!(
                    matches!(value, Some(Wire::DoubleValue(_))),
                    "{context}: {} is {value:?}",
                    kv.key
                );
                checked += 1;
            }
            (Some(CatalogType::Int), value) => {
                assert!(
                    matches!(value, Some(Wire::IntValue(_))),
                    "{context}: {} is {value:?}",
                    kv.key
                );
                checked += 1;
            }
            _ => {}
        }
    }
    checked
}

fn check_inference_span(requests: &[Received], context: &str) {
    let mut span = None;
    for request in requests {
        for scope in request
            .traces()
            .resource_spans
            .into_iter()
            .flat_map(|r| r.scope_spans)
        {
            if scope.scope.as_ref().map(|s| s.name.as_str()) != Some(SCOPE_NAME) {
                continue;
            }
            for s in scope.spans {
                assert_catalog_types(&s.attributes, context);
                for event in &s.events {
                    assert_catalog_types(&event.attributes, context);
                }
                if find(&s.attributes, attr::REQUEST_TEMPERATURE).is_some() {
                    span = Some(s);
                }
            }
        }
    }
    let span = span.unwrap_or_else(|| panic!("{context}: no inference span"));
    let a = &span.attributes;
    let expect = [
        (attr::REQUEST_TEMPERATURE, Wire::DoubleValue(0.0)),
        (attr::REQUEST_TOP_P, Wire::DoubleValue(1.0)),
        (attr::REQUEST_PRESENCE_PENALTY, Wire::DoubleValue(0.0)),
        (attr::REQUEST_SEED, Wire::IntValue(7)),
        (attr::REQUEST_MAX_TOKENS, Wire::IntValue(64)),
        (attr::RESPONSE_TIME_TO_FIRST_CHUNK, Wire::DoubleValue(1.0)),
        (attr::RESPONSE_TIME_TO_FIRST_TOKEN, Wire::DoubleValue(1.0)),
        (attr::RESPONSE_TIME_PER_OUTPUT_TOKEN, Wire::DoubleValue(2.0)),
        (attr::RESOURCE_CPU_UTILIZATION, Wire::DoubleValue(1.0)),
        (attr::RESOURCE_BATTERY_LEVEL, Wire::DoubleValue(1.0)),
        (attr::USAGE_INPUT_TOKENS, Wire::IntValue(2)),
        (attr::USAGE_OUTPUT_TOKENS, Wire::IntValue(3)),
        (attr::SERVER_PORT, Wire::IntValue(443)),
    ];
    for (key, value) in expect {
        assert_eq!(find(a, key), Some(&value), "{context}: {key}");
    }
}

#[test]
fn trace_catalog_types_via_the_pipeline_and_the_paired_exporter() {
    // R-WIRE-1 (protobuf) and R-WIRE-2 (JSON).
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let pipeline = ObservabilityPipeline::build(server.traces_config(protocol)).unwrap();
        instrument(&pipeline.observer());
        pipeline.shutdown(Duration::from_secs(5)).unwrap();
        check_inference_span(
            &server.requests_to("/v1/traces"),
            &format!("pipeline {protocol:?}"),
        );

        let server = Server::ok();
        let exporter = exporters(&server.traces_config(protocol))
            .unwrap()
            .traces
            .unwrap();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter)
            .build();
        instrument(&Observer::builder().tracer_provider(&provider).build());
        provider.shutdown().unwrap();
        check_inference_span(
            &server.requests_to("/v1/traces"),
            &format!("standalone {protocol:?}"),
        );
    }
}

fn tool_result(observer: &Observer, name: &str, result: Value) {
    let tool = observer.start_tool(ToolCallInfo::new(name), StartOptions::default());
    tool.record_result(&result);
    tool.end();
}

fn records(requests: &[Received]) -> Vec<(String, Vec<WireKeyValue>)> {
    requests
        .iter()
        .flat_map(|r| r.logs().resource_logs)
        .flat_map(|r| r.scope_logs)
        .flat_map(|s| {
            let scope = s.scope.map(|s| s.name).unwrap_or_default();
            s.log_records
                .into_iter()
                .map(move |l| (scope.clone(), l.attributes))
        })
        .collect()
}

fn tool_record<'a>(records: &'a [(String, Vec<WireKeyValue>)], name: &str) -> &'a [WireKeyValue] {
    &records
        .iter()
        .find(|(_, a)| find(a, attr::TOOL_NAME) == Some(&Wire::StringValue(name.into())))
        .unwrap_or_else(|| panic!("no record for tool {name}"))
        .1
}

/// The JSON value of attribute `key` of the log record carrying tool `name`.
fn json_tool_attribute(body: &[u8], name: &str, key: &str) -> Value {
    let doc: Value = serde_json::from_slice(body).unwrap();
    for resource in doc["resourceLogs"].as_array().unwrap() {
        for scope in resource["scopeLogs"].as_array().unwrap() {
            for record in scope["logRecords"].as_array().unwrap() {
                let attributes = record["attributes"].as_array().unwrap();
                let get = |k: &str| {
                    attributes
                        .iter()
                        .find(|a| a["key"] == k)
                        .map(|a| a["value"].clone())
                };
                if get(attr::TOOL_NAME) == Some(json!({"stringValue": name})) {
                    return get(key).unwrap();
                }
            }
        }
    }
    panic!("no JSON record for tool {name}")
}

#[test]
fn log_records_carry_catalog_types_and_structured_nulls() {
    // R-WIRE-3: score 1 -> doubleValue, usage -> intValue, null -> AnyValue
    // without a value (`{}` in JSON), "" -> stringValue "", at top level, in
    // maps and in lists.
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let config = server
            .traces_config(protocol)
            .with_logs(Some(server.endpoint("logs", protocol)));
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        let observer = pipeline
            .observer_with(ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent));
        let target = instrument(&observer).expect("sampled inference");
        assert!(observer.record_evaluation(
            &target,
            &Evaluation {
                name: "quality".into(),
                score_value: Some(1.0),
                ..Evaluation::default()
            },
        ));
        tool_result(
            &observer,
            "nested",
            json!({"null": null, "empty": "", "list": [null, "", {"inner": null}], "map": {"a": null, "b": ""}}),
        );
        tool_result(&observer, "top-null", Value::Null);
        tool_result(&observer, "top-empty", json!(""));
        pipeline.shutdown(Duration::from_secs(5)).unwrap();

        let requests = server.requests_to("/v1/logs");
        let records = records(&requests);
        let mut checked = 0;
        for (scope, attributes) in &records {
            assert_eq!(scope, SCOPE_NAME);
            checked += assert_catalog_types(attributes, &format!("logs {protocol:?}"));
        }
        assert!(checked >= 3, "{protocol:?}: {checked}");
        let evaluation = records
            .iter()
            .find(|(_, a)| find(a, attr::EVALUATION_SCORE_VALUE).is_some())
            .expect("evaluation record");
        assert_eq!(
            find(&evaluation.1, attr::EVALUATION_SCORE_VALUE),
            Some(&Wire::DoubleValue(1.0))
        );
        let details = records
            .iter()
            .find(|(_, a)| find(a, attr::USAGE_INPUT_TOKENS).is_some())
            .expect("inference details record");
        assert_eq!(
            find(&details.1, attr::USAGE_INPUT_TOKENS),
            Some(&Wire::IntValue(2))
        );
        assert_eq!(
            find(&details.1, attr::USAGE_OUTPUT_TOKENS),
            Some(&Wire::IntValue(3))
        );

        // Top level.
        let top_null = tool_record(&records, "top-null");
        let value = top_null
            .iter()
            .find(|kv| kv.key == attr::TOOL_CALL_RESULT)
            .and_then(|kv| kv.value.as_ref())
            .expect("a null result is present");
        assert_eq!(value.value, None, "{protocol:?}: top-level null");
        assert_eq!(
            find(tool_record(&records, "top-empty"), attr::TOOL_CALL_RESULT),
            Some(&Wire::StringValue(String::new()))
        );
        // Maps and lists.
        let Some(Wire::KvlistValue(map)) =
            find(tool_record(&records, "nested"), attr::TOOL_CALL_RESULT)
        else {
            panic!("{protocol:?}: nested result is not a map")
        };
        let entry = |key: &str| {
            map.values
                .iter()
                .find(|kv| kv.key == key)
                .and_then(|kv| kv.value.clone())
                .unwrap_or_else(|| panic!("{key}"))
        };
        assert_eq!(entry("null").value, None);
        assert_eq!(entry("empty").value, Some(Wire::StringValue(String::new())));
        let Some(Wire::ArrayValue(list)) = entry("list").value else {
            panic!("list")
        };
        assert_eq!(list.values[0].value, None);
        assert_eq!(list.values[1].value, Some(Wire::StringValue(String::new())));
        let Some(Wire::KvlistValue(inner)) = &list.values[2].value else {
            panic!("inner map")
        };
        assert_eq!(inner.values[0].value.as_ref().unwrap().value, None);
        let Some(Wire::KvlistValue(nested)) = entry("map").value else {
            panic!("map")
        };
        assert_eq!(nested.values.len(), 2);

        if protocol == Protocol::HttpJson {
            // The JSON text itself: null is `{}`, "" is a stringValue.
            let body = &requests
                .iter()
                .find(|r| String::from_utf8_lossy(&r.body).contains("top-null"))
                .unwrap()
                .body;
            assert_eq!(
                json_tool_attribute(body, "top-null", attr::TOOL_CALL_RESULT),
                json!({})
            );
            assert_eq!(
                json_tool_attribute(body, "top-empty", attr::TOOL_CALL_RESULT),
                json!({"stringValue": ""})
            );
            let nested = json_tool_attribute(body, "nested", attr::TOOL_CALL_RESULT);
            let values = nested["kvlistValue"]["values"].as_array().unwrap();
            let get = |k: &str| values.iter().find(|v| v["key"] == k).unwrap()["value"].clone();
            assert_eq!(get("null"), json!({}));
            assert_eq!(get("empty"), json!({"stringValue": ""}));
            assert_eq!(get("list")["arrayValue"]["values"][0], json!({}));
        }
    }
}

/// The bucket boundaries the catalog assigns to histogram `name`.
fn catalog_bounds(name: &str) -> Vec<f64> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("contract/catalog.json");
    let catalog: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let kind = catalog["metrics"][name]["buckets"]
        .as_str()
        .unwrap_or_else(|| panic!("{name} has no catalog buckets"));
    let bounds: Vec<f64> = catalog["bucket_boundaries"][kind]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let constants = match kind {
        "duration" => DURATION_BUCKETS.to_vec(),
        "tokens" => TOKEN_BUCKETS.to_vec(),
        other => panic!("unknown bucket set {other}"),
    };
    assert_eq!(bounds, constants, "{kind}");
    bounds
}

#[test]
fn metrics_use_catalog_types_and_bucket_boundaries() {
    // R-WIRE-4: counters asInt, histograms with double sums and the catalog
    // boundaries, `server.port` dimension intValue.
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let config = server
            .traces_config(protocol)
            .with_traces(None)
            .with_metrics(Some(server.endpoint("metrics", protocol)));
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        instrument(&pipeline.observer());
        pipeline.force_flush(Duration::from_secs(5)).unwrap();
        pipeline.shutdown(Duration::from_secs(5)).unwrap();

        let requests = server.requests_to("/v1/metrics");
        assert!(!requests.is_empty());
        let request = requests[0].metrics();
        let mut seen = BTreeMap::new();
        for scope in request
            .resource_metrics
            .iter()
            .flat_map(|r| &r.scope_metrics)
        {
            assert_eq!(scope.scope.as_ref().unwrap().name, SCOPE_NAME);
            for m in &scope.metrics {
                let points: Vec<&[WireKeyValue]> = match m.data.as_ref().unwrap() {
                    Data::Sum(sum) => {
                        for p in &sum.data_points {
                            assert!(
                                matches!(p.value, Some(Number::AsInt(_))),
                                "{protocol:?} {}: {:?}",
                                m.name,
                                p.value
                            );
                        }
                        sum.data_points
                            .iter()
                            .map(|p| p.attributes.as_slice())
                            .collect()
                    }
                    Data::Histogram(h) => {
                        let bounds = catalog_bounds(&m.name);
                        for p in &h.data_points {
                            assert_eq!(p.explicit_bounds, bounds, "{protocol:?} {}", m.name);
                            assert!(p.sum.is_some(), "{}", m.name);
                        }
                        h.data_points
                            .iter()
                            .map(|p| p.attributes.as_slice())
                            .collect()
                    }
                    other => panic!("unexpected metric data {other:?}"),
                };
                for attributes in &points {
                    assert_catalog_types(attributes, &m.name);
                }
                seen.insert(
                    m.name.clone(),
                    points.iter().map(|a| a.to_vec()).collect::<Vec<_>>(),
                );
            }
        }
        for name in [
            metric::USAGE_INPUT_TOKENS,
            metric::INFERENCE_DURATION,
            metric::OPERATION_INPUT_TOKENS,
            metric::TIME_TO_FIRST_CHUNK,
        ] {
            assert!(seen.contains_key(name), "{protocol:?}: {name} missing");
        }
        let duration = &seen[metric::INFERENCE_DURATION][0];
        assert_eq!(
            find(duration, attr::SERVER_PORT),
            Some(&Wire::IntValue(443))
        );
        // The counter value itself is the integer token count.
        let Data::Sum(sum) = request.resource_metrics[0].scope_metrics[0]
            .metrics
            .iter()
            .find(|m| m.name == metric::USAGE_INPUT_TOKENS)
            .unwrap()
            .data
            .as_ref()
            .unwrap()
        else {
            panic!("sum")
        };
        assert_eq!(sum.data_points[0].value, Some(Number::AsInt(2)));
    }
}

#[derive(Debug, Default)]
struct Seen(Mutex<Vec<(String, AnyValue)>>);

#[derive(Debug)]
struct Recording(Arc<Seen>);

impl LogProcessor for Recording {
    fn emit(&self, data: &mut SdkLogRecord, _scope: &InstrumentationScope) {
        let mut seen = self.0 .0.lock().unwrap();
        for (k, v) in data.attributes_iter() {
            seen.push((k.as_str().to_owned(), v.clone()));
        }
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }
}

#[test]
fn the_paired_log_exporter_turns_the_carrier_a_processor_sees_into_null() {
    // R-WIRE-5: a preceding processor observes `Bytes(empty)`; the paired
    // exporter (standalone, application-owned provider) emits a real null;
    // a foreign OTLP encoder emits `bytesValue ""`.
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let config = server
            .traces_config(protocol)
            .with_traces(None)
            .with_logs(Some(server.endpoint("logs", protocol)));
        let exporter = exporters(&config).unwrap().logs.unwrap();
        let seen = Arc::new(Seen::default());
        let foreign = InMemoryLogExporter::default();
        let provider = SdkLoggerProvider::builder()
            .with_log_processor(Recording(seen.clone()))
            .with_simple_exporter(foreign.clone())
            .with_simple_exporter(exporter)
            .build();
        let observer = Observer::builder()
            .logger_provider(&provider)
            .content_policy(ContentPolicy::default().with_capture(ContentCapture::EventOnly))
            .build();
        tool_result(&observer, "carrier", json!([null, ""]));
        // The in-memory exporter forgets its records on shutdown: read first.
        let emitted = foreign.get_emitted_logs().unwrap();
        provider.shutdown().unwrap();

        let seen = seen.0.lock().unwrap();
        let observed = &seen
            .iter()
            .find(|(k, _)| k == attr::TOOL_CALL_RESULT)
            .unwrap()
            .1;
        let AnyValue::ListAny(items) = observed else {
            panic!("{observed:?}")
        };
        assert_eq!(items[0], AnyValue::Bytes(Box::default()));
        assert_eq!(items[1], AnyValue::String("".into()));

        let records = records(&server.requests_to("/v1/logs"));
        let Some(Wire::ArrayValue(list)) =
            find(tool_record(&records, "carrier"), attr::TOOL_CALL_RESULT)
        else {
            panic!("{protocol:?}: no array")
        };
        assert_eq!(list.values[0].value, None, "{protocol:?}: null");
        assert_eq!(list.values[1].value, Some(Wire::StringValue(String::new())));

        let record = emitted
            .iter()
            .find(|l| {
                l.record
                    .attributes_iter()
                    .any(|(k, _)| k.as_str() == attr::TOOL_CALL_RESULT)
            })
            .unwrap();
        let foreign_wire = opentelemetry_proto::tonic::logs::v1::LogRecord::from(&record.record);
        let Some(Wire::ArrayValue(list)) = find(&foreign_wire.attributes, attr::TOOL_CALL_RESULT)
        else {
            panic!("foreign array")
        };
        assert_eq!(list.values[0].value, Some(Wire::BytesValue(Vec::new())));
    }
}

#[test]
fn other_instrumentation_scopes_are_not_rewritten() {
    // R-WIRE-6: only scope `xybrid-observability` is type-enforced or
    // carrier-decoded; spans, log records and metrics of other scopes keep
    // what the application recorded.
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let pipeline = ObservabilityPipeline::build(server.all_config(protocol)).unwrap();
        let tracer = pipeline.tracer_provider().unwrap().tracer("app");
        let mut span = tracer.start("app-span");
        span.set_attribute(KeyValue::new(attr::REQUEST_TEMPERATURE, 1_i64));
        span.set_attribute(KeyValue::new(attr::SERVER_PORT, 443.0));
        span.end();
        instrument(&pipeline.observer());
        let logger = pipeline.logger_provider().unwrap().logger("app");
        let mut record = logger.create_log_record();
        record.add_attribute(attr::EVALUATION_SCORE_VALUE, 1_i64);
        record.add_attribute("carrier", AnyValue::Bytes(Box::default()));
        logger.emit(record);
        let meter = pipeline.meter_provider().unwrap().meter("app");
        meter
            .u64_counter("app.requests")
            .build()
            .add(1, &[KeyValue::new(attr::SERVER_PORT, 443.0)]);
        pipeline.shutdown(Duration::from_secs(5)).unwrap();

        let mut spans = BTreeMap::new();
        for scope in server
            .requests_to("/v1/traces")
            .iter()
            .flat_map(|r| r.traces().resource_spans)
            .flat_map(|r| r.scope_spans)
        {
            let name = scope.scope.unwrap().name;
            for s in scope.spans {
                spans.insert((name.clone(), s.name.clone()), s.attributes);
            }
        }
        let app = &spans[&("app".to_owned(), "app-span".to_owned())];
        assert_eq!(
            find(app, attr::REQUEST_TEMPERATURE),
            Some(&Wire::IntValue(1))
        );
        assert_eq!(
            find(app, attr::SERVER_PORT),
            Some(&Wire::DoubleValue(443.0))
        );
        assert!(spans.keys().any(|(scope, _)| scope == SCOPE_NAME));

        let records = records(&server.requests_to("/v1/logs"));
        let (_, app_log) = records.iter().find(|(s, _)| s == "app").unwrap();
        assert_eq!(
            find(app_log, attr::EVALUATION_SCORE_VALUE),
            Some(&Wire::IntValue(1))
        );
        assert_eq!(
            find(app_log, "carrier"),
            Some(&Wire::BytesValue(Vec::new()))
        );

        let mut app_points = 0;
        for scope in server
            .requests_to("/v1/metrics")
            .iter()
            .flat_map(|r| r.metrics().resource_metrics)
            .flat_map(|r| r.scope_metrics)
            .filter(|s| s.scope.as_ref().unwrap().name == "app")
        {
            for m in scope.metrics {
                let Some(Data::Sum(sum)) = m.data else {
                    panic!("sum")
                };
                for p in sum.data_points {
                    assert_eq!(
                        find(&p.attributes, attr::SERVER_PORT),
                        Some(&Wire::DoubleValue(443.0))
                    );
                    app_points += 1;
                }
            }
        }
        assert_eq!(app_points, 1, "{protocol:?}");
    }
}

fn sent_headers(request: &Received) -> BTreeMap<String, String> {
    request
        .headers
        .iter()
        .filter(|(k, _)| !matches!(k.as_str(), "host" | "content-length"))
        // The HTTP client's generic default when no Accept was configured;
        // the package adds no Accept of its own.
        .filter(|(k, v)| !(k == "accept" && v == "*/*"))
        .cloned()
        .collect()
}

#[test]
fn the_preset_sends_only_its_credential_and_generic_endpoints_merge_env_headers() {
    // R-CFG-1 on the wire: environment headers (including another
    // Authorization) never reach a preset request.
    let environment = EnvironmentSource::vars([
        (
            "OTEL_EXPORTER_OTLP_HEADERS",
            "Authorization=Basic%20x,x-honeycomb-team=k",
        ),
        ("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x-other=y"),
        ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
        (
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://127.0.0.1:9/v1/traces",
        ),
    ]);
    let server = Server::ok();
    let preset = OtlpConfig::xybrid_traces_to(&format!("{}/v1/traces", server.url), "sk-test")
        .with_environment(environment.clone());
    let pipeline = ObservabilityPipeline::build(preset).unwrap();
    instrument(&pipeline.observer());
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    let requests = server.requests_to("/v1/traces");
    assert_eq!(requests.len(), 1);
    assert_eq!(
        sent_headers(&requests[0]),
        BTreeMap::from([
            ("authorization".to_owned(), "Bearer sk-test".to_owned()),
            (
                "content-type".to_owned(),
                "application/x-protobuf".to_owned()
            ),
        ])
    );

    // R-CFG-2 on the wire: a generic endpoint merges the environment under
    // its programmatic headers (signal-specific over general).
    let server = Server::ok();
    let config = server
        .traces_config(Protocol::HttpProtobuf)
        .with_traces(Some(
            server
                .endpoint("traces", Protocol::HttpProtobuf)
                .with_header("X-Other", "mine"),
        ))
        .with_environment(environment);
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    instrument(&pipeline.observer());
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    let requests = server.requests_to("/v1/traces");
    assert_eq!(
        sent_headers(&requests[0]),
        BTreeMap::from([
            ("authorization".to_owned(), "Basic x".to_owned()),
            ("x-honeycomb-team".to_owned(), "k".to_owned()),
            ("x-other".to_owned(), "mine".to_owned()),
            (
                "content-type".to_owned(),
                "application/x-protobuf".to_owned()
            ),
        ])
    );
}
