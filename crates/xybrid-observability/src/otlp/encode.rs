//! OTLP encoding owned by the package: opentelemetry-proto messages, catalog
//! type enforcement and null-carrier decoding for scope
//! `xybrid-observability` (D13, D14), grouping for final-size splitting
//! (D27), partial-success decoding (D26) and admission estimates (D32).

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};

use opentelemetry::logs::{AnyValue as LogValue, LogRecord as _};
use opentelemetry::{Array, InstrumentationScope, Value};
use opentelemetry_proto::tonic::collector::logs::v1::{
    ExportLogsServiceRequest, ExportLogsServiceResponse,
};
use opentelemetry_proto::tonic::collector::metrics::v1::{
    ExportMetricsServiceRequest, ExportMetricsServiceResponse,
};
use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use opentelemetry_proto::tonic::common::v1::{
    any_value, AnyValue, InstrumentationScope as ProtoScope, KeyValue as ProtoKeyValue,
};
use opentelemetry_proto::tonic::logs::v1::{LogRecord as ProtoLogRecord, ResourceLogs, ScopeLogs};
use opentelemetry_proto::tonic::metrics::v1::metric::Data;
use opentelemetry_proto::tonic::resource::v1::Resource as ProtoResource;
use opentelemetry_proto::tonic::trace::v1::{ResourceSpans, ScopeSpans, Span as ProtoSpan};
use opentelemetry_proto::transform::common::tonic::ResourceAttributesWithSchema;
use opentelemetry_sdk::logs::SdkLogRecord;
use opentelemetry_sdk::metrics::data::ResourceMetrics;
use opentelemetry_sdk::trace::SpanData;
use opentelemetry_sdk::Resource;
use prost::Message;
use serde::Serialize;

use super::config::Protocol;
use crate::semconv::SCOPE_NAME;
use crate::wire::{self, CatalogType};

/// An OTLP signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Signal {
    Traces,
    Metrics,
    Logs,
}

impl Signal {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Traces => "traces",
            Self::Metrics => "metrics",
            Self::Logs => "logs",
        }
    }
}

/// The resource of trace and log requests.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResourceInfo {
    resource: ProtoResource,
    schema_url: String,
}

impl ResourceInfo {
    pub(crate) fn from_sdk(resource: &Resource) -> Self {
        let converted = ResourceAttributesWithSchema::from(resource);
        Self {
            resource: ProtoResource {
                attributes: converted.attributes.0,
                dropped_attributes_count: 0,
                entity_refs: Vec::new(),
            },
            schema_url: converted.schema_url.unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Type enforcement and null decoding (scope `xybrid-observability` only).

fn enforce_value(key: &str, value: &mut AnyValue) {
    match (wire::attribute_type(key), &value.value) {
        (Some(CatalogType::Double), Some(any_value::Value::IntValue(n))) => {
            value.value = Some(any_value::Value::DoubleValue(*n as f64));
        }
        (Some(CatalogType::Int), Some(any_value::Value::DoubleValue(f))) => {
            if let Some(n) = wire::integral(*f) {
                value.value = Some(any_value::Value::IntValue(n));
            }
        }
        _ => {}
    }
}

/// Applies catalog `int`/`double` types to top-level attributes.
pub(crate) fn enforce_attributes(attributes: &mut [ProtoKeyValue]) {
    for kv in attributes {
        if let Some(value) = kv.value.as_mut() {
            enforce_value(&kv.key, value);
        }
    }
}

/// Turns the Rust null carrier (`bytesValue` empty) into an `AnyValue` with no
/// value set, at every level.
pub(crate) fn decode_null_carrier(value: &mut AnyValue) {
    match &mut value.value {
        Some(any_value::Value::BytesValue(bytes)) if bytes.is_empty() => value.value = None,
        Some(any_value::Value::ArrayValue(array)) => {
            array.values.iter_mut().for_each(decode_null_carrier);
        }
        Some(any_value::Value::KvlistValue(list)) => list
            .values
            .iter_mut()
            .filter_map(|kv| kv.value.as_mut())
            .for_each(decode_null_carrier),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Items grouped by scope.

/// An instrumentation scope as exported.
#[derive(Clone, Debug)]
pub(crate) struct ScopeEntry {
    scope: ProtoScope,
    schema_url: String,
}

/// Items in their original order, each tagged with its scope.
#[derive(Debug)]
pub(crate) struct Scoped<M> {
    pub scopes: Vec<ScopeEntry>,
    pub items: Vec<(usize, M)>,
}

/// A trace or log item type and its request message.
pub(crate) trait Grouped: Sized + Send + 'static {
    type Request: Message + Serialize + Send;
    /// One request with a single resource and one group per scope.
    fn wrap(resource: &ResourceInfo, groups: Vec<(ScopeEntry, Vec<Self>)>) -> Self::Request;
    /// The groups of a request built by [`Grouped::wrap`], in order.
    fn unwrap(request: Self::Request) -> Vec<Vec<Self>>;
}

impl Grouped for ProtoSpan {
    type Request = ExportTraceServiceRequest;

    fn wrap(resource: &ResourceInfo, groups: Vec<(ScopeEntry, Vec<Self>)>) -> Self::Request {
        ExportTraceServiceRequest {
            resource_spans: vec![ResourceSpans {
                resource: Some(resource.resource.clone()),
                scope_spans: groups
                    .into_iter()
                    .map(|(entry, spans)| ScopeSpans {
                        scope: Some(entry.scope),
                        spans,
                        schema_url: entry.schema_url,
                    })
                    .collect(),
                schema_url: resource.schema_url.clone(),
            }],
        }
    }

    fn unwrap(request: Self::Request) -> Vec<Vec<Self>> {
        request
            .resource_spans
            .into_iter()
            .flat_map(|r| r.scope_spans)
            .map(|s| s.spans)
            .collect()
    }
}

impl Grouped for ProtoLogRecord {
    type Request = ExportLogsServiceRequest;

    fn wrap(resource: &ResourceInfo, groups: Vec<(ScopeEntry, Vec<Self>)>) -> Self::Request {
        ExportLogsServiceRequest {
            resource_logs: vec![ResourceLogs {
                resource: Some(resource.resource.clone()),
                scope_logs: groups
                    .into_iter()
                    .map(|(entry, log_records)| ScopeLogs {
                        scope: Some(entry.scope),
                        log_records,
                        schema_url: entry.schema_url,
                    })
                    .collect(),
                schema_url: resource.schema_url.clone(),
            }],
        }
    }

    fn unwrap(request: Self::Request) -> Vec<Vec<Self>> {
        request
            .resource_logs
            .into_iter()
            .flat_map(|r| r.scope_logs)
            .map(|s| s.log_records)
            .collect()
    }
}

/// A request built from some items, with what is needed to take them back.
pub(crate) struct Part<M: Grouped> {
    pub request: M::Request,
    order: Vec<usize>,
    groups: Vec<usize>,
}

impl<M: Grouped> Part<M> {
    pub(crate) fn len(&self) -> usize {
        self.order.len()
    }
}

/// Builds one request: groups follow the first appearance of their scope and
/// items keep their original order within each group.
pub(crate) fn build<M: Grouped>(
    resource: &ResourceInfo,
    scopes: &[ScopeEntry],
    items: Vec<(usize, M)>,
) -> Part<M> {
    let mut group_of: Vec<Option<usize>> = vec![None; scopes.len()];
    let mut groups: Vec<(usize, Vec<M>)> = Vec::new();
    let mut order = Vec::with_capacity(items.len());
    for (scope, item) in items {
        order.push(scope);
        let index = *group_of[scope].get_or_insert_with(|| {
            groups.push((scope, Vec::new()));
            groups.len() - 1
        });
        groups[index].1.push(item);
    }
    let group_scopes = groups.iter().map(|(s, _)| *s).collect();
    let request = M::wrap(
        resource,
        groups
            .into_iter()
            .map(|(s, items)| (scopes[s].clone(), items))
            .collect(),
    );
    Part {
        request,
        order,
        groups: group_scopes,
    }
}

/// Takes the items of a part back, in their original order.
pub(crate) fn unbuild<M: Grouped>(part: Part<M>, scope_count: usize) -> Vec<(usize, M)> {
    let mut by_scope: Vec<VecDeque<M>> = (0..scope_count).map(|_| VecDeque::new()).collect();
    for (scope, items) in part.groups.iter().zip(M::unwrap(part.request)) {
        by_scope[*scope].extend(items);
    }
    part.order
        .into_iter()
        .filter_map(|scope| by_scope[scope].pop_front().map(|item| (scope, item)))
        .collect()
}

/// The final request body, or `None` when encoding failed.
pub(crate) fn encode<R: Message + Serialize>(request: &R, protocol: Protocol) -> Option<Vec<u8>> {
    match protocol {
        Protocol::HttpProtobuf => Some(request.encode_to_vec()),
        Protocol::HttpJson => serde_json::to_vec(request).ok(),
    }
}

struct ScopeTable {
    index: HashMap<InstrumentationScope, usize>,
    scopes: Vec<ScopeEntry>,
}

impl ScopeTable {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            scopes: Vec::new(),
        }
    }

    fn get(&mut self, scope: &InstrumentationScope) -> usize {
        if let Some(&i) = self.index.get(scope) {
            return i;
        }
        let i = self.scopes.len();
        self.scopes.push(ScopeEntry {
            scope: (scope, None::<Cow<'static, str>>).into(),
            schema_url: scope.schema_url().unwrap_or_default().to_owned(),
        });
        self.index.insert(scope.clone(), i);
        i
    }

    fn enforced(&self, index: usize) -> bool {
        self.scopes[index].scope.name == SCOPE_NAME
    }
}

/// Converts spans, enforcing catalog types on span and event attributes of
/// the package's scope.
pub(crate) fn prepare_spans(batch: Vec<SpanData>) -> Scoped<ProtoSpan> {
    let mut table = ScopeTable::new();
    let mut items = Vec::with_capacity(batch.len());
    for span in batch {
        let scope = table.get(&span.instrumentation_scope);
        let mut proto = ProtoSpan::from(span);
        if table.enforced(scope) {
            enforce_attributes(&mut proto.attributes);
            for event in &mut proto.events {
                enforce_attributes(&mut event.attributes);
            }
        }
        items.push((scope, proto));
    }
    Scoped {
        scopes: table.scopes,
        items,
    }
}

/// Converts log records (the exported scope name is the record's target when
/// set), decoding the null carrier and enforcing catalog types on records of
/// the package's scope.
pub(crate) fn prepare_logs<'a>(
    records: impl Iterator<Item = (&'a SdkLogRecord, &'a InstrumentationScope)>,
) -> Scoped<ProtoLogRecord> {
    let mut table = ScopeTable::new();
    let mut items = Vec::new();
    for (record, scope) in records {
        let scope = match record.target() {
            Some(target) => table.get(
                &InstrumentationScope::builder(target.clone())
                    .with_version(scope.version().unwrap_or_default().to_owned())
                    .with_schema_url(scope.schema_url().unwrap_or_default().to_owned())
                    .with_attributes(scope.attributes().cloned())
                    .build(),
            ),
            None => table.get(scope),
        };
        let mut proto = if record.observed_timestamp().is_some() {
            ProtoLogRecord::from(record)
        } else {
            // A record that was never emitted through a logger.
            let mut owned = record.clone();
            owned.set_observed_timestamp(std::time::SystemTime::now());
            ProtoLogRecord::from(&owned)
        };
        if table.enforced(scope) {
            for kv in &mut proto.attributes {
                if let Some(value) = kv.value.as_mut() {
                    decode_null_carrier(value);
                    enforce_value(&kv.key, value);
                }
            }
            if let Some(body) = proto.body.as_mut() {
                decode_null_carrier(body);
            }
        }
        items.push((scope, proto));
    }
    Scoped {
        scopes: table.scopes,
        items,
    }
}

/// Converts one metrics collection; returns the request and its number of
/// data points.
pub(crate) fn prepare_metrics(metrics: &ResourceMetrics) -> (ExportMetricsServiceRequest, u64) {
    let mut request = ExportMetricsServiceRequest::from(metrics);
    let mut points = 0u64;
    for resource in &mut request.resource_metrics {
        for scope in &mut resource.scope_metrics {
            let enforced = scope.scope.as_ref().is_some_and(|s| s.name == SCOPE_NAME);
            for metric in &mut scope.metrics {
                let attributes: Vec<&mut Vec<ProtoKeyValue>> = match metric.data.as_mut() {
                    Some(Data::Gauge(g)) => g
                        .data_points
                        .iter_mut()
                        .map(|p| &mut p.attributes)
                        .collect(),
                    Some(Data::Sum(s)) => s
                        .data_points
                        .iter_mut()
                        .map(|p| &mut p.attributes)
                        .collect(),
                    Some(Data::Histogram(h)) => h
                        .data_points
                        .iter_mut()
                        .map(|p| &mut p.attributes)
                        .collect(),
                    Some(Data::ExponentialHistogram(h)) => h
                        .data_points
                        .iter_mut()
                        .map(|p| &mut p.attributes)
                        .collect(),
                    Some(Data::Summary(s)) => s
                        .data_points
                        .iter_mut()
                        .map(|p| &mut p.attributes)
                        .collect(),
                    None => Vec::new(),
                };
                points += attributes.len() as u64;
                if enforced {
                    for list in attributes {
                        enforce_attributes(list);
                    }
                }
            }
        }
    }
    (request, points)
}

/// Rejected item count of a 2xx response body (OTLP partial success); 0 when
/// the body is empty or does not decode.
pub(crate) fn rejected(signal: Signal, protocol: Protocol, body: &[u8]) -> u64 {
    let count = match protocol {
        Protocol::HttpProtobuf => match signal {
            Signal::Traces => ExportTraceServiceResponse::decode(body)
                .ok()
                .and_then(|r| r.partial_success)
                .map(|p| p.rejected_spans),
            Signal::Logs => ExportLogsServiceResponse::decode(body)
                .ok()
                .and_then(|r| r.partial_success)
                .map(|p| p.rejected_log_records),
            Signal::Metrics => ExportMetricsServiceResponse::decode(body)
                .ok()
                .and_then(|r| r.partial_success)
                .map(|p| p.rejected_data_points),
        },
        Protocol::HttpJson => {
            let field = match signal {
                Signal::Traces => "rejectedSpans",
                Signal::Logs => "rejectedLogRecords",
                Signal::Metrics => "rejectedDataPoints",
            };
            serde_json::from_slice::<serde_json::Value>(body)
                .ok()
                .and_then(|v| {
                    let n = v.get("partialSuccess")?.get(field)?.clone();
                    match n {
                        serde_json::Value::Number(n) => n.as_i64(),
                        serde_json::Value::String(s) => s.trim().parse::<i64>().ok(),
                        _ => None,
                    }
                })
        }
    };
    count.map_or(0, |n| u64::try_from(n).unwrap_or(0))
}

// ---------------------------------------------------------------------------
// Admission estimates (D32.3). Protobuf uses the contract formula; JSON uses
// larger per-node constants (field names, hex ids, decimal numbers) and the
// exact escaped length of every string, so the estimate stays at or above the
// item's share of the encoded request in both encodings.

#[derive(Clone, Copy, Debug)]
pub(crate) struct Estimator {
    json: bool,
    base: u64,
    node: u64,
    scalar: u64,
    element: u64,
    event: u64,
}

impl Estimator {
    pub(crate) fn new(protocol: Protocol) -> Self {
        match protocol {
            Protocol::HttpProtobuf => Self {
                json: false,
                base: 512,
                node: 16,
                scalar: 8,
                element: 8,
                event: 128,
            },
            Protocol::HttpJson => Self {
                json: true,
                base: 1024,
                node: 64,
                scalar: 32,
                element: 32,
                event: 256,
            },
        }
    }

    /// A string's encoded size: its UTF-8 length for protobuf; for JSON its
    /// escaped length with quotes (serde_json writes `\"`, `\\` and the short
    /// control escapes as two bytes and other control characters as six).
    fn string(&self, s: &str) -> u64 {
        if !self.json {
            return s.len() as u64;
        }
        let mut len = s.len() as u64 + 2;
        for b in s.bytes() {
            match b {
                b'"' | b'\\' | b'\n' | b'\r' | b'\t' | 0x08 | 0x0c => len += 1,
                0x00..=0x1f => len += 5,
                _ => {}
            }
        }
        len
    }

    /// Bytes are base64 in OTLP/JSON.
    fn bytes(&self, len: usize) -> u64 {
        let len = len as u64;
        if self.json {
            len.div_ceil(3) * 4 + 2
        } else {
            len
        }
    }

    fn value(&self, value: &Value) -> u64 {
        match value {
            Value::String(s) => self.string(s.as_str()),
            Value::Bool(_) | Value::I64(_) | Value::F64(_) => self.scalar,
            Value::Array(Array::String(v)) => v
                .iter()
                .map(|s| self.string(s.as_str()) + self.element)
                .sum(),
            Value::Array(Array::Bool(v)) => (self.scalar + self.element) * v.len() as u64,
            Value::Array(Array::I64(v)) => (self.scalar + self.element) * v.len() as u64,
            Value::Array(Array::F64(v)) => (self.scalar + self.element) * v.len() as u64,
            other => self.string(&other.to_string()) + self.node,
        }
    }

    fn attributes<'a>(&self, attributes: impl Iterator<Item = &'a opentelemetry::KeyValue>) -> u64 {
        attributes
            .map(|kv| self.node + self.string(kv.key.as_str()) + self.value(&kv.value))
            .sum()
    }

    fn any(&self, value: &LogValue) -> u64 {
        match value {
            LogValue::String(s) => self.string(s.as_str()),
            LogValue::Int(_) | LogValue::Double(_) | LogValue::Boolean(_) => self.scalar,
            LogValue::Bytes(b) => self.bytes(b.len()),
            LogValue::ListAny(items) => items.iter().map(|v| self.any(v) + self.element).sum(),
            LogValue::Map(map) => map
                .iter()
                .map(|(k, v)| self.node + self.string(k.as_str()) + self.any(v))
                .sum(),
            _ => self.scalar,
        }
    }

    fn trace_state(&self, state: &opentelemetry::trace::TraceState) -> u64 {
        if state == &Default::default() {
            0
        } else {
            self.string(&state.header())
        }
    }

    /// Charge every item's scope independently: distinct scopes can be as
    /// large as record attributes, and grouping happens only after admission.
    fn scope(&self, scope: &InstrumentationScope, name: &str) -> u64 {
        self.node
            + self.string(name)
            + scope.version().map_or(0, |v| self.string(v))
            + scope.schema_url().map_or(0, |url| self.string(url))
            + self.attributes(scope.attributes())
    }

    /// Conservative size of a span's share of an encoded request.
    pub(crate) fn span(&self, span: &SpanData) -> u64 {
        let status = match &span.status {
            opentelemetry::trace::Status::Error { description } => {
                self.node + self.string(description)
            }
            _ => 0,
        };
        self.base
            + self.scope(
                &span.instrumentation_scope,
                span.instrumentation_scope.name(),
            )
            + self.string(&span.name)
            + self.trace_state(span.span_context.trace_state())
            + status
            + self.attributes(span.attributes.iter())
            + span
                .events
                .iter()
                .map(|e| self.event + self.string(&e.name) + self.attributes(e.attributes.iter()))
                .sum::<u64>()
            + span
                .links
                .iter()
                .map(|l| {
                    self.event
                        + self.trace_state(l.span_context.trace_state())
                        + self.attributes(l.attributes.iter())
                })
                .sum::<u64>()
    }

    /// Conservative size of a log record's share of an encoded request.
    pub(crate) fn log(&self, record: &SdkLogRecord, scope: &InstrumentationScope) -> u64 {
        self.base
            // The queued item keeps the original scope even when the target
            // overrides its exported name. Reserve the derived scope too;
            // prepare_logs clones its version, schema URL and attributes.
            + self.scope(scope, scope.name())
            + record.target().map_or(0, |target| self.scope(scope, target))
            + record.event_name().map_or(0, |n| self.string(n))
            + record.severity_text().map_or(0, |n| self.string(n))
            + record.target().map_or(0, |t| self.string(t))
            + record.body().map_or(0, |b| self.any(b))
            + record
                .attributes_iter()
                .map(|(k, v)| self.node + self.string(k.as_str()) + self.any(v))
                .sum::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(name: &str) -> ScopeEntry {
        ScopeEntry {
            scope: ProtoScope {
                name: name.into(),
                ..ProtoScope::default()
            },
            schema_url: String::new(),
        }
    }

    fn span(name: &str) -> ProtoSpan {
        ProtoSpan {
            name: name.into(),
            ..ProtoSpan::default()
        }
    }

    #[test]
    fn build_groups_by_scope_and_unbuild_restores_order() {
        let scopes = vec![scope("a"), scope("b")];
        let items = vec![
            (0, span("1")),
            (1, span("2")),
            (0, span("3")),
            (1, span("4")),
        ];
        let part = build(&ResourceInfo::default(), &scopes, items);
        assert_eq!(part.len(), 4);
        let groups = &part.request.resource_spans[0].scope_spans;
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].scope.as_ref().unwrap().name, "a");
        let names: Vec<_> = groups[0].spans.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["1", "3"]);
        let back = unbuild(part, scopes.len());
        let names: Vec<_> = back.iter().map(|(_, s)| s.name.as_str()).collect();
        assert_eq!(names, ["1", "2", "3", "4"]);
        assert_eq!(back[1].0, 1);
    }

    #[test]
    fn null_carrier_and_types_are_decoded_on_the_wire_model() {
        let mut v = AnyValue {
            value: Some(any_value::Value::KvlistValue(
                opentelemetry_proto::tonic::common::v1::KeyValueList {
                    values: vec![ProtoKeyValue {
                        key: "k".into(),
                        value: Some(AnyValue {
                            value: Some(any_value::Value::BytesValue(Vec::new())),
                        }),
                        key_strindex: 0,
                    }],
                },
            )),
        };
        decode_null_carrier(&mut v);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(
            json,
            r#"{"kvlistValue":{"values":[{"key":"k","value":{}}]}}"#
        );
        let mut attrs = vec![
            ProtoKeyValue {
                key: "gen_ai.request.temperature".into(),
                value: Some(AnyValue {
                    value: Some(any_value::Value::IntValue(0)),
                }),
                key_strindex: 0,
            },
            ProtoKeyValue {
                key: "server.port".into(),
                value: Some(AnyValue {
                    value: Some(any_value::Value::DoubleValue(443.0)),
                }),
                key_strindex: 0,
            },
        ];
        enforce_attributes(&mut attrs);
        assert_eq!(
            attrs[0].value.as_ref().unwrap().value,
            Some(any_value::Value::DoubleValue(0.0))
        );
        assert_eq!(
            attrs[1].value.as_ref().unwrap().value,
            Some(any_value::Value::IntValue(443))
        );
    }

    /// The item's share of a request: the body with the item minus the body
    /// without it (which also drops the scope group, so this over-counts).
    fn share<M: Grouped + Clone>(scoped: &Scoped<M>, protocol: Protocol) -> u64 {
        let resource = ResourceInfo::default();
        let with = build::<M>(&resource, &scoped.scopes, scoped.items.clone());
        let without = build::<M>(&resource, &scoped.scopes, Vec::new());
        let with = encode(&with.request, protocol).unwrap().len();
        let without = encode(&without.request, protocol).unwrap().len();
        (with - without) as u64
    }

    #[test]
    fn admission_estimates_cover_the_encoded_share_of_adversarial_items() {
        // R-BUF-2 (estimate half), F6: control characters (6-byte JSON
        // escapes), quotes, non-ASCII, status descriptions, trace state on
        // the span and on links, extreme numbers and every array kind.
        use opentelemetry::logs::{LogRecord as _, Logger as _, LoggerProvider as _};
        use opentelemetry::trace::{
            Link, Span as _, SpanContext, SpanId, Status, TraceContextExt as _, TraceFlags,
            TraceId, TraceState, Tracer as _, TracerProvider as _,
        };
        use opentelemetry::{Context, KeyValue, StringValue};
        use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
        use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};

        let nasty = "\u{1}\"\\\n\u{1f}\u{7f}\u{e9}/".repeat(64);
        let state = TraceState::from_key_value([("vendor", "v".repeat(200))]).unwrap();
        let remote = SpanContext::new(
            TraceId::from(7u128),
            SpanId::from(9u64),
            TraceFlags::SAMPLED,
            true,
            state.clone(),
        );
        let spans = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(spans.clone())
            .build();
        let tracer = provider.tracer("estimates");
        let parent = Context::new().with_remote_span_context(remote.clone());
        let attributes = vec![
            KeyValue::new(nasty.clone(), nasty.clone()),
            KeyValue::new("i", i64::MIN),
            KeyValue::new("f", -1.797_693_134_862_315_7e308_f64),
            KeyValue::new("tiny", 5e-324_f64),
            KeyValue::new("b", true),
            KeyValue::new("empty", ""),
            KeyValue::new(
                "strings",
                opentelemetry::Value::Array(opentelemetry::Array::String(vec![
                    StringValue::from(nasty.clone()),
                    StringValue::from(""),
                ])),
            ),
            KeyValue::new(
                "ints",
                opentelemetry::Value::Array(opentelemetry::Array::I64(vec![i64::MIN; 9])),
            ),
            KeyValue::new(
                "floats",
                opentelemetry::Value::Array(opentelemetry::Array::F64(vec![-1.0e-300; 9])),
            ),
            KeyValue::new(
                "bools",
                opentelemetry::Value::Array(opentelemetry::Array::Bool(vec![false; 9])),
            ),
        ];
        let mut span = tracer.build_with_context(
            tracer
                .span_builder(nasty.clone())
                .with_attributes(attributes.clone())
                .with_links(vec![
                    Link::new(remote.clone(), attributes.clone(), 0),
                    Link::new(remote, Vec::new(), 0),
                ]),
            &parent,
        );
        span.add_event(nasty.clone(), attributes.clone());
        span.add_event("", Vec::new());
        span.set_status(Status::error(nasty.clone()));
        span.end();
        tracer.start("x").end();
        let finished = spans.get_finished_spans().unwrap();
        assert_eq!(finished.len(), 2);

        let logs = InMemoryLogExporter::default();
        let logger_provider = SdkLoggerProvider::builder()
            .with_simple_exporter(logs.clone())
            .build();
        let logger = logger_provider.logger("estimates");
        let nested = LogValue::Map(Box::new(
            [
                (
                    opentelemetry::Key::from(nasty.clone()),
                    LogValue::ListAny(Box::new(vec![
                        LogValue::Bytes(Box::default()),
                        LogValue::Bytes(Box::new(vec![0xff; 100])),
                        LogValue::Int(i64::MIN),
                        LogValue::Double(-1.797_693_134_862_315_7e308),
                        LogValue::Boolean(true),
                        LogValue::String(nasty.clone().into()),
                        LogValue::ListAny(Box::default()),
                        LogValue::Map(Box::default()),
                    ])),
                ),
                (opentelemetry::Key::from("e"), LogValue::Map(Box::default())),
            ]
            .into_iter()
            .collect(),
        ));
        let mut record = logger.create_log_record();
        record.set_event_name("estimates.event");
        record.set_severity_text("DEBUG");
        record.set_target(nasty.clone());
        record.set_body(nested.clone());
        record.add_attribute(nasty.clone(), nested.clone());
        record.add_attribute("n", LogValue::Bytes(Box::default()));
        record.set_trace_context(TraceId::from(7u128), SpanId::from(9u64), None);
        logger.emit(record);
        let mut plain = logger.create_log_record();
        plain.set_body(LogValue::from(""));
        logger.emit(plain);
        let emitted = logs.get_emitted_logs().unwrap();
        assert_eq!(emitted.len(), 2);

        for protocol in [Protocol::HttpProtobuf, Protocol::HttpJson] {
            let estimator = Estimator::new(protocol);
            for span in &finished {
                let estimate = estimator.span(span);
                let actual = share(&prepare_spans(vec![span.clone()]), protocol);
                assert!(
                    estimate >= actual,
                    "{protocol:?} span {:?}: estimate {estimate} < encoded {actual}",
                    span.name.len()
                );
            }
            for log in &emitted {
                let estimate = estimator.log(&log.record, &log.instrumentation);
                let actual = share(
                    &prepare_logs(std::iter::once((&log.record, &log.instrumentation))),
                    protocol,
                );
                assert!(
                    estimate >= actual,
                    "{protocol:?} log: estimate {estimate} < encoded {actual}"
                );
            }
        }
    }

    #[test]
    fn admission_estimates_cover_every_scope_field_and_log_target_override() {
        use opentelemetry::logs::{LogRecord as _, Logger as _, LoggerProvider as _};
        use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
        use opentelemetry::KeyValue;
        use opentelemetry_sdk::logs::SdkLoggerProvider;
        use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};

        let memory = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(memory.clone())
            .build();
        provider.tracer("app").start("small").end();
        let span = memory.get_finished_spans().unwrap().pop().unwrap();
        let logger_provider = SdkLoggerProvider::builder().build();
        let logger = logger_provider.logger("app");
        // Large enough that a missing scope field cannot hide in the fixed
        // envelope allowance; JSON also exercises six-byte control escapes.
        let large = "\u{1}é\"\\".repeat(4096);
        let scopes = [
            InstrumentationScope::builder(large.clone()).build(),
            InstrumentationScope::builder("app")
                .with_version(large.clone())
                .build(),
            InstrumentationScope::builder("app")
                .with_schema_url(large.clone())
                .build(),
            InstrumentationScope::builder("app")
                .with_attributes([KeyValue::new(large.clone(), large.clone())])
                .build(),
        ];
        for protocol in [Protocol::HttpProtobuf, Protocol::HttpJson] {
            let estimator = Estimator::new(protocol);
            for (field, scope) in scopes.iter().enumerate() {
                let mut scoped_span = span.clone();
                scoped_span.instrumentation_scope = scope.clone();
                let estimate = estimator.span(&scoped_span);
                let actual = share(&prepare_spans(vec![scoped_span]), protocol);
                assert!(
                    estimate >= actual,
                    "{protocol:?} scope field {field}: {estimate} < {actual}"
                );

                for target in [None, Some("short-target".to_owned()), Some(large.clone())] {
                    let mut record = logger.create_log_record();
                    if let Some(target) = target {
                        record.set_target(target);
                    }
                    let estimate = estimator.log(&record, scope);
                    let actual = share(&prepare_logs(std::iter::once((&record, scope))), protocol);
                    assert!(
                        estimate >= actual,
                        "{protocol:?} log scope field {field}: {estimate} < {actual}"
                    );
                    // A short target replaces the wire name, but the queued
                    // record still retains the original (possibly huge) scope.
                    assert!(
                        estimate >= large.len() as u64,
                        "retained scope field {field} omitted"
                    );
                }
            }
        }
    }

    #[test]
    fn partial_success_decodes_numbers_and_strings() {
        let pb = ExportTraceServiceResponse {
            partial_success: Some(
                opentelemetry_proto::tonic::collector::trace::v1::ExportTracePartialSuccess {
                    rejected_spans: 2,
                    error_message: "no".into(),
                },
            ),
        }
        .encode_to_vec();
        assert_eq!(rejected(Signal::Traces, Protocol::HttpProtobuf, &pb), 2);
        assert_eq!(rejected(Signal::Traces, Protocol::HttpProtobuf, b""), 0);
        assert_eq!(
            rejected(Signal::Traces, Protocol::HttpProtobuf, b"\xff\xff"),
            0
        );
        assert_eq!(
            rejected(
                Signal::Logs,
                Protocol::HttpJson,
                br#"{"partialSuccess":{"rejectedLogRecords":"3"}}"#
            ),
            3
        );
        assert_eq!(
            rejected(
                Signal::Metrics,
                Protocol::HttpJson,
                br#"{"partialSuccess":{"rejectedDataPoints":1}}"#
            ),
            1
        );
        assert_eq!(rejected(Signal::Metrics, Protocol::HttpJson, b"{}"), 0);
        assert_eq!(
            rejected(Signal::Metrics, Protocol::HttpJson, b"not json"),
            0
        );
    }
}
