import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  ATTEMPT_SPAN_NAME,
  ATTR,
  ATTRIBUTE_TYPES,
  BUCKET_BOUNDARIES,
  CONTRACT_VERSION,
  CUSTOM_INFERENCE_OPERATION,
  catalogAttributeType,
  DEFAULTS,
  ENUMS,
  ENV,
  EVENTS,
  EXTENSION_NAMESPACE,
  GENAI_SEMCONV_REVISION,
  INFERENCE_OPERATIONS,
  INSTRUMENTATION_SCOPE_NAME,
  JSON_ATTRIBUTES,
  METRIC_DIMENSIONS,
  METRICS,
  OPERATIONS,
  PRESETS,
  WIRE,
} from "../src/index.ts";

type Catalog = {
  contract_version: string;
  genai_semconv: { revision: string };
  instrumentation_scope: { name: string };
  extension_namespace: string;
  operations: Record<string, unknown>;
  inference_operations: string[];
  custom_inference_operation: Record<string, unknown>;
  attempt_span_name: string;
  attributes: Record<string, Record<string, string>>;
  enums: Record<string, string[]>;
  metrics: Record<string, unknown>;
  bucket_boundaries: Record<string, number[]>;
  events: Record<string, unknown>;
  metric_dimensions: Record<string, unknown>;
  defaults: Record<string, unknown>;
  env: Record<string, string>;
  presets: Record<string, unknown>;
  wire: Record<string, unknown>;
};

const catalog = JSON.parse(
  readFileSync(
    fileURLToPath(
      new URL("../../../crates/xybrid-observability/contract/catalog.json", import.meta.url),
    ),
    "utf8",
  ),
) as Catalog;

const withoutComment = (value: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(value).filter(([key]) => key !== "$comment"));

describe("catalog parity", () => {
  test("versions and names", () => {
    expect(CONTRACT_VERSION).toBe(catalog.contract_version);
    expect(GENAI_SEMCONV_REVISION).toBe(catalog.genai_semconv.revision);
    expect(INSTRUMENTATION_SCOPE_NAME).toBe(catalog.instrumentation_scope.name);
    expect(EXTENSION_NAMESPACE).toBe(catalog.extension_namespace);
    expect(ATTEMPT_SPAN_NAME).toBe(catalog.attempt_span_name);
  });

  test("operations", () => expect(OPERATIONS).toEqual(catalog.operations as never));
  test("inference operations", () => {
    expect([...INFERENCE_OPERATIONS] as string[]).toEqual(catalog.inference_operations);
    expect(CUSTOM_INFERENCE_OPERATION).toEqual(
      withoutComment(catalog.custom_inference_operation) as never,
    );
  });
  test("wire", () => expect(WIRE).toEqual(withoutComment(catalog.wire) as never));
  test("attribute value types", () => {
    const groups = ["standard", "content", "schema_content", "extension"];
    const merged = Object.assign(
      {},
      ...groups.map((group) => withoutComment(catalog.attributes[group] ?? {})),
    );
    expect(ATTRIBUTE_TYPES).toEqual(merged);
    expect(catalogAttributeType("gen_ai.prompt.variable.city")).toBe("string");
    expect(catalogAttributeType("gen_ai.request.temperature")).toBe("double");
    expect(catalogAttributeType("app.custom")).toBeUndefined();
  });
  test("metrics", () => expect(METRICS).toEqual(catalog.metrics as never));
  test("bucket boundaries", () =>
    expect(BUCKET_BOUNDARIES).toEqual(catalog.bucket_boundaries as never));
  test("events", () => expect(EVENTS).toEqual(catalog.events as never));
  test("metric dimensions", () =>
    expect(METRIC_DIMENSIONS).toEqual(withoutComment(catalog.metric_dimensions) as never));
  test("enums", () => expect(ENUMS).toEqual(catalog.enums as never));
  test("defaults", () => expect(DEFAULTS).toEqual(catalog.defaults as never));
  test("env", () => expect(ENV).toEqual(catalog.env as never));
  test("presets", () => expect(PRESETS).toEqual(catalog.presets as never));

  test("every catalog attribute has a constant and no constant is unknown", () => {
    const groups = ["standard", "content", "schema_content", "extension"];
    const expected = new Set(
      groups.flatMap((group) =>
        Object.keys(withoutComment(catalog.attributes[group] ?? {})).map((key) =>
          key === "gen_ai.prompt.variable.{key}" ? ATTR.PROMPT_VARIABLE_PREFIX : key,
        ),
      ),
    );
    expect(new Set<string>(Object.values(ATTR))).toEqual(expected);
  });

  test("JSON-typed attributes match the catalog", () => {
    const json = Object.entries({
      ...catalog.attributes["content"],
      ...catalog.attributes["schema_content"],
    })
      .filter(([, type]) => type === "json")
      .map(([key]) => key);
    expect(new Set(JSON_ATTRIBUTES)).toEqual(new Set(json));
  });
});
