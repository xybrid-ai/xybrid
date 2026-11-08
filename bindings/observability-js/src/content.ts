import { DEFAULTS } from "./catalog.ts";

/** Any JSON value. */
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue };

export type Modality = "text" | "image" | "audio" | "video" | (string & {});

export type TextPart = { readonly type: "text"; readonly content: string };
export type ReasoningPart = { readonly type: "reasoning"; readonly content: string };
export type ToolCallPart = {
  readonly type: "tool_call";
  readonly id?: string;
  readonly name: string;
  /** Omitted from the recorded part when absent or `null`. */
  readonly arguments?: JsonValue;
};
export type ToolCallResponsePart = {
  readonly type: "tool_call_response";
  readonly id?: string;
  /** Always recorded; `null` when absent. */
  readonly response: JsonValue;
};
export type UriPart = {
  readonly type: "uri";
  readonly modality: Modality;
  readonly mime_type?: string;
  readonly uri: string;
};
export type FilePart = {
  readonly type: "file";
  readonly modality: Modality;
  readonly mime_type?: string;
  readonly file_id: string;
};
/** Media described by metadata only; raw bytes are never accepted. */
export type MediaPart = {
  readonly type: "media";
  readonly modality: Modality;
  readonly mime_type?: string;
  readonly size_bytes?: number;
  readonly duration_s?: number;
  readonly width?: number;
  readonly height?: number;
};
/** Any other part type is passed through with `type` first; binary values are rejected. */
export type GenericPart = { readonly type: string; readonly [key: string]: JsonValue | undefined };

export type MessagePart =
  | TextPart
  | ReasoningPart
  | ToolCallPart
  | ToolCallResponsePart
  | UriPart
  | FilePart
  | MediaPart
  | GenericPart;

export type InputMessage = {
  readonly role: string;
  readonly parts: readonly MessagePart[];
  readonly name?: string;
};
export type OutputMessage = {
  readonly role: string;
  readonly parts: readonly MessagePart[];
  readonly finish_reason: string;
};

/**
 * Callback redactor. `kind` is the attribute key (`gen_ai.input.messages`,
 * `gen_ai.prompt.variable.<name>`, `error.message`, `xybrid.genai.feedback.comment`, …) and
 * `value` is the package's canonical copy (a string for string kinds), never the caller's object.
 *
 * Return the value to record, `undefined` to drop it silently, or `null` for JSON null. A thrown
 * error, a returned promise or thenable, or an output that is not JSON data (or not a string for
 * string kinds) omits that value and marks it in `xybrid.genai.content.truncated_attributes`;
 * later values are redacted normally.
 */
export type Redactor = (kind: string, value: JsonValue) => JsonValue | undefined;

/**
 * Declarative redaction rule, applied after the callback redactor. `replace_literal` replaces
 * every non-overlapping occurrence of `find` (left to right) in every JSON string value, never
 * in object keys; rules run in list order, each on the previous rule's output. `omit_attribute`
 * drops exactly the named attribute key (case-sensitive) before any other processing.
 */
export type RedactionRule =
  | { readonly type: "replace_literal"; readonly find: string; readonly replace: string }
  | { readonly type: "omit_attribute"; readonly key: string };

const RULE_ERROR = "content.rules";

const toRedactionRule = (rule: unknown, where: string): RedactionRule => {
  if (typeof rule !== "object" || rule === null || Array.isArray(rule)) {
    throw new TypeError(`${where} must be a rule object.`);
  }
  const fields = rule as Record<string, unknown>;
  if (fields["type"] === "replace_literal") {
    const find = fields["find"];
    const replace = fields["replace"];
    if (typeof find !== "string" || find.length === 0) {
      throw new TypeError(`${where}.find must be a non-empty string.`);
    }
    if (typeof replace !== "string") throw new TypeError(`${where}.replace must be a string.`);
    return Object.freeze({ type: "replace_literal", find, replace });
  }
  if (fields["type"] === "omit_attribute") {
    const key = fields["key"];
    if (typeof key !== "string" || key.length === 0) {
      throw new TypeError(`${where}.key must be a non-empty string.`);
    }
    return Object.freeze({ type: "omit_attribute", key });
  }
  throw new TypeError(`${where}.type must be "replace_literal" or "omit_attribute".`);
};

/**
 * Validates declarative redaction rules and returns frozen canonical copies (`type` first).
 * Throws `TypeError` for anything but an array of valid rules: unknown `type`, empty `find` or
 * `key`, or non-string fields. Extra members are ignored.
 */
export const toRedactionRules = (rules: unknown): readonly RedactionRule[] => {
  if (!Array.isArray(rules)) throw new TypeError(`${RULE_ERROR} must be an array of rules.`);
  const validated: RedactionRule[] = [];
  for (let index = 0; index < rules.length; index += 1) {
    validated.push(toRedactionRule(rules[index], `${RULE_ERROR}[${index}]`));
  }
  return Object.freeze(validated);
};

/** UTF-8 byte length without allocating; a lone surrogate counts as U+FFFD (3 bytes). */
export const utf8Length = (text: string): number => {
  // Most serialized content is ASCII. The engine can scan it without a JS iteration per byte.
  if (/^\p{ASCII}*$/u.test(text)) return text.length;
  let bytes = 0;
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    if (code < 0x80) bytes += 1;
    else if (code < 0x800) bytes += 2;
    else if (code >= 0xd800 && code <= 0xdbff && index + 1 < text.length) {
      const next = text.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4;
        index += 1;
      } else bytes += 3;
    } else bytes += 3;
  }
  return bytes;
};

/** Longest prefix of at most `maxBytes` UTF-8 bytes that does not split a code point. */
export const utf8Prefix = (text: string, maxBytes: number): string => {
  let bytes = 0;
  let end = 0;
  while (end < text.length) {
    const code = text.charCodeAt(end);
    let width = 1;
    let units = 1;
    if (code >= 0x80 && code < 0x800) width = 2;
    else if (code >= 0xd800 && code <= 0xdbff && end + 1 < text.length) {
      const next = text.charCodeAt(end + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        width = 4;
        units = 2;
      } else width = 3;
    } else if (code >= 0x800) width = 3;
    if (bytes + width > maxBytes) break;
    bytes += width;
    end += units;
  }
  return text.slice(0, end);
};

/** Sets an own enumerable data property, so a `__proto__` key stays an ordinary key. */
const setOwn = (target: Record<string, unknown>, key: string, value: unknown): void => {
  if (key === "__proto__") {
    Object.defineProperty(target, key, {
      value,
      enumerable: true,
      writable: true,
      configurable: true,
    });
  } else target[key] = value;
};

// ---------------------------------------------------------------------------------------------
// Preflight, canonical clone and validation (CONTRACT.md "Content pipeline").

/** Why a value failed closed. Never carries content. */
export type ContentFailure =
  | "depth"
  | "traversal"
  | "unrepresentable"
  | "redactor_error"
  | "redactor_output"
  | "rules";

class Failure {
  constructor(readonly reason: ContentFailure) {}
}

const OMIT: unique symbol = Symbol("omit");
type Walked = JsonValue | typeof OMIT;

/** Canonical shapes: plain JSON or the contract's message model. */
export type ContentShape = "json" | "string" | "input_messages" | "output_messages" | "parts";

const NODE = DEFAULTS.preflight_node_bytes;
const MAX_DEPTH = DEFAULTS.max_json_depth;

const isBinary = (value: object): boolean =>
  value instanceof ArrayBuffer ||
  ArrayBuffer.isView(value) ||
  (typeof SharedArrayBuffer !== "undefined" && value instanceof SharedArrayBuffer) ||
  (typeof Blob !== "undefined" && value instanceof Blob);

const PART_KEYS: Readonly<Record<string, readonly string[]>> = {
  text: ["type", "content"],
  reasoning: ["type", "content"],
  tool_call: ["type", "id", "name", "arguments"],
  tool_call_response: ["type", "id", "response"],
  uri: ["type", "modality", "mime_type", "uri"],
  file: ["type", "modality", "mime_type", "file_id"],
  media: ["type", "modality", "mime_type", "size_bytes", "duration_s", "width", "height"],
};

/**
 * One bounded walk over the canonical JSON form of a caller's value. With `build` false it only
 * measures (preflight); with `build` true it also returns the canonical copy. Either way it
 * stops at the first violation: depth > 128, cost > the traversal budget, or a value JSON cannot
 * represent (cycle, BigInt, binary data, throwing getter or `toJSON`). Otherwise it follows
 * `JSON.stringify`.
 */
class CanonicalWalk {
  cost = 0;
  readonly #limit: number;
  readonly #build: boolean;
  readonly #ancestors = new Set<object>();

  constructor(limit: number, build: boolean) {
    this.#limit = limit;
    this.#build = build;
  }

  #add(bytes: number): void {
    this.cost += bytes;
    if (this.cost > this.#limit) throw new Failure("traversal");
  }

  #string(text: string): string {
    // UTF-8 length >= UTF-16 length: an over-long string fails without being scanned.
    if (this.cost + NODE + text.length > this.#limit) throw new Failure("traversal");
    this.#add(NODE + utf8Length(text));
    return text;
  }

  #enter(value: object, level: number): void {
    if (level > MAX_DEPTH) throw new Failure("depth");
    if (this.#ancestors.has(value)) throw new Failure("unrepresentable");
    this.#ancestors.add(value);
    this.#add(NODE);
  }

  #leave(value: object): void {
    this.#ancestors.delete(value);
  }

  #member(target: Record<string, unknown> | undefined, key: string, value: JsonValue): void {
    this.#add(NODE + utf8Length(key));
    if (target !== undefined) setOwn(target, key, value);
  }

  /** `JSON.stringify` semantics for one value at nesting `level` (containers above it). */
  json(input: unknown, key: string, level: number): Walked {
    let value = input;
    if (typeof value === "bigint") throw new Failure("unrepresentable");
    if (typeof value === "object" && value !== null) {
      if (isBinary(value)) throw new Failure("unrepresentable");
      const toJSON = (value as { toJSON?: unknown }).toJSON;
      if (typeof toJSON === "function") value = toJSON.call(value, key);
    }
    if (typeof value === "object" && value !== null) {
      if (value instanceof Number) value = Number(value);
      else if (value instanceof String) value = String(value);
      else if (value instanceof Boolean) value = value.valueOf();
      else if (typeof BigInt !== "undefined" && value instanceof BigInt) {
        throw new Failure("unrepresentable");
      }
    }
    if (value === null || typeof value === "boolean") {
      this.#add(NODE);
      return value;
    }
    if (typeof value === "number") {
      this.#add(NODE);
      return Number.isFinite(value) ? value : null;
    }
    if (typeof value === "string") return this.#string(value);
    if (typeof value === "bigint") throw new Failure("unrepresentable");
    if (typeof value !== "object") return OMIT; // undefined, function, symbol
    if (isBinary(value)) throw new Failure("unrepresentable");
    this.#enter(value, level + 1);
    let result: JsonValue;
    if (Array.isArray(value)) {
      const items: JsonValue[] | undefined = this.#build ? [] : undefined;
      for (let index = 0; index < value.length; index += 1) {
        const item = this.json(value[index], String(index), level + 1);
        const kept = item === OMIT ? this.#nullInArray() : item;
        items?.push(kept);
      }
      result = items ?? null;
    } else {
      const target: Record<string, unknown> | undefined = this.#build ? {} : undefined;
      for (const field of Object.keys(value)) {
        const item = this.json((value as Record<string, unknown>)[field], field, level + 1);
        if (item !== OMIT) this.#member(target, field, item);
      }
      result = (target as JsonValue | undefined) ?? null;
    }
    this.#leave(value);
    return result;
  }

  #nullInArray(): null {
    this.#add(NODE);
    return null;
  }

  /** Top-level value: `undefined`, functions and symbols become `null`. */
  root(value: unknown, shape: ContentShape): JsonValue {
    switch (shape) {
      case "string":
        if (typeof value !== "string") throw new Failure("unrepresentable");
        return this.#string(value);
      case "input_messages":
        return this.#messages(value, false);
      case "output_messages":
        return this.#messages(value, true);
      case "parts":
        return this.#parts(value, 0);
      default: {
        const result = this.json(value, "", 0);
        return result === OMIT ? this.#nullInArray() : result;
      }
    }
  }

  #messages(value: unknown, output: boolean): JsonValue {
    if (!Array.isArray(value)) throw new Failure("unrepresentable");
    this.#enter(value, 1);
    const items: JsonValue[] | undefined = this.#build ? [] : undefined;
    for (const message of value) {
      if (typeof message !== "object" || message === null || Array.isArray(message)) {
        throw new Failure("unrepresentable");
      }
      this.#enter(message, 2);
      const fields = message as Record<string, unknown>;
      const target: Record<string, unknown> | undefined = this.#build ? {} : undefined;
      const role = this.json(fields["role"], "role", 2);
      if (role !== OMIT) this.#member(target, "role", role);
      this.#member(target, "parts", this.#parts(fields["parts"], 2));
      const last = output ? "finish_reason" : "name";
      const lastValue = fields[last];
      if (output || lastValue !== undefined) {
        const walked = this.json(lastValue, last, 2);
        if (walked !== OMIT) this.#member(target, last, walked);
      }
      this.#leave(message);
      items?.push((target as JsonValue | undefined) ?? null);
    }
    this.#leave(value);
    return items ?? null;
  }

  #parts(value: unknown, level: number): JsonValue {
    if (!Array.isArray(value)) throw new Failure("unrepresentable");
    this.#enter(value, level + 1);
    const items: JsonValue[] | undefined = this.#build ? [] : undefined;
    for (let index = 0; index < value.length; index += 1) {
      // Walk even during preflight: optional-call arguments would skip the entire part.
      const part = this.#part(value[index], String(index), level + 1);
      items?.push(part);
    }
    this.#leave(value);
    return items ?? null;
  }

  #part(value: unknown, key: string, level: number): JsonValue {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      const walked = this.json(value, key, level);
      return walked === OMIT ? this.#nullInArray() : walked;
    }
    if (isBinary(value)) throw new Failure("unrepresentable");
    this.#enter(value, level + 1);
    const fields = value as Record<string, unknown>;
    const target: Record<string, unknown> | undefined = this.#build ? {} : undefined;
    const type = fields["type"];
    const keys =
      typeof type === "string" && Object.hasOwn(PART_KEYS, type) ? PART_KEYS[type] : undefined;
    if (keys !== undefined) {
      for (const field of keys) {
        let item = fields[field];
        if (field === "response" && item === undefined) item = null;
        if (item === undefined || (item === null && field !== "response")) continue;
        let walked = this.json(item, field, level + 1);
        if (walked === OMIT && field === "response") walked = this.#nullInArray();
        if (field === "arguments" && walked === null) {
          // Arguments that serialize to null (NaN, a toJSON returning null) are omitted too.
          this.cost -= NODE;
          continue;
        }
        if (walked !== OMIT) this.#member(target, field, walked);
      }
    } else {
      // Generic part: `type` first, then the other members as JSON.stringify would write them.
      const walkedType = this.json(type, "type", level + 1);
      if (walkedType !== OMIT) this.#member(target, "type", walkedType);
      for (const field of Object.keys(fields)) {
        if (field === "type") continue;
        const walked = this.json(fields[field], field, level + 1);
        if (walked !== OMIT) this.#member(target, field, walked);
      }
    }
    this.#leave(value);
    return (target as JsonValue | undefined) ?? null;
  }
}

const isThenable = (value: unknown): boolean =>
  (typeof value === "object" || typeof value === "function") &&
  value !== null &&
  typeof (value as { then?: unknown }).then === "function";

/**
 * Validates and copies callback output: only `null`, booleans, finite numbers, strings, arrays
 * and plain objects (object members that are `undefined` are skipped, as `JSON.stringify` does),
 * no cycles, depth <= 128, cost <= the traversal budget.
 */
class OutputWalk {
  cost = 0;
  readonly #limit: number;
  readonly #ancestors = new Set<object>();

  constructor(limit: number) {
    this.#limit = limit;
  }

  #add(bytes: number): void {
    this.cost += bytes;
    if (this.cost > this.#limit) throw new Failure("redactor_output");
  }

  value(value: unknown, level: number): JsonValue {
    if (value === null || typeof value === "boolean") {
      this.#add(NODE);
      return value;
    }
    if (typeof value === "number") {
      if (!Number.isFinite(value)) throw new Failure("redactor_output");
      this.#add(NODE);
      return value;
    }
    if (typeof value === "string") {
      if (this.cost + NODE + value.length > this.#limit) throw new Failure("redactor_output");
      this.#add(NODE + utf8Length(value));
      return value;
    }
    if (typeof value !== "object" || isThenable(value)) throw new Failure("redactor_output");
    const prototype = Object.getPrototypeOf(value);
    const array = Array.isArray(value);
    if (!array && prototype !== Object.prototype && prototype !== null) {
      throw new Failure("redactor_output");
    }
    if (level + 1 > MAX_DEPTH || this.#ancestors.has(value)) throw new Failure("redactor_output");
    this.#ancestors.add(value);
    this.#add(NODE);
    let result: JsonValue;
    if (array) {
      const items: JsonValue[] = [];
      for (let index = 0; index < value.length; index += 1) {
        items.push(this.value(value[index], level + 1));
      }
      result = items;
    } else {
      const target: Record<string, unknown> = {};
      for (const field of Object.keys(value)) {
        const item = (value as Record<string, unknown>)[field];
        if (item === undefined) continue;
        this.#add(NODE + utf8Length(field));
        setOwn(target, field, this.value(item, level + 1));
      }
      result = target as JsonValue;
    }
    this.#ancestors.delete(value);
    return result;
  }
}

/** Preflight cost of a JSON value (CONTRACT.md "Content pipeline"). */
export const jsonCost = (value: JsonValue): number => {
  if (value === null || typeof value !== "object") {
    return typeof value === "string" ? NODE + utf8Length(value) : NODE;
  }
  let cost = NODE;
  if (Array.isArray(value)) {
    for (const item of value) cost += jsonCost(item);
  } else {
    for (const [key, item] of Object.entries(value))
      cost += NODE + utf8Length(key) + jsonCost(item);
  }
  return cost;
};

/** Counts non-overlapping occurrences of `find`, scanning left to right. */
const occurrences = (text: string, find: string): number => {
  let count = 0;
  let from = text.indexOf(find);
  while (from !== -1) {
    count += 1;
    from = text.indexOf(find, from + find.length);
  }
  return count;
};

type ReplaceRule = Extract<RedactionRule, { type: "replace_literal" }>;

/**
 * Applies the replacement rules in order to every string value of `value` (owned by the caller
 * and rewritten in place; each string gets every rule in list order, which equals applying the
 * rules one after another to the whole document). Fails when a single string, after any rule,
 * would cost more than the traversal budget (checked before that string is built), or when the
 * final document cost exceeds it. Both checks are independent of traversal order (contract
 * amendment A1), so Rust and JS agree whatever order they visit object members in.
 */
const applyReplacements = (
  value: JsonValue,
  rules: readonly ReplaceRule[],
  startCost: number,
  limit: number,
): JsonValue => {
  let cost = startCost;
  const rewrite = (text: string): string => {
    let result = text;
    let length = utf8Length(text);
    for (const rule of rules) {
      const count = occurrences(result, rule.find);
      if (count === 0) continue;
      const next = length + count * (utf8Length(rule.replace) - utf8Length(rule.find));
      if (NODE + next > limit) throw new Failure("rules");
      result = result.split(rule.find).join(rule.replace);
      cost += next - length;
      length = next;
    }
    return result;
  };
  const visit = (node: JsonValue): JsonValue => {
    if (typeof node === "string") return rewrite(node);
    if (node === null || typeof node !== "object") return node;
    if (Array.isArray(node)) {
      const items = node as JsonValue[];
      for (let index = 0; index < items.length; index += 1) {
        items[index] = visit(items[index] as JsonValue);
      }
      return items;
    }
    const record = node as Record<string, JsonValue>;
    for (const key of Object.keys(record)) setOwn(record, key, visit(record[key] as JsonValue));
    return record;
  };
  const result = visit(value);
  if (cost > limit) throw new Failure("rules");
  return result;
};

/** Outcome of the content pipeline before truncation. */
export type ContentOutcome =
  | { readonly kind: "value"; readonly value: JsonValue }
  | { readonly kind: "dropped" }
  | { readonly kind: "failed"; readonly reason: ContentFailure };

export type ContentProcessorOptions = {
  readonly redactor?: Redactor | undefined;
  readonly rules?: readonly RedactionRule[] | undefined;
  readonly maxTraversalBytes: number;
};

/**
 * The content pipeline up to (not including) truncation: omit rules, preflight before cloning,
 * canonical clone, callback redactor, output validation, replacement rules. It never throws for
 * content and never reports content in its outcome.
 */
export class ContentProcessor {
  readonly #redactor: Redactor | undefined;
  readonly #omit: ReadonlySet<string>;
  readonly #replace: readonly ReplaceRule[];
  readonly #limit: number;

  constructor(options: ContentProcessorOptions) {
    this.#redactor = options.redactor;
    const rules = options.rules ?? [];
    this.#omit = new Set(
      rules.flatMap((rule) => (rule.type === "omit_attribute" ? [rule.key] : [])),
    );
    this.#replace = rules.filter((rule): rule is ReplaceRule => rule.type === "replace_literal");
    this.#limit = options.maxTraversalBytes;
  }

  /** Whether an `omit_attribute` rule names `kind`. */
  omits(kind: string): boolean {
    return this.#omit.has(kind);
  }

  process(kind: string, raw: unknown, shape: ContentShape): ContentOutcome {
    if (this.#omit.has(kind)) return { kind: "dropped" };
    let value: JsonValue;
    let cost: number;
    try {
      // Preflight measures the caller's value without copying it; only then is it cloned. The
      // clone re-checks every limit, so a value that changes between the walks still fails.
      new CanonicalWalk(this.#limit, false).root(raw, shape);
      const clone = new CanonicalWalk(this.#limit, true);
      value = clone.root(raw, shape);
      cost = clone.cost;
    } catch (error) {
      return {
        kind: "failed",
        reason: error instanceof Failure ? error.reason : "unrepresentable",
      };
    }
    if (this.#redactor !== undefined) {
      let output: unknown;
      try {
        output = this.#redactor(kind, value);
      } catch {
        // Never report the thrown value: it may contain content.
        return { kind: "failed", reason: "redactor_error" };
      }
      if (output === undefined) return { kind: "dropped" };
      if (isThenable(output)) {
        // A rejected promise must not surface as an unhandled rejection.
        Promise.resolve(output).catch(() => undefined);
        return { kind: "failed", reason: "redactor_output" };
      }
      if (shape === "string" && typeof output !== "string") {
        return { kind: "failed", reason: "redactor_output" };
      }
      try {
        const walk = new OutputWalk(this.#limit);
        value = walk.value(output, 0);
        cost = walk.cost;
      } catch {
        return { kind: "failed", reason: "redactor_output" };
      }
    }
    if (this.#replace.length > 0) {
      try {
        value = applyReplacements(value, this.#replace, cost, this.#limit);
      } catch {
        return { kind: "failed", reason: "rules" };
      }
    }
    return { kind: "value", value };
  }
}

// ---------------------------------------------------------------------------------------------
// Truncation (CONTRACT.md "Truncation"), unchanged output, linear array step.

const maxStringBytes = (value: JsonValue): number => {
  if (typeof value === "string") return utf8Length(value);
  if (value === null || typeof value !== "object") return 0;
  let max = 0;
  for (const field of Array.isArray(value) ? value : Object.values(value)) {
    max = Math.max(max, maxStringBytes(field));
  }
  return max;
};

/** Truncates one string to `limit` bytes plus the marker when it is longer than `limit`. */
export const truncateString = (text: string, limit: number): string =>
  utf8Length(text) > limit ? `${utf8Prefix(text, limit)}${DEFAULTS.truncation_marker}` : text;

const truncateStrings = (value: JsonValue, limit: number): JsonValue => {
  if (typeof value === "string") return truncateString(value, limit);
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map((item: JsonValue) => truncateStrings(item, limit));
  const result: Record<string, unknown> = {};
  for (const [key, field] of Object.entries(value))
    setOwn(result, key, truncateStrings(field, limit));
  return result as JsonValue;
};

export type Bounded =
  | {
      readonly kind: "kept";
      readonly value: JsonValue;
      readonly json: string;
      readonly truncated: boolean;
    }
  | { readonly kind: "omitted" };

/**
 * The contract's deterministic truncation (CONTRACT.md "Truncation"): halve the longest string
 * budget, then drop the oldest array elements, then omit. Lengths are UTF-8 bytes of the
 * compact JSON serialization. The array step serializes each element once (linear work).
 */
export const boundJson = (original: JsonValue, budget: number): Bounded => {
  let json = JSON.stringify(original);
  let length = utf8Length(json);
  if (length <= budget) return { kind: "kept", value: original, json, truncated: false };
  let value = original;
  let limit = maxStringBytes(original);
  while (length > budget && limit > DEFAULTS.truncation_min_string_bytes) {
    limit = Math.floor(limit / 2);
    value = truncateStrings(original, limit);
    json = JSON.stringify(value);
    length = utf8Length(json);
  }
  if (length > budget && Array.isArray(value)) {
    const items = value as readonly JsonValue[];
    const encoded = items.map((item) => JSON.stringify(item));
    const sizes = encoded.map(utf8Length);
    let start = 0;
    while (length > budget && items.length - start > 1) {
      length -= (sizes[start] ?? 0) + 1;
      start += 1;
    }
    if (start > 0) {
      value = items.slice(start);
      json = `[${encoded.slice(start).join(",")}]`;
    }
  }
  if (length > budget) return { kind: "omitted" };
  return { kind: "kept", value, json, truncated: true };
};

/** Bounds a plain string to `limit` code points (error messages). */
export const boundCodePoints = (text: string, limit: number): string => {
  let count = 0;
  let index = 0;
  while (index < text.length) {
    if (count === limit) return text.slice(0, index);
    const code = text.charCodeAt(index);
    const pair =
      code >= 0xd800 && code <= 0xdbff && index + 1 < text.length ? text.charCodeAt(index + 1) : 0;
    index += pair >= 0xdc00 && pair <= 0xdfff ? 2 : 1;
    count += 1;
  }
  return text;
};

/** The reference redactor used by fixtures: replaces substrings in every JSON string value. */
export const substringRedactor = (
  substrings: readonly string[],
  replacement: string = DEFAULTS.redaction_replacement,
): Redactor => {
  const replace = (value: JsonValue): JsonValue => {
    if (typeof value === "string") {
      let result = value;
      for (const substring of substrings) {
        if (substring.length > 0) result = result.split(substring).join(replacement);
      }
      return result;
    }
    if (value === null || typeof value !== "object") return value;
    if (Array.isArray(value)) return value.map(replace);
    const result: Record<string, unknown> = {};
    for (const [key, field] of Object.entries(value)) setOwn(result, key, replace(field));
    return result as JsonValue;
  };
  return (_kind, value) => replace(value);
};
