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
  readonly arguments?: JsonValue;
};
export type ToolCallResponsePart = {
  readonly type: "tool_call_response";
  readonly id?: string;
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
/** Any other part type is passed through unchanged with `type` first. */
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
 * Replaces content before truncation and serialization. `kind` is the attribute key.
 * If the callback throws, that content is omitted and the operation continues.
 */
export type Redactor = (kind: string, value: unknown) => unknown;

const PART_KEYS: Readonly<Record<string, readonly string[]>> = {
  text: ["type", "content"],
  reasoning: ["type", "content"],
  tool_call: ["type", "id", "name", "arguments"],
  tool_call_response: ["type", "id", "response"],
  uri: ["type", "modality", "mime_type", "uri"],
  file: ["type", "modality", "mime_type", "file_id"],
  media: ["type", "modality", "mime_type", "size_bytes", "duration_s", "width", "height"],
};

const ordered = (value: Record<string, unknown>, keys: readonly string[]): JsonValue => {
  const result: Record<string, JsonValue> = {};
  for (const key of keys) {
    const field = value[key];
    if (field !== undefined && field !== null) result[key] = toJson(field);
  }
  return result;
};

/** Copies a JSON-compatible value, dropping `undefined` members and non-JSON leaves. */
export const toJson = (value: unknown): JsonValue => {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "bigint") return Number(value);
  if (Array.isArray(value)) return value.map((item) => (item === undefined ? null : toJson(item)));
  if (typeof value === "object") {
    const result: Record<string, JsonValue> = {};
    for (const [key, field] of Object.entries(value as Record<string, unknown>)) {
      if (field !== undefined && typeof field !== "function") result[key] = toJson(field);
    }
    return result;
  }
  return null;
};

/** Serializes a part with the contract's key order (CONTRACT.md "Message model"). */
export const canonicalPart = (part: MessagePart): JsonValue => {
  const keys = PART_KEYS[part.type];
  if (keys !== undefined) return ordered(part as Record<string, unknown>, keys);
  const { type, ...rest } = part as GenericPart;
  return { type, ...(toJson(rest) as Record<string, JsonValue>) };
};

export const canonicalParts = (parts: readonly MessagePart[]): JsonValue =>
  parts.map(canonicalPart);

export const canonicalInputMessages = (messages: readonly InputMessage[]): JsonValue =>
  messages.map((message) => {
    const result: Record<string, JsonValue> = {
      role: message.role,
      parts: canonicalParts(message.parts),
    };
    if (message.name !== undefined) result["name"] = message.name;
    return result;
  });

export const canonicalOutputMessages = (messages: readonly OutputMessage[]): JsonValue =>
  messages.map((message) => ({
    role: message.role,
    parts: canonicalParts(message.parts),
    finish_reason: message.finish_reason,
  }));

/** UTF-8 byte length without allocating; a lone surrogate counts as U+FFFD (3 bytes). */
export const utf8Length = (text: string): number => {
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

const maxStringBytes = (value: JsonValue): number => {
  if (typeof value === "string") return utf8Length(value);
  if (Array.isArray(value))
    return value.reduce((max: number, item) => Math.max(max, maxStringBytes(item)), 0);
  if (value !== null && typeof value === "object") {
    let max = 0;
    for (const field of Object.values(value)) max = Math.max(max, maxStringBytes(field));
    return max;
  }
  return 0;
};

/** Truncates one string to `limit` bytes plus the marker when it is longer than `limit`. */
export const truncateString = (text: string, limit: number): string =>
  utf8Length(text) > limit ? `${utf8Prefix(text, limit)}${DEFAULTS.truncation_marker}` : text;

const truncateStrings = (value: JsonValue, limit: number): JsonValue => {
  if (typeof value === "string") return truncateString(value, limit);
  if (Array.isArray(value)) return value.map((item) => truncateStrings(item, limit));
  if (value !== null && typeof value === "object") {
    const result: Record<string, JsonValue> = {};
    for (const [key, field] of Object.entries(value)) result[key] = truncateStrings(field, limit);
    return result;
  }
  return value;
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
 * compact JSON serialization.
 */
export const boundJson = (original: JsonValue, budget: number): Bounded => {
  let json = JSON.stringify(original);
  if (utf8Length(json) <= budget) return { kind: "kept", value: original, json, truncated: false };
  let value = original;
  let limit = maxStringBytes(original);
  while (utf8Length(json) > budget && limit > DEFAULTS.truncation_min_string_bytes) {
    limit = Math.floor(limit / 2);
    value = truncateStrings(original, limit);
    json = JSON.stringify(value);
  }
  if (utf8Length(json) > budget && Array.isArray(value)) {
    const items = [...value];
    while (utf8Length(json) > budget && items.length > 1) {
      items.shift();
      json = JSON.stringify(items);
    }
    value = items;
  }
  if (utf8Length(json) > budget) return { kind: "omitted" };
  return { kind: "kept", value, json, truncated: true };
};

/** The reference redactor used by fixtures: replaces substrings in every JSON string value. */
export const substringRedactor = (
  substrings: readonly string[],
  replacement: string = DEFAULTS.redaction_replacement,
): Redactor => {
  const replace = (value: unknown): unknown => {
    if (typeof value === "string") {
      let result = value;
      for (const substring of substrings) {
        if (substring.length > 0) result = result.split(substring).join(replacement);
      }
      return result;
    }
    if (Array.isArray(value)) return value.map(replace);
    if (value !== null && typeof value === "object") {
      const result: Record<string, unknown> = {};
      for (const [key, field] of Object.entries(value)) result[key] = replace(field);
      return result;
    }
    return value;
  };
  return (_kind, value) => replace(value);
};
