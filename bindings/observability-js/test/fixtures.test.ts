import { describe, expect, test } from "bun:test";
import { firstDifference, loadFixtures, runFixture } from "./support/fixtures.ts";

const fixtures = loadFixtures();

describe("contract fixtures", () => {
  test("the shared fixture directory is present", () => {
    expect(fixtures.length).toBeGreaterThan(0);
  });

  for (const fixture of fixtures) {
    test(fixture.name, async () => {
      // runFixture rejects a null expectation, unknown fields and failed inline assertions.
      const actual = await runFixture(fixture);
      expect(firstDifference(actual, fixture.expect)).toBeNull();
    });
  }

  test("content_unrecorded_span_event_budget without a logger charges only the span copy", async () => {
    // As in Rust, an observer with no logger has no event destination: the event copy is never
    // charged, so the span copy (95 bytes) fits the 300-byte budget and nothing is truncated.
    const fixture = fixtures.find((f) => f.name === "content_unrecorded_span_event_budget");
    if (fixture === undefined) throw new Error("missing content_unrecorded_span_event_budget");
    const actual = await runFixture({ ...fixture, config: { ...fixture.config, logs: false } });
    expect(actual.logs).toEqual([]);
    const recorded = actual.spans.find((span) => span.id === "recorded");
    expect(recorded?.attributes["gen_ai.input.messages"]).toEqual([
      { role: "user", parts: [{ type: "text", content: "hi" }] },
    ]);
    expect(Object.keys(recorded?.attributes ?? {})).not.toContain(
      "xybrid.genai.content.truncated_attributes",
    );
  });
});
