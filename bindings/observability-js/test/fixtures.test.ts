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
});
