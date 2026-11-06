import { describe, expect, test } from "bun:test";
import { firstDifference, loadFixtures, runFixture } from "./support/fixtures.ts";

const fixtures = loadFixtures();

describe("contract fixtures", () => {
  test("the shared fixture directory is present", () => {
    expect(fixtures.length).toBeGreaterThan(0);
  });

  for (const fixture of fixtures) {
    test(fixture.name, async () => {
      const actual = await runFixture(fixture);
      // Every fixture runs; expectations are asserted once the contract owner fills them.
      expect(actual.spans.length + actual.metrics.length).toBeGreaterThan(0);
      if (fixture.expect !== null) {
        expect(firstDifference(actual, fixture.expect)).toBeNull();
      }
    });
  }
});
