import { expect, spyOn, test } from "bun:test";
import { epochClock } from "../src/internal/clock.ts";

test("operation clocks use the current epoch anchor and monotonic elapsed time", () => {
  const wall = spyOn(Date, "now").mockReturnValue(1_800_000_000_000);
  const monotonic = spyOn(performance, "now").mockReturnValue(100);
  try {
    const first = epochClock();
    wall.mockReturnValue(1_700_000_000_000);
    monotonic.mockReturnValue(125);
    expect(first()).toBe(1_800_000_000_025);
    // Each subsequent operation anchors to the then-current wall clock.
    const second = epochClock();
    monotonic.mockReturnValue(150);
    expect(first()).toBe(1_800_000_000_050);
    expect(second()).toBe(1_700_000_000_025);
  } finally {
    wall.mockRestore();
    monotonic.mockRestore();
  }
});
