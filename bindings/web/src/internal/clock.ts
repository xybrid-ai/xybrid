/** Epoch milliseconds anchored to the current wall clock, monotonic for one operation. */
export const epochClock = (): (() => number) => {
  const wall = Date.now();
  const monotonic = performance.now();
  return () => wall + performance.now() - monotonic;
};
