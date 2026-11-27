"""The wrapper's Jinja closure is listed twice: once for Cargo, once for Bazel.

Both lists must name the same llama.cpp sources, or the two builds render
decision prompts with different code.
"""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VENDOR = "vendor/llama-cpp/"


def cargo_sources():
    """The sources build.rs adds to the wrapper for the Jinja closure."""
    text = (ROOT / "crates/llama-cpp-sys/build.rs").read_text()
    block = re.search(
        r"// Audited minimal Jinja closure.*?for source in \[(.*?)\]", text, re.S
    )
    if block is None:
        raise AssertionError("build.rs: the Jinja closure list was not found")
    return sorted(re.findall(r'"([^"]+)"', block.group(1)))


def bazel_sources():
    """The `.cpp` sources of `//:llama_decision_jinja`, relative to llama.cpp."""
    text = (ROOT / "BUILD.bazel").read_text()
    rule = re.search(
        r'name = "llama_decision_jinja",.*?srcs = \[(.*?)\]', text, re.S
    )
    if rule is None:
        raise AssertionError("BUILD.bazel: //:llama_decision_jinja was not found")
    paths = re.findall(r'"([^"]+)"', rule.group(1))
    outside = [path for path in paths if not path.startswith(VENDOR)]
    if outside:
        raise AssertionError(f"//:llama_decision_jinja: sources outside llama.cpp: {outside}")
    return sorted(path[len(VENDOR):] for path in paths if path.endswith(".cpp"))


class DecisionJinjaSourceTests(unittest.TestCase):
    def test_cargo_and_bazel_compile_the_same_jinja_sources(self):
        cargo = cargo_sources()
        self.assertTrue(cargo, "build.rs lists no Jinja source")
        self.assertTrue(all(path.endswith(".cpp") for path in cargo), cargo)
        self.assertEqual(cargo, bazel_sources())


if __name__ == "__main__":
    unittest.main()
