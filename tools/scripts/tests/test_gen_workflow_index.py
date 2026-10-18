"""Unit tests for the workflow index generator (no repository checkout required)."""

import contextlib
import io
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gen_workflow_index as gen

SAMPLE = textwrap.dedent(
    """\
    # Purpose: Build things.
    name: "CI: Sample"

    on:
      push:
        branches: [master]
        paths-ignore:
          - '**.md'
      pull_request:
        types: [closed]
        branches:
          - master
          - 'release/**'
      schedule:
        - cron: "17 6 * * 1"
      workflow_dispatch:
        inputs:
          tag:
            description: 'Tag'   # a comment
            required: true

    concurrency:
      group: sample-${{ github.ref }}
      cancel-in-progress: true

    permissions:
      contents: read

    jobs:
      build:
        runs-on: ubuntu-latest
        permissions:
          contents: write   # release assets
          id-token: write
          checks: read
        steps:
          - run: echo hi
    """
)


def write(directory: Path, file: str, text: str) -> Path:
    path = directory / file
    path.write_text(text, encoding="utf-8")
    return path


def make(file="a.yml", name="CI: A", purpose="Does A.", concurrency=()):
    return gen.Workflow(
        file=file,
        name=name,
        purpose=purpose,
        triggers=("push to `master`",),
        writes=(),
        concurrency=tuple(concurrency),
    )


class ParseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def parse(self, text: str) -> "gen.Workflow":
        return gen.parse_workflow(write(self.dir, "w.yml", textwrap.dedent(text)))

    def test_reads_name_purpose_triggers_writes_and_concurrency(self):
        workflow = gen.parse_workflow(write(self.dir, "sample.yml", SAMPLE))
        self.assertEqual(workflow.file, "sample.yml")
        self.assertEqual(workflow.name, "CI: Sample")
        self.assertEqual(workflow.purpose, "Build things.")
        self.assertEqual(
            workflow.triggers,
            (
                "push to `master`",
                "pull_request (closed) to `master`, `release/**`",
                "schedule (`17 6 * * 1`)",
                "workflow_dispatch",
            ),
        )
        self.assertEqual(workflow.writes, ("contents", "id-token"))
        self.assertEqual(workflow.concurrency, ("sample-${{ github.ref }}",))
        self.assertEqual(workflow.group, "CI")

    def test_lists_events_in_a_fixed_order_whatever_the_file_order(self):
        workflow = self.parse(
            """\
            # Purpose: x
            name: "CI: X"
            on:
              workflow_dispatch:
              schedule:
                - cron: '0 1 * * *'
              pull_request:
              push:
            """
        )
        self.assertEqual(workflow.triggers, ("push", "pull_request", "schedule (`0 1 * * *`)", "workflow_dispatch"))

    def test_renders_tag_filters(self):
        workflow = self.parse(
            """\
            # Purpose: x
            name: "Release: X"
            on:
              push:
                tags: ["v*"]
            """
        )
        self.assertEqual(workflow.triggers, ("push of tags `v*`",))

    def test_reads_inline_on(self):
        self.assertEqual(self.parse('# Purpose: x\nname: "CI: X"\non: push\n').triggers, ("push",))
        self.assertEqual(
            self.parse('# Purpose: x\nname: "CI: X"\non: [push, pull_request]\n').triggers,
            ("push", "pull_request"),
        )

    def test_permission_scalars(self):
        self.assertEqual(self.parse('name: "CI: X"\non: push\npermissions: read-all\n').writes, ())
        self.assertEqual(self.parse('name: "CI: X"\non: push\npermissions: write-all\n').writes, ("all",))

    def test_purpose_must_sit_directly_above_name(self):
        workflow = self.parse('# Purpose: x\n\nname: "CI: X"\non: push\n')
        self.assertIsNone(workflow.purpose)

    def test_name_on_the_first_line_does_not_borrow_the_last_line(self):
        # lines[-1] is the line "above" index 0 in Python; it must not count.
        workflow = self.parse('name: "CI: X"\non: push\n# Purpose: trailing\n')
        self.assertIsNone(workflow.purpose)

    def test_refuses_shapes_it_cannot_read(self):
        cases = {
            "an ignore filter changes what a row means": 'name: "CI: X"\non:\n  push:\n    branches-ignore: [docs]\n',
            "a schedule with no cron": 'name: "CI: X"\non:\n  schedule:\n    - foo: bar\n',
            "an escaped quote in a scalar": 'name: "CI: \\"X\\""\non: push\n',
            "no on:": 'name: "CI: X"\n',
        }
        for label, text in cases.items():
            with self.subTest(label):
                with self.assertRaises(gen.Unsupported) as raised:
                    gen.parse_workflow(write(self.dir, "w.yml", text))
                self.assertIn("w.yml", str(raised.exception))


class ProblemsTests(unittest.TestCase):
    def test_accepts_conforming_workflows(self):
        self.assertEqual(gen.problems([make("a.yml", "CI: A"), make("b.yml", "SDK: B")]), [])

    def test_flags_each_rule(self):
        found = gen.problems(
            [
                make("no-group.yml", name="Just a name"),
                make("wrong-group.yml", name="Docs: Site"),
                make("no-purpose.yml", name="CI: C", purpose=None),
                make("keyed-by-name.yml", name="CI: D", concurrency=["${{ github.workflow }}-${{ github.ref }}"]),
                make("dup-1.yml", name="CI: E"),
                make("dup-2.yml", name="CI: E"),
            ]
        )
        joined = "\n".join(found)
        self.assertIn("no-group.yml: name 'Just a name' must be 'Group: Subject'", joined)
        self.assertIn("wrong-group.yml: name 'Docs: Site' must be 'Group: Subject'", joined)
        self.assertIn("no-purpose.yml: add '# Purpose:", joined)
        self.assertIn("keyed-by-name.yml: concurrency group", joined)
        self.assertIn("dup-2.yml: name 'CI: E' is already used by dup-1.yml", joined)
        self.assertEqual(len(found), 5)


class RenderTests(unittest.TestCase):
    def test_orders_groups_then_names_and_escapes_pipes(self):
        table = gen.render(
            [
                make("z.yml", "SDK: Zed"),
                make("b.yml", "CI: beta"),
                make("a.yml", "CI: Alpha", purpose="Left | right."),
            ]
        )
        self.assertLess(table.index("### CI"), table.index("### SDK"))
        self.assertLess(table.index("CI: Alpha"), table.index("CI: beta"))
        self.assertIn("Left \\| right.", table)
        self.assertNotIn("### Release", table)  # empty groups are omitted
        self.assertTrue(table.endswith("\n"))


class SpliceTests(unittest.TestCase):
    def test_replaces_only_between_the_markers(self):
        readme = f"# Title\n\n{gen.BEGIN}\nold\n{gen.END}\n\nafter\n"
        result = gen.splice(readme, "new table\n")
        self.assertEqual(result, f"# Title\n\n{gen.BEGIN}\n\nnew table\n\n{gen.END}\n\nafter\n")
        self.assertEqual(gen.splice(result, "new table\n"), result)  # idempotent

    def test_requires_both_markers_once_and_in_order(self):
        for label, readme in {
            "missing": "# Title\n",
            "duplicated": f"{gen.BEGIN}\n{gen.END}\n{gen.BEGIN}\n{gen.END}\n",
            "reversed": f"{gen.END}\n{gen.BEGIN}\n",
        }.items():
            with self.subTest(label):
                with self.assertRaises(gen.Unsupported):
                    gen.splice(readme, "t\n")


class MainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.readme = self.dir / "README.md"
        self.readme.write_text(f"# Workflows\n\n{gen.BEGIN}\n{gen.END}\n", encoding="utf-8")
        write(self.dir, "sample.yml", SAMPLE)
        for target, value in (("WORKFLOWS_DIR", self.dir), ("README", self.readme)):
            patcher = patch.object(gen, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = gen.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_check_reports_a_stale_table_and_writing_repairs_it(self):
        before = self.readme.read_text(encoding="utf-8")
        code, _, err = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("stale", err)
        self.assertEqual(self.readme.read_text(encoding="utf-8"), before)  # --check writes nothing

        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("updated", out)
        self.assertIn("| [`sample.yml`](sample.yml) | CI: Sample |", self.readme.read_text(encoding="utf-8"))

        code, out, _ = self.run_main("--check")
        self.assertEqual((code, out.strip()), (0, "workflow index is current"))

    def test_rule_violations_fail_and_leave_the_readme_alone(self):
        write(self.dir, "bad.yml", "name: Bad name\non: push\n")
        before = self.readme.read_text(encoding="utf-8")
        for argv in (["--check"], []):
            with self.subTest(argv=argv):
                code, _, err = self.run_main(*argv)
                self.assertEqual(code, 1)
                self.assertIn("bad.yml: name 'Bad name'", err)
        self.assertEqual(self.readme.read_text(encoding="utf-8"), before)

    def test_unreadable_yaml_fails_with_the_file_name(self):
        write(self.dir, "odd.yml", 'name: "CI: Odd"\non:\n  push:\n    tags-ignore: [v0]\n')
        code, _, err = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("odd.yml", err)
        self.assertIn("tags-ignore", err)


if __name__ == "__main__":
    unittest.main()
