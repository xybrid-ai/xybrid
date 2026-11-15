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


def make(file="a.yml", name="CI: A", purpose="Does A.", concurrency=(), events=("push",), calls=()):
    return gen.Workflow(
        file=file,
        name=name,
        purpose=purpose,
        triggers=tuple("workflow_call" if event == "workflow_call" else f"{event} to `master`" for event in events),
        writes=(),
        concurrency=tuple(concurrency),
        events=tuple(events),
        calls=tuple(calls),
    )


def allowed(*workflows, pull_request=()):
    """problems() with every given file allowed and `pull_request` allowed to declare PR events."""
    return gen.problems(list(workflows), {w.file for w in workflows}, set(pull_request))


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

    def test_reads_a_lane_file_and_the_jobs_that_call_it(self):
        lane = self.parse(
            """\
            # Purpose: x
            name: "SDK: Lane"
            on:
              workflow_call:
                inputs:
                  lanes:
                    type: string
                    required: false
                secrets:
                  TOKEN:
                    required: false
                outputs:
                  digest:
                    value: ${{ jobs.build.outputs.digest }}
              schedule:
                - cron: '0 1 * * *'
              workflow_dispatch:
            """
        )
        self.assertEqual(lane.events, ("workflow_call", "schedule", "workflow_dispatch"))
        self.assertEqual(lane.triggers, ("workflow_call", "schedule (`0 1 * * *`)", "workflow_dispatch"))
        caller = self.parse(
            """\
            # Purpose: x
            name: "CI: Caller"
            on: pull_request
            jobs:
              lane:
                uses: ./.github/workflows/lane.yml   # the lane
                secrets: inherit
              quoted:
                uses: "./.github/workflows/other.yml"
              again:
                uses: ./.github/workflows/lane.yml
              step:
                runs-on: ubuntu-latest
                steps:
                  - uses: ./.github/actions/setup-rust
            """
        )
        self.assertEqual(caller.calls, ("lane.yml", "other.yml"))

    def test_refuses_keys_an_event_does_not_take(self):
        cases = {
            "secrets outside workflow_call": 'name: "CI: X"\non:\n  push:\n    secrets: {}\n',
            "filters on workflow_call": 'name: "CI: X"\non:\n  workflow_call:\n    paths: [a]\n',
        }
        for label, text in cases.items():
            with self.subTest(label):
                with self.assertRaises(gen.Unsupported):
                    gen.parse_workflow(write(self.dir, "w.yml", text))

    def test_refuses_shapes_it_cannot_read(self):
        cases = {
            "an ignore filter changes what a row means": 'name: "CI: X"\non:\n  push:\n    branches-ignore: [docs]\n',
            "a schedule with no cron": 'name: "CI: X"\non:\n  schedule:\n    - foo: bar\n',
            "an escaped quote in a scalar": 'name: "CI: \\"X\\""\non: push\n',
            "no on:": 'name: "CI: X"\n',
            "a flow-mapping on: hides its events": 'name: "CI: X"\non: { pull_request: { branches: [master] } }\n',
        }
        for label, text in cases.items():
            with self.subTest(label):
                with self.assertRaises(gen.Unsupported) as raised:
                    gen.parse_workflow(write(self.dir, "w.yml", text))
                self.assertIn("w.yml", str(raised.exception))


class ProblemsTests(unittest.TestCase):
    def test_accepts_conforming_workflows(self):
        self.assertEqual(allowed(make("a.yml", "CI: A"), make("b.yml", "SDK: B")), [])

    def test_flags_each_rule(self):
        found = allowed(
            make("no-group.yml", name="Just a name"),
            make("wrong-group.yml", name="Docs: Site"),
            make("no-purpose.yml", name="CI: C", purpose=None),
            make("keyed-by-name.yml", name="CI: D", concurrency=["${{ github.workflow }}-${{ github.ref }}"]),
            make("dup-1.yml", name="CI: E"),
            make("dup-2.yml", name="CI: E"),
        )
        joined = "\n".join(found)
        self.assertIn("no-group.yml: name 'Just a name' must be 'Group: Subject'", joined)
        self.assertIn("wrong-group.yml: name 'Docs: Site' must be 'Group: Subject'", joined)
        self.assertIn("no-purpose.yml: add '# Purpose:", joined)
        self.assertIn("keyed-by-name.yml: concurrency group", joined)
        self.assertIn("dup-2.yml: name 'CI: E' is already used by dup-1.yml", joined)
        self.assertEqual(len(found), 5)


class AllowlistTests(unittest.TestCase):
    def test_a_file_missing_from_the_allowlist_fails(self):
        found = gen.problems([make("ci.yml", "CI: Workspace"), make("new-lane.yml", "CI: New")], {"ci.yml"}, set())
        self.assertEqual(len(found), 1)
        self.assertIn("new-lane.yml: not an allowed workflow file", found[0])
        self.assertIn("Adding CI", found[0])

    def test_an_allowed_file_that_is_gone_fails_so_the_list_shrinks(self):
        found = gen.problems([make("ci.yml", "CI: Workspace")], {"ci.yml", "retired.yml"}, set())
        self.assertEqual(found, ["WORKFLOW_FILES lists retired.yml, which does not exist; remove it"])

    def test_only_listed_files_declare_pull_request_events(self):
        workflows = [
            make("ci.yml", "CI: Workspace", events=("push", "pull_request")),
            make("lane.yml", "CI: Lane", events=("push", "pull_request")),
            make("bot.yml", "Community: Bot", events=("pull_request_target",)),
        ]
        found = allowed(*workflows, pull_request={"ci.yml"})
        self.assertEqual(len(found), 2)
        self.assertIn("lane.yml: declares pull_request, but only ci.yml and bazel.yml validate", found[0])
        self.assertIn("bot.yml: declares pull_request_target", found[1])
        self.assertEqual(allowed(*workflows, pull_request={"ci.yml", "lane.yml", "bot.yml"}), [])

    def test_a_pull_request_entry_without_the_event_fails_so_the_list_shrinks(self):
        workflows = [make("ci.yml", "CI: Workspace", events=("pull_request",)), make("lane.yml", "CI: Lane")]
        self.assertEqual(
            allowed(*workflows, pull_request={"ci.yml", "lane.yml"}),
            ["PULL_REQUEST_FILES lists lane.yml, which declares no pull-request event; remove it"],
        )
        found = gen.problems(workflows, {"ci.yml", "lane.yml"}, {"ci.yml", "gone.yml"})
        self.assertIn("WORKFLOW_FILES does not allow", "\n".join(found))

    def test_a_lane_file_is_called_and_declares_no_push_or_pull_request(self):
        caller = make("ci.yml", "CI: Workspace", events=("pull_request",), calls=("lane.yml",))
        good = make("lane.yml", "SDK: Lane", events=("workflow_call", "schedule", "workflow_dispatch"))
        self.assertEqual(allowed(caller, good, pull_request={"ci.yml"}), [])

        pushed = make("lane.yml", "SDK: Lane", events=("push", "workflow_call"))
        found = allowed(caller, pushed, pull_request={"ci.yml"})
        self.assertEqual(len(found), 1)
        self.assertIn("lane.yml: a lane file (workflow_call)", found[0])
        self.assertIn("not push", found[0])

        orphan = make("orphan.yml", "SDK: Orphan", events=("workflow_call",))
        self.assertEqual(
            allowed(caller, good, orphan, pull_request={"ci.yml"}),
            ["orphan.yml: declares workflow_call, but no workflow calls it"],
        )

    def test_a_call_names_an_existing_lane_file(self):
        caller = make("ci.yml", "CI: Workspace", calls=("missing.yml", "plain.yml"))
        found = allowed(caller, make("plain.yml", "CI: Plain"))
        self.assertEqual(
            found,
            [
                "ci.yml: calls missing.yml, which does not exist",
                "ci.yml: calls plain.yml, which does not declare workflow_call",
            ],
        )

    def test_the_repository_allowlists_are_consistent(self):
        self.assertLessEqual(gen.PULL_REQUEST_FILES, gen.WORKFLOW_FILES)
        self.assertIn("ci.yml", gen.PULL_REQUEST_FILES)
        self.assertIn("bazel.yml", gen.PULL_REQUEST_FILES)


class MacosBudgetTests(unittest.TestCase):
    CI = textwrap.dedent(
        """\
        # Purpose: x
        name: "CI: Workspace"
        on: pull_request
        jobs:
          linux:
            runs-on: ubuntu-latest
          mac:
            runs-on: macos-15
          legs:
            name: legs (${{ matrix.name }})
            runs-on: ${{ matrix.os }}
            strategy:
              fail-fast: false
              matrix:
                include:
                  - name: linux
                    os: ubuntu-latest
                  - name: mac
                    os: macos-latest
                  - name: windows
                    os: windows-latest
          lane:
            name: Lane
            uses: ./.github/workflows/lane.yml
        """
    )
    LANE = textwrap.dedent(
        """\
        # Purpose: x
        name: "SDK: Lane"
        on:
          workflow_call:
        jobs:
          ios:
            runs-on: macos-26
          android:
            runs-on: ubuntu-latest
        """
    )

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        write(self.dir, "ci.yml", self.CI)
        write(self.dir, "lane.yml", self.LANE)

    def test_counts_jobs_matrix_legs_and_called_lanes(self):
        self.assertEqual(
            gen.macos_jobs(self.dir),
            ["ci.yml:mac", "ci.yml:legs (macos-latest)", "lane.yml:ios"],
        )

    def test_the_budget_is_exact_both_ways(self):
        self.assertEqual(gen.macos_problems(self.dir, 3), [])
        over = gen.macos_problems(self.dir, 2)
        self.assertEqual(len(over), 1)
        self.assertIn("3 macOS jobs", over[0])
        self.assertIn("over MACOS_JOB_BUDGET (2)", over[0])
        under = gen.macos_problems(self.dir, 4)
        self.assertIn("lower MACOS_JOB_BUDGET from 4 to 3", under[0])

    def test_labels_match_case_insensitively(self):
        write(self.dir, "lane.yml", self.LANE.replace("macos-26", "macOS-latest"))
        self.assertIn("lane.yml:ios", gen.macos_jobs(self.dir))

    def test_a_runner_it_cannot_count_is_refused(self):
        # Each shape would otherwise count zero and slip past the budget.
        shapes = {
            "an expression": "runs-on: ${{ inputs.runner }}",
            "a label list": "runs-on: [macos-15]",
            "a group mapping": "runs-on:\n      group: mac\n      labels: macos-15",
            "a matrix axis": "runs-on: ${{ matrix.os }}\n    strategy:\n      matrix:\n"
            "        os: [ubuntu-latest, macos-latest]\n        include:\n          - os: ubuntu-latest\n            extra: one",
            "a flow-mapping leg": "runs-on: ${{ matrix.os }}\n    strategy:\n      matrix:\n"
            "        include:\n          - { os: macos-latest }",
        }
        for label, runner in shapes.items():
            with self.subTest(label):
                write(self.dir, "lane.yml", self.LANE.replace("runs-on: macos-26", runner))
                with self.assertRaises(gen.Unsupported):
                    gen.macos_jobs(self.dir)

    def test_jobs_at_another_indent_or_remote_calls_are_refused(self):
        head, jobs = self.LANE.split("jobs:\n")
        write(self.dir, "lane.yml", head + "jobs:\n" + "".join("  " + line for line in jobs.splitlines(True)))
        with self.assertRaises(gen.Unsupported):
            gen.macos_jobs(self.dir)
        write(self.dir, "lane.yml", self.LANE)
        write(self.dir, "ci.yml", self.CI + "  remote:\n    uses: org/repo/.github/workflows/x.yml@main\n")
        with self.assertRaises(gen.Unsupported):
            gen.macos_jobs(self.dir)

    def test_the_repository_pipeline_is_within_budget(self):
        self.assertEqual(gen.macos_problems(gen.WORKFLOWS_DIR), [])


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

    def test_a_lane_file_row_names_its_callers(self):
        table = gen.render(
            [
                make("ci.yml", "CI: Workspace", events=("pull_request",), calls=("lane.yml",)),
                make("lane.yml", "SDK: Lane", events=("workflow_call", "workflow_dispatch")),
                make("idle.yml", "SDK: Idle", events=("workflow_call",)),
            ]
        )
        self.assertIn("| [`lane.yml`](lane.yml) | SDK: Lane | called by `ci.yml`, workflow_dispatch to `master` |", table)
        self.assertIn("| [`idle.yml`](idle.yml) | SDK: Idle | workflow_call |", table)


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
        for target, value in (
            ("WORKFLOWS_DIR", self.dir),
            ("README", self.readme),
            ("WORKFLOW_FILES", frozenset({"sample.yml"})),
            ("PULL_REQUEST_FILES", frozenset({"sample.yml"})),
            ("MACOS_JOB_BUDGET", 0),
        ):
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

    def test_an_unlisted_file_fails_the_check(self):
        write(self.dir, "new-lane.yml", '# Purpose: x\nname: "CI: New lane"\non: pull_request\n')
        code, _, err = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("new-lane.yml: not an allowed workflow file", err)
        self.assertIn("new-lane.yml: declares pull_request", err)

    def test_unreadable_yaml_fails_with_the_file_name(self):
        write(self.dir, "odd.yml", 'name: "CI: Odd"\non:\n  push:\n    tags-ignore: [v0]\n')
        code, _, err = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("odd.yml", err)
        self.assertIn("tags-ignore", err)


if __name__ == "__main__":
    unittest.main()
