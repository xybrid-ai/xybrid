"""Tests of oracle.py's case-file collection: corpus checks, receipts and output containment.

Stdlib only; no model, no build. Run: python3 -m unittest discover tools/conformance/decision/tests
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1]
ORACLE = TOOL / "oracle.py"
REPO_CASES = TOOL.parents[2] / "integration-tests" / "fixtures" / "decision" / "cases"

spec = importlib.util.spec_from_file_location("decision_oracle", ORACLE)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

REQUEST = '{"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}'


def case(case_id="a-case", request=REQUEST, **extra):
    doc = {"schema_version": 1, "id": case_id, "profile": "lfm3-d1/text/v1", "request_text": request}
    doc.update(extra)
    return doc


class Corpus:
    """A temporary corpus directory, filled with case files given as documents or raw text."""

    def __init__(self, test):
        self.dir = Path(tempfile.mkdtemp())
        test.addCleanup(subprocess.run, ["rm", "-rf", str(self.dir)])

    def add(self, name, doc=None, text=None, raw=None):
        path = self.dir / name
        if raw is not None:
            path.write_bytes(raw)
        else:
            path.write_text(text if text is not None else json.dumps(doc), encoding="utf-8")
        return path


def run_cli(*args):
    return subprocess.run([sys.executable, "-I", str(ORACLE), *args], capture_output=True, text=True)


class CaseFileTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(self)

    def refused(self, name, doc=None, text=None, raw=None):
        path = self.corpus.add(name, doc, text, raw)
        with self.assertRaises(oracle.CaseFileError) as e:
            oracle.read_case_file(path)
        return str(e.exception)

    def test_an_array_root_is_refused_not_a_crash(self):
        self.assertIn("root is a JSON list", self.refused("array.json", text="[1, 2]"))

    def test_a_lone_surrogate_in_the_request_is_refused(self):
        text = json.dumps(case(request="\ud800"))  # json.dumps writes the escape \ud800
        self.assertIn("lone surrogate", self.refused("surrogate.json", text=text))

    def test_duplicate_envelope_keys_are_refused(self):
        text = '{"schema_version": 1, "id": "a", "id": "b", "profile": "lfm3-d1/text/v1", "request_text": "{}"}'
        self.assertIn("'id' twice", self.refused("dup.json", text=text))

    def test_an_id_that_could_leave_the_output_directory_is_refused(self):
        self.assertIn("id must match", self.refused("escape.json", case("../escaped-review-observation")))
        self.assertIn("id must match", self.refused("upper.json", case("Upper")))

    def test_envelope_fields_are_typed(self):
        self.assertIn("schema_version", self.refused("v.json", case(schema_version="1")))
        self.assertIn("schema_version", self.refused("vb.json", case(schema_version=True)))
        self.assertIn("profile", self.refused("p.json", case(profile="other")))
        self.assertIn("request_text", self.refused("r.json", case(request={"state": "s"})))
        self.assertIn("has_media", self.refused("m.json", case(has_media="yes")))
        self.assertIn("metadata", self.refused("md.json", case(metadata=["set"])))
        self.assertIn("metadata.set", self.refused("ms.json", case(metadata={"set": {"k=v": "f32:1"}})))
        self.assertIn("f32", self.refused("mf.json", case(metadata={"set": {"k": "f32:one"}})))
        self.assertIn("token_types", self.refused("mt.json", case(metadata={"token_types": {"x": 3}})))
        self.assertIn("upstream", self.refused("u.json", case(upstream=[])))
        bad_scores = {"evaluate": {"source": "synthetic", "raw_scores": [[1.0]]}}
        self.assertIn("raw_scores", self.refused("s.json", case(upstream=bad_scores)))

    def test_non_json_constants_and_bytes_are_refused(self):
        self.assertIn("NaN", self.refused("nan.json", text='{"schema_version": NaN}'))
        self.assertIn("not UTF-8", self.refused("latin1.json", raw=b'{"id": "\xe9"}'))
        self.assertIn("not JSON", self.refused("broken.json", text="{"))

    def test_malformed_and_duplicate_key_requests_are_kept_verbatim(self):
        for request in ('{"state": "s", "state": "t", "questions": {}}', "state: refund please", ""):
            path = self.corpus.add("ok.json", case(request=request))
            got = oracle.read_case_file(path)
            self.assertEqual(got["request_text"], request)
            self.assertEqual(got["request_sha256"], hashlib.sha256(request.encode("utf-8")).hexdigest())
            self.assertEqual(got["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_synthetic_scores_metadata_and_media_are_read(self):
        doc = case(metadata={"set": {"lfm3.decision.temperature.image": "f32:2.5"}}, has_media=True,
                   upstream={"evaluate": {"source": "synthetic", "raw_scores": [["0x3f800000"]]}})
        got = oracle.read_case_file(self.corpus.add("syn.json", doc))
        self.assertEqual(got["select_logits"], [["0x3f800000"]])
        self.assertTrue(got["has_media"])
        self.assertEqual(got["metadata"], doc["metadata"])

    def test_every_committed_case_reads(self):
        entries, cases = oracle.load_corpus(REPO_CASES)
        self.assertEqual([e for e in entries if e.get("status") == "invalid"], [])
        self.assertEqual(len(cases), len(list(REPO_CASES.glob("*.json"))))


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(self)

    def test_a_missing_corpus_stops_the_run(self):
        with self.assertRaises(SystemExit):
            oracle.load_corpus(self.corpus.dir / "missing")

    def test_an_empty_corpus_stops_the_run(self):
        with self.assertRaises(SystemExit):
            oracle.load_corpus(self.corpus.dir)

    def test_bad_files_get_error_receipts_and_are_not_scheduled(self):
        self.corpus.add("good.json", case("good"))
        self.corpus.add("array.json", text="[]")
        self.corpus.add("escape.json", case("../escaped-review-observation"))
        self.corpus.add("surrogate.json", text=json.dumps(case("sur", request="\ud800")))
        entries, cases = oracle.load_corpus(self.corpus.dir)
        self.assertEqual([c["id"] for _, c in cases], ["good"])
        invalid = {e["file"]: e for e in entries if e.get("status") == "invalid"}
        self.assertEqual(set(invalid), {"array.json", "escape.json", "surrogate.json"})
        self.assertTrue(all(e["error"] and e["file_sha256"] for e in invalid.values()))
        self.assertTrue(all("id" not in e for e in invalid.values()))

    def test_files_sharing_an_id_are_all_refused(self):
        self.corpus.add("one.json", case("same"))
        self.corpus.add("two.json", case("same", request="{}"))
        self.corpus.add("other.json", case("other"))
        entries, cases = oracle.load_corpus(self.corpus.dir)
        self.assertEqual([c["id"] for _, c in cases], ["other"])
        self.assertEqual(sorted(e["file"] for e in entries if e.get("status") == "invalid"), ["one.json", "two.json"])

    def test_the_selection_must_match_what_was_declared(self):
        self.corpus.add("a.json", case("a"))
        self.corpus.add("b.json", case("b"))
        with self.assertRaises(SystemExit):
            oracle.load_corpus(self.corpus.dir, only=["c"])
        with self.assertRaises(SystemExit):
            oracle.load_corpus(self.corpus.dir, expect=3)
        with self.assertRaises(SystemExit):
            oracle.load_corpus(self.corpus.dir, only=["a"], expect=2)
        entries, cases = oracle.load_corpus(self.corpus.dir, only=["a"], expect=1)
        self.assertEqual([c["id"] for _, c in cases], ["a"])
        self.assertEqual([e["status"] for e in entries if e.get("id") == "b"], ["not_selected"])
        self.assertEqual(len(oracle.load_corpus(self.corpus.dir, expect=2)[1]), 2)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.root)])

    def test_the_output_directory_must_be_new_or_empty(self):
        self.assertEqual(oracle.check_out_dir(self.root / "new", self.root / "corpus"), self.root / "new")
        (self.root / "used").mkdir()
        (self.root / "used" / "collection.json").write_text("{}")
        with self.assertRaises(SystemExit):
            oracle.check_out_dir(self.root / "used", self.root / "corpus")

    def test_the_output_is_neither_the_corpus_nor_the_fixtures(self):
        with self.assertRaises(SystemExit):
            oracle.check_out_dir(self.root, self.root)
        with self.assertRaises(SystemExit):
            oracle.check_out_dir(oracle.FIXTURES / "observations", self.root)

    def test_nothing_is_written_outside_the_output_directory(self):
        out = self.root / "out"
        out.mkdir()
        self.assertEqual(oracle.contained(out, "a.0f.json"), out / "a.0f.json")
        for name in ("../escaped-review-observation.json", "sub/x.json", "/tmp/x.json"):
            with self.assertRaises(SystemExit):
                oracle.contained(out, name)


class ReceiptTests(unittest.TestCase):
    def test_a_developer_run_is_never_official(self):
        self.assertEqual(oracle.collector_record([], True), {"official": False, "developer_run": True, "problems": []})
        self.assertTrue(oracle.collector_record([], False)["official"])
        self.assertFalse(oracle.collector_record(["stale"], False)["official"])

    def test_the_recipe_names_every_producing_file(self):
        recipe = oracle.recipe_hashes()
        self.assertEqual(set(recipe), set(oracle.RECIPE))
        self.assertIn("oracle.py", recipe)
        self.assertEqual(recipe["oracle.py"], hashlib.sha256(ORACLE.read_bytes()).hexdigest())

    def test_the_inputs_identity_changes_with_anything_that_changes_an_observation(self):
        artifact = {"sha256": "a" * 64}
        base = {"request_sha256": "r" * 64, "has_media": False, "metadata": None, "select_logits": None}
        digest = oracle.canonical_sha256(oracle.case_inputs(base, "d1-2b-q4km", artifact, None))
        self.assertEqual(digest, oracle.canonical_sha256(oracle.case_inputs(dict(base), "d1-2b-q4km", artifact, None)))
        changes = [dict(base, has_media=True), dict(base, metadata={"remove": ["k"]}), dict(base, select_logits=[["0x00000000"]]),
                   dict(base, request_sha256="s" * 64)]
        for changed in changes:
            self.assertNotEqual(digest, oracle.canonical_sha256(oracle.case_inputs(changed, "d1-2b-q4km", artifact, None)))
        self.assertNotEqual(digest, oracle.canonical_sha256(oracle.case_inputs(base, "d1-2b-f16", {"sha256": "b" * 64}, None)))
        with tempfile.NamedTemporaryFile() as variant:
            variant.write(b"gguf")
            variant.flush()
            with_variant = oracle.case_inputs(base, "d1-2b-q4km", artifact, Path(variant.name))
        self.assertEqual(with_variant["model"]["variant_sha256"], hashlib.sha256(b"gguf").hexdigest())
        self.assertNotEqual(digest, oracle.canonical_sha256(with_variant))


class CommandLineTests(unittest.TestCase):
    """The checks main() makes before it looks for executables or a model."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.root)])

    def test_a_missing_corpus_fails_and_writes_nothing(self):
        out = self.root / "out"
        result = run_cli("--collect", str(self.root / "missing"), "--out", str(out))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no such directory", result.stderr)
        self.assertFalse(out.exists())

    def test_collect_and_check_do_not_combine(self):
        result = run_cli("--collect", str(self.root), "--check", "--out", str(self.root / "out"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not combine", result.stderr)
        self.assertFalse((self.root / "out").exists())

    def test_collect_needs_an_output_directory(self):
        result = run_cli("--collect", str(self.root))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs --out", result.stderr)


if __name__ == "__main__":
    unittest.main()
