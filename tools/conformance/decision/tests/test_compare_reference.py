"""Reference comparison keeps preparation disagreements separate from drift."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_reference as comparison
from hf_reference import digest


class Comparison(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.hf = self.root / "hf"
        self.native = self.root / "native"
        self.hf.mkdir()
        self.native.mkdir()
        self.hf_case = {
            "id": "sample", "split": "held_out", "status": "complete", "request_sha256": "request",
            "passes": [{"question": "q", "prompt": "same", "token_ids": [1],
                        "selected": {"groups": [[2], [3]], "raw_scores": ["0x3f800000", "0x00000000"], "log_probabilities": ["0xbf000000", "0xbfc00000"]},
                        "result": {"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}}}}],
        }
        self.native_case = {
            "request_sha256": "request", "upstream": {
                "prepare": {"passes": [{"question": "q", "prompt": "same", "token_ids": [1], "select": [2, 3]}]},
                "evaluate": {"raw_scores": [["0x40000000", "0x00000000"]]},
                "finalize": {"answers": {"q": {"type": "choice", "choice": "b", "probabilities": {"a": 0.4, "b": 0.6}}}},
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    def stage(self):
        hf_file = self.hf / "sample.json"
        hf_file.write_text(json.dumps(self.hf_case))
        (self.native / "sample.observation.json").write_text(json.dumps(self.native_case))
        (self.hf / "collection.json").write_text(json.dumps({
            "counts": {"selected": 1, "executed": 1},
            "settings": {"set_size": 1, "batch_rows": 1},
            "cases": [{"id": "sample", "sha256": digest(hf_file)}],
        }))
        (self.native / "collection.json").write_text(json.dumps({
            "status": "complete", "collector": {"official": True}, "corpus": {"selected": 1},
            "artifact": {"id": "fixture"},
            "cases": [{"id": "sample", "status": "observed", "observation": "sample.observation.json"}],
        }))

    def test_matching_preparation_records_errors_and_changed_answers_without_promoting(self):
        self.stage()
        report = comparison.compare(self.hf, self.native, 1)
        self.assertFalse(report["promotion_qualified"])
        self.assertEqual(report["matching_preparation_maxima"]["absolute_selected_logit"], 1.0)
        self.assertAlmostEqual(report["matching_preparation_maxima"]["absolute_probability"], 0.2)
        self.assertEqual(report["counts"]["changed_selected_answers"], 1)

    def test_prompt_difference_cannot_be_absorbed_into_numeric_error(self):
        self.hf_case["passes"][0]["prompt"] = "different semantics"
        self.stage()
        report = comparison.compare(self.hf, self.native, 1)
        self.assertEqual(report["counts"]["preparation_differences"], 1)
        self.assertNotIn("max_observed_abs_logit", report["cases"][0]["questions"][0])
        self.assertEqual(report["counts"]["changed_selected_answers"], 1)

    def test_hf_refusals_are_explicit_case_results(self):
        self.hf_case.update(status="failed", error_type="ValueError", error="fixture refusal", passes=[])
        self.stage()
        report = comparison.compare(self.hf, self.native, 1)
        self.assertEqual(report["counts"]["hf_refused"], 1)
        self.assertEqual(report["counts"]["cases"], 1)

    def test_missing_expected_execution_cannot_pass(self):
        self.stage()
        with self.assertRaisesRegex(ValueError, "expected case count"):
            comparison.compare(self.hf, self.native, 2)

    def test_request_identity_must_match(self):
        self.native_case["request_sha256"] = "another request"
        self.stage()
        with self.assertRaisesRegex(ValueError, "requests differ"):
            comparison.compare(self.hf, self.native, 1)

    def test_normalized_readouts_cannot_be_mislabeled_raw_logits(self):
        del self.hf_case["passes"][0]["selected"]["log_probabilities"]
        self.stage()
        with self.assertRaisesRegex(ValueError, "separate raw-head"):
            comparison.compare(self.hf, self.native, 1)

    def test_partial_comparison_retains_full_source_receipt_counts(self):
        self.stage()
        receipt = json.loads((self.hf / "collection.json").read_text())
        receipt["counts"].update(selected=2, executed=2)
        receipt["cases"].append({"id": "unselected", "sha256": "not-read"})
        (self.hf / "collection.json").write_text(json.dumps(receipt))
        report = comparison.compare(self.hf, self.native, 1, {"sample"})
        self.assertEqual(report["source_case_counts"], {"hf": 2, "native": 1})
        self.assertEqual(report["selection"], ["sample"])
        self.assertEqual(report["counts"]["cases"], 1)
        with self.assertRaisesRegex(ValueError, "absent from a reference"):
            comparison.compare(self.hf, self.native, 1, {"unknown"})


if __name__ == "__main__":
    unittest.main()
