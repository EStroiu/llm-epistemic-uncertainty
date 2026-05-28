import csv
import json
import os
import shutil
import tempfile
import unittest

from uncertainty_benchmark import run_benchmark


class TestBenchmarkSmoke(unittest.TestCase):
    def test_dry_run_benchmark(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="uncertainty_bench_")
        try:
            dataset = os.path.join("tests", "data", "tiny_fever.jsonl")
            metrics = run_benchmark(
                dataset_path=dataset,
                output_dir=temp_dir,
                model="FAST.gpt-oss:120b",
                max_examples=3,
                n_samples=3,
                sample_temperature=0.7,
                prob_temperature=0.0,
                nli_pairs=2,
                variant="numeric",
                dry_run=True,
            )
            self.assertEqual(metrics["num_examples"], 3)
            self.assertIn("error_detection_auroc", metrics)
            self.assertIn("avg_clarity_score", metrics)
            self.assertIn("avg_num_claims", metrics)
            self.assertIn("avg_mean_claim_uncertainty", metrics)
            self.assertIn("total_claim_contradictions", metrics)

            run_dir = os.path.join(temp_dir, metrics["run_id"])
            with open(os.path.join(run_dir, "summary.csv"), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertIn("num_claims", rows[0])
            self.assertIn("mean_claim_uncertainty", rows[0])
            self.assertIn("max_claim_uncertainty", rows[0])
            self.assertIn("min_claim_certainty", rows[0])

            with open(os.path.join(run_dir, "examples", "example_0001.json"), encoding="utf-8") as f:
                example = json.load(f)
            self.assertIn("claim_uncertainties", example)
            self.assertIn("claims", example["uncertainty_payload"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

