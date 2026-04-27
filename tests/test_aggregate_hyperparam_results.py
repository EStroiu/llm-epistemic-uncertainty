import json
import os
import shutil
import tempfile
import unittest

from aggregate_hyperparam_results import collect_rows, write_csv


class TestAggregateResults(unittest.TestCase):
    def test_collect_rows_and_write_csv(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="agg_runs_")
        try:
            run_dir = os.path.join(temp_dir, "multi_estimator_numeric_20260101_000000")
            os.makedirs(os.path.join(run_dir, "examples"), exist_ok=True)

            metrics = {
                "run_id": "multi_estimator_numeric_20260101_000000",
                "variant": "numeric",
                "model": "FAST.gpt-oss:120b",
                "num_examples": 2,
                "error_detection_auroc": 0.61,
                "nei_detection_auroc": 0.58,
                "spearman_uncertainty_error": 0.21,
                "ece_confidence": 0.18,
                "avg_clarity_score": 0.72,
                "avg_interpretability_score": 0.81,
            }
            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f)

            ex = {
                "nli_relations": ["entailment", "neutral"],
                "uncertainty_payload": {"metadata": {"n_samples": 3}},
            }
            with open(os.path.join(run_dir, "examples", "example_0001.json"), "w", encoding="utf-8") as f:
                json.dump(ex, f)

            rows = collect_rows(temp_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["variant"], "numeric")
            self.assertEqual(rows[0]["n_samples"], 3)
            self.assertEqual(rows[0]["nli_pairs"], 2)

            out_csv = os.path.join(temp_dir, "comparison.csv")
            write_csv(out_csv, rows)
            self.assertTrue(os.path.exists(out_csv))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

