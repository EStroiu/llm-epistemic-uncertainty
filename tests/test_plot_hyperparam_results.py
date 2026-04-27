import csv
import os
import shutil
import tempfile
import unittest

from plot_hyperparam_results import _filter_rows, _group_stats, _read_rows


class TestPlotHyperparamResults(unittest.TestCase):
    def test_filter_and_group_stats(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="plot_results_")
        try:
            csv_path = os.path.join(temp_dir, "hyperparam_comparison.csv")
            rows = [
                {
                    "run_id": "r1",
                    "variant": "numeric",
                    "model": "FAST.gpt-oss:120b",
                    "num_examples": "10",
                    "n_samples": "4",
                    "nli_pairs": "2",
                    "error_detection_auroc": "0.8",
                    "ece_confidence": "0.2",
                    "avg_clarity_score": "0.30",
                    "avg_interpretability_score": "0.45",
                },
                {
                    "run_id": "r2",
                    "variant": "numeric",
                    "model": "FAST.gpt-oss:120b",
                    "num_examples": "10",
                    "n_samples": "4",
                    "nli_pairs": "2",
                    "error_detection_auroc": "0.6",
                    "ece_confidence": "0.3",
                    "avg_clarity_score": "0.35",
                    "avg_interpretability_score": "0.50",
                },
                {
                    "run_id": "tiny",
                    "variant": "numeric",
                    "model": "FAST.gpt-oss:120b",
                    "num_examples": "3",
                    "n_samples": "4",
                    "nli_pairs": "2",
                    "error_detection_auroc": "0.0",
                    "ece_confidence": "0.5",
                    "avg_clarity_score": "1.0",
                    "avg_interpretability_score": "1.0",
                },
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            loaded = _read_rows(csv_path)
            filtered = _filter_rows(loaded, variant="numeric", model="FAST.gpt-oss:120b", min_examples=10)
            self.assertEqual(len(filtered), 2)

            stats = _group_stats(filtered, "error_detection_auroc")
            key = "numeric|n=4|pairs=2"
            self.assertIn(key, stats)
            self.assertAlmostEqual(stats[key]["mean"], 0.7, places=6)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

