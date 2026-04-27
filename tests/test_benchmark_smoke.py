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
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

