import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_example_json(run_dir: str) -> Optional[Dict[str, Any]]:
    examples_dir = os.path.join(run_dir, "examples")
    if not os.path.isdir(examples_dir):
        return None
    candidates = sorted([x for x in os.listdir(examples_dir) if x.endswith(".json")])
    if not candidates:
        return None
    return _read_json(os.path.join(examples_dir, candidates[0]))


def _infer_n_samples(example_payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not example_payload:
        return None
    meta = (
        example_payload.get("uncertainty_payload", {})
        .get("metadata", {})
    )
    n = meta.get("n_samples")
    return int(n) if isinstance(n, (int, float)) else None


def _infer_nli_pairs(example_payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not example_payload:
        return None
    rels = example_payload.get("nli_relations", [])
    return len(rels) if isinstance(rels, list) else None


def collect_rows(experiments_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(experiments_dir):
        return rows

    for run_name in sorted(os.listdir(experiments_dir)):
        run_dir = os.path.join(experiments_dir, run_name)
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.isdir(run_dir) or not os.path.isfile(metrics_path):
            continue

        metrics = _read_json(metrics_path)
        example_payload = _first_example_json(run_dir)

        row = {
            "run_id": metrics.get("run_id", run_name),
            "variant": metrics.get("variant"),
            "model": metrics.get("model"),
            "dataset_path": metrics.get("dataset_path"),
            "max_examples": metrics.get("max_examples", metrics.get("num_examples")),
            "num_examples": metrics.get("num_examples"),
            "n_samples": metrics.get("n_samples", _infer_n_samples(example_payload)),
            "nli_pairs": metrics.get("nli_pairs", _infer_nli_pairs(example_payload)),
            "sample_temperature": metrics.get("sample_temperature"),
            "prob_temperature": metrics.get("prob_temperature"),
            "error_detection_auroc": metrics.get("error_detection_auroc"),
            "nei_detection_auroc": metrics.get("nei_detection_auroc"),
            "spearman_uncertainty_error": metrics.get("spearman_uncertainty_error"),
            "ece_confidence": metrics.get("ece_confidence"),
            "avg_clarity_score": metrics.get("avg_clarity_score"),
            "avg_interpretability_score": metrics.get("avg_interpretability_score"),
            "run_dir": run_name,
        }
        rows.append(row)

    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No run metrics found in experiments directory.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark run metrics into one CSV for hyperparameter comparison.")
    parser.add_argument("--experiments-dir", default="experiments", help="Directory containing run folders.")
    parser.add_argument("--output-csv", default="experiments/hyperparam_comparison.csv", help="Path to output CSV file.")
    args = parser.parse_args()

    rows = collect_rows(args.experiments_dir)
    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} runs to {args.output_csv}")


if __name__ == "__main__":
    main()

