import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _filter_rows(
    rows: List[Dict[str, str]],
    *,
    variant: Optional[str],
    model: Optional[str],
    min_examples: int,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        if variant and r.get("variant") != variant:
            continue
        if model and r.get("model") != model:
            continue
        n = _to_float(r.get("num_examples", ""))
        if n is None or n < min_examples:
            continue
        out.append(r)
    return out


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return {"mean": mean, "std": math.sqrt(var)}


def _group_stats(rows: List[Dict[str, str]], metric: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        key = f"{r.get('variant')}|n={r.get('n_samples')}|pairs={r.get('nli_pairs')}"
        v = _to_float(r.get(metric, ""))
        if v is not None:
            groups[key].append(v)
    return {k: _mean_std(vs) for k, vs in groups.items() if vs}


def _make_bar_plot(
    *,
    output_path: str,
    title: str,
    ylabel: str,
    stats: Dict[str, Dict[str, float]],
) -> None:
    if not stats:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = sorted(stats.keys())
    means = [stats[k]["mean"] for k in labels]
    stds = [stats[k]["std"] for k in labels]

    plt.figure(figsize=(max(8, 0.7 * len(labels)), 4.8))
    x = list(range(len(labels)))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _make_run_scatter(
    *,
    output_path: str,
    rows: List[Dict[str, str]],
) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    ece = []
    auroc = []
    labels = []
    for r in rows:
        x = _to_float(r.get("ece_confidence", ""))
        y = _to_float(r.get("error_detection_auroc", ""))
        if x is None or y is None:
            continue
        ece.append(x)
        auroc.append(y)
        labels.append(r.get("run_id", "run"))

    if not ece:
        return

    plt.figure(figsize=(6.8, 5.0))
    plt.scatter(ece, auroc, alpha=0.75)
    for i, name in enumerate(labels):
        plt.annotate(name[-8:], (ece[i], auroc[i]), fontsize=8, alpha=0.8)
    plt.xlabel("ECE (lower is better)")
    plt.ylabel("Error-detection AUROC (higher is better)")
    plt.title("Calibration vs uncertainty quality by run")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregated hyperparameter benchmark results.")
    parser.add_argument("--input-csv", default="experiments/hyperparam_comparison.csv")
    parser.add_argument("--output-dir", default="experiments/plots")
    parser.add_argument("--variant", help="Optional filter: none/brief/numeric/calibrated")
    parser.add_argument("--model", help="Optional model filter")
    parser.add_argument("--min-examples", type=int, default=10, help="Ignore tiny smoke runs")
    args = parser.parse_args()

    rows = _read_rows(args.input_csv)
    rows = _filter_rows(
        rows,
        variant=args.variant,
        model=args.model,
        min_examples=args.min_examples,
    )
    if not rows:
        raise SystemExit("No rows left after filtering. Check filters or input CSV.")

    os.makedirs(args.output_dir, exist_ok=True)

    _make_bar_plot(
        output_path=os.path.join(args.output_dir, "error_detection_auroc_by_setting.png"),
        title="Error detection AUROC by setting (mean +/- std)",
        ylabel="AUROC",
        stats=_group_stats(rows, "error_detection_auroc"),
    )
    _make_bar_plot(
        output_path=os.path.join(args.output_dir, "ece_by_setting.png"),
        title="ECE by setting (mean +/- std)",
        ylabel="ECE",
        stats=_group_stats(rows, "ece_confidence"),
    )
    _make_bar_plot(
        output_path=os.path.join(args.output_dir, "clarity_by_setting.png"),
        title="Clarity by setting (mean +/- std)",
        ylabel="Clarity score",
        stats=_group_stats(rows, "avg_clarity_score"),
    )
    _make_bar_plot(
        output_path=os.path.join(args.output_dir, "interpretability_by_setting.png"),
        title="Interpretability by setting (mean +/- std)",
        ylabel="Interpretability score",
        stats=_group_stats(rows, "avg_interpretability_score"),
    )
    _make_run_scatter(
        output_path=os.path.join(args.output_dir, "ece_vs_auroc_scatter.png"),
        rows=rows,
    )

    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()

