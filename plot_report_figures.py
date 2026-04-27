import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


VARIANT_ORDER = ["none", "brief", "numeric", "calibrated"]
LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_int(value: str) -> Optional[int]:
    f = _to_float(value)
    if f is None:
        return None
    return int(f)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _pilot_rows(rows: List[Dict[str, str]], *, model: Optional[str], min_examples: int, final_min_examples: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        if model and r.get("model") != model:
            continue
        n = _to_int(r.get("num_examples", ""))
        if n is None:
            continue
        if n < min_examples:
            continue
        if n >= final_min_examples:
            continue
        out.append(r)
    return out


def _final_rows(rows: List[Dict[str, str]], *, model: Optional[str], variant: str, min_examples: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        if model and r.get("model") != model:
            continue
        if r.get("variant") != variant:
            continue
        n = _to_int(r.get("num_examples", ""))
        if n is None or n < min_examples:
            continue
        out.append(r)
    out.sort(key=lambda x: x.get("run_id", ""))
    return out


def _plot_hyperparam_heatmap(rows: List[Dict[str, str]], output_path: str) -> None:
    metrics = [
        ("error_detection_auroc", "Err AUROC"),
        ("nei_detection_auroc", "NEI AUROC"),
        ("ece_confidence", "ECE"),
        ("avg_clarity_score", "Clarity"),
        ("avg_interpretability_score", "Interpretability"),
    ]

    # Collect averages per variant.
    matrix: List[List[float]] = []
    for variant in VARIANT_ORDER:
        row_vals: List[float] = []
        v_rows = [r for r in rows if r.get("variant") == variant]
        for metric_key, _ in metrics:
            vals = [_to_float(r.get(metric_key, "")) for r in v_rows]
            vals = [v for v in vals if v is not None]
            m = _mean(vals)
            row_vals.append(m if m is not None else float("nan"))
        matrix.append(row_vals)

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], rotation=20, ha="right")
    ax.set_yticks(range(len(VARIANT_ORDER)))
    ax.set_yticklabels(VARIANT_ORDER)
    ax.set_title("Hyperparameter search summary (means by variant)")
    for i in range(len(VARIANT_ORDER)):
        for j in range(len(metrics)):
            value = matrix[i][j]
            txt = "NA" if value != value else f"{value:.3f}"  # NaN check: x != x
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_final_trends(rows: List[Dict[str, str]], output_path: str) -> None:
    run_labels = [r.get("run_id", "")[-6:] for r in rows]
    x = list(range(len(rows)))

    err_auroc = [_to_float(r.get("error_detection_auroc", "")) or 0.0 for r in rows]
    nei_auroc = [_to_float(r.get("nei_detection_auroc", "")) or 0.0 for r in rows]
    ece = [_to_float(r.get("ece_confidence", "")) or 0.0 for r in rows]
    clarity = [_to_float(r.get("avg_clarity_score", "")) or 0.0 for r in rows]
    interpretability = [_to_float(r.get("avg_interpretability_score", "")) or 0.0 for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))

    ax = axes[0]
    ax.plot(x, err_auroc, marker="o", linewidth=2.0, label="Err AUROC")
    ax.plot(x, nei_auroc, marker="s", linewidth=2.0, label="NEI AUROC")
    ax.plot(x, ece, marker="^", linewidth=2.0, label="ECE")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_title("Final runs: uncertainty metrics")
    ax.set_xlabel("Run (timestamp suffix)")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(x, clarity, marker="o", linewidth=2.0, label="Clarity")
    ax2.plot(x, interpretability, marker="s", linewidth=2.0, label="Interpretability")
    ax2.set_xticks(x)
    ax2.set_xticklabels(run_labels, rotation=0)
    ax2.set_ylim(0, 1)
    ax2.set_title("Final runs: communication metrics")
    ax2.set_xlabel("Run (timestamp suffix)")
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _load_confusion_from_run_summary(summary_csv_path: str) -> List[List[int]]:
    mat = [[0 for _ in LABEL_ORDER] for _ in LABEL_ORDER]
    rows = _read_csv_rows(summary_csv_path)
    idx = {label: i for i, label in enumerate(LABEL_ORDER)}

    for r in rows:
        gold = r.get("gold_label", "")
        pred = r.get("predicted_label", "")
        if gold not in idx or pred not in idx:
            continue
        mat[idx[gold]][idx[pred]] += 1
    return mat


def _sum_matrices(mats: List[List[List[int]]]) -> List[List[int]]:
    out = [[0 for _ in LABEL_ORDER] for _ in LABEL_ORDER]
    for m in mats:
        for i in range(len(LABEL_ORDER)):
            for j in range(len(LABEL_ORDER)):
                out[i][j] += m[i][j]
    return out


def _plot_confusion_matrix(matrix: List[List[int]], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Final runs confusion matrix (aggregated)")

    max_value = max(max(row) for row in matrix) if matrix else 1
    threshold = max_value / 2.0
    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            value = matrix[i][j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _collect_uncertainty_by_gold_from_run_summary(summary_csv_path: str) -> Dict[str, List[float]]:
    rows = _read_csv_rows(summary_csv_path)
    out: Dict[str, List[float]] = {label: [] for label in LABEL_ORDER}
    for r in rows:
        gold = r.get("gold_label", "")
        unc = _to_float(r.get("uncertainty", ""))
        if gold in out and unc is not None:
            out[gold].append(unc)
    return out


def _merge_uncertainty_buckets(items: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {label: [] for label in LABEL_ORDER}
    for item in items:
        for label in LABEL_ORDER:
            out[label].extend(item.get(label, []))
    return out


def _plot_uncertainty_distribution_by_gold(unc_by_gold: Dict[str, List[float]], output_path: str) -> None:
    data = [unc_by_gold.get(label, []) for label in LABEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bp = ax.boxplot(
        data,
        tick_labels=LABEL_ORDER,
        showfliers=False,
        patch_artist=True,
        widths=0.6,
    )

    colors = ["#2ecc71", "#3498db", "#e67e22"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Uncertainty")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Uncertainty distribution by gold FEVER label (final runs aggregated)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report-ready figures for FEVER unified uncertainty.")
    parser.add_argument("--input-csv", default="experiments/hyperparam_comparison.csv")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--output-dir", default="experiments/plots_report")
    parser.add_argument("--model", default="FAST.gpt-oss:120b")
    parser.add_argument("--pilot-min-examples", type=int, default=10)
    parser.add_argument("--final-min-examples", type=int, default=200)
    parser.add_argument("--final-variant", default="numeric")
    args = parser.parse_args()

    all_rows = _read_csv_rows(args.input_csv)
    pilot_rows = _pilot_rows(
        all_rows,
        model=args.model,
        min_examples=args.pilot_min_examples,
        final_min_examples=args.final_min_examples,
    )
    final_rows = _final_rows(
        all_rows,
        model=args.model,
        variant=args.final_variant,
        min_examples=args.final_min_examples,
    )

    if not pilot_rows:
        raise SystemExit("No pilot rows found. Check --pilot-min-examples / --final-min-examples / --model.")
    if not final_rows:
        raise SystemExit("No final rows found. Check --final-variant / --final-min-examples / --model.")

    os.makedirs(args.output_dir, exist_ok=True)

    _plot_hyperparam_heatmap(
        pilot_rows,
        os.path.join(args.output_dir, "fig1_hyperparam_heatmap.png"),
    )
    _plot_final_trends(
        final_rows,
        os.path.join(args.output_dir, "fig2_final_runs_trends.png"),
    )

    matrices = []
    uncertainty_buckets = []
    for r in final_rows:
        run_dir = r.get("run_dir", "")
        summary_csv = os.path.join(args.experiments_dir, run_dir, "summary.csv")
        if os.path.exists(summary_csv):
            matrices.append(_load_confusion_from_run_summary(summary_csv))
            uncertainty_buckets.append(_collect_uncertainty_by_gold_from_run_summary(summary_csv))

    if not matrices:
        raise SystemExit("No summary.csv files found for final runs.")

    conf = _sum_matrices(matrices)
    _plot_confusion_matrix(
        conf,
        os.path.join(args.output_dir, "fig3_confusion_matrix_final_agg.png"),
    )
    merged_unc = _merge_uncertainty_buckets(uncertainty_buckets)
    _plot_uncertainty_distribution_by_gold(
        merged_unc,
        os.path.join(args.output_dir, "fig4_uncertainty_by_gold_boxplot.png"),
    )

    print(f"Saved report figures to: {args.output_dir}")
    print("- fig1_hyperparam_heatmap.png")
    print("- fig2_final_runs_trends.png")
    print("- fig3_confusion_matrix_final_agg.png")
    print("- fig4_uncertainty_by_gold_boxplot.png")


if __name__ == "__main__":
    main()

