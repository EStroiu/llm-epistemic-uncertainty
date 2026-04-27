import argparse
import csv
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

VARIANT_ORDER = ["none", "brief", "numeric", "calibrated"]
VARIANT_COLORS = {
    "none": "#5d6d7e",
    "brief": "#2980b9",
    "numeric": "#27ae60",
    "calibrated": "#8e44ad",
}


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


def _to_int(value: str) -> Optional[int]:
    f = _to_float(value)
    if f is None:
        return None
    return int(f)


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _filter_min_examples(rows: List[Dict[str, str]], min_examples: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        n = _to_int(r.get("num_examples", ""))
        if n is not None and n >= min_examples:
            out.append(r)
    return out


def _filter_rows(
    rows: List[Dict[str, str]],
    *,
    variant: Optional[str],
    model: Optional[str],
    min_examples: int,
) -> List[Dict[str, str]]:
    out = _filter_min_examples(rows, min_examples)
    out = _filter_model(out, model)
    if variant:
        out = [r for r in out if r.get("variant") == variant]
    return out


def _filter_model(rows: List[Dict[str, str]], model: Optional[str]) -> List[Dict[str, str]]:
    if not model:
        return rows
    return [r for r in rows if r.get("model") == model]


def _short_run_label(run_id: str) -> str:
    m = re.search(r"_(\d{8}_\d{6})$", run_id or "")
    if m:
        return m.group(1)[-6:]
    return (run_id or "?")[-10:]


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def _aggregate_by_variant(rows: List[Dict[str, str]], metric: str) -> Tuple[List[str], List[float], List[float]]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        v = r.get("variant", "")
        val = _to_float(r.get(metric, ""))
        if val is not None:
            groups[v].append(val)
    labels = [v for v in VARIANT_ORDER if v in groups]
    if not labels:
        labels = sorted(groups.keys())
    means = []
    stds = []
    for lb in labels:
        m, s = _mean_std(groups[lb])
        means.append(m)
        stds.append(s)
    return labels, means, stds


def _bar_by_variant(
    ax,
    rows: List[Dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    labels, means, stds = _aggregate_by_variant(rows, metric)
    if not labels:
        ax.set_visible(False)
        return
    x = range(len(labels))
    colors = [VARIANT_COLORS.get(lb, "#3498db") for lb in labels]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="#2c3e50", linewidth=0.8, alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    if ylim:
        lo, hi = ylim
        if hi is None:
            ax.set_ylim(bottom=lo if lo is not None else 0)
        elif lo is not None:
            pad = (hi - lo) * 0.08 if hi > lo else 0.05
            ax.set_ylim(lo - pad, hi + pad)
        else:
            pad = hi * 0.05 if hi > 0 else 0.05
            ax.set_ylim(0, hi + pad)
    else:
        ax.set_ylim(bottom=0)


def _runs_subplot(ax, rows: List[Dict[str, str]], title: str) -> None:
    if len(rows) < 2:
        ax.text(0.5, 0.5, "Need 2+ runs\n(same filter)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    rows_sorted = sorted(rows, key=lambda r: r.get("run_id", ""))
    xs = list(range(len(rows_sorted)))
    labels = [_short_run_label(r.get("run_id", "")) for r in rows_sorted]
    auroc = [_to_float(r.get("error_detection_auroc", "")) or 0 for r in rows_sorted]
    ece = [_to_float(r.get("ece_confidence", "")) or 0 for r in rows_sorted]
    w = 0.35
    ax.bar([i - w / 2 for i in xs], auroc, width=w, label="err AUROC", color="#27ae60", edgecolor="#1e8449")
    ax.bar([i + w / 2 for i in xs], ece, width=w, label="ECE", color="#e67e22", edgecolor="#ca6f1e")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("value")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)


def _scatter_runs(ax, rows: List[Dict[str, str]]) -> None:
    for v in VARIANT_ORDER:
        pts = [r for r in rows if r.get("variant") == v]
        xs: List[float] = []
        ys: List[float] = []
        for r in pts:
            x = _to_float(r.get("ece_confidence", ""))
            y = _to_float(r.get("error_detection_auroc", ""))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if not xs:
            continue
        c = VARIANT_COLORS.get(v, "#7f8c8d")
        ax.scatter(xs, ys, s=55, c=c, label=v, edgecolors="#2c3e50", linewidths=0.5, alpha=0.85)
    ax.set_xlabel("ECE (lower better)")
    ax.set_ylabel("Error AUROC (higher better)")
    ax.set_title("Runs: calibration vs ranking", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)


def _make_dashboard(
    *,
    output_path: str,
    rows_all: List[Dict[str, str]],
    rows_detail: List[Dict[str, str]],
    detail_title: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle("FEVER unified benchmark — summary", fontsize=13, fontweight="600")

    _bar_by_variant(axes[0, 0], rows_all, "error_detection_auroc", "Error detection AUROC", "AUROC", (0, 1))
    _bar_by_variant(axes[0, 1], rows_all, "ece_confidence", "Expected calibration error", "ECE", (0, None))
    _bar_by_variant(axes[0, 2], rows_all, "nei_detection_auroc", "NEI vs rest AUROC", "AUROC", (0, 1))

    sp_vals = []
    for r in rows_all:
        v = _to_float(r.get("spearman_uncertainty_error", ""))
        if v is not None:
            sp_vals.append(v)
    if sp_vals:
        spearman_lo = min(sp_vals)
        spearman_hi = max(sp_vals)
        pad = max(0.08, (spearman_hi - spearman_lo) * 0.15)
        sp_ylim: Optional[Tuple[float, float]] = (spearman_lo - pad, spearman_hi + pad)
    else:
        sp_ylim = None
    _bar_by_variant(
        axes[1, 0],
        rows_all,
        "spearman_uncertainty_error",
        "Spearman (uncertainty vs error)",
        "rho",
        sp_ylim,
    )
    _bar_by_variant(axes[1, 1], rows_all, "avg_clarity_score", "Clarity (prompt proxy)", "score", (0, 1))
    _bar_by_variant(axes[1, 2], rows_all, "avg_interpretability_score", "Interpretability", "score", (0, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()

    fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig2.suptitle("Repeated runs & scatter", fontsize=12, fontweight="600")
    _runs_subplot(axes2[0], rows_detail, detail_title)
    _scatter_runs(axes2[1], rows_all)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    base, ext = os.path.splitext(output_path)
    path2 = base + "_runs" + ext
    plt.savefig(path2, dpi=160, bbox_inches="tight")
    plt.close()


def _make_bar_plot_legacy(
    *,
    output_path: str,
    title: str,
    ylabel: str,
    stats: Dict[str, Dict[str, float]],
) -> None:
    if not stats:
        return
    import matplotlib.pyplot as plt

    labels = sorted(stats.keys())
    means = [stats[k]["mean"] for k in labels]
    stds = [stats[k]["std"] for k in labels]

    plt.figure(figsize=(max(8, 0.55 * len(labels)), 4.5))
    x = list(range(len(labels)))
    plt.bar(x, means, yerr=stds, capsize=4, color="#3498db", edgecolor="#2c3e50")
    plt.xticks(x, labels, rotation=22, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _group_stats(rows: List[Dict[str, str]], metric: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        key = f"{r.get('variant')}|n={r.get('n_samples')}|pairs={r.get('nli_pairs')}"
        v = _to_float(r.get(metric, ""))
        if v is not None:
            groups[key].append(v)
    out: Dict[str, Dict[str, float]] = {}
    for k, vs in groups.items():
        m, s = _mean_std(vs)
        out[k] = {"mean": m, "std": s}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregated hyperparameter benchmark results.")
    parser.add_argument("--input-csv", default="experiments/hyperparam_comparison.csv")
    parser.add_argument("--output-dir", default="experiments/plots")
    parser.add_argument("--variant", help="Optional: only these rows for run-detail panel (dashboard still uses all rows for variant bars)")
    parser.add_argument("--model", help="Optional model filter")
    parser.add_argument("--min-examples", type=int, default=10, help="Min num_examples to include in plots")
    parser.add_argument(
        "--legacy-single-bars",
        action="store_true",
        help="Also write old per-metric bar charts (often one bar if filter is narrow)",
    )
    args = parser.parse_args()

    rows = _read_rows(args.input_csv)
    rows = _filter_model(rows, args.model)

    rows_meeting = _filter_min_examples(rows, args.min_examples)
    if not rows_meeting:
        raise SystemExit("No rows with num_examples >= min-examples. Lower --min-examples.")

    os.makedirs(args.output_dir, exist_ok=True)

    detail_rows = rows_meeting
    if args.variant:
        detail_rows = [r for r in rows_meeting if r.get("variant") == args.variant]
    detail_title = f"Runs ({args.variant or 'all'}) n≥{args.min_examples}"

    dash_path = os.path.join(args.output_dir, "hyperparam_dashboard.png")
    _make_dashboard(
        output_path=dash_path,
        rows_all=rows_meeting,
        rows_detail=detail_rows,
        detail_title=detail_title,
    )

    if args.legacy_single_bars:
        filt = rows_meeting
        if args.variant:
            filt = [r for r in rows_meeting if r.get("variant") == args.variant]
        if filt:
            _make_bar_plot_legacy(
                output_path=os.path.join(args.output_dir, "error_detection_auroc_by_setting.png"),
                title="Error detection AUROC by setting (mean +/- std)",
                ylabel="AUROC",
                stats=_group_stats(filt, "error_detection_auroc"),
            )
            _make_bar_plot_legacy(
                output_path=os.path.join(args.output_dir, "ece_by_setting.png"),
                title="ECE by setting (mean +/- std)",
                ylabel="ECE",
                stats=_group_stats(filt, "ece_confidence"),
            )
            _make_bar_plot_legacy(
                output_path=os.path.join(args.output_dir, "clarity_by_setting.png"),
                title="Clarity by setting (mean +/- std)",
                ylabel="Clarity score",
                stats=_group_stats(filt, "avg_clarity_score"),
            )
            _make_bar_plot_legacy(
                output_path=os.path.join(args.output_dir, "interpretability_by_setting.png"),
                title="Interpretability by setting (mean +/- std)",
                ylabel="Interpretability score",
                stats=_group_stats(filt, "avg_interpretability_score"),
            )

    print(f"Saved dashboard: {dash_path}")
    print(f"Saved runs panel: {os.path.join(args.output_dir, 'hyperparam_dashboard_runs.png')}")


if __name__ == "__main__":
    main()
