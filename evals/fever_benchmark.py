import argparse
import csv
from datetime import datetime
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Allow importing the existing uncertainty pipeline from project root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uncertainty_logit_gap import (  # noqa: E402
    NEBULA_BASE_URL,
    _ensure_dir,
    _save_confidence_html,
    run_logit_gap_claim_detection_live,
)

LABEL_SUPPORTS = "SUPPORTED"
LABEL_REFUTES = "REFUTED"
LABEL_NEI = "NOT_ENOUGH_INFO"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _normalize_label(text: str) -> str:
    t = (text or "").strip().upper()
    t = t.replace("_", " ")

    if "NOT ENOUGH" in t or "NEI" in t:
        return LABEL_NEI
    if "SUPPORT" in t:
        return LABEL_SUPPORTS
    if "REFUT" in t:
        return LABEL_REFUTES
    return "UNKNOWN"


def _extract_predicted_label(answer_text: str) -> str:
    lines = [ln.strip() for ln in (answer_text or "").splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        m = re.search(r"(SUPPORTED|SUPPORTS|REFUTED|REFUTES|NOT[ _-]?ENOUGH[ _-]?INFO|NEI)", first, re.IGNORECASE)
        if m:
            return _normalize_label(m.group(1))
    return _normalize_label(answer_text)


def _build_prompt(claim: str) -> str:
    return (
        "Task: classify the following factual claim into exactly one label: "
        "SUPPORTED, REFUTED, or NOT_ENOUGH_INFO.\n"
        "Output format strictly:\n"
        "LABEL: <SUPPORTED|REFUTED|NOT_ENOUGH_INFO>\n"
        "REASON: <one short sentence>\n\n"
        f"CLAIM: {claim}"
    )


def _sample_uncertainty(result: Dict[str, Any]) -> Optional[float]:
    claim_spans = result.get("claim_spans", []) or []
    fragility = [c.get("fragility_score") for c in claim_spans if c.get("fragility_score") is not None]
    if fragility:
        v = sum(float(x) for x in fragility) / len(fragility)
        return max(0.0, min(1.0, v))

    token_signals = result.get("token_signals", []) or []
    gaps = [t.get("prob_gap") for t in token_signals if t.get("prob_gap") is not None]
    if gaps:
        mean_gap = sum(float(x) for x in gaps) / len(gaps)
        v = 1.0 - max(0.0, min(1.0, mean_gap))
        return max(0.0, min(1.0, v))

    return None


def _compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    valid = [r for r in rows if r.get("uncertainty") is not None]
    if not valid:
        metrics["num_rows"] = len(rows)
        metrics["num_valid_uncertainty"] = 0
        return metrics

    metrics["num_rows"] = len(rows)
    metrics["num_valid_uncertainty"] = len(valid)

    by_label: Dict[str, List[float]] = {LABEL_SUPPORTS: [], LABEL_REFUTES: [], LABEL_NEI: []}
    for r in valid:
        by_label.setdefault(r["gold_label"], []).append(float(r["uncertainty"]))

    label_stats = {}
    for label, vals in by_label.items():
        if vals:
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
            label_stats[label] = {"n": len(vals), "mean_uncertainty": mean, "std_uncertainty": std}
        else:
            label_stats[label] = {"n": 0, "mean_uncertainty": None, "std_uncertainty": None}
    metrics["label_stats"] = label_stats

    y_unc = [float(r["uncertainty"]) for r in valid]
    y_nei = [1 if r["gold_label"] == LABEL_NEI else 0 for r in valid]
    y_err = [1 if not r["is_correct"] else 0 for r in valid]

    try:
        from sklearn.metrics import roc_auc_score

        metrics["auroc_nei_detection"] = (
            roc_auc_score(y_nei, y_unc) if len(set(y_nei)) > 1 else None
        )
        metrics["auroc_error_detection"] = (
            roc_auc_score(y_err, y_unc) if len(set(y_err)) > 1 else None
        )
    except Exception:
        metrics["auroc_nei_detection"] = None
        metrics["auroc_error_detection"] = None

    try:
        from scipy.stats import spearmanr

        rho, pval = spearmanr(y_unc, y_err)
        metrics["spearman_uncertainty_vs_error"] = rho
        metrics["spearman_pvalue"] = pval
    except Exception:
        metrics["spearman_uncertainty_vs_error"] = None
        metrics["spearman_pvalue"] = None

    # Calibration-style buckets using confidence = 1 - uncertainty.
    buckets = []
    n_bins = 10
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bucket_rows = []
        for r in valid:
            conf = 1.0 - float(r["uncertainty"])
            if i < n_bins - 1:
                in_bucket = lo <= conf < hi
            else:
                in_bucket = lo <= conf <= hi
            if in_bucket:
                bucket_rows.append(r)

        if not bucket_rows:
            buckets.append({"bin": i, "count": 0, "mean_confidence": None, "accuracy": None})
            continue

        confs = [1.0 - float(r["uncertainty"]) for r in bucket_rows]
        accs = [1.0 if r["is_correct"] else 0.0 for r in bucket_rows]
        buckets.append(
            {
                "bin": i,
                "count": len(bucket_rows),
                "mean_confidence": sum(confs) / len(confs),
                "accuracy": sum(accs) / len(accs),
            }
        )

    total = sum(b["count"] for b in buckets)
    ece = 0.0
    if total > 0:
        for b in buckets:
            if b["count"] == 0:
                continue
            ece += abs(float(b["accuracy"]) - float(b["mean_confidence"])) * (b["count"] / total)
    metrics["ece_confidence_from_fragility"] = ece if total > 0 else None
    metrics["calibration_buckets"] = buckets

    return metrics


def _save_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _plot_report(run_dir: str, rows: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    if not rows:
        return

    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        return

    # 1) Uncertainty by gold label (mean +/- std).
    labels = [LABEL_SUPPORTS, LABEL_REFUTES, LABEL_NEI]
    stats = metrics.get("label_stats", {})
    means = [stats.get(lb, {}).get("mean_uncertainty") or 0.0 for lb in labels]
    stds = [stats.get(lb, {}).get("std_uncertainty") or 0.0 for lb in labels]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel("Mean uncertainty")
    plt.title("Uncertainty by FEVER gold label")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "uncertainty_by_label.pdf"))
    plt.close()

    # 2) Correct vs incorrect uncertainty.
    corr = [float(r["uncertainty"]) for r in rows if r.get("uncertainty") is not None and r["is_correct"]]
    err = [float(r["uncertainty"]) for r in rows if r.get("uncertainty") is not None and not r["is_correct"]]

    vals = [sum(corr) / len(corr) if corr else 0.0, sum(err) / len(err) if err else 0.0]
    stds = []
    for arr in [corr, err]:
        if arr:
            m = sum(arr) / len(arr)
            stds.append(math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr)))
        else:
            stds.append(0.0)

    plt.figure(figsize=(6, 4))
    plt.bar(["correct", "incorrect"], vals, yerr=stds, capsize=5)
    plt.ylabel("Mean uncertainty")
    plt.title("Uncertainty vs correctness")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "uncertainty_correct_vs_incorrect.pdf"))
    plt.close()

    # 3) Reliability-style plot.
    buckets = metrics.get("calibration_buckets", [])
    xs = []
    acc = []
    conf = []
    for b in buckets:
        if b.get("count", 0) == 0:
            continue
        xs.append(float(b["bin"]) / 10.0 + 0.05)
        acc.append(float(b["accuracy"]))
        conf.append(float(b["mean_confidence"]))

    if xs:
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(xs, acc, marker="o", label="accuracy")
        plt.plot(xs, conf, marker="s", label="mean confidence")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Confidence bucket center")
        plt.ylabel("Value")
        plt.title("Calibration-style reliability plot")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "calibration_reliability.pdf"))
        plt.close()

    # 4) Confusion matrix (gold vs predicted labels).
    cm_labels = [LABEL_SUPPORTS, LABEL_REFUTES, LABEL_NEI, "UNKNOWN"]
    index = {lb: i for i, lb in enumerate(cm_labels)}
    mat = [[0 for _ in cm_labels] for _ in cm_labels]
    for r in rows:
        g = r.get("gold_label", "UNKNOWN")
        p = r.get("pred_label", "UNKNOWN")
        gi = index.get(g, index["UNKNOWN"])
        pi = index.get(p, index["UNKNOWN"])
        mat[gi][pi] += 1

    plt.figure(figsize=(7, 6))
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar()
    ticks = list(range(len(cm_labels)))
    plt.xticks(ticks, cm_labels, rotation=25, ha="right")
    plt.yticks(ticks, cm_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("Gold label")
    plt.title("FEVER confusion matrix")
    for i in range(len(cm_labels)):
        for j in range(len(cm_labels)):
            plt.text(j, i, str(mat[i][j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.pdf"))
    plt.close()

    # 5) Uncertainty distribution by gold label (histograms).
    valid_rows = [r for r in rows if r.get("uncertainty") is not None]
    if valid_rows:
        plt.figure(figsize=(8, 4.8))
        bins = [i / 20 for i in range(21)]
        for lb in [LABEL_SUPPORTS, LABEL_REFUTES, LABEL_NEI]:
            vals = [float(r["uncertainty"]) for r in valid_rows if r.get("gold_label") == lb]
            if vals:
                plt.hist(vals, bins=bins, alpha=0.45, label=lb)
        plt.xlabel("Uncertainty")
        plt.ylabel("Count")
        plt.title("Uncertainty distributions by FEVER gold label")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "uncertainty_distribution_by_label.pdf"))
        plt.close()

    # 6) ROC curves for NEI detection and error detection.
    if valid_rows:
        try:
            from sklearn.metrics import roc_curve, roc_auc_score

            y_unc = [float(r["uncertainty"]) for r in valid_rows]
            y_nei = [1 if r["gold_label"] == LABEL_NEI else 0 for r in valid_rows]
            y_err = [1 if not r["is_correct"] else 0 for r in valid_rows]

            plt.figure(figsize=(6.5, 5.0))
            plotted = False

            if len(set(y_nei)) > 1:
                fpr, tpr, _ = roc_curve(y_nei, y_unc)
                auc = roc_auc_score(y_nei, y_unc)
                plt.plot(fpr, tpr, label=f"NEI vs non-NEI (AUROC={auc:.3f})")
                plotted = True

            if len(set(y_err)) > 1:
                fpr, tpr, _ = roc_curve(y_err, y_unc)
                auc = roc_auc_score(y_err, y_unc)
                plt.plot(fpr, tpr, label=f"Error detection (AUROC={auc:.3f})")
                plotted = True

            if plotted:
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                plt.title("ROC curves for uncertainty informativeness")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "roc_curves_uncertainty.pdf"))
                plt.close()
            else:
                plt.close()
        except Exception:
            pass

    # 7) Uncertainty vs claim length.
    if valid_rows:
        xs = [len(str(r.get("claim", ""))) for r in valid_rows]
        ys = [float(r["uncertainty"]) for r in valid_rows]
        if xs and ys:
            plt.figure(figsize=(7, 4.5))
            plt.scatter(xs, ys, alpha=0.65)
            plt.xlabel("Claim length (characters)")
            plt.ylabel("Uncertainty")
            plt.title("Uncertainty vs claim length")
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "uncertainty_vs_claim_length.pdf"))
            plt.close()


def _load_fever_rows(
    *,
    submodule_dir: str,
    config_name: str,
    split_name: str,
    max_examples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    # Newer versions of `datasets` no longer support dataset scripts for FEVER.
    # Load FEVER from Hub parquet conversion files instead.
    parquet_glob = f"hf://datasets/fever/fever@refs/convert/parquet/{config_name}/{split_name}/*.parquet"
    try:
        ds = load_dataset("parquet", data_files={"data": parquet_glob}, split="data")
    except Exception as exc:
        raise SystemExit(
            "Failed to load FEVER parquet files from Hugging Face Hub. "
            "Check network access and split/config values. "
            f"config={config_name}, split={split_name}. Original error: {exc}"
        ) from exc

    unique: Dict[int, Dict[str, Any]] = {}
    for item in ds:
        cid = int(item["id"])
        if cid in unique:
            continue
        label = _normalize_label(item.get("label", ""))
        if label not in {LABEL_SUPPORTS, LABEL_REFUTES, LABEL_NEI}:
            continue
        unique[cid] = {
            "id": cid,
            "claim": str(item.get("claim", "")),
            "label": label,
            "evidence_wiki_url": item.get("evidence_wiki_url", ""),
            "evidence_sentence_id": item.get("evidence_sentence_id", -1),
        }

    rows = list(unique.values())
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    if max_examples > 0:
        rows = rows[:max_examples]
    return rows


def run_benchmark(args: argparse.Namespace) -> str:
    api_key = args.api_key or os.getenv("NEBULA_API_KEY")
    if not api_key:
        raise SystemExit("Missing NEBULA_API_KEY in .env or --api-key")
    if not args.model:
        raise SystemExit("Missing --model")

    run_id = f"fever_{args.config}_{args.split}_{_now()}"
    run_dir = os.path.join(args.output_dir, run_id)
    json_dir = os.path.join(run_dir, "result_jsons")
    sentence_dir = os.path.join(run_dir, "sentence_confidence")
    _ensure_dir(run_dir)
    _ensure_dir(json_dir)
    _ensure_dir(sentence_dir)

    fever_rows = _load_fever_rows(
        submodule_dir=args.fever_submodule,
        config_name=args.config,
        split_name=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    summary_rows: List[Dict[str, Any]] = []

    for run_idx in range(1, args.num_runs + 1):
        print(f"\n[run {run_idx}/{args.num_runs}]")
        for i, sample in enumerate(fever_rows, start=1):
            claim = sample["claim"]
            prompt = _build_prompt(claim)
            result = run_logit_gap_claim_detection_live(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                system_prompt=args.system,
                user_prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_logprobs=args.top_logprobs,
            )

            pred_label = _extract_predicted_label(result.get("answer_text", ""))
            uncertainty = _sample_uncertainty(result)
            is_correct = pred_label == sample["label"]

            row = {
                "run_index": run_idx,
                "sample_index": i,
                "claim_id": sample["id"],
                "gold_label": sample["label"],
                "pred_label": pred_label,
                "is_correct": is_correct,
                "uncertainty": uncertainty,
                "token_logprobs_available": result.get("provider_capabilities", {}).get("token_logprobs_available", False),
                "num_claim_spans": len(result.get("claim_spans", []) or []),
                "num_high_claims": sum(1 for c in (result.get("claim_spans", []) or []) if c.get("severity") == "high"),
                "mean_claim_fragility": uncertainty,
                "claim": claim,
            }
            summary_rows.append(row)

            record = {
                "run_index": run_idx,
                "sample": sample,
                "prompt": prompt,
                "prediction": {
                    "label": pred_label,
                    "is_correct": is_correct,
                    "uncertainty": uncertainty,
                },
                "analysis": result,
            }
            safe_name = f"result_run{run_idx:02d}_id{sample['id']}.json"
            with open(os.path.join(json_dir, safe_name), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            html_name = f"confidence_run{run_idx:02d}_id{sample['id']}.html"
            _save_confidence_html(os.path.join(sentence_dir, html_name), result, prompt)

            print(
                f"[saved] run={run_idx} sample={i}/{len(fever_rows)} id={sample['id']} "
                f"gold={sample['label']} pred={pred_label} uncertainty={uncertainty}"
            )

    _save_rows_csv(os.path.join(run_dir, "summary.csv"), summary_rows)
    metrics = _compute_metrics(summary_rows)

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    config = {
        "run_id": run_id,
        "dataset": "FEVER",
        "fever_submodule": args.fever_submodule,
        "config": args.config,
        "split": args.split,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "model": args.model,
        "temperature": args.temperature,
        "num_runs": args.num_runs,
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "base_url": args.base_url,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    _plot_report(run_dir, summary_rows, metrics)
    print(f"\nFEVER benchmark artifacts saved in: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FEVER benchmark for logit-gap uncertainty informativeness")
    parser.add_argument("--fever-submodule", default=os.path.join(ROOT_DIR, "external", "fever"))
    parser.add_argument("--config", default="v1.0", help="FEVER dataset config: v1.0 or v2.0")
    parser.add_argument("--split", default="labelled_dev", help="Split to evaluate (e.g., labelled_dev, paper_dev, validation)")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-url", default=NEBULA_BASE_URL)
    parser.add_argument("--api-key", help="Optional API key override")
    parser.add_argument("--model", required=True)
    parser.add_argument("--system", default="You are a careful fact-checking assistant.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--top-logprobs", type=int, default=5)

    parser.add_argument("--output-dir", default=os.path.join(ROOT_DIR, "experiments"))
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.num_runs < 1:
        raise SystemExit("--num-runs must be >= 1")

    if not os.path.isdir(args.fever_submodule):
        print(
            "Warning: FEVER submodule directory was not found. "
            "Continuing because dataset loading uses Hugging Face Hub directly."
        )

    run_benchmark(args)


if __name__ == "__main__":
    main()
