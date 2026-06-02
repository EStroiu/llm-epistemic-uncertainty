#!/usr/bin/env python3
"""FEVER benchmark variant that estimates uncertainty via ensemble self-consistency.

The metric is based on repeated sampled outputs for the same claim across a
temperature ensemble. The claim-level uncertainty combines label entropy,
label disagreement, and reason disagreement so it can detect instability even
when one temperature produces a confident-looking majority answer.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime
import json
import math
import os
import re
import sys
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> None:
        return None

# Allow importing the existing FEVER benchmark helpers from the project root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evals.fever_benchmark import (  # noqa: E402
    LABEL_NEI,
    LABEL_REFUTES,
    LABEL_SUPPORTS,
    _build_prompt,
    _compute_metrics,
    _extract_predicted_label,
    _load_fever_rows,
)
from uncertainty_logit_gap import (  # noqa: E402
    NEBULA_BASE_URL,
    _ensure_dir,
    run_logit_gap_claim_detection_live,
)


REASON_LINE_RE = re.compile(r"^\s*REASON\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
REASON_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]+\b", re.IGNORECASE)
STOPWORD_SET = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "hers", "him", "his", "i", "if", "in", "into", "is",
    "it", "its", "may", "might", "must", "not", "of", "on", "or", "our",
    "out", "over", "she", "should", "so", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "under",
    "up", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _save_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_evaluation_metrics_csv(path: str, metrics: Dict[str, Any]) -> None:
    scalar_keys = [
        "num_rows",
        "num_valid_uncertainty",
        "auroc_nei_detection",
        "auroc_error_detection",
        "spearman_uncertainty_vs_error",
        "spearman_pvalue",
        "ece_confidence_from_self_consistency",
        "calibrated_confidence_ece",
        "calibrated_confidence_brier",
        "aurc",
        "accuracy_at_80pct_coverage",
        "accuracy_at_90pct_coverage",
        "balanced_accuracy",
        "macro_f1",
        "method",
        "samples_per_claim",
        "temperature",
        "temperatures",
    ]
    rows = []
    for key in scalar_keys:
        value = metrics.get(key)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        rows.append({"metric": key, "value": value})
    _save_rows_csv(path, rows)


def _extract_reason(answer_text: str) -> str:
    match = REASON_LINE_RE.search(answer_text or "")
    if not match:
        return ""
    return match.group(1).strip()


def _normalize_reason(reason_text: str) -> str:
    text = (reason_text or "").strip().lower()
    text = re.sub(r"[\W_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _reason_token_set(reason_text: str) -> set[str]:
    tokens = []
    for token in REASON_TOKEN_RE.findall((reason_text or "").lower()):
        if token in STOPWORD_SET:
            continue
        if len(token) < 3:
            continue
        tokens.append(token)
    return set(tokens)


def _normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 1:
        return 0.0

    probs = [count / total for count in counts.values() if count > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    max_classes = max(1, min(len(counts), total))
    if max_classes <= 1:
        return 0.0
    return float(entropy / math.log(max_classes))


def _majority_fraction(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(counts.most_common(1)[0][1] / total)


def _pairwise_jaccard_disagreement(texts: List[str]) -> float:
    token_sets = [ts for ts in (_reason_token_set(text) for text in texts) if ts]
    if len(token_sets) <= 1:
        return 0.0

    pairwise = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a = token_sets[i]
            b = token_sets[j]
            union = a | b
            if not union:
                continue
            pairwise.append(1.0 - (len(a & b) / len(union)))

    if not pairwise:
        return 0.0
    return float(sum(pairwise) / len(pairwise))


def _parse_temperatures(value: str) -> List[float]:
    parts = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not parts:
        return [0.5]
    temps = [float(item) for item in parts]
    unique_temps = []
    seen = set()
    for temp in temps:
        if temp in seen:
            continue
        seen.add(temp)
        unique_temps.append(temp)
    return unique_temps


def _sample_for_claim(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    temperatures: List[float],
    max_tokens: int,
    top_logprobs: int,
    samples_per_claim: int,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    sample_idx = 0
    for temperature in temperatures:
        for _ in range(samples_per_claim):
            sample_idx += 1
            result = run_logit_gap_claim_detection_live(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_logprobs=top_logprobs,
                signal_granularity="token",
            )
            answer_text = str(result.get("answer_text", "") or "")
            pred_label = _extract_predicted_label(answer_text)
            reason_text = _extract_reason(answer_text)

            samples.append(
                {
                    "sample_index": sample_idx,
                    "temperature": temperature,
                    "pred_label": pred_label,
                    "reason_text": reason_text,
                    "normalized_reason": _normalize_reason(reason_text),
                    "finish_reason": result.get("finish_reason"),
                    "answer_text": answer_text,
                    "token_logprobs_available": result.get("provider_capabilities", {}).get("token_logprobs_available", False),
                    "analysis": result,
                }
            )
    return samples


def _aggregate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counts: Counter[str] = Counter(sample["pred_label"] for sample in samples)
    reason_counts: Counter[str] = Counter(sample["normalized_reason"] or "<empty>" for sample in samples)
    reason_texts = [sample["reason_text"] for sample in samples]

    label_entropy = _normalized_entropy(label_counts)
    reason_entropy = _normalized_entropy(reason_counts)
    label_disagreement = 1.0 - _majority_fraction(label_counts)
    reason_disagreement = _pairwise_jaccard_disagreement(reason_texts)
    reason_diversity = 1.0 - _majority_fraction(reason_counts)
    format_failure_rate = sum(
        1
        for sample in samples
        if sample["pred_label"] == "UNKNOWN"
        or not sample["reason_text"]
        or str(sample.get("finish_reason", "")).lower() == "length"
    ) / len(samples)

    # Mix label instability with reason disagreement so paraphrases still count.
    uncertainty = _clamp01(
        0.4 * label_entropy
        + 0.2 * label_disagreement
        + 0.3 * reason_disagreement
        + 0.1 * format_failure_rate
    )
    majority_label, majority_label_count = label_counts.most_common(1)[0]
    majority_reason, majority_reason_count = reason_counts.most_common(1)[0]

    per_temperature = {}
    for sample in samples:
        temp_key = str(sample.get("temperature", ""))
        per_temperature.setdefault(temp_key, []).append(sample)

    temp_label_stability = 0.0
    temp_reason_stability = 0.0
    if per_temperature:
        temp_label_scores = []
        temp_reason_scores = []
        for temp_samples in per_temperature.values():
            temp_label_counts = Counter(sample["pred_label"] for sample in temp_samples)
            temp_reason_counts = Counter(sample["normalized_reason"] or "<empty>" for sample in temp_samples)
            temp_label_scores.append(_majority_fraction(temp_label_counts))
            temp_reason_scores.append(_majority_fraction(temp_reason_counts))
        temp_label_stability = sum(temp_label_scores) / len(temp_label_scores)
        temp_reason_stability = sum(temp_reason_scores) / len(temp_reason_scores)

    return {
        "label_counts": dict(label_counts),
        "reason_counts": dict(reason_counts),
        "label_entropy": label_entropy,
        "reason_entropy": reason_entropy,
        "label_disagreement": label_disagreement,
        "reason_disagreement": reason_disagreement,
        "reason_diversity": reason_diversity,
        "format_failure_rate": format_failure_rate,
        "temperature_label_stability": temp_label_stability,
        "temperature_reason_stability": temp_reason_stability,
        "temperature_count": len(per_temperature),
        "majority_label": majority_label,
        "majority_label_fraction": majority_label_count / len(samples),
        "majority_reason": majority_reason,
        "majority_reason_fraction": majority_reason_count / len(samples),
        "uncertainty": uncertainty,
        "confidence": 1.0 - uncertainty,
        "num_samples": len(samples),
        "num_distinct_labels": len(label_counts),
        "num_distinct_reasons": len(reason_counts),
    }


def _plot_report(run_dir: str, rows: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    if not rows:
        return

    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        return

    labels = [LABEL_SUPPORTS, LABEL_REFUTES, LABEL_NEI]
    stats = metrics.get("label_stats", {})
    means = [stats.get(lb, {}).get("mean_uncertainty") or 0.0 for lb in labels]
    stds = [stats.get(lb, {}).get("std_uncertainty") or 0.0 for lb in labels]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel("Mean uncertainty")
    plt.title("Self-consistency uncertainty by FEVER gold label")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "uncertainty_by_label.pdf"))
    plt.close()

    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        xs.append(float(row.get("majority_label_fraction", 0.0)))
        ys.append(float(row.get("uncertainty", 0.0)))

    if xs and ys:
        plt.figure(figsize=(7, 4))
        plt.scatter(xs, ys, alpha=0.7)
        plt.xlabel("Majority label fraction")
        plt.ylabel("Uncertainty")
        plt.title("Agreement vs self-consistency uncertainty")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "agreement_vs_uncertainty.pdf"))
        plt.close()


def _risk_coverage_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("uncertainty") is not None]
    if not valid:
        return {"aurc": None, "risk_coverage_curve": [], "accuracy_at_80pct_coverage": None, "accuracy_at_90pct_coverage": None}

    ordered = sorted(valid, key=lambda r: float(r["uncertainty"]))
    total = len(ordered)
    curve = []
    correct_so_far = 0
    risks = []
    for idx, row in enumerate(ordered, start=1):
        if row.get("is_correct"):
            correct_so_far += 1
        coverage = idx / total
        accuracy = correct_so_far / idx
        risk = 1.0 - accuracy
        risks.append(risk)
        curve.append({"coverage": coverage, "accuracy": accuracy, "risk": risk})

    aurc = sum(risks) / len(risks)

    def _accuracy_at_coverage(coverage_target: float) -> float:
        k = max(1, int(math.ceil(total * coverage_target)))
        kept = ordered[:k]
        return sum(1 for row in kept if row.get("is_correct")) / len(kept)

    return {
        "aurc": aurc,
        "risk_coverage_curve": curve,
        "accuracy_at_80pct_coverage": _accuracy_at_coverage(0.8),
        "accuracy_at_90pct_coverage": _accuracy_at_coverage(0.9),
    }


def _class_balance_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("uncertainty") is not None]
    if not valid:
        return {"balanced_accuracy": None, "macro_f1": None}

    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score
    except Exception:
        return {"balanced_accuracy": None, "macro_f1": None}

    y_true = [r["gold_label"] for r in valid]
    y_pred = [r["pred_label"] for r in valid]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _calibrate_confidence_oof(rows: List[Dict[str, Any]]) -> List[float]:
    valid_idx = [i for i, row in enumerate(rows) if row.get("uncertainty") is not None]
    if len(valid_idx) < 4:
        return [float(row.get("confidence", 0.5) or 0.5) for row in rows]

    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import StratifiedKFold
    except Exception:
        return [float(row.get("confidence", 0.5) or 0.5) for row in rows]

    y = [1 if rows[i].get("is_correct") else 0 for i in valid_idx]
    x = [float(rows[i].get("confidence", 0.5) or 0.5) for i in valid_idx]
    calibrated = [float(row.get("confidence", 0.5) or 0.5) for row in rows]

    class_counts = Counter(y)
    min_class_count = min(class_counts.values()) if class_counts else 0
    n_splits = min(5, min_class_count, len(valid_idx))
    if n_splits < 2:
        mean_y = sum(y) / len(y)
        for idx in valid_idx:
            calibrated[idx] = mean_y
        return calibrated

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_pos, test_pos in splitter.split(x, y):
        train_x = [x[i] for i in train_pos]
        train_y = [y[i] for i in train_pos]
        test_positions = [valid_idx[i] for i in test_pos]

        if len(set(train_y)) < 2:
            fallback = sum(train_y) / len(train_y)
            for row_idx in test_positions:
                calibrated[row_idx] = fallback
            continue

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(train_x, train_y)
        for local_idx, row_idx in zip(test_pos, test_positions):
            calibrated[row_idx] = float(model.predict([x[local_idx]])[0])

    return calibrated


def _calibration_metrics(rows: List[Dict[str, Any]], confidence_key: str) -> Dict[str, Any]:
    valid = [r for r in rows if r.get(confidence_key) is not None]
    if not valid:
        return {"ece": None, "brier": None}

    y_true = [1 if r.get("is_correct") else 0 for r in valid]
    confs = [float(r.get(confidence_key, 0.5) or 0.5) for r in valid]

    buckets = []
    n_bins = 10
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i < n_bins - 1:
            mask = [lo <= c < hi for c in confs]
        else:
            mask = [lo <= c <= hi for c in confs]
        bucket_idx = [j for j, flag in enumerate(mask) if flag]
        if not bucket_idx:
            buckets.append({"bin": i, "count": 0, "mean_confidence": None, "accuracy": None})
            continue
        bucket_confs = [confs[j] for j in bucket_idx]
        bucket_accs = [y_true[j] for j in bucket_idx]
        mean_conf = sum(bucket_confs) / len(bucket_confs)
        mean_acc = sum(bucket_accs) / len(bucket_accs)
        buckets.append({"bin": i, "count": len(bucket_idx), "mean_confidence": mean_conf, "accuracy": mean_acc})
        ece += abs(mean_conf - mean_acc) * (len(bucket_idx) / len(valid))

    brier = sum((confs[i] - y_true[i]) ** 2 for i in range(len(valid))) / len(valid)
    return {"ece": ece, "brier": brier, "calibration_buckets": buckets}


def run_benchmark(args: argparse.Namespace) -> str:
    api_key = args.api_key or os.getenv("NEBULA_API_KEY")
    if not api_key:
        raise SystemExit("Missing NEBULA_API_KEY in .env or --api-key")
    if not args.model:
        raise SystemExit("Missing --model")

    run_id = f"fever_selfconsistency_{args.config}_{args.split}_{_now()}"
    run_dir = os.path.join(args.output_dir, run_id)
    result_dir = os.path.join(run_dir, "result_jsons")
    _ensure_dir(run_dir)
    _ensure_dir(result_dir)

    fever_rows = _load_fever_rows(
        submodule_dir=args.fever_submodule,
        config_name=args.config,
        split_name=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    temperatures = _parse_temperatures(getattr(args, "temperatures", ""))
    summary_rows: List[Dict[str, Any]] = []

    for run_idx in range(1, args.num_runs + 1):
        print(f"\n[run {run_idx}/{args.num_runs}]")
        for i, sample in enumerate(fever_rows, start=1):
            claim = sample["claim"]
            prompt = _build_prompt(claim)

            sampled_outputs = _sample_for_claim(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                system_prompt=args.system,
                prompt=prompt,
                temperatures=temperatures,
                max_tokens=args.max_tokens,
                top_logprobs=args.top_logprobs,
                samples_per_claim=args.samples_per_claim,
            )
            aggregate = _aggregate_samples(sampled_outputs)
            pred_label = aggregate["majority_label"]
            is_correct = pred_label == sample["label"]

            row = {
                "run_index": run_idx,
                "sample_index": i,
                "claim_id": sample["id"],
                "gold_label": sample["label"],
                "pred_label": pred_label,
                "is_correct": is_correct,
                "uncertainty": aggregate["uncertainty"],
                "confidence": aggregate["confidence"],
                "label_entropy": aggregate["label_entropy"],
                "reason_diversity": aggregate["reason_diversity"],
                "format_failure_rate": aggregate["format_failure_rate"],
                "majority_label_fraction": aggregate["majority_label_fraction"],
                "majority_reason_fraction": aggregate["majority_reason_fraction"],
                "num_samples": aggregate["num_samples"],
                "num_distinct_labels": aggregate["num_distinct_labels"],
                "num_distinct_reasons": aggregate["num_distinct_reasons"],
                "temperature": args.temperature,
                "temperatures": ",".join(str(t) for t in temperatures),
                "samples_per_claim": args.samples_per_claim,
                "claim": claim,
            }
            summary_rows.append(row)

            record = {
                "run_index": run_idx,
                "sample": sample,
                "prompt": prompt,
                "samples": sampled_outputs,
                "prediction": {
                    "label": pred_label,
                    "is_correct": is_correct,
                    "uncertainty": aggregate["uncertainty"],
                    "confidence": aggregate["confidence"],
                },
                "analysis": aggregate,
            }
            safe_name = f"result_run{run_idx:02d}_id{sample['id']}.json"
            with open(os.path.join(result_dir, safe_name), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            print(
                f"[saved] run={run_idx} sample={i}/{len(fever_rows)} id={sample['id']} "
                f"gold={sample['label']} pred={pred_label} uncertainty={aggregate['uncertainty']:.3f}"
            )

    calibrated_confidence = _calibrate_confidence_oof(summary_rows)
    for row, cal_conf in zip(summary_rows, calibrated_confidence):
        row["calibrated_confidence"] = cal_conf
        row["calibrated_uncertainty"] = 1.0 - cal_conf

    _save_rows_csv(os.path.join(run_dir, "summary.csv"), summary_rows)

    metrics = _compute_metrics(summary_rows)
    metrics["ece_confidence_from_self_consistency"] = metrics.pop("ece_confidence_from_fragility", None)
    metrics.update(_risk_coverage_metrics(summary_rows))
    metrics.update(_class_balance_metrics(summary_rows))
    calibrated_metrics = _calibration_metrics(summary_rows, "calibrated_confidence")
    metrics["calibrated_confidence_ece"] = calibrated_metrics["ece"]
    metrics["calibrated_confidence_brier"] = calibrated_metrics["brier"]
    metrics["calibrated_confidence_buckets"] = calibrated_metrics["calibration_buckets"]

    metrics["method"] = "self_consistency"
    metrics["samples_per_claim"] = args.samples_per_claim
    metrics["temperature"] = args.temperature
    metrics["temperatures"] = temperatures
    metrics["num_samples_per_claim_total"] = args.samples_per_claim * len(temperatures)

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    _save_evaluation_metrics_csv(os.path.join(run_dir, "evaluation_metrics.csv"), metrics)

    with open(os.path.join(run_dir, "risk_coverage_curve.json"), "w", encoding="utf-8") as f:
        json.dump(metrics.get("risk_coverage_curve", []), f, ensure_ascii=False, indent=2)

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
        "temperatures": temperatures,
        "num_runs": args.num_runs,
        "samples_per_claim": args.samples_per_claim,
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "base_url": args.base_url,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    _plot_report(run_dir, summary_rows, metrics)
    print(f"\nFEVER self-consistency artifacts saved in: {run_dir}")
    print(f"AURC: {metrics.get('aurc')}")
    print(f"Accuracy @ 80% coverage: {metrics.get('accuracy_at_80pct_coverage')}")
    print(f"Accuracy @ 90% coverage: {metrics.get('accuracy_at_90pct_coverage')}")
    print(f"Balanced accuracy: {metrics.get('balanced_accuracy')}")
    print(f"Macro F1: {metrics.get('macro_f1')}")
    print(f"Calibrated ECE: {metrics.get('calibrated_confidence_ece')}")
    print(f"Calibrated Brier: {metrics.get('calibrated_confidence_brier')}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FEVER benchmark with self-consistency uncertainty.")
    parser.add_argument("--fever-submodule", default=os.path.join(ROOT_DIR, "external", "fever"))
    parser.add_argument("--config", default="v1.0", help="FEVER dataset config: v1.0 or v2.0")
    parser.add_argument("--split", default="labelled_dev", help="Split to evaluate (e.g., labelled_dev, paper_dev, validation)")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-url", default=NEBULA_BASE_URL)
    parser.add_argument("--api-key", help="Optional API key override")
    parser.add_argument("--model", required=True)
    parser.add_argument("--system", default="You are a careful fact-checking assistant.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Fallback temperature used if --temperatures is omitted")
    parser.add_argument("--temperatures", default="0.2,0.5,0.8", help="Comma-separated temperature ensemble, e.g. 0.2,0.5,0.8")
    parser.add_argument("--samples-per-claim", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-logprobs", type=int, default=5)

    parser.add_argument("--output-dir", default=os.path.join(ROOT_DIR, "experiments"))
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.num_runs < 1:
        raise SystemExit("--num-runs must be >= 1")
    if args.samples_per_claim < 2:
        raise SystemExit("--samples-per-claim must be >= 2")
    if not str(args.temperatures).strip():
        args.temperatures = str(args.temperature)

    run_benchmark(args)


if __name__ == "__main__":
    main()