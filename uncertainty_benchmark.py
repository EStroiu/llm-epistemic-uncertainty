import argparse
import csv
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> bool:
        return False
from claim_extraction import extract_atomic_claims
from claim_judging import judge_claim_against_answer
from claim_uncertainty import build_claim_consistency_estimator, summarize_claim_uncertainties
from uncertainty_estimators import (
    FEVER_LABELS,
    estimate_nli_consistency,
    estimate_sampling_variance,
    estimate_self_disagreement,
    majority_vote_label,
    parse_label,
    parse_probabilities,
    select_random_pairs,
)
from uncertainty_prompting import evaluate_clarity_interpretability, get_variant_instruction
from uncertainty_schema import EstimatorOutput, build_uncertainty_payload


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_claim_label(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    claim = row.get("claim") or row.get("question") or row.get("input")
    label = row.get("label") or row.get("gold_label") or row.get("gold")
    if claim is None or label is None:
        return None

    label_norm = str(label).upper().replace(" ", "_")
    if label_norm not in FEVER_LABELS:
        # Keep only FEVER-like rows for a clean benchmark.
        return None
    return str(claim), label_norm


def _ask_for_answer(client: Any, model: str, claim: str, temperature: float, variant: str) -> str:
    instruction = get_variant_instruction(variant)
    prompt = (
        f"{instruction}\n\n"
        "Task: verify this claim and output one label from "
        "SUPPORTED / REFUTED / NOT_ENOUGH_INFO.\n"
        f"Claim: {claim}\n"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful fact verification assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=220,
    )
    return response.choices[0].message.content or ""


def _ask_for_probs(client: Any, model: str, claim: str, temperature: float) -> Dict[str, float]:
    prompt = (
        "Return class probabilities for this claim verification task.\n"
        "Output exactly these lines:\n"
        "SUPPORTED: <float 0..1>\n"
        "REFUTED: <float 0..1>\n"
        "NOT_ENOUGH_INFO: <float 0..1>\n\n"
        f"Claim: {claim}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You estimate class probabilities carefully."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=120,
    )
    text = response.choices[0].message.content or ""
    return parse_probabilities(text)


def _ask_nli_relation(client: Any, model: str, claim: str, answer_a: str, answer_b: str) -> str:
    prompt = (
        "Compare two answers to the same claim.\n"
        "Return one word only: entailment, contradiction, or neutral.\n\n"
        f"Claim: {claim}\n"
        f"Answer A: {answer_a}\n"
        f"Answer B: {answer_b}\n"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an NLI judge."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    text = (response.choices[0].message.content or "").strip().lower()
    if "contradiction" in text:
        return "contradiction"
    if "entailment" in text:
        return "entailment"
    return "neutral"


def _mock_answer(claim: str, variant: str, sample_idx: int) -> str:
    label_cycle = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO", "SUPPORTED"]
    label = label_cycle[sample_idx % len(label_cycle)]
    base = f"{label}. Short reason for claim: {claim[:80]}."
    if variant in {"numeric", "calibrated"}:
        conf = 70 - (sample_idx * 10)
        base += f" Confidence: {max(10, conf)}%."
    if variant == "calibrated":
        base += " Uncertainty reason: limited evidence."
    return base


def _mock_probs(sample_idx: int) -> Dict[str, float]:
    vals = [
        {"SUPPORTED": 0.68, "REFUTED": 0.20, "NOT_ENOUGH_INFO": 0.12},
        {"SUPPORTED": 0.30, "REFUTED": 0.58, "NOT_ENOUGH_INFO": 0.12},
        {"SUPPORTED": 0.31, "REFUTED": 0.24, "NOT_ENOUGH_INFO": 0.45},
        {"SUPPORTED": 0.51, "REFUTED": 0.32, "NOT_ENOUGH_INFO": 0.17},
    ]
    return vals[sample_idx % len(vals)]


def _mock_nli_relation(i: int, j: int) -> str:
    if (i + j) % 3 == 0:
        return "contradiction"
    if (i + j) % 2 == 0:
        return "entailment"
    return "neutral"


def _rankdata(values: Sequence[float]) -> List[float]:
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    den = math.sqrt(vx * vy)
    if den <= 0:
        return 0.0
    return cov / den


def auroc_binary(scores: Sequence[float], labels: Sequence[int]) -> float:
    n = len(scores)
    if n == 0 or n != len(labels):
        return 0.5
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = 0.0
    for i in pos:
        for j in neg:
            total += 1.0
            if scores[i] > scores[j]:
                wins += 1.0
            elif scores[i] == scores[j]:
                wins += 0.5
    return wins / total if total > 0 else 0.5


def expected_calibration_error(conf: Sequence[float], correct: Sequence[int], bins: int = 10) -> float:
    if not conf or len(conf) != len(correct):
        return 0.0
    ece = 0.0
    n = len(conf)
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (c >= lo and (c < hi or (b == bins - 1 and c <= hi)))]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        avg_c = sum(conf[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - avg_c)
    return ece


def _mean(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _claim_summary_stats(claim_uncertainties: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_vals = [
        float(c["final_uncertainty"])
        for c in claim_uncertainties
        if c.get("final_uncertainty") is not None
    ]
    semantic_vals = [
        float(c["semantic_uncertainty"])
        for c in claim_uncertainties
        if c.get("semantic_uncertainty") is not None
    ]
    certainty_vals = [
        float(c["final_certainty"])
        for c in claim_uncertainties
        if c.get("final_certainty") is not None
    ]

    return {
        "num_claims": len(claim_uncertainties),
        "mean_claim_uncertainty": _mean(final_vals),
        "max_claim_uncertainty": max(final_vals) if final_vals else None,
        "min_claim_certainty": min(certainty_vals) if certainty_vals else None,
        "mean_semantic_uncertainty": _mean(semantic_vals),
        "total_claim_supports": sum(int(c.get("support_count", 0)) for c in claim_uncertainties),
        "total_claim_contradictions": sum(int(c.get("contradiction_count", 0)) for c in claim_uncertainties),
        "total_claim_not_addressed": sum(int(c.get("not_addressed_count", 0)) for c in claim_uncertainties),
        "num_high_claims": sum(1 for c in claim_uncertainties if c.get("severity") == "high"),
        "num_medium_claims": sum(1 for c in claim_uncertainties if c.get("severity") == "medium"),
        "num_low_claims": sum(1 for c in claim_uncertainties if c.get("severity") == "low"),
    }


def run_benchmark(
    *,
    dataset_path: str,
    output_dir: str,
    model: str,
    max_examples: int,
    n_samples: int,
    sample_temperature: float,
    prob_temperature: float,
    nli_pairs: int,
    variant: str,
    dry_run: bool,
) -> Dict[str, Any]:
    load_dotenv()
    rows = _read_jsonl(dataset_path)
    pairs = [_extract_claim_label(r) for r in rows]
    pairs = [p for p in pairs if p is not None]
    pairs = pairs[:max_examples]
    if not pairs:
        raise RuntimeError("No valid rows found in dataset (need claim + FEVER-style label).")

    client = None
    if not dry_run:
        from nebula_prompting import get_nebula_client

        client = get_nebula_client()
    run_id = f"multi_estimator_{variant}_{_timestamp()}"
    run_dir = os.path.join(output_dir, run_id)
    _ensure_dir(run_dir)
    _ensure_dir(os.path.join(run_dir, "examples"))

    all_uncertainty: List[float] = []
    all_error: List[int] = []
    all_nei: List[int] = []
    all_conf: List[float] = []
    all_correct: List[int] = []
    clarity_vals: List[float] = []
    interp_vals: List[float] = []
    claim_count_vals: List[float] = []
    mean_claim_uncertainty_vals: List[float] = []
    max_claim_uncertainty_vals: List[float] = []
    min_claim_certainty_vals: List[float] = []
    claim_support_total = 0
    claim_contradiction_total = 0
    claim_not_addressed_total = 0
    summary_rows: List[Dict[str, Any]] = []

    for idx, (claim, gold_label) in enumerate(pairs, start=1):
        answer_samples: List[str] = []
        label_samples: List[str] = []
        prob_samples: List[Dict[str, float]] = []
        clarity_scores: List[float] = []
        interp_scores: List[float] = []

        for s_idx in range(n_samples):
            if dry_run:
                ans = _mock_answer(claim, variant, s_idx)
            else:
                ans = _ask_for_answer(client, model, claim, sample_temperature, variant)
            label = parse_label(ans)
            probs = _mock_probs(s_idx) if dry_run else _ask_for_probs(client, model, claim, prob_temperature)
            ci = evaluate_clarity_interpretability(ans)

            answer_samples.append(ans)
            label_samples.append(label)
            prob_samples.append(probs)
            clarity_scores.append(ci["clarity_score"])
            interp_scores.append(ci["interpretability_score"])

        rels: List[str] = []
        selected = select_random_pairs(answer_samples, max_pairs=nli_pairs, seed=idx + 123)
        for i, j in selected:
            if dry_run:
                rels.append(_mock_nli_relation(i, j))
            else:
                rels.append(_ask_nli_relation(client, model, claim, answer_samples[i], answer_samples[j]))

        est_sampling = estimate_sampling_variance(prob_samples)
        est_self = estimate_self_disagreement(answer_samples, label_samples)
        est_nli = estimate_nli_consistency(rels)

        main_answer = answer_samples[0] if answer_samples else ""
        alternative_answers = answer_samples[1:]
        atomic_claims = [c.to_dict() for c in extract_atomic_claims(main_answer)]
        claim_judgments: List[Dict[str, Any]] = []
        for extracted_claim in atomic_claims:
            for alt_idx, alt_answer in enumerate(alternative_answers, start=1):
                judgment = judge_claim_against_answer(
                    client=client,
                    model=model,
                    claim=extracted_claim["text"],
                    answer_text=alt_answer,
                    dry_run=dry_run,
                )
                row = judgment.to_dict()
                row["claim_id"] = extracted_claim["claim_id"]
                row["sample_index"] = alt_idx
                claim_judgments.append(row)

        claim_uncertainties = summarize_claim_uncertainties(atomic_claims, claim_judgments)
        claim_stats = _claim_summary_stats(claim_uncertainties)
        est_claims = build_claim_consistency_estimator(claim_uncertainties)
        estimators: List[EstimatorOutput] = [est_sampling, est_self, est_nli, est_claims]

        predicted = majority_vote_label(label_samples)
        payload = build_uncertainty_payload(
            content=main_answer,
            estimators=estimators,
            claims=claim_uncertainties,
            prompt_variant=variant,
            expression=variant,
            metadata={
                "claim": claim,
                "gold_label": gold_label,
                "predicted_label": predicted,
                "n_samples": n_samples,
                "atomic_claims": atomic_claims,
                "claim_judgments": claim_judgments,
                "claim_uncertainties": claim_uncertainties,
            },
        )

        u = payload["uncertainty"]["overall_uncertainty"]
        c = payload["uncertainty"]["overall_confidence"]
        err = int(predicted != gold_label)
        nei = int(gold_label == "NOT_ENOUGH_INFO")

        all_uncertainty.append(u)
        all_error.append(err)
        all_nei.append(nei)
        all_conf.append(c)
        all_correct.append(1 - err)
        clarity_vals.append(sum(clarity_scores) / len(clarity_scores))
        interp_vals.append(sum(interp_scores) / len(interp_scores))
        claim_count_vals.append(float(claim_stats["num_claims"]))
        if claim_stats["mean_claim_uncertainty"] is not None:
            mean_claim_uncertainty_vals.append(float(claim_stats["mean_claim_uncertainty"]))
        if claim_stats["max_claim_uncertainty"] is not None:
            max_claim_uncertainty_vals.append(float(claim_stats["max_claim_uncertainty"]))
        if claim_stats["min_claim_certainty"] is not None:
            min_claim_certainty_vals.append(float(claim_stats["min_claim_certainty"]))
        claim_support_total += int(claim_stats["total_claim_supports"])
        claim_contradiction_total += int(claim_stats["total_claim_contradictions"])
        claim_not_addressed_total += int(claim_stats["total_claim_not_addressed"])

        ex_payload = {
            "id": idx,
            "claim": claim,
            "gold_label": gold_label,
            "predicted_label": predicted,
            "answer_samples": answer_samples,
            "label_samples": label_samples,
            "nli_relations": rels,
            "uncertainty_payload": payload,
            "atomic_claims": atomic_claims,
            "claim_judgments": claim_judgments,
            "claim_uncertainties": claim_uncertainties,
            "claim_stats": claim_stats,
            "clarity": {
                "avg_clarity_score": sum(clarity_scores) / len(clarity_scores),
                "avg_interpretability_score": sum(interp_scores) / len(interp_scores),
            },
        }
        _save_json(os.path.join(run_dir, "examples", f"example_{idx:04d}.json"), ex_payload)

        summary_rows.append(
            {
                "id": idx,
                "gold_label": gold_label,
                "predicted_label": predicted,
                "error": err,
                "uncertainty": u,
                "confidence": c,
                "clarity_score": ex_payload["clarity"]["avg_clarity_score"],
                "interpretability_score": ex_payload["clarity"]["avg_interpretability_score"],
                **claim_stats,
            }
        )
        print(f"[{idx}/{len(pairs)}] done: gold={gold_label} pred={predicted} uncertainty={u:.3f}")

    metrics = {
        "run_id": run_id,
        "model": model,
        "dataset_path": dataset_path,
        "variant": variant,
        "max_examples": max_examples,
        "n_samples": n_samples,
        "nli_pairs": nli_pairs,
        "sample_temperature": sample_temperature,
        "prob_temperature": prob_temperature,
        "num_examples": len(summary_rows),
        "error_detection_auroc": auroc_binary(all_uncertainty, all_error),
        "nei_detection_auroc": auroc_binary(all_uncertainty, all_nei),
        "spearman_uncertainty_error": spearman_correlation(all_uncertainty, [float(e) for e in all_error]),
        "ece_confidence": expected_calibration_error(all_conf, all_correct, bins=10),
        "avg_clarity_score": sum(clarity_vals) / len(clarity_vals) if clarity_vals else 0.0,
        "avg_interpretability_score": sum(interp_vals) / len(interp_vals) if interp_vals else 0.0,
        "avg_num_claims": sum(claim_count_vals) / len(claim_count_vals) if claim_count_vals else 0.0,
        "avg_mean_claim_uncertainty": _mean(mean_claim_uncertainty_vals),
        "avg_max_claim_uncertainty": _mean(max_claim_uncertainty_vals),
        "avg_min_claim_certainty": _mean(min_claim_certainty_vals),
        "total_claim_supports": claim_support_total,
        "total_claim_contradictions": claim_contradiction_total,
        "total_claim_not_addressed": claim_not_addressed_total,
    }

    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            w.writeheader()
            w.writerows(summary_rows)
    _save_json(os.path.join(run_dir, "metrics.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified benchmark for sampling variance, self-disagreement, and NLI consistency.")
    parser.add_argument("--dataset-path", required=True, help="Path to JSONL file with claim + label")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--model", default="FAST.gpt-oss:120b")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--prob-temperature", type=float, default=0.0)
    parser.add_argument("--nli-pairs", type=int, default=4)
    parser.add_argument("--variant", default="numeric", choices=["none", "brief", "numeric", "calibrated"])
    parser.add_argument("--dry-run", action="store_true", help="Run quick smoke benchmark without API calls")
    args = parser.parse_args()

    metrics = run_benchmark(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model=args.model,
        max_examples=args.max_examples,
        n_samples=args.n_samples,
        sample_temperature=args.sample_temperature,
        prob_temperature=args.prob_temperature,
        nli_pairs=args.nli_pairs,
        variant=args.variant,
        dry_run=args.dry_run,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

