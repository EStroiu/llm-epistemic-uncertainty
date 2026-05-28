from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

from claim_extraction import extract_atomic_claims_with_llm
from claim_judging import judge_claim_against_answer
from claim_uncertainty import build_claim_consistency_estimator, summarize_claim_uncertainties
from nebula_prompting import get_nebula_client
from uncertainty_estimators import (
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

MODE_CLAIM_VERIFICATION = "claim_verification"
MODE_FREEFORM = "freeform"
VALID_MODES = {MODE_CLAIM_VERIFICATION, MODE_FREEFORM}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_payload(payload: Dict[str, Any], output_dir: str = "outputs", prefix: str = "freeform") -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe_prefix = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in prefix).strip("_")
    if not safe_prefix:
        safe_prefix = "uncertainty"
    path = os.path.join(output_dir, f"{safe_prefix}_{_timestamp()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _get_freeform_variant_instruction(variant: str) -> str:
    if variant == "brief":
        return "Answer concisely. If uncertain, say so briefly."
    if variant == "numeric":
        return "Answer concisely. If useful, include a brief confidence statement at the end."
    if variant == "calibrated":
        return (
            "Answer concisely. If useful, include a brief confidence statement and one short reason "
            "for any uncertainty at the end."
        )
    return "Answer concisely."


def _ask_for_answer(client: Any, model: str, query: str, temperature: float, variant: str) -> str:
    instruction = get_variant_instruction(variant)
    prompt = (
        f"{instruction}\n\n"
        "Task: verify this claim and output one label from "
        "SUPPORTED / REFUTED / NOT_ENOUGH_INFO.\n"
        f"Claim: {query}\n"
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


def _ask_freeform_answer(client: Any, model: str, query: str, temperature: float, variant: str) -> str:
    instruction = _get_freeform_variant_instruction(variant)
    prompt = (
        f"{instruction}\n\n"
        "Answer the user prompt directly. Prefer a concise answer with concrete factual claims.\n\n"
        f"User prompt: {query}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful, concise assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=260,
    )
    return response.choices[0].message.content or ""


def _ask_freeform_answer_with_logprobs(
    client: Any,
    model: str,
    query: str,
    temperature: float,
    variant: str,
    top_logprobs: int,
) -> tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        from uncertainty_logit_gap import analyze_response_for_logit_gap_claims
    except Exception as exc:
        return _ask_freeform_answer(client, model, query, temperature, variant), None, str(exc)

    instruction = _get_freeform_variant_instruction(variant)
    prompt = (
        f"{instruction}\n\n"
        "Answer the user prompt directly. Prefer a concise answer with concrete factual claims.\n\n"
        f"User prompt: {query}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful, concise assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=260,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
    except Exception as exc:
        return _ask_freeform_answer(client, model, query, temperature, variant), None, str(exc)

    answer = response.choices[0].message.content or ""
    return answer, analyze_response_for_logit_gap_claims(response, requested_top_logprobs=top_logprobs), None


def _ask_for_probs(client: Any, model: str, query: str, temperature: float) -> Dict[str, float]:
    prompt = (
        "Return class probabilities for this claim verification task.\n"
        "Output exactly these lines:\n"
        "SUPPORTED: <float 0..1>\n"
        "REFUTED: <float 0..1>\n"
        "NOT_ENOUGH_INFO: <float 0..1>\n\n"
        f"Claim: {query}"
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


def _ask_nli_relation(client: Any, model: str, query: str, answer_a: str, answer_b: str) -> str:
    prompt = (
        "Compare two answers to the same user query.\n"
        "Return one word only: entailment, contradiction, or neutral.\n\n"
        f"User query: {query}\n"
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


def _mock_answer(query: str, variant: str, sample_idx: int) -> str:
    label_cycle = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO", "SUPPORTED"]
    label = label_cycle[sample_idx % len(label_cycle)]
    base = f"{label}. Short reason for claim: {query[:80]}."
    if variant in {"numeric", "calibrated"}:
        conf = 72 - (sample_idx * 9)
        base += f" Confidence: {max(10, conf)}%."
    if variant == "calibrated":
        base += " Uncertainty reason: limited evidence."
    return base


def _mock_freeform_answer(query: str, variant: str, sample_idx: int) -> str:
    answers = [
        (
            "Marie Curie won two Nobel Prizes. She was born in Warsaw. "
            "She conducted pioneering research on radioactivity."
        ),
        (
            "Marie Curie received two Nobel Prizes. She was born in Warsaw. "
            "Her research helped establish the study of radioactivity."
        ),
        (
            "Marie Curie won Nobel Prizes in physics and chemistry. She was born in Warsaw. "
            "She researched radioactivity."
        ),
        (
            "Marie Curie won two Nobel Prizes. She was born in Warsaw. "
            "She made major discoveries related to radioactivity."
        ),
    ]
    return answers[sample_idx % len(answers)]


def _mock_probs(sample_idx: int) -> Dict[str, float]:
    vals = [
        {"SUPPORTED": 0.70, "REFUTED": 0.17, "NOT_ENOUGH_INFO": 0.13},
        {"SUPPORTED": 0.28, "REFUTED": 0.58, "NOT_ENOUGH_INFO": 0.14},
        {"SUPPORTED": 0.32, "REFUTED": 0.25, "NOT_ENOUGH_INFO": 0.43},
        {"SUPPORTED": 0.51, "REFUTED": 0.30, "NOT_ENOUGH_INFO": 0.19},
    ]
    return vals[sample_idx % len(vals)]


def _mock_nli_relation(i: int, j: int) -> str:
    if (i + j) % 3 == 0:
        return "contradiction"
    if (i + j) % 2 == 0:
        return "entailment"
    return "neutral"


def _build_answer_sample_record(
    *,
    sample_idx: int,
    answer_text: str,
    label: str,
    probabilities: Dict[str, float],
    clarity_interpretability: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "sample_index": sample_idx,
        "role": "main" if sample_idx == 0 else "alternative",
        "answer_text": answer_text,
        "label": label,
        "probabilities": probabilities,
        "clarity_score": clarity_interpretability["clarity_score"],
        "interpretability_score": clarity_interpretability["interpretability_score"],
    }


def _claim_logit_metrics_from_analysis(claim: Dict[str, Any], analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not analysis:
        return {}
    token_signals = analysis.get("token_signals", []) or []
    start = int(claim.get("start", 0) or 0)
    end = int(claim.get("end", 0) or 0)
    span_tokens = [
        t for t in token_signals
        if int(t.get("end", 0) or 0) > start and int(t.get("start", 0) or 0) < end
    ]
    gaps = [float(t["prob_gap"]) for t in span_tokens if t.get("prob_gap") is not None]
    ents = [float(t["entropy_topk"]) for t in span_tokens if t.get("entropy_topk") is not None]
    if not gaps and not ents:
        return {}

    min_gap = min(gaps) if gaps else None
    mean_gap = sum(gaps) / len(gaps) if gaps else None
    mean_ent = sum(ents) / len(ents) if ents else None

    fragility = None
    if mean_gap is not None or mean_ent is not None:
        import math

        gap_component = 1.0 - (mean_gap if mean_gap is not None else 0.5)
        ent_norm = min(1.0, max(0.0, mean_ent / math.log(5.0))) if mean_ent is not None else 0.0
        fragility = max(0.0, min(1.0, 0.65 * gap_component + 0.35 * ent_norm))

    return {
        "token_count": len(span_tokens),
        "min_prob_gap": min_gap,
        "mean_prob_gap": mean_gap,
        "mean_entropy_topk": mean_ent,
        "logit_fragility": fragility,
        "logit_uncertainty": fragility,
        "logit_confidence": (1.0 - fragility) if fragility is not None else None,
    }


def infer_uncertainty(
    *,
    query: str,
    model: str = "FAST.gpt-oss:120b",
    variant: str = "numeric",
    n_samples: int = 4,
    nli_pairs: int = 2,
    sample_temperature: float = 0.7,
    prob_temperature: float = 0.0,
    dry_run: bool = False,
    mode: str = MODE_CLAIM_VERIFICATION,
    top_logprobs: int = 5,
) -> Dict[str, Any]:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if nli_pairs < 0:
        raise ValueError("nli_pairs must be >= 0")
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(VALID_MODES))}")

    client = None if dry_run else get_nebula_client()

    answer_samples: List[str] = []
    label_samples: List[str] = []
    prob_samples: List[Dict[str, float]] = []
    answer_sample_records: List[Dict[str, Any]] = []
    clarity_scores: List[float] = []
    interp_scores: List[float] = []
    main_logit_analysis: Optional[Dict[str, Any]] = None
    logprob_error: Optional[str] = None

    for s_idx in range(n_samples):
        if mode == MODE_FREEFORM:
            if dry_run:
                ans = _mock_freeform_answer(query, variant, s_idx)
            else:
                if s_idx == 0:
                    ans, main_logit_analysis, logprob_error = _ask_freeform_answer_with_logprobs(
                        client,
                        model,
                        query,
                        sample_temperature,
                        variant,
                        top_logprobs,
                    )
                else:
                    ans = _ask_freeform_answer(client, model, query, sample_temperature, variant)
            probs = {}
            label = "UNKNOWN"
        else:
            if dry_run:
                ans = _mock_answer(query, variant, s_idx)
                probs = _mock_probs(s_idx)
            else:
                ans = _ask_for_answer(client, model, query, sample_temperature, variant)
                probs = _ask_for_probs(client, model, query, prob_temperature)
            label = parse_label(ans)

        ci = evaluate_clarity_interpretability(ans)
        answer_samples.append(ans)
        label_samples.append(label)
        prob_samples.append(probs)
        answer_sample_records.append(
            _build_answer_sample_record(
                sample_idx=s_idx,
                answer_text=ans,
                label=label,
                probabilities=probs,
                clarity_interpretability=ci,
            )
        )
        clarity_scores.append(ci["clarity_score"])
        interp_scores.append(ci["interpretability_score"])

    rels: List[str] = []
    selected = select_random_pairs(answer_samples, max_pairs=nli_pairs, seed=123)
    for i, j in selected:
        if dry_run:
            rels.append(_mock_nli_relation(i, j))
        else:
            rels.append(_ask_nli_relation(client, model, query, answer_samples[i], answer_samples[j]))

    predicted = majority_vote_label(label_samples) if mode == MODE_CLAIM_VERIFICATION else "UNKNOWN"
    answer_text = answer_samples[0] if answer_samples else ""
    main_answer = answer_sample_records[0] if answer_sample_records else None
    alternative_answers = answer_sample_records[1:]

    extracted_claims = extract_atomic_claims_with_llm(
        client=client,
        model=model,
        answer_text=answer_text,
        dry_run=dry_run,
    )
    atomic_claims = [claim.to_dict() for claim in extracted_claims]
    if main_logit_analysis is not None:
        for claim in atomic_claims:
            claim.update(_claim_logit_metrics_from_analysis(claim, main_logit_analysis))
    claim_judgments: List[Dict[str, Any]] = []
    for claim in atomic_claims:
        for alt in alternative_answers:
            judgment = judge_claim_against_answer(
                client=client,
                model=model,
                claim=claim["text"],
                answer_text=alt["answer_text"],
                dry_run=dry_run,
            )
            row = judgment.to_dict()
            row["claim_id"] = claim["claim_id"]
            row["sample_index"] = alt["sample_index"]
            claim_judgments.append(row)

    claim_uncertainties = summarize_claim_uncertainties(atomic_claims, claim_judgments)

    est_self = estimate_self_disagreement(answer_samples, label_samples)
    est_nli = estimate_nli_consistency(rels)
    est_claims = build_claim_consistency_estimator(claim_uncertainties)
    estimators: List[EstimatorOutput] = [est_self, est_nli, est_claims]
    if mode == MODE_CLAIM_VERIFICATION:
        estimators.insert(0, estimate_sampling_variance(prob_samples))

    payload = build_uncertainty_payload(
        content=answer_text,
        estimators=estimators,
        claims=claim_uncertainties,
        prompt_variant=variant,
        expression=variant,
        metadata={
            "query": query,
            "predicted_label": predicted,
            "n_samples": n_samples,
            "nli_pairs": nli_pairs,
            "sample_temperature": sample_temperature,
            "prob_temperature": prob_temperature,
            "top_logprobs": top_logprobs,
            "model": model,
            "source": "uncertainty_runtime",
            "dry_run": dry_run,
            "mode": mode,
            "token_logprobs_available": bool(
                main_logit_analysis
                and main_logit_analysis.get("provider_capabilities", {}).get("token_logprobs_available")
            ),
            "logprob_error": logprob_error,
            "main_answer": main_answer,
            "alternative_answers": alternative_answers,
            "answer_samples": answer_sample_records,
            "num_alternative_answers": len(alternative_answers),
            "atomic_claims": atomic_claims,
            "claim_judgments": claim_judgments,
            "claim_uncertainties": claim_uncertainties,
            "logit_analysis": main_logit_analysis,
        },
    )

    payload["metadata"]["clarity_score"] = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
    payload["metadata"]["interpretability_score"] = (
        sum(interp_scores) / len(interp_scores) if interp_scores else 0.0
    )
    payload["metadata"]["nli_relations"] = rels
    payload["metadata"]["label_samples"] = label_samples
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run uncertainty inference for a prompt and print claim-level scores.")
    parser.add_argument("--query", required=True, help="User prompt or claim to send to the model")
    parser.add_argument("--model", default="FAST.gpt-oss:120b")
    parser.add_argument("--mode", default=MODE_FREEFORM, choices=sorted(VALID_MODES))
    parser.add_argument("--variant", default="numeric", choices=["none", "brief", "numeric", "calibrated"])
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--nli-pairs", type=int, default=2)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--prob-temperature", type=float, default=0.0)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the full JSON payload")
    parser.add_argument("--save", action="store_true", help="Save the full JSON payload to --output-dir")
    parser.add_argument("--output-dir", default="outputs", help="Directory used with --save")
    parser.add_argument("--output-prefix", default="freeform", help="Filename prefix used with --save")
    args = parser.parse_args()

    payload = infer_uncertainty(
        query=args.query,
        model=args.model,
        mode=args.mode,
        variant=args.variant,
        n_samples=args.n_samples,
        nli_pairs=args.nli_pairs,
        sample_temperature=args.sample_temperature,
        prob_temperature=args.prob_temperature,
        top_logprobs=args.top_logprobs,
        dry_run=args.dry_run,
    )
    if args.save:
        out_path = save_payload(payload, output_dir=args.output_dir, prefix=args.output_prefix)
        print(f"\nSaved result: {out_path}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("\n=== Answer ===")
    print(payload["content"])
    print("\n=== Claims ===")
    print(json.dumps(payload["claims"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

