from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

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
        "Compare two answers to the same claim.\n"
        "Return one word only: entailment, contradiction, or neutral.\n\n"
        f"Claim: {query}\n"
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
) -> Dict[str, Any]:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if nli_pairs < 0:
        raise ValueError("nli_pairs must be >= 0")
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    client = None if dry_run else get_nebula_client()

    answer_samples: List[str] = []
    label_samples: List[str] = []
    prob_samples: List[Dict[str, float]] = []
    clarity_scores: List[float] = []
    interp_scores: List[float] = []

    for s_idx in range(n_samples):
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
        clarity_scores.append(ci["clarity_score"])
        interp_scores.append(ci["interpretability_score"])

    rels: List[str] = []
    selected = select_random_pairs(answer_samples, max_pairs=nli_pairs, seed=123)
    for i, j in selected:
        if dry_run:
            rels.append(_mock_nli_relation(i, j))
        else:
            rels.append(_ask_nli_relation(client, model, query, answer_samples[i], answer_samples[j]))

    est_sampling = estimate_sampling_variance(prob_samples)
    est_self = estimate_self_disagreement(answer_samples, label_samples)
    est_nli = estimate_nli_consistency(rels)
    estimators: List[EstimatorOutput] = [est_sampling, est_self, est_nli]

    predicted = majority_vote_label(label_samples)
    answer_text = answer_samples[0] if answer_samples else ""
    payload = build_uncertainty_payload(
        content=answer_text,
        estimators=estimators,
        prompt_variant=variant,
        expression=variant,
        metadata={
            "query": query,
            "predicted_label": predicted,
            "n_samples": n_samples,
            "nli_pairs": nli_pairs,
            "sample_temperature": sample_temperature,
            "prob_temperature": prob_temperature,
            "model": model,
            "source": "uncertainty_runtime",
            "dry_run": dry_run,
        },
    )

    payload["metadata"]["clarity_score"] = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
    payload["metadata"]["interpretability_score"] = (
        sum(interp_scores) / len(interp_scores) if interp_scores else 0.0
    )
    payload["metadata"]["nli_relations"] = rels
    payload["metadata"]["label_samples"] = label_samples
    return payload

