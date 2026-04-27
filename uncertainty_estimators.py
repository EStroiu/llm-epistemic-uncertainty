import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from uncertainty_schema import EstimatorOutput


FEVER_LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
NLI_LABELS = ["entailment", "contradiction", "neutral"]


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def normalized_entropy(probs: Dict[str, float]) -> float:
    vals = [max(0.0, float(v)) for v in probs.values()]
    s = sum(vals)
    if s <= 0:
        return 1.0
    vals = [v / s for v in vals]
    h = 0.0
    for p in vals:
        if p > 0:
            h -= p * math.log(p)
    max_h = math.log(len(vals)) if vals else 1.0
    if max_h <= 0:
        return 0.0
    return _clip01(h / max_h)


def parse_label(text: str) -> str:
    up = (text or "").upper()
    if "NOT_ENOUGH_INFO" in up or "NOT ENOUGH INFO" in up:
        return "NOT_ENOUGH_INFO"
    if "SUPPORTED" in up:
        return "SUPPORTED"
    if "REFUTED" in up:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def parse_probabilities(text: str) -> Dict[str, float]:
    text = text or ""
    clean = text.replace("%", "")
    probs: Dict[str, float] = {k: 0.0 for k in FEVER_LABELS}
    for key in FEVER_LABELS:
        pat = re.compile(rf"{key}\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
        m = pat.search(clean)
        if m:
            probs[key] = float(m.group(1))
    total = sum(probs.values())
    if total <= 0:
        return {k: 1.0 / len(FEVER_LABELS) for k in FEVER_LABELS}
    if total > 1.5:
        probs = {k: v / 100.0 for k, v in probs.items()}
    total = sum(max(0.0, v) for v in probs.values())
    if total <= 0:
        return {k: 1.0 / len(FEVER_LABELS) for k in FEVER_LABELS}
    return {k: max(0.0, v) / total for k, v in probs.items()}


def token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def jaccard_similarity(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta and not tb:
        return 1.0
    den = len(ta | tb)
    if den == 0:
        return 1.0
    return len(ta & tb) / den


def estimate_sampling_variance(prob_samples: Sequence[Dict[str, float]]) -> EstimatorOutput:
    if not prob_samples:
        return EstimatorOutput(
            name="sampling_variance",
            uncertainty=0.5,
            confidence=0.5,
            details={"reason": "no_samples"},
            weight=1.0,
        )

    n = len(prob_samples)
    means: Dict[str, float] = {}
    variances: Dict[str, float] = {}

    for label in FEVER_LABELS:
        xs = [float(p.get(label, 0.0)) for p in prob_samples]
        mu = sum(xs) / n
        means[label] = mu
        variances[label] = sum((x - mu) ** 2 for x in xs) / n

    avg_var = sum(variances.values()) / len(FEVER_LABELS)
    max_var = 2.0 / 9.0
    variance_component = _clip01(avg_var / max_var) if max_var > 0 else 0.5
    entropy_component = normalized_entropy(means)
    uncertainty = _clip01(0.6 * variance_component + 0.4 * entropy_component)

    return EstimatorOutput(
        name="sampling_variance",
        uncertainty=uncertainty,
        confidence=1.0 - uncertainty,
        details={
            "n_samples": n,
            "mean_probs": means,
            "label_variances": variances,
            "avg_variance": avg_var,
            "entropy_component": entropy_component,
        },
        weight=1.0,
    )


def estimate_self_disagreement(answer_samples: Sequence[str], label_samples: Optional[Sequence[str]] = None) -> EstimatorOutput:
    answers = [a or "" for a in answer_samples]
    if len(answers) <= 1:
        return EstimatorOutput(
            name="self_disagreement",
            uncertainty=0.5,
            confidence=0.5,
            details={"reason": "not_enough_samples"},
            weight=1.0,
        )

    pair_sims: List[float] = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            pair_sims.append(jaccard_similarity(answers[i], answers[j]))
    mean_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 1.0
    lexical_disagreement = _clip01(1.0 - mean_sim)

    label_disagreement = 0.0
    if label_samples:
        counts = Counter(label_samples)
        majority = max(counts.values()) if counts else 0
        label_disagreement = 1.0 - (majority / len(label_samples)) if len(label_samples) > 0 else 0.0

    uncertainty = _clip01(0.55 * lexical_disagreement + 0.45 * label_disagreement)
    return EstimatorOutput(
        name="self_disagreement",
        uncertainty=uncertainty,
        confidence=1.0 - uncertainty,
        details={
            "n_answers": len(answers),
            "pairwise_mean_similarity": mean_sim,
            "lexical_disagreement": lexical_disagreement,
            "label_disagreement": label_disagreement,
        },
        weight=1.0,
    )


def estimate_nli_consistency(nli_relations: Sequence[str]) -> EstimatorOutput:
    cleaned = [r.lower().strip() for r in nli_relations if r]
    if not cleaned:
        return EstimatorOutput(
            name="nli_consistency",
            uncertainty=0.5,
            confidence=0.5,
            details={"reason": "no_nli_relations"},
            weight=1.0,
        )

    counts = Counter(cleaned)
    n = len(cleaned)
    contradiction_rate = counts.get("contradiction", 0) / n
    neutral_rate = counts.get("neutral", 0) / n
    uncertainty = _clip01(contradiction_rate + 0.5 * neutral_rate)

    return EstimatorOutput(
        name="nli_consistency",
        uncertainty=uncertainty,
        confidence=1.0 - uncertainty,
        details={
            "n_pairs": n,
            "label_counts": dict(counts),
            "contradiction_rate": contradiction_rate,
            "neutral_rate": neutral_rate,
        },
        weight=1.0,
    )


def majority_vote_label(labels: Sequence[str]) -> str:
    if not labels:
        return "NOT_ENOUGH_INFO"
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def select_random_pairs(items: Sequence[Any], max_pairs: int, seed: int = 42) -> List[Tuple[int, int]]:
    idx_pairs: List[Tuple[int, int]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            idx_pairs.append((i, j))
    if len(idx_pairs) <= max_pairs:
        return idx_pairs
    rnd = random.Random(seed)
    rnd.shuffle(idx_pairs)
    return idx_pairs[:max_pairs]

