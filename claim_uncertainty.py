from __future__ import annotations

from typing import Any, Dict, List, Optional

from claim_judging import CONTRADICTS, NOT_ADDRESSED, SUPPORTS
from uncertainty_schema import EstimatorOutput


DEFAULT_SEMANTIC_WEIGHT = 0.65
DEFAULT_LOGIT_WEIGHT = 0.35


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _severity_from_uncertainty(uncertainty: Optional[float]) -> str:
    if uncertainty is None:
        return "unknown"
    if uncertainty >= 0.75:
        return "high"
    if uncertainty >= 0.45:
        return "medium"
    return "low"


def _uncertainty_types(
    *,
    contradiction_rate: float,
    not_addressed_rate: float,
    logit_uncertainty: Optional[float],
) -> List[str]:
    types: List[str] = []
    if contradiction_rate > 0.0:
        types.append("model_disagreement")
    if not_addressed_rate > 0.0:
        types.append("not_consistently_addressed")
    if logit_uncertainty is not None and logit_uncertainty >= 0.45:
        types.append("low_logit_margin")
    if not types:
        types.append("stable_across_samples")
    return types


def combine_semantic_and_logit_uncertainty(
    *,
    semantic_uncertainty: Optional[float],
    logit_uncertainty: Optional[float],
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    logit_weight: float = DEFAULT_LOGIT_WEIGHT,
) -> Optional[float]:
    weighted: List[tuple[float, float]] = []
    if semantic_uncertainty is not None:
        weighted.append((semantic_weight, _clip01(semantic_uncertainty)))
    if logit_uncertainty is not None:
        weighted.append((logit_weight, _clip01(logit_uncertainty)))
    if not weighted:
        return None
    total = sum(w for w, _ in weighted)
    if total <= 0:
        return None
    return _clip01(sum(w * v for w, v in weighted) / total)


def summarize_claim_uncertainties(
    atomic_claims: List[Dict[str, Any]],
    claim_judgments: List[Dict[str, Any]],
    *,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    logit_weight: float = DEFAULT_LOGIT_WEIGHT,
) -> List[Dict[str, Any]]:
    by_claim: Dict[str, List[Dict[str, Any]]] = {}
    for judgment in claim_judgments:
        claim_id = str(judgment.get("claim_id", ""))
        if claim_id:
            by_claim.setdefault(claim_id, []).append(judgment)

    summaries: List[Dict[str, Any]] = []
    for claim in atomic_claims:
        claim_id = str(claim.get("claim_id", ""))
        judgments = by_claim.get(claim_id, [])
        total = len(judgments)
        support_count = sum(1 for j in judgments if j.get("judgment") == SUPPORTS)
        contradiction_count = sum(1 for j in judgments if j.get("judgment") == CONTRADICTS)
        not_addressed_count = sum(1 for j in judgments if j.get("judgment") == NOT_ADDRESSED)

        if total > 0:
            support_rate = support_count / total
            contradiction_rate = contradiction_count / total
            not_addressed_rate = not_addressed_count / total
            semantic_uncertainty: Optional[float] = _clip01(contradiction_rate + 0.5 * not_addressed_rate)
        else:
            support_rate = 0.0
            contradiction_rate = 0.0
            not_addressed_rate = 0.0
            semantic_uncertainty = None

        raw_logit_uncertainty = claim.get("logit_uncertainty")
        logit_uncertainty = float(raw_logit_uncertainty) if raw_logit_uncertainty is not None else None
        final_uncertainty = combine_semantic_and_logit_uncertainty(
            semantic_uncertainty=semantic_uncertainty,
            logit_uncertainty=logit_uncertainty,
            semantic_weight=semantic_weight,
            logit_weight=logit_weight,
        )
        final_certainty = (1.0 - final_uncertainty) if final_uncertainty is not None else None

        summaries.append(
            {
                **claim,
                "support_count": support_count,
                "contradiction_count": contradiction_count,
                "not_addressed_count": not_addressed_count,
                "judgment_count": total,
                "support_rate": support_rate,
                "contradiction_rate": contradiction_rate,
                "not_addressed_rate": not_addressed_rate,
                "semantic_uncertainty": semantic_uncertainty,
                "semantic_certainty": (1.0 - semantic_uncertainty) if semantic_uncertainty is not None else None,
                "final_uncertainty": final_uncertainty,
                "final_certainty": final_certainty,
                "certainty": final_certainty,
                "uncertainty": final_uncertainty,
                "severity": _severity_from_uncertainty(final_uncertainty),
                "uncertainty_type": _uncertainty_types(
                    contradiction_rate=contradiction_rate,
                    not_addressed_rate=not_addressed_rate,
                    logit_uncertainty=logit_uncertainty,
                ),
            }
        )

    return summaries


def build_claim_consistency_estimator(claim_summaries: List[Dict[str, Any]]) -> EstimatorOutput:
    vals = [c.get("final_uncertainty") for c in claim_summaries if c.get("final_uncertainty") is not None]
    if not vals:
        uncertainty = 0.5
        confidence = 0.5
    else:
        uncertainties = [float(v) for v in vals]
        mean_uncertainty = sum(uncertainties) / len(uncertainties)
        max_uncertainty = max(uncertainties)
        uncertainty = _clip01(0.7 * mean_uncertainty + 0.3 * max_uncertainty)
        confidence = 1.0 - uncertainty

    high = sum(1 for c in claim_summaries if c.get("severity") == "high")
    medium = sum(1 for c in claim_summaries if c.get("severity") == "medium")
    low = sum(1 for c in claim_summaries if c.get("severity") == "low")

    return EstimatorOutput(
        name="claim_semantic_consistency",
        uncertainty=uncertainty,
        confidence=confidence,
        details={
            "num_claims": len(claim_summaries),
            "num_high_claims": high,
            "num_medium_claims": medium,
            "num_low_claims": low,
            "semantic_weight": DEFAULT_SEMANTIC_WEIGHT,
            "logit_weight": DEFAULT_LOGIT_WEIGHT,
        },
        weight=1.25,
    )
