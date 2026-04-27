from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0.0"


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _reliability_from_uncertainty(uncertainty: float) -> str:
    if uncertainty >= 0.75:
        return "low_reliability"
    if uncertainty >= 0.45:
        return "medium_reliability"
    return "high_reliability"


@dataclass
class EstimatorOutput:
    name: str
    uncertainty: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["uncertainty"] = _clip01(payload["uncertainty"])
        payload["confidence"] = _clip01(payload["confidence"])
        payload["weight"] = max(0.0, float(payload["weight"]))
        return payload


def combine_estimators(estimators: List[EstimatorOutput]) -> Dict[str, Any]:
    if not estimators:
        return {
            "overall_uncertainty": 0.5,
            "overall_confidence": 0.5,
            "reliability": "medium_reliability",
            "estimators": [],
        }

    prepared = [e.to_dict() for e in estimators]
    total_weight = sum(e["weight"] for e in prepared)

    if total_weight <= 0:
        total_weight = float(len(prepared))
        for e in prepared:
            e["weight"] = 1.0

    weighted_uncertainty = sum(e["uncertainty"] * e["weight"] for e in prepared) / total_weight
    weighted_confidence = sum(e["confidence"] * e["weight"] for e in prepared) / total_weight
    weighted_uncertainty = _clip01(weighted_uncertainty)
    weighted_confidence = _clip01(weighted_confidence)

    return {
        "overall_uncertainty": weighted_uncertainty,
        "overall_confidence": weighted_confidence,
        "reliability": _reliability_from_uncertainty(weighted_uncertainty),
        "estimators": prepared,
    }


def build_uncertainty_payload(
    *,
    content: str,
    estimators: List[EstimatorOutput],
    prompt_variant: Optional[str] = None,
    expression: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = combine_estimators(estimators)
    metadata = metadata or {}

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "uncertainty": {
            "overall_uncertainty": summary["overall_uncertainty"],
            "overall_confidence": summary["overall_confidence"],
            "reliability": summary["reliability"],
            "estimators": summary["estimators"],
            "prompt_variant": prompt_variant or "default",
            "uncertainty_expression": expression or "none",
        },
        "metadata": metadata,
    }

