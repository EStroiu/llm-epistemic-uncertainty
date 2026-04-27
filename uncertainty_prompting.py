import re
from typing import Dict


PROMPT_VARIANTS = {
    "none": "Answer the claim verification task. Return label and concise reason.",
    "brief": (
        "Answer the claim verification task. Return label and concise reason. "
        "If unsure, state uncertainty in one short sentence."
    ),
    "numeric": (
        "Answer the claim verification task. Return label and concise reason. "
        "Then add: Confidence: <0-100>%."
    ),
    "calibrated": (
        "Answer the claim verification task. Return label and concise reason. "
        "Then add: Confidence: <0-100>%. "
        "Also include one short uncertainty reason (for example: missing evidence, conflicting evidence)."
    ),
}


def get_variant_instruction(variant: str) -> str:
    return PROMPT_VARIANTS.get(variant, PROMPT_VARIANTS["none"])


def evaluate_clarity_interpretability(answer_text: str) -> Dict[str, float]:
    text = (answer_text or "").strip()
    lower = text.lower()
    token_count = len(re.findall(r"\w+", text))

    has_numeric_conf = bool(re.search(r"confidence\s*:\s*\d{1,3}\s*%?", lower))
    has_uncertainty_cue = any(
        cue in lower
        for cue in [
            "not sure",
            "uncertain",
            "missing evidence",
            "conflicting evidence",
            "insufficient evidence",
            "limited evidence",
        ]
    )
    has_label = any(lbl in text.upper() for lbl in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"])

    # Very lightweight proxies, but stable for quick benchmark loops.
    explicitness = 0.0
    if has_label:
        explicitness += 0.4
    if has_numeric_conf:
        explicitness += 0.4
    if has_uncertainty_cue:
        explicitness += 0.2

    if token_count == 0:
        brevity = 0.0
    elif token_count <= 70:
        brevity = 1.0
    elif token_count <= 120:
        brevity = 0.7
    else:
        brevity = 0.4

    interpretability = 0.6 * explicitness + 0.4 * brevity
    return {
        "clarity_score": round(explicitness, 4),
        "interpretability_score": round(interpretability, 4),
        "has_numeric_confidence": float(has_numeric_conf),
        "has_uncertainty_cue": float(has_uncertainty_cue),
        "token_count": float(token_count),
    }

