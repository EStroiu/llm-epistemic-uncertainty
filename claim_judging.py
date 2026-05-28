from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Set


SUPPORTS = "SUPPORTS"
CONTRADICTS = "CONTRADICTS"
NOT_ADDRESSED = "NOT_ADDRESSED"
JUDGMENT_LABELS = {SUPPORTS, CONTRADICTS, NOT_ADDRESSED}
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9]+")
NEGATION_RE = re.compile(r"\b(?:not|never|no|false|incorrect|refuted|cannot|can't|isn't|aren't|wasn't|weren't)\b", re.IGNORECASE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


@dataclass
class ClaimJudgment:
    claim: str
    answer_text: str
    judgment: str
    confidence: float
    reason: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _content_tokens(text: str) -> Set[str]:
    return {t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1}


def parse_judgment_label(text: str) -> str:
    up = (text or "").upper()
    if "CONTRADICT" in up or "REFUTE" in up:
        return CONTRADICTS
    if "NOT_ADDRESSED" in up or "NOT ADDRESSED" in up or "NEUTRAL" in up or "IRRELEVANT" in up:
        return NOT_ADDRESSED
    if "SUPPORT" in up or "ENTAIL" in up:
        return SUPPORTS
    return NOT_ADDRESSED


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = JSON_BLOCK_RE.search(text or "")
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def parse_judge_response(text: str) -> ClaimJudgment:
    obj = _extract_json_object(text)
    if obj:
        label = parse_judgment_label(str(obj.get("judgment", "")))
        try:
            confidence = float(obj.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        reason = str(obj.get("reason", "") or "")
        return ClaimJudgment(
            claim="",
            answer_text="",
            judgment=label,
            confidence=_clip01(confidence),
            reason=reason,
            source="llm_judge",
        )

    return ClaimJudgment(
        claim="",
        answer_text="",
        judgment=parse_judgment_label(text),
        confidence=0.5,
        reason="parsed_from_free_text",
        source="llm_judge",
    )


def mock_judge_claim_against_answer(claim: str, answer_text: str) -> ClaimJudgment:
    claim_tokens = _content_tokens(claim)
    answer_tokens = _content_tokens(answer_text)
    if not claim_tokens or not answer_tokens:
        judgment = NOT_ADDRESSED
        overlap = 0.0
    else:
        overlap = len(claim_tokens & answer_tokens) / len(claim_tokens)
        claim_negated = bool(NEGATION_RE.search(claim or ""))
        answer_negated = bool(NEGATION_RE.search(answer_text or ""))
        if overlap >= 0.55 and claim_negated != answer_negated:
            judgment = CONTRADICTS
        elif overlap >= 0.45:
            judgment = SUPPORTS
        else:
            judgment = NOT_ADDRESSED

    confidence = 0.5 + 0.5 * overlap if judgment != NOT_ADDRESSED else max(0.5, 1.0 - overlap)
    return ClaimJudgment(
        claim=claim,
        answer_text=answer_text,
        judgment=judgment,
        confidence=_clip01(confidence),
        reason=f"mock_token_overlap={overlap:.3f}",
        source="mock_judge",
    )


def judge_claim_against_answer(
    *,
    client: Any,
    model: str,
    claim: str,
    answer_text: str,
    dry_run: bool = False,
) -> ClaimJudgment:
    if dry_run:
        return mock_judge_claim_against_answer(claim, answer_text)

    prompt = (
        "Judge whether the answer supports, contradicts, or does not address the claim.\n"
        "Return ONLY valid JSON with this exact shape:\n"
        '{"judgment":"SUPPORTS|CONTRADICTS|NOT_ADDRESSED","confidence":0.0,"reason":"short reason"}\n\n'
        f"Claim: {claim}\n\n"
        f"Answer: {answer_text}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict claim verification judge."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    text = response.choices[0].message.content or ""
    parsed = parse_judge_response(text)
    parsed.claim = claim
    parsed.answer_text = answer_text
    return parsed
