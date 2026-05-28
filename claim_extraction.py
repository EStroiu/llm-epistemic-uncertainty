from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)
WORD_RE = re.compile(r"\b[\w'-]+\b")


@dataclass
class AtomicClaim:
    claim_id: str
    text: str
    source_text: str
    start: int
    end: int
    sentence_start: int
    sentence_end: int
    extraction_method: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _clean_claim_text(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    cleaned = cleaned.lstrip("-*0123456789. ")
    cleaned = cleaned.strip(" ,;:")
    cleaned = re.sub(r"([.!?])[\"'”’]+$", r"\1", cleaned)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _extract_json_payload(text: str) -> Any:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = JSON_BLOCK_RE.search(text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _claim_texts_from_payload(payload: Any) -> List[str]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("claims", "atomic_claims"):
            if key in payload:
                return _claim_texts_from_payload(payload[key])
        if "text" in payload:
            return [str(payload["text"])]
        if "claim" in payload:
            return [str(payload["claim"])]
        return []
    if isinstance(payload, list):
        out: List[str] = []
        for item in payload:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                val = item.get("text", item.get("claim", ""))
                if val:
                    out.append(str(val))
        return out
    return []


def _locate_claim_span(answer_text: str, claim_text: str, search_start: int = 0) -> Tuple[int, int]:
    answer = answer_text or ""
    claim = claim_text.strip()
    if not answer or not claim:
        return 0, 0

    direct = answer.find(claim, search_start)
    if direct < 0:
        direct = answer.find(claim)
    if direct >= 0:
        return direct, direct + len(claim)

    words = WORD_RE.findall(claim)
    if len(words) >= 4:
        anchor = " ".join(words[:4])
        match = re.search(re.escape(anchor).replace(r"\ ", r"\s+"), answer, re.IGNORECASE)
        if match:
            return match.start(), min(len(answer), match.start() + len(claim))

    return 0, min(len(answer), len(claim))


def claims_from_texts(answer_text: str, claim_texts: List[str], *, method: str, max_claims: int = 20) -> List[AtomicClaim]:
    claims: List[AtomicClaim] = []
    cursor = 0
    seen = set()
    for raw in claim_texts:
        claim_text = _clean_claim_text(raw)
        key = claim_text.lower()
        if not claim_text or _word_count(claim_text) < 2 or key in seen:
            continue
        seen.add(key)
        start, end = _locate_claim_span(answer_text, claim_text, cursor)
        cursor = max(cursor, end)
        claims.append(
            AtomicClaim(
                claim_id=f"c{len(claims) + 1}",
                text=claim_text,
                source_text=(answer_text or "")[start:end].strip(),
                start=start,
                end=end,
                sentence_start=start,
                sentence_end=end,
                extraction_method=method,
            )
        )
        if len(claims) >= max_claims:
            break
    return claims


def parse_claim_extraction_response(answer_text: str, response_text: str, *, max_claims: int = 20) -> List[AtomicClaim]:
    payload = _extract_json_payload(response_text)
    return claims_from_texts(
        answer_text,
        _claim_texts_from_payload(payload),
        method="llm_claim_extraction",
        max_claims=max_claims,
    )


def mock_extract_atomic_claims(answer_text: str, *, max_claims: int = 20) -> List[AtomicClaim]:
    text = answer_text or ""
    if "Marie Curie" in text:
        claim_texts = [
            "Marie Curie won two Nobel Prizes.",
            "She was born in Warsaw.",
            "She conducted pioneering research on radioactivity.",
        ]
    else:
        claim_texts = [text]
    return claims_from_texts(answer_text, claim_texts, method="mock_llm_claim_extraction", max_claims=max_claims)


def extract_atomic_claims_with_llm(
    *,
    client: Any,
    model: str,
    answer_text: str,
    max_claims: int = 20,
    dry_run: bool = False,
) -> List[AtomicClaim]:
    if dry_run:
        return mock_extract_atomic_claims(answer_text, max_claims=max_claims)

    prompt = (
        "Extract atomic factual claims from the answer below.\n"
        "Rules:\n"
        "- Use semantic atomicity, not punctuation. Do not split just because there is a period.\n"
        "- Ignore confidence statements, hedges, formatting, labels, and explanations of uncertainty.\n"
        "- Include only factual claims made by the answer.\n"
        "- Preserve enough context that pronouns are resolved.\n"
        "- Return ONLY JSON in this exact shape: {\"claims\":[{\"text\":\"...\"}]}\n\n"
        f"Answer:\n{answer_text}"
    )
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You extract atomic factual claims as JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 360,
    }
    try:
        response = client.chat.completions.create(**params, response_format={"type": "json_object"})
    except Exception:
        response = client.chat.completions.create(**params)
    response_text = response.choices[0].message.content or ""
    claims = parse_claim_extraction_response(answer_text, response_text, max_claims=max_claims)
    if claims:
        return claims

    retry_prompt = (
        "Return a JSON array of strings. Each string must be one factual claim made in the answer. "
        "Do not include confidence statements. Do not split claims only because of punctuation.\n\n"
        f"Answer:\n{answer_text}"
    )
    retry = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You extract factual claims. Return JSON only."},
            {"role": "user", "content": retry_prompt},
        ],
        temperature=0.0,
        max_tokens=360,
    )
    retry_text = retry.choices[0].message.content or ""
    return parse_claim_extraction_response(answer_text, retry_text, max_claims=max_claims)


def extract_atomic_claims(text: str, *, max_claims: int = 20) -> List[AtomicClaim]:
    """Compatibility wrapper for tests/dry-run.

    Live runtime extraction should call extract_atomic_claims_with_llm so claim
    boundaries are decided semantically by an LLM, not by punctuation.
    """
    return mock_extract_atomic_claims(text, max_claims=max_claims)
