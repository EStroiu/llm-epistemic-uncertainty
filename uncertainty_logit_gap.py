import argparse
import csv
from datetime import datetime
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # allows running tests without optional deps installed globally
    def load_dotenv() -> bool:
        return False

from uncertainty_schema import EstimatorOutput, build_uncertainty_payload


NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:19|20)\d{2})\b")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.MULTILINE)
NEBULA_BASE_URL = "https://nebula.cs.vu.nl/api/"


@dataclass
class TokenSignal:
    token: str
    start: int
    end: int
    top1_logprob: float
    top2_logprob: Optional[float]
    prob_gap: Optional[float]
    logprob_gap: Optional[float]
    entropy_topk: Optional[float]


@dataclass
class ClaimSpan:
    text: str
    start: int
    end: int
    reason: List[str]
    min_prob_gap: Optional[float]
    mean_prob_gap: Optional[float]
    mean_entropy_topk: Optional[float]
    fragility_score: Optional[float]
    severity: str


@dataclass
class Highlight:
    text: str
    start: int
    end: int
    severity: str
    reason: str


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_topk_probs(logprobs: List[float]) -> List[float]:
    if not logprobs:
        return []
    max_lp = max(logprobs)
    exps = [math.exp(lp - max_lp) for lp in logprobs]
    s = sum(exps)
    if s <= 0:
        return [0.0 for _ in exps]
    return [v / s for v in exps]


def _extract_message_text(response: Any) -> str:
    choices = _obj_get(response, "choices", [])
    if not choices:
        return ""
    message = _obj_get(choices[0], "message", None)
    return _obj_get(message, "content", "") or ""


def _extract_token_logprobs(response: Any) -> List[Any]:
    choices = _obj_get(response, "choices", [])
    if not choices:
        return []

    first_choice = choices[0]
    # OpenAI-compatible chat format usually uses: choice.logprobs.content
    choice_logprobs = _obj_get(first_choice, "logprobs", None)
    if choice_logprobs is None:
        return []

    content = _obj_get(choice_logprobs, "content", None)
    return content or []


def _extract_usage(response: Any) -> Any:
    usage = _obj_get(response, "usage", None)
    if usage is None:
        return None
    return usage.model_dump() if hasattr(usage, "model_dump") else usage


def _compute_token_signals(token_infos: List[Any]) -> List[TokenSignal]:
    signals: List[TokenSignal] = []
    cursor = 0

    for tok in token_infos:
        token = _obj_get(tok, "token", "") or ""
        top1_logprob = float(_obj_get(tok, "logprob", float("nan")))

        top_entries = _obj_get(tok, "top_logprobs", None) or []
        top_values: List[Tuple[str, float]] = []

        # Include top1 candidate itself
        if not math.isnan(top1_logprob):
            top_values.append((token, top1_logprob))

        for entry in top_entries:
            t = _obj_get(entry, "token", "") or ""
            lp = _obj_get(entry, "logprob", None)
            if lp is None:
                continue
            lp = float(lp)
            if not any(abs(lp - v[1]) < 1e-12 and t == v[0] for v in top_values):
                top_values.append((t, lp))

        top_values.sort(key=lambda x: x[1], reverse=True)

        top2_logprob: Optional[float] = None
        logprob_gap: Optional[float] = None
        prob_gap: Optional[float] = None
        entropy_topk: Optional[float] = None

        if len(top_values) >= 2:
            top2_logprob = top_values[1][1]
            logprob_gap = top_values[0][1] - top_values[1][1]

        if top_values:
            probs = _normalize_topk_probs([lp for _, lp in top_values])
            if len(probs) >= 2:
                prob_gap = probs[0] - probs[1]
            if probs:
                entropy_topk = -sum(p * math.log(p + 1e-12) for p in probs)

        start = cursor
        end = cursor + len(token)
        cursor = end

        signals.append(
            TokenSignal(
                token=token,
                start=start,
                end=end,
                top1_logprob=top1_logprob,
                top2_logprob=top2_logprob,
                prob_gap=prob_gap,
                logprob_gap=logprob_gap,
                entropy_topk=entropy_topk,
            )
        )

    return signals


def _extract_claim_spans(text: str) -> List[Tuple[int, int, str, List[str]]]:
    spans: List[Tuple[int, int, str, List[str]]] = []

    for m in SENTENCE_RE.finditer(text):
        sent = m.group(0)
        sent_clean = sent.strip()
        if not sent_clean:
            continue

        reasons: List[str] = []
        if NUMBER_RE.search(sent_clean):
            reasons.append("contains_number")
        if DATE_RE.search(sent_clean):
            reasons.append("contains_date")
        if ENTITY_RE.search(sent_clean):
            reasons.append("entity_like_phrase")

        if reasons:
            # Trim leading/trailing whitespace in sentence span while preserving offsets
            leading_ws = len(sent) - len(sent.lstrip())
            trailing_ws = len(sent) - len(sent.rstrip())
            start = m.start() + leading_ws
            end = m.end() - trailing_ws
            spans.append((start, end, text[start:end], reasons))

    return spans


def _tokens_in_span(tokens: List[TokenSignal], start: int, end: int) -> List[TokenSignal]:
    return [t for t in tokens if t.end > start and t.start < end]


def _severity_from_fragility(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _aggregate_claim_scores(claims: List[Tuple[int, int, str, List[str]]], tokens: List[TokenSignal]) -> List[ClaimSpan]:
    result: List[ClaimSpan] = []

    for start, end, claim_text, reasons in claims:
        claim_tokens = _tokens_in_span(tokens, start, end)

        gaps = [t.prob_gap for t in claim_tokens if t.prob_gap is not None]
        ents = [t.entropy_topk for t in claim_tokens if t.entropy_topk is not None]

        min_gap = min(gaps) if gaps else None
        mean_gap = sum(gaps) / len(gaps) if gaps else None
        mean_ent = sum(ents) / len(ents) if ents else None

        # Simple fragility score in [0, 1] (approx):
        # - lower gap => more fragile
        # - higher top-k entropy => more fragile (normalized by ln(k), approximated with ln(5))
        fragility: Optional[float] = None
        if mean_gap is not None or mean_ent is not None:
            gap_component = 1.0 - (mean_gap if mean_gap is not None else 0.5)
            ent_norm = 0.0
            if mean_ent is not None:
                ent_norm = min(1.0, max(0.0, mean_ent / math.log(5.0)))
            fragility = max(0.0, min(1.0, 0.65 * gap_component + 0.35 * ent_norm))

        severity = _severity_from_fragility(fragility)
        # Escalate if very low minimum gap appears inside the claim
        if min_gap is not None and min_gap < 0.1:
            severity = "high"
        elif min_gap is not None and min_gap < 0.2 and severity == "low":
            severity = "medium"

        result.append(
            ClaimSpan(
                text=claim_text,
                start=start,
                end=end,
                reason=reasons,
                min_prob_gap=min_gap,
                mean_prob_gap=mean_gap,
                mean_entropy_topk=mean_ent,
                fragility_score=fragility,
                severity=severity,
            )
        )

    return result


def _make_fragile_highlights(tokens: List[TokenSignal], claims: List[ClaimSpan]) -> List[Highlight]:
    highlights: List[Highlight] = []

    for claim in claims:
        if claim.severity not in {"medium", "high"}:
            continue

        claim_tokens = _tokens_in_span(tokens, claim.start, claim.end)
        for t in claim_tokens:
            if t.prob_gap is None:
                continue
            if t.prob_gap < 0.2:
                tok_text = t.token
                if not tok_text.strip():
                    continue
                highlights.append(
                    Highlight(
                        text=tok_text,
                        start=t.start,
                        end=t.end,
                        severity="high" if t.prob_gap < 0.1 else "medium",
                        reason="low_token_gap",
                    )
                )

    return highlights


def analyze_response_for_logit_gap_claims(response: Any, *, requested_top_logprobs: int = 5) -> Dict[str, Any]:
    """Analyze an OpenAI-compatible chat response object/dict for token- and claim-level fragility."""
    text = _extract_message_text(response)
    token_infos = _extract_token_logprobs(response)

    has_token_scores = len(token_infos) > 0
    token_signals = _compute_token_signals(token_infos) if has_token_scores else []

    claim_spans_raw = _extract_claim_spans(text)
    claim_spans = _aggregate_claim_scores(claim_spans_raw, token_signals) if has_token_scores else []
    highlights = _make_fragile_highlights(token_signals, claim_spans) if has_token_scores else []

    # If no token-level scores are available, still return claim candidates for fallback pathways.
    fallback_claim_candidates = []
    if not has_token_scores:
        for start, end, ctext, reasons in claim_spans_raw:
            fallback_claim_candidates.append(
                {
                    "text": ctext,
                    "start": start,
                    "end": end,
                    "reason": reasons,
                }
            )

    mean_claim_fragility: Optional[float] = None
    if claim_spans:
        fragility_values = [c.fragility_score for c in claim_spans if c.fragility_score is not None]
        if fragility_values:
            mean_claim_fragility = sum(fragility_values) / len(fragility_values)

    fallback_uncertainty: Optional[float] = None
    if token_signals and mean_claim_fragility is None:
        token_gaps = [t.prob_gap for t in token_signals if t.prob_gap is not None]
        if token_gaps:
            fallback_uncertainty = 1.0 - max(0.0, min(1.0, sum(token_gaps) / len(token_gaps)))

    overall_uncertainty = mean_claim_fragility
    if overall_uncertainty is None:
        overall_uncertainty = fallback_uncertainty
    if overall_uncertainty is None:
        overall_uncertainty = 0.5

    estimator = EstimatorOutput(
        name="logit_gap_fragility",
        uncertainty=overall_uncertainty,
        confidence=1.0 - overall_uncertainty,
        details={
            "token_logprobs_available": has_token_scores,
            "num_claim_spans": len(claim_spans),
            "num_highlights": len(highlights),
            "mean_claim_fragility": mean_claim_fragility,
            "fallback_uncertainty": fallback_uncertainty,
        },
        weight=1.0,
    )

    uncertainty_payload = build_uncertainty_payload(
        content=text,
        estimators=[estimator],
        prompt_variant="default",
        expression="metadata_only",
        metadata={
            "source": "uncertainty_logit_gap.py",
            "requested_top_logprobs": requested_top_logprobs,
        },
    )

    return {
        "answer_text": text,
        "provider_capabilities": {
            "token_logprobs_available": has_token_scores,
            "requested_top_logprobs": requested_top_logprobs,
            "signal_source": "token_logprobs" if has_token_scores else "none",
            "fallback_recommended": not has_token_scores,
        },
        "usage": _extract_usage(response),
        "token_signals": [asdict(t) for t in token_signals],
        "claim_spans": [asdict(c) for c in claim_spans],
        "highlights": [asdict(h) for h in highlights],
        "fallback_claim_candidates": fallback_claim_candidates,
        "uncertainty_payload": uncertainty_payload,
    }


def run_logit_gap_claim_detection_live(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 220,
    temperature: float = 0.2,
    top_logprobs: int = 5,
) -> Dict[str, Any]:
    """
    Optional live execution using any OpenAI-compatible endpoint.
    This function is independent from project-specific wrappers.
    """
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for live mode") from exc

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
        stream=False,
    )
    return analyze_response_for_logit_gap_claims(response, requested_top_logprobs=top_logprobs)


def _pretty_print_result(result: Dict[str, Any]) -> None:
    print("\n=== Answer ===")
    print(result["answer_text"])

    caps = result["provider_capabilities"]
    print("\n=== Capabilities ===")
    print(caps)

    if caps["token_logprobs_available"]:
        print("\n=== Claim Fragility ===")
        for i, c in enumerate(result["claim_spans"], start=1):
            score = c["fragility_score"]
            score_txt = "None" if score is None else f"{score:.3f}"
            print(f"{i}. [{c['severity']}] score={score_txt} :: {c['text']}")
        print("\n=== Highlights (fragile tokens) ===")
        for h in result["highlights"][:25]:
            print(f"- {h['text']!r} ({h['start']},{h['end']}) {h['severity']}")
    else:
        print("\nToken-level scores were not returned by this model/provider.")
        print("Claim candidates (for fallback analysis):")
        for c in result["fallback_claim_candidates"]:
            print(f"- {c['text']}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_prompts_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    # JSON list support
    if path.endswith(".json"):
        obj = json.loads(raw)
        if not isinstance(obj, list):
            raise ValueError("JSON prompts file must be a list of strings")
        prompts = [str(x).strip() for x in obj if str(x).strip()]
        return prompts

    # Plain text: one prompt per non-empty line
    prompts = [line.strip() for line in raw.splitlines() if line.strip()]
    return prompts


def _parse_temperatures(value: str) -> List[float]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        return [0.2]
    return [float(x) for x in items]


def _result_summary_row(
    *,
    run_id: str,
    prompt_id: int,
    model: str,
    temperature: float,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    claim_spans = result.get("claim_spans", [])
    highs = sum(1 for c in claim_spans if c.get("severity") == "high")
    mediums = sum(1 for c in claim_spans if c.get("severity") == "medium")
    lows = sum(1 for c in claim_spans if c.get("severity") == "low")

    fragility_values = [c.get("fragility_score") for c in claim_spans if c.get("fragility_score") is not None]
    mean_fragility = (sum(fragility_values) / len(fragility_values)) if fragility_values else None

    return {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "model": model,
        "temperature": temperature,
        "token_logprobs_available": result.get("provider_capabilities", {}).get("token_logprobs_available", False),
        "answer_len_chars": len(result.get("answer_text", "")),
        "num_claim_spans": len(claim_spans),
        "num_high_claims": highs,
        "num_medium_claims": mediums,
        "num_low_claims": lows,
        "num_highlights": len(result.get("highlights", [])),
        "mean_claim_fragility": mean_fragility,
    }


def _save_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_visualizations(output_dir: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        note_path = os.path.join(output_dir, "PLOTS_NOT_CREATED.txt")
        with open(note_path, "w", encoding="utf-8") as f:
            f.write("matplotlib not available. Install with: pip install matplotlib\n")
        return

    # 1) Average fragility by temperature
    by_temp: Dict[float, List[float]] = {}
    for r in rows:
        v = r.get("mean_claim_fragility")
        if v is None:
            continue
        t = float(r["temperature"])
        by_temp.setdefault(t, []).append(float(v))

    if by_temp:
        temps = sorted(by_temp.keys())
        means = [sum(by_temp[t]) / len(by_temp[t]) for t in temps]
        plt.figure(figsize=(7, 4))
        plt.bar([str(t) for t in temps], means)
        plt.xlabel("Temperature")
        plt.ylabel("Average claim fragility")
        plt.title("Fragility by temperature")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fragility_by_temperature.png"), dpi=140)
        plt.close()

    # 2) Severity counts (stacked per temperature)
    sev_counts: Dict[float, Dict[str, int]] = {}
    for r in rows:
        t = float(r["temperature"])
        cur = sev_counts.setdefault(t, {"low": 0, "medium": 0, "high": 0})
        cur["low"] += int(r.get("num_low_claims", 0))
        cur["medium"] += int(r.get("num_medium_claims", 0))
        cur["high"] += int(r.get("num_high_claims", 0))

    if sev_counts:
        temps = sorted(sev_counts.keys())
        lows = [sev_counts[t]["low"] for t in temps]
        meds = [sev_counts[t]["medium"] for t in temps]
        highs = [sev_counts[t]["high"] for t in temps]

        plt.figure(figsize=(8, 4.5))
        x = list(range(len(temps)))
        plt.bar(x, lows, label="low")
        plt.bar(x, meds, bottom=lows, label="medium")
        bottoms = [l + m for l, m in zip(lows, meds)]
        plt.bar(x, highs, bottom=bottoms, label="high")
        plt.xticks(x, [str(t) for t in temps])
        plt.xlabel("Temperature")
        plt.ylabel("Total claim count")
        plt.title("Claim severities by temperature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "claim_severity_by_temperature.png"), dpi=140)
        plt.close()

    # 3) Scatter answer length vs mean fragility
    xs = []
    ys = []
    for r in rows:
        v = r.get("mean_claim_fragility")
        if v is None:
            continue
        xs.append(float(r.get("answer_len_chars", 0)))
        ys.append(float(v))

    if xs and ys:
        plt.figure(figsize=(7, 4))
        plt.scatter(xs, ys, alpha=0.7)
        plt.xlabel("Answer length (chars)")
        plt.ylabel("Mean claim fragility")
        plt.title("Answer length vs fragility")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "answer_length_vs_fragility.png"), dpi=140)
        plt.close()


def _run_batch_live(
    *,
    output_dir: str,
    experiment_name: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    prompts: List[str],
    temperatures: List[float],
    max_tokens: int,
    top_logprobs: int,
) -> None:
    _ensure_dir(output_dir)

    run_id = f"{experiment_name}_{_timestamp()}"
    run_dir = os.path.join(output_dir, run_id)
    _ensure_dir(run_dir)

    rows: List[Dict[str, Any]] = []

    for p_idx, prompt in enumerate(prompts, start=1):
        for temp in temperatures:
            result = run_logit_gap_claim_detection_live(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_logprobs=top_logprobs,
            )

            safe_temp = str(temp).replace(".", "_")
            single_path = os.path.join(run_dir, f"result_prompt{p_idx:03d}_t{safe_temp}.json")
            _save_json(single_path, result)

            row = _result_summary_row(
                run_id=run_id,
                prompt_id=p_idx,
                model=model,
                temperature=temp,
                result=result,
            )
            rows.append(row)

            print(
                f"[saved] prompt={p_idx} temp={temp} claims={row['num_claim_spans']} "
                f"high={row['num_high_claims']} mean_frag={row['mean_claim_fragility']}"
            )

    summary_csv = os.path.join(run_dir, "summary.csv")
    _save_summary_csv(summary_csv, rows)

    config_json = {
        "run_id": run_id,
        "model": model,
        "base_url": base_url,
        "num_prompts": len(prompts),
        "temperatures": temperatures,
        "max_tokens": max_tokens,
        "top_logprobs": top_logprobs,
    }
    _save_json(os.path.join(run_dir, "run_config.json"), config_json)

    _create_visualizations(run_dir, rows)
    print(f"\nExperiment artifacts saved in: {run_dir}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Minimal independent logit-gap claim detection tool.")
    parser.add_argument("--response-json", help="Path to a saved OpenAI-compatible chat response JSON file")
    parser.add_argument("--base-url", default=NEBULA_BASE_URL, help="OpenAI-compatible base URL (defaults to Nebula)")
    parser.add_argument("--api-key", help="Optional API key override (defaults to NEBULA_API_KEY from .env)")
    parser.add_argument("--model", help="Model name for live mode")
    parser.add_argument("--system", default="You are a factual assistant. Keep answers concise and grounded.")
    parser.add_argument("--user", help="User prompt for live mode")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--temperatures", default="0.2", help="Comma-separated temperatures for batch mode, e.g. 0.0,0.2,0.5")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--prompts-file", help="Batch mode: .txt (one prompt per line) or .json (list of prompts)")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--experiment-name", default="idea2_logit_gap")
    parser.add_argument("--save-single", action="store_true", help="Save single-run result JSON in output dir")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("NEBULA_API_KEY")

    if args.prompts_file:
        missing = []
        if not api_key:
            missing.append("NEBULA_API_KEY in .env (or --api-key)")
        if not args.model:
            missing.append("--model")
        if missing:
            raise SystemExit("Missing arguments for batch live mode: " + ", ".join(missing))

        prompts = _load_prompts_file(args.prompts_file)
        if not prompts:
            raise SystemExit("No prompts found in --prompts-file")

        temperatures = _parse_temperatures(args.temperatures)
        _run_batch_live(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=args.system,
            prompts=prompts,
            temperatures=temperatures,
            max_tokens=args.max_tokens,
            top_logprobs=args.top_logprobs,
        )
        return

    if args.response_json:
        with open(args.response_json, "r", encoding="utf-8") as f:
            response = json.load(f)
        result = analyze_response_for_logit_gap_claims(response, requested_top_logprobs=args.top_logprobs)
    else:
        missing = []
        if not api_key:
            missing.append("NEBULA_API_KEY in .env (or --api-key)")
        if not args.model:
            missing.append("--model")
        if not args.user:
            missing.append("--user")
        if missing:
            raise SystemExit(
                "Missing arguments for live mode: "
                + ", ".join(missing)
                + ". Or provide --response-json for offline analysis."
            )

        result = run_logit_gap_claim_detection_live(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=args.system,
            user_prompt=args.user,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_logprobs=args.top_logprobs,
        )

    _pretty_print_result(result)

    if args.save_single:
        _ensure_dir(args.output_dir)
        out_path = os.path.join(
            args.output_dir,
            f"single_{args.experiment_name}_{_timestamp()}.json",
        )
        _save_json(out_path, result)
        print(f"\nSaved single-run result: {out_path}")


if __name__ == "__main__":
    main()
