import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

# Allow imports from project root when running `python evals/...`.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uncertainty_logit_gap import NEBULA_BASE_URL, _ensure_dir

from evals.fever_benchmark import (
    LABEL_NEI,
    LABEL_REFUTES,
    LABEL_SUPPORTS,
    _compute_metrics,
    _load_fever_rows,
    _normalize_label,
    _now,
    _plot_report,
    _save_rows_csv,
)


LABELS = [LABEL_SUPPORTS, LABEL_REFUTES, LABEL_NEI]
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_message_text(response: Any) -> str:
    choices = _obj_get(response, "choices", [])
    if not choices:
        return ""
    first = choices[0]
    message = _obj_get(first, "message", {}) or {}
    content = _obj_get(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                txt = part.get("text") or part.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        if parts:
            return "\n".join(parts).strip()

    # Some providers return text only in reasoning fields.
    reasoning = (
        _obj_get(message, "reasoning_content")
        or _obj_get(_obj_get(message, "provider_specific_fields", {}), "reasoning_content")
        or _obj_get(_obj_get(message, "provider_specific_fields", {}), "reasoning")
    )
    if isinstance(reasoning, str):
        return reasoning.strip()

    # Fallback for providers that place text at choice level.
    choice_reasoning = (
        _obj_get(_obj_get(first, "provider_specific_fields", {}), "reasoning")
        or _obj_get(_obj_get(first, "provider_specific_fields", {}), "reasoning_content")
    )
    if isinstance(choice_reasoning, str):
        return choice_reasoning.strip()
    return ""


def _build_prompt(claim: str) -> str:
    return (
        "You are solving FEVER claim verification.\n"
        "Return ONLY valid JSON with this exact structure. No explanations.\n"
        "{\n"
        '  "label_probs": {\n'
        '    "SUPPORTED": <float>,\n'
        '    "REFUTED": <float>,\n'
        '    "NOT_ENOUGH_INFO": <float>\n'
        "  },\n"
        '  "predicted_label": "<SUPPORTED|REFUTED|NOT_ENOUGH_INFO>",\n'
        '  "reason": "<one short sentence>"\n'
        "}\n"
        "Requirements:\n"
        "- All probabilities must be between 0 and 1.\n"
        "- Probabilities must sum to 1.\n"
        "- Use decimal numbers (e.g. 0.72).\n"
        "- Output JSON only, no markdown.\n\n"
        f"Claim: {claim}"
    )


def _build_repair_prompt(claim: str) -> str:
    return (
        "Output exactly one JSON object and nothing else.\n"
        "Schema:\n"
        '{"label_probs":{"SUPPORTED":0.0,"REFUTED":0.0,"NOT_ENOUGH_INFO":0.0},'
        '"predicted_label":"SUPPORTED","reason":"..."}\n'
        "Probabilities must sum to 1.\n"
        f"Claim: {claim}"
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _normalize_probs(raw_probs: Dict[str, Any]) -> Optional[Dict[str, float]]:
    mapped: Dict[str, float] = {}
    for key, val in raw_probs.items():
        label = _normalize_label(str(key))
        if label not in LABELS:
            continue
        try:
            mapped[label] = float(val)
        except Exception:
            continue

    if len(mapped) != 3:
        return None
    if any(v < 0 for v in mapped.values()):
        return None

    total = sum(mapped.values())
    if total <= 0:
        return None
    return {k: max(0.0, v / total) for k, v in mapped.items()}


def _fallback_parse(text: str) -> Tuple[Dict[str, float], str]:
    label = _normalize_label(text)
    if label not in LABELS:
        return ({lb: 1.0 / 3.0 for lb in LABELS}, "UNKNOWN")
    probs = {lb: 0.0 for lb in LABELS}
    probs[label] = 1.0
    return (probs, label)


def _parse_output(text: str) -> Tuple[Dict[str, float], str, bool]:
    obj = _extract_json(text)
    if not obj:
        probs, label = _fallback_parse(text)
        return probs, label, False

    probs_obj = obj.get("label_probs", {})
    probs = _normalize_probs(probs_obj) if isinstance(probs_obj, dict) else None
    predicted = _normalize_label(str(obj.get("predicted_label", "")))

    if probs is None:
        probs, label = _fallback_parse(text)
        return probs, label, False

    if predicted not in LABELS:
        predicted = max(probs, key=probs.get)
    return probs, predicted, True


def _entropy_uncertainty(probs: Dict[str, float]) -> float:
    entropy = 0.0
    for label in LABELS:
        p = max(0.0, min(1.0, float(probs.get(label, 0.0))))
        if p > 0:
            entropy -= p * math.log(p)
    return max(0.0, min(1.0, entropy / math.log(3.0)))


def _request_label_probs(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    # Try strict JSON response mode first (if backend supports it),
    # then fall back to plain chat completion.
    try:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            stream=False,
        )
    except Exception:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )


def run_benchmark(args: argparse.Namespace) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit("Missing dependency 'openai'. Install project requirements first.") from exc

    api_key = args.api_key or os.getenv("NEBULA_API_KEY")
    if not api_key:
        raise SystemExit("Missing NEBULA_API_KEY in .env or --api-key")
    if not args.model:
        raise SystemExit("Missing --model")

    if args.quick_run:
        args.max_examples = min(args.max_examples, args.quick_examples)
        args.num_runs = 1
        args.max_tokens = min(args.max_tokens, 120)
        print(
            f"[quick-run] max_examples={args.max_examples}, "
            f"num_runs={args.num_runs}, max_tokens={args.max_tokens}"
        )

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    run_id = f"label_ambiguity_fever_{args.config}_{args.split}_{_now()}"
    run_dir = os.path.join(args.output_dir, run_id)
    json_dir = os.path.join(run_dir, "result_jsons")
    _ensure_dir(run_dir)
    _ensure_dir(json_dir)

    fever_rows = _load_fever_rows(
        submodule_dir=args.fever_submodule,
        config_name=args.config,
        split_name=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    summary_rows = []

    for run_idx in range(1, args.num_runs + 1):
        print(f"\n[run {run_idx}/{args.num_runs}]")
        for i, sample in enumerate(fever_rows, start=1):
            prompt = _build_prompt(sample["claim"])
            response = _request_label_probs(
                client=client,
                model=args.model,
                system_prompt=args.system,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            answer_text = _extract_message_text(response)
            probs, pred_label, parse_ok = _parse_output(answer_text)

            if not parse_ok and args.retry_parse_failures:
                repair_prompt = _build_repair_prompt(sample["claim"])
                repair_response = _request_label_probs(
                    client=client,
                    model=args.model,
                    system_prompt=args.system,
                    prompt=repair_prompt,
                    temperature=0.0,
                    max_tokens=max(90, min(args.max_tokens, 180)),
                )
                repair_text = _extract_message_text(repair_response)
                repair_probs, repair_pred, repair_ok = _parse_output(repair_text)
                if repair_ok:
                    response = repair_response
                    answer_text = repair_text
                    probs, pred_label, parse_ok = repair_probs, repair_pred, repair_ok

            uncertainty = _entropy_uncertainty(probs)
            is_correct = pred_label == sample["label"]

            row = {
                "run_index": run_idx,
                "sample_index": i,
                "claim_id": sample["id"],
                "gold_label": sample["label"],
                "pred_label": pred_label,
                "is_correct": is_correct,
                "uncertainty": uncertainty,
                "parse_ok": parse_ok,
                "p_supported": probs[LABEL_SUPPORTS],
                "p_refuted": probs[LABEL_REFUTES],
                "p_nei": probs[LABEL_NEI],
                "claim": sample["claim"],
            }
            summary_rows.append(row)

            response_dump = response.model_dump() if hasattr(response, "model_dump") else {}
            record = {
                "run_index": run_idx,
                "sample": sample,
                "prompt": prompt,
                "answer_text": answer_text,
                "parsed": {
                    "probs": probs,
                    "predicted_label": pred_label,
                    "parse_ok": parse_ok,
                    "uncertainty": uncertainty,
                    "is_correct": is_correct,
                },
                "response": response_dump,
            }
            out_name = f"result_run{run_idx:02d}_id{sample['id']}.json"
            with open(os.path.join(json_dir, out_name), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            print(
                f"[saved] run={run_idx} sample={i}/{len(fever_rows)} id={sample['id']} "
                f"gold={sample['label']} pred={pred_label} uncertainty={uncertainty:.4f}"
            )

    _save_rows_csv(os.path.join(run_dir, "summary.csv"), summary_rows)
    metrics = _compute_metrics(summary_rows)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    config = {
        "run_id": run_id,
        "method": "label_ambiguity_entropy",
        "dataset": "FEVER",
        "config": args.config,
        "split": args.split,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "model": args.model,
        "temperature": args.temperature,
        "num_runs": args.num_runs,
        "max_tokens": args.max_tokens,
        "base_url": args.base_url,
        "quick_run": args.quick_run,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    _plot_report(run_dir, summary_rows, metrics)
    print(f"\nLabel-ambiguity benchmark artifacts saved in: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FEVER benchmark with label-ambiguity uncertainty (entropy)")
    parser.add_argument("--fever-submodule", default=os.path.join(ROOT_DIR, "external", "fever"))
    parser.add_argument("--config", default="v1.0")
    parser.add_argument("--split", default="labelled_dev")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-url", default=NEBULA_BASE_URL)
    parser.add_argument("--api-key", help="Optional API key override")
    parser.add_argument("--model", required=True)
    parser.add_argument("--system", default="You are a careful fact-checking assistant.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--output-dir", default=os.path.join(ROOT_DIR, "experiments"))

    parser.add_argument("--quick-run", action="store_true", help="Fast pilot run (~1 minute depending on API latency)")
    parser.add_argument("--quick-examples", type=int, default=10, help="Max examples when --quick-run is used")
    parser.add_argument(
        "--retry-parse-failures",
        action="store_true",
        default=True,
        help="Retry once with a stricter prompt when parsing fails (default: enabled)",
    )
    parser.add_argument(
        "--no-retry-parse-failures",
        action="store_false",
        dest="retry_parse_failures",
        help="Disable retry on parse failure",
    )
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.num_runs < 1:
        raise SystemExit("--num-runs must be >= 1")
    if args.max_examples < 1:
        raise SystemExit("--max-examples must be >= 1")

    run_benchmark(args)


if __name__ == "__main__":
    main()
