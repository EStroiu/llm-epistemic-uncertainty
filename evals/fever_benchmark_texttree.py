import argparse
import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text


HEDGE_PATTERNS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bappears?\b",
    r"\bseems?\b",
    r"\bunclear\b",
    r"\buncertain\b",
    r"\bcannot\s+verify\b",
    r"\bcan't\s+verify\b",
    r"\bno\s+evidence\b",
    r"\bnot\s+enough\s+info\b",
    r"\bnot\s+enough\s+information\b",
    r"\bunknown\b",
    r"\binsufficient\b",
    r"\bmight\b",
    r"\bcould\b",
]

CERTAINTY_PATTERNS = [
    r"\bdefinitely\b",
    r"\bcertainly\b",
    r"\bclearly\b",
    r"\bevidently\b",
    r"\bundoubtedly\b",
]

ENTITY_TOKEN_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_\-]+\b")
DIGIT_RE = re.compile(r"\d")
QUOTE_RE = re.compile(r'["“”\']')
LABEL_LINE_RE = re.compile(r"^\s*LABEL\s*[:=-]\s*(SUPPORTED|REFUTED|NOT_ENOUGH_INFO|NEI)\b", re.IGNORECASE | re.MULTILINE)
REASON_LINE_RE = re.compile(r"^\s*REASON\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _count_matches(patterns: List[str], text: str) -> int:
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return total


def _safe_len_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _extract_reason(answer_text: str) -> str:
    match = REASON_LINE_RE.search(answer_text or "")
    if not match:
        return ""
    return match.group(1).strip()


def _extract_pred_line(answer_text: str) -> str:
    match = LABEL_LINE_RE.search(answer_text or "")
    return match.group(1).upper() if match else "UNKNOWN"


def _feature_row_from_result(result_path: str) -> Dict[str, Any]:
    with open(result_path, "r") as f:
        payload = json.load(f)

    sample = payload.get("sample", {}) or {}
    prediction = payload.get("prediction", {}) or {}
    analysis = payload.get("analysis", {}) or {}

    answer_text = str(analysis.get("answer_text", "") or "")
    claim = str(sample.get("claim", "") or "")
    reason = _extract_reason(answer_text)
    pred_label = str(prediction.get("label", "UNKNOWN") or "UNKNOWN")
    finish_reason = str(analysis.get("finish_reason", "") or "").lower()

    hedge_count = _count_matches(HEDGE_PATTERNS, answer_text + "\n" + reason)
    certainty_count = _count_matches(CERTAINTY_PATTERNS, answer_text + "\n" + reason)
    label_present = 1 if LABEL_LINE_RE.search(answer_text or "") else 0
    reason_present = 1 if REASON_LINE_RE.search(answer_text or "") else 0
    format_ok = 1 if label_present and reason_present and finish_reason != "length" else 0

    reason_words = _safe_len_words(reason)
    answer_words = _safe_len_words(answer_text)
    claim_words = _safe_len_words(claim)
    reason_entities = len(ENTITY_TOKEN_RE.findall(reason))
    reason_digits = len(DIGIT_RE.findall(reason))
    reason_quotes = len(QUOTE_RE.findall(reason))
    claim_entities = len(ENTITY_TOKEN_RE.findall(claim))
    claim_digits = len(DIGIT_RE.findall(claim))
    claim_quotes = len(QUOTE_RE.findall(claim))

    features = {
        "sample_id": sample.get("id"),
        "gold_label": sample.get("label"),
        "pred_label": pred_label,
        "is_correct": bool(prediction.get("is_correct", False)),
        "error": 0 if prediction.get("is_correct", False) else 1,
        "uncertainty": float(prediction.get("uncertainty", 0.0) or 0.0),
        "answer_words": answer_words,
        "claim_words": claim_words,
        "reason_words": reason_words,
        "reason_entities": reason_entities,
        "reason_digits": reason_digits,
        "reason_quotes": reason_quotes,
        "claim_entities": claim_entities,
        "claim_digits": claim_digits,
        "claim_quotes": claim_quotes,
        "hedge_count": hedge_count,
        "certainty_count": certainty_count,
        "hedge_minus_certainty": hedge_count - certainty_count,
        "label_present": label_present,
        "reason_present": reason_present,
        "format_ok": format_ok,
        "finish_reason_length": 1 if finish_reason == "length" else 0,
        "finish_reason_stop": 1 if finish_reason == "stop" else 0,
        "pred_supported": 1 if pred_label == "SUPPORTED" else 0,
        "pred_refuted": 1 if pred_label == "REFUTED" else 0,
        "pred_nei": 1 if pred_label in {"NOT_ENOUGH_INFO", "NEI"} else 0,
        "specificity_reason": reason_entities + reason_digits + reason_quotes,
        "specificity_claim": claim_entities + claim_digits + claim_quotes,
        "reason_to_answer_ratio": (reason_words / answer_words) if answer_words else 0.0,
    }
    return features


def _load_dataframe(run_dir: str) -> pd.DataFrame:
    result_dir = os.path.join(run_dir, "result_jsons")
    rows = []
    for file_name in sorted(os.listdir(result_dir)):
        if not file_name.endswith(".json"):
            continue
        path = os.path.join(result_dir, file_name)
        try:
            rows.append(_feature_row_from_result(path))
        except Exception as exc:  # pragma: no cover
            print(f"Skipping {path}: {exc}")
    return pd.DataFrame(rows)


def _ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins, right=True) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        bin_prob = float(probs[mask].mean())
        bin_acc = float(y_true[mask].mean())
        ece += abs(bin_prob - bin_acc) * (mask.sum() / len(probs))
    return float(ece)


def _run_cv(df: pd.DataFrame, features: List[str], random_state: int = 42) -> Dict[str, Any]:
    X = df[features].fillna(0.0).astype(float).values
    y = df["error"].astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    aucs: List[float] = []
    eces: List[float] = []
    baseline_aucs: List[float] = []
    baseline_eces: List[float] = []
    all_importances = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        tree = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        )
        fitted_tree = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        )
        fitted_tree.fit(X_train, y_train)
        all_importances.append(getattr(fitted_tree, "feature_importances_", np.zeros(len(features))))

        calibrator = CalibratedClassifierCV(estimator=tree, method="sigmoid", cv=3)
        calibrator.fit(X_train, y_train)

        probs = calibrator.predict_proba(X_test)[:, 1]
        aucs.append(float(roc_auc_score(y_test, probs)))
        eces.append(_ece(y_test, probs))

        baseline_scores = df.iloc[test_idx]["uncertainty"].fillna(0.0).astype(float).values
        baseline_aucs.append(float(roc_auc_score(y_test, baseline_scores)))
        baseline_eces.append(_ece(y_test, baseline_scores))

    mean_importances = np.mean(np.asarray(all_importances), axis=0).tolist() if all_importances else [0.0] * len(features)
    ranked_features = sorted(zip(features, mean_importances), key=lambda x: x[1], reverse=True)

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ece_mean": float(np.mean(eces)),
        "baseline_auc_mean": float(np.mean(baseline_aucs)),
        "baseline_ece_mean": float(np.mean(baseline_eces)),
        "ranked_features": ranked_features,
    }


def _fit_and_export_tree(df: pd.DataFrame, features: List[str], out_path: str) -> str:
    X = df[features].fillna(0.0).astype(float).values
    y = df["error"].astype(int).values

    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X, y)

    text_rules = export_text(tree, feature_names=features)
    with open(out_path, "w") as f:
        f.write(text_rules)
    return text_rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpretable text-based uncertainty estimator for FEVER outputs.")
    parser.add_argument("--run-dir", required=True, help="Path to a FEVER experiment directory containing result_jsons/ and summary.csv.")
    parser.add_argument("--output-prefix", default="texttree_uncertainty", help="Prefix for saved outputs.")
    args = parser.parse_args()

    df = _load_dataframe(args.run_dir)
    if df.empty:
        raise SystemExit("No result JSONs found or failed to parse any samples.")

    feature_cols = [
        "answer_words",
        "claim_words",
        "reason_words",
        "reason_entities",
        "reason_digits",
        "reason_quotes",
        "claim_entities",
        "claim_digits",
        "claim_quotes",
        "hedge_count",
        "certainty_count",
        "hedge_minus_certainty",
        "label_present",
        "reason_present",
        "format_ok",
        "finish_reason_length",
        "finish_reason_stop",
        "pred_supported",
        "pred_refuted",
        "pred_nei",
        "specificity_reason",
        "specificity_claim",
        "reason_to_answer_ratio",
    ]

    results = _run_cv(df, feature_cols)

    metrics_path = os.path.join(args.run_dir, f"{args.output_prefix}_results.json")
    rules_path = os.path.join(args.run_dir, f"{args.output_prefix}_rules.txt")
    tree_text = _fit_and_export_tree(df, feature_cols, rules_path)

    payload = {
        "n": int(len(df)),
        "features": feature_cols,
        **results,
        "tree_rules_path": rules_path,
    }

    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print("\nDecision tree rules:\n")
    print(tree_text)


if __name__ == "__main__":
    main()
