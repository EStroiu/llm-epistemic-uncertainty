import argparse
import csv
from datetime import datetime
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from nebula_prompting import get_nebula_client  # noqa: E402


LABEL_SUPPORTS = "SUPPORTED"
LABEL_REFUTES = "REFUTED"
LABEL_NEI = "NOT_ENOUGH_INFO"

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


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_label(text: str) -> str:
    t = (text or "").strip().upper().replace("_", " ")
    if "NOT ENOUGH" in t or "NEI" in t:
        return LABEL_NEI
    if "SUPPORT" in t:
        return LABEL_SUPPORTS
    if "REFUT" in t:
        return LABEL_REFUTES
    return "UNKNOWN"


def _extract_predicted_label(answer_text: str) -> str:
    text = (answer_text or "").strip()
    if not text:
        return "UNKNOWN"

    match = re.search(
        r"LABEL\s*[:=-]\s*(SUPPORTED|SUPPORTS|REFUTED|REFUTES|NOT[ _-]?ENOUGH[ _-]?INFO|NEI)",
        text,
        re.IGNORECASE,
    )
    if match:
        return _normalize_label(match.group(1))

    match = re.search(
        r"\b(SUPPORTED|SUPPORTS|REFUTED|REFUTES|NOT[ _-]?ENOUGH[ _-]?INFO|NEI)\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return _normalize_label(match.group(1))

    return "UNKNOWN"


def _build_prompt(claim: str) -> str:
    return (
        "Classify the following factual claim into exactly one label: "
        "SUPPORTED, REFUTED, or NOT_ENOUGH_INFO.\n\n"
        "Return exactly two lines and nothing else:\n"
        "LABEL: <SUPPORTED|REFUTED|NOT_ENOUGH_INFO>\n"
        "REASON: <one short sentence>\n\n"
        f"CLAIM: {claim}"
    )


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


def _feature_row(answer_text: str, claim: str, pred_label: str, finish_reason: Optional[str]) -> Dict[str, Any]:
    reason = _extract_reason(answer_text)
    finish_reason = str(finish_reason or "").lower()

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

    return {
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


def _load_fever_rows(*, config_name: str, split_name: str, max_examples: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    parquet_glob = f"hf://datasets/fever/fever@refs/convert/parquet/{config_name}/{split_name}/*.parquet"
    try:
        ds = load_dataset("parquet", data_files={"data": parquet_glob}, split="data")
    except Exception as exc:
        raise SystemExit(
            "Failed to load FEVER parquet files from Hugging Face Hub. "
            f"config={config_name}, split={split_name}. Original error: {exc}"
        ) from exc

    unique: Dict[int, Dict[str, Any]] = {}
    for item in ds:
        cid = int(item["id"])
        if cid in unique:
            continue
        label = _normalize_label(item.get("label", ""))
        if label not in {LABEL_SUPPORTS, LABEL_REFUTES, LABEL_NEI}:
            continue
        unique[cid] = {
            "id": cid,
            "claim": str(item.get("claim", "")),
            "label": label,
            "evidence_wiki_url": item.get("evidence_wiki_url", ""),
            "evidence_sentence_id": item.get("evidence_sentence_id", -1),
        }

    rows = list(unique.values())
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    if max_examples > 0:
        rows = rows[:max_examples]
    return rows


def _call_nebula(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    client = get_nebula_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    choice = response.choices[0]
    return {
        "answer_text": choice.message.content or "",
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": getattr(response, "usage", None),
    }


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


def _calibration_buckets(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins, right=True) - 1
    out = []
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            out.append({"bin": i, "count": 0, "mean_confidence": None, "accuracy": None})
        else:
            out.append({
                "bin": i,
                "count": int(mask.sum()),
                "mean_confidence": float(probs[mask].mean()),
                "accuracy": float(y_true[mask].mean()),
            })
    return out


def _bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = 42):
    rng = np.random.RandomState(seed)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            val = metric_fn(y_true[idx], probs[idx])
            if val is None:
                continue
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                continue
            stats.append(float(val))
        except Exception:
            continue
    if not stats:
        return {"mean": None, "ci_lower": None, "ci_upper": None}
    a = float(np.percentile(stats, 2.5))
    b = float(np.percentile(stats, 97.5))
    return {"mean": float(np.mean(stats)), "ci_lower": a, "ci_upper": b}


def _run_cv(df: pd.DataFrame, feature_cols: List[str], random_state: int = 42) -> Dict[str, Any]:
    X = df[feature_cols].fillna(0.0).astype(float).values
    y = df["error"].astype(int).values

    class_counts = np.bincount(y)
    if len(class_counts) < 2 or class_counts.min() < 2 or len(df) < 10:
        fitted_tree = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
        )
        fitted_tree.fit(X, y)
        probs = fitted_tree.predict_proba(X)[:, 1]
        ranked_features = sorted(
            zip(feature_cols, getattr(fitted_tree, "feature_importances_", np.zeros(len(feature_cols)))),
            key=lambda x: x[1],
            reverse=True,
        )
        return {
            "oof_probs": probs.tolist(),
            "auc_mean": float(roc_auc_score(y, probs)) if len(set(y)) > 1 else None,
            "auc_std": 0.0,
            "ece_mean": float(_ece(y, probs)),
            "ranked_features": ranked_features,
            "calibration_buckets": _calibration_buckets(y, probs),
            "fallback": True,
        }

    n_splits = min(5, int(class_counts.min()))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_probs = np.zeros(len(df), dtype=float)
    importances = []
    fold_aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fitted_tree = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        )
        fitted_tree.fit(X_train, y_train)
        importances.append(getattr(fitted_tree, "feature_importances_", np.zeros(len(feature_cols))))

        min_train_class = int(np.bincount(y_train).min())
        calib_cv = 3 if min_train_class >= 3 else 2

        calibrator = CalibratedClassifierCV(
            estimator=DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=random_state,
            ),
            method="sigmoid",
            cv=calib_cv,
        )
        calibrator.fit(X_train, y_train)
        probs = calibrator.predict_proba(X_test)[:, 1]
        oof_probs[test_idx] = probs
        fold_aucs.append(float(roc_auc_score(y_test, probs)))

    mean_importances = np.mean(np.asarray(importances), axis=0).tolist() if importances else [0.0] * len(feature_cols)
    ranked_features = sorted(zip(feature_cols, mean_importances), key=lambda x: x[1], reverse=True)

    return {
        "oof_probs": oof_probs.tolist(),
        "auc_mean": float(roc_auc_score(y, oof_probs)),
        "auc_std": float(np.std(fold_aucs)),
        "ece_mean": float(_ece(y, oof_probs)),
        "ranked_features": ranked_features,
        "calibration_buckets": _calibration_buckets(y, oof_probs),
    }


def _fit_and_export_tree(df: pd.DataFrame, feature_cols: List[str], out_path: str) -> str:
    X = df[feature_cols].fillna(0.0).astype(float).values
    y = df["error"].astype(int).values

    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X, y)

    text_rules = export_text(tree, feature_names=feature_cols)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text_rules)
    return text_rules


def _fit_final_model(df: pd.DataFrame, feature_cols: List[str], random_state: int = 42) -> Dict[str, Any]:
    X = df[feature_cols].fillna(0.0).astype(float).values
    y = df["error"].astype(int).values

    fitted_tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=random_state,
    )
    fitted_tree.fit(X, y)

    class_counts = np.bincount(y)
    ranked_features = sorted(
        zip(feature_cols, getattr(fitted_tree, "feature_importances_", np.zeros(len(feature_cols)))),
        key=lambda x: x[1],
        reverse=True,
    )

    if len(class_counts) < 2 or class_counts.min() < 2 or len(df) < 10:
        return {
            "predictor": fitted_tree,
            "ranked_features": ranked_features,
            "fallback": True,
        }

    min_train_class = int(class_counts.min())
    calib_cv = 3 if min_train_class >= 3 else 2
    calibrator = CalibratedClassifierCV(
        estimator=DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        ),
        method="sigmoid",
        cv=calib_cv,
    )
    calibrator.fit(X, y)

    return {
        "predictor": calibrator,
        "ranked_features": ranked_features,
        "fallback": False,
    }


def _plot_figures(run_dir: str, df: pd.DataFrame, feature_cols: List[str]) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Label-wise box + jitter
    order = [LABEL_SUPPORTS, LABEL_REFUTES, LABEL_NEI]
    values = [df.loc[df["gold_label"] == lab, "texttree_uncertainty"].dropna().values for lab in order]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(values, tick_labels=order, showfliers=False)
    for i, vals in enumerate(values, start=1):
        if len(vals) == 0:
            continue
        x = np.random.normal(i, 0.04, size=len(vals))
        ax.scatter(x, vals, s=7, alpha=0.25, color="black")
    ax.set_title("Text-tree uncertainty by FEVER label")
    ax.set_xlabel("Gold label")
    ax.set_ylabel("Uncertainty")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "labelwise_uncertainty.png"), dpi=200)
    fig.savefig(os.path.join(fig_dir, "labelwise_uncertainty.pdf"))
    plt.close(fig)

    ranked = df.attrs.get("ranked_features", [])
    if ranked:
        feat = [k for k, _ in ranked[:10]][::-1]
        imp = [v for _, v in ranked[:10]][::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feat, imp, color="#2f855a")
        ax.set_title("Top text-tree feature importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "texttree_feature_importance.png"), dpi=200)
        fig.savefig(os.path.join(fig_dir, "texttree_feature_importance.pdf"))
        plt.close(fig)

    # ROC curve for detecting errors
    y_err = (~df["is_correct"]).astype(int).values
    probs = df["texttree_uncertainty"].fillna(0.0).values
    try:
        fpr, tpr, _ = roc_curve(y_err, probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve (error detection)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "roc_error_detection.png"), dpi=200)
        fig.savefig(os.path.join(fig_dir, "roc_error_detection.pdf"))
        plt.close(fig)
    except Exception:
        pass

    # Calibration (reliability) diagram
    try:
        prob_true, prob_pred = calibration_curve(y_err, probs, n_bins=10)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(prob_pred, prob_true, marker="o")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Reliability diagram")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "calibration_reliability.png"), dpi=200)
        fig.savefig(os.path.join(fig_dir, "calibration_reliability.pdf"))
        plt.close(fig)
    except Exception:
        pass

    # Score distribution (hist + density)
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(probs, bins=30, kde=True, ax=ax, color="#3182ce")
        ax.set_title("Distribution of text-tree uncertainty scores")
        ax.set_xlabel("Uncertainty")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "uncertainty_distribution.png"), dpi=200)
        fig.savefig(os.path.join(fig_dir, "uncertainty_distribution.pdf"))
        plt.close(fig)
    except Exception:
        pass

    # Risk-Coverage (selective risk): keep most confident samples first
    try:
        n = len(probs)
        order_idx = np.argsort(probs)  # ascending uncertainty -> most confident first
        sorted_err = y_err[order_idx]
        cum_counts = np.arange(1, n + 1)
        cum_risk = np.cumsum(sorted_err) / cum_counts  # risk at coverage k/n
        coverage = cum_counts / n
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(coverage, cum_risk, marker=".")
        ax.set_xlabel("Coverage (fraction retained, most confident first)")
        ax.set_ylabel("Risk (error rate)")
        ax.set_title("Risk vs Coverage (selective risk)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "risk_coverage.png"), dpi=200)
        fig.savefig(os.path.join(fig_dir, "risk_coverage.pdf"))
        plt.close(fig)
    except Exception:
        pass


def run_benchmark(args: argparse.Namespace) -> str:
    api_key = os.getenv("NEBULA_API_KEY")
    if not api_key:
        raise SystemExit("Missing NEBULA_API_KEY in .env or environment")
    if not args.model:
        raise SystemExit("Missing --model")

    run_id = f"fever_texttree_{args.config}_{args.split}_{_now()}"
    run_dir = os.path.join(args.output_dir, run_id)
    json_dir = os.path.join(run_dir, "result_jsons")
    _ensure_dir(run_dir)
    _ensure_dir(json_dir)

    fever_rows = _load_fever_rows(
        config_name=args.config,
        split_name=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    summary_rows: List[Dict[str, Any]] = []
    raw_records: List[Dict[str, Any]] = []

    for run_idx in range(1, args.num_runs + 1):
        print(f"\n[run {run_idx}/{args.num_runs}]")
        for i, sample in enumerate(fever_rows, start=1):
            prompt = _build_prompt(sample["claim"])
            result = _call_nebula(
                model=args.model,
                system_prompt=args.system,
                user_prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            answer_text = str(result.get("answer_text", "") or "")
            pred_label = _extract_predicted_label(answer_text)
            is_correct = pred_label == sample["label"]
            finish_reason = result.get("finish_reason")
            features = _feature_row(answer_text, sample["claim"], pred_label, finish_reason)

            row = {
                "run_index": run_idx,
                "sample_index": i,
                "claim_id": sample["id"],
                "gold_label": sample["label"],
                "pred_label": pred_label,
                "is_correct": is_correct,
                "error": 0 if is_correct else 1,
                "finish_reason": finish_reason,
                "claim": sample["claim"],
            }
            row.update(features)
            summary_rows.append(row)

            record = {
                "run_index": run_idx,
                "sample": sample,
                "prompt": prompt,
                "prediction": {
                    "label": pred_label,
                    "is_correct": is_correct,
                },
                "analysis": {
                    "answer_text": answer_text,
                    "finish_reason": finish_reason,
                    "features": features,
                },
            }
            raw_records.append(record)
            safe_name = f"result_run{run_idx:02d}_id{sample['id']}.json"
            with open(os.path.join(json_dir, safe_name), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            print(f"[saved] run={run_idx} sample={i}/{len(fever_rows)} id={sample['id']} gold={sample['label']} pred={pred_label}")

    df = pd.DataFrame(summary_rows)
    if df.empty:
        raise SystemExit("No benchmark rows were generated.")

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

    unique_claims = (
        df.sort_values(["claim_id", "run_index"])
        .drop_duplicates(subset=["claim_id"], keep="first")
        .reset_index(drop=True)
    )

    try:
        train_claims, test_claims = train_test_split(
            unique_claims,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=unique_claims["error"] if unique_claims["error"].nunique() > 1 else None,
        )
    except Exception:
        train_claims = unique_claims.copy()
        test_claims = unique_claims.iloc[0:0].copy()

    train_claims = train_claims.reset_index(drop=True)
    test_claims = test_claims.reset_index(drop=True)

    cv_results = _run_cv(train_claims, feature_cols)
    train_claims["texttree_uncertainty"] = cv_results["oof_probs"]
    train_claims["split_role"] = "train"

    final_model = _fit_final_model(train_claims, feature_cols)
    if len(test_claims) > 0:
        test_X = test_claims[feature_cols].fillna(0.0).astype(float).values
        test_claims["texttree_uncertainty"] = final_model["predictor"].predict_proba(test_X)[:, 1]
        test_claims["split_role"] = "test"

    scored_claims = pd.concat([train_claims, test_claims], ignore_index=True)
    claim_to_score = dict(zip(scored_claims["claim_id"], scored_claims["texttree_uncertainty"]))
    claim_to_split = dict(zip(scored_claims["claim_id"], scored_claims["split_role"]))

    df["texttree_uncertainty"] = df["claim_id"].map(claim_to_score)
    df["split_role"] = df["claim_id"].map(claim_to_split)

    # Persist scores and metrics.
    _save_rows_csv(os.path.join(run_dir, "summary.csv"), df.to_dict(orient="records"))

    train_y_err = train_claims["error"].astype(int).values
    train_probs = train_claims["texttree_uncertainty"].fillna(0.0).values
    train_y_nei = (train_claims["gold_label"] == LABEL_NEI).astype(int).values

    train_metrics = {
        "num_rows": int(len(train_claims)),
        "auroc_error_detection": cv_results.get("auc_mean"),
        "ece": cv_results.get("ece_mean"),
        "auc_std": cv_results.get("auc_std"),
        "calibration_buckets": cv_results.get("calibration_buckets"),
        "auroc_nei_detection": None,
        "spearman_uncertainty_vs_error": None,
        "brier_score": None,
    }
    try:
        train_metrics["auroc_nei_detection"] = float(roc_auc_score(train_y_nei, train_probs)) if len(set(train_y_nei)) > 1 else None
    except Exception:
        train_metrics["auroc_nei_detection"] = None
    try:
        train_metrics["spearman_uncertainty_vs_error"] = float(pd.Series(train_probs).corr(pd.Series(train_y_err), method="spearman"))
    except Exception:
        train_metrics["spearman_uncertainty_vs_error"] = None
    try:
        train_metrics["brier_score"] = float(brier_score_loss(train_y_err, train_probs))
    except Exception:
        train_metrics["brier_score"] = None

    holdout_metrics = None
    if len(test_claims) > 0:
        test_y_err = test_claims["error"].astype(int).values
        test_probs = test_claims["texttree_uncertainty"].fillna(0.0).values
        test_y_nei = (test_claims["gold_label"] == LABEL_NEI).astype(int).values
        holdout_metrics = {
            "num_rows": int(len(test_claims)),
            "auroc_error_detection": None,
            "ece": float(_ece(test_y_err, test_probs)),
            "calibration_buckets": _calibration_buckets(test_y_err, test_probs),
            "auroc_nei_detection": None,
            "spearman_uncertainty_vs_error": None,
            "brier_score": None,
        }
        try:
            holdout_metrics["auroc_error_detection"] = float(roc_auc_score(test_y_err, test_probs)) if len(set(test_y_err)) > 1 else None
        except Exception:
            holdout_metrics["auroc_error_detection"] = None
        try:
            holdout_metrics["auroc_nei_detection"] = float(roc_auc_score(test_y_nei, test_probs)) if len(set(test_y_nei)) > 1 else None
        except Exception:
            holdout_metrics["auroc_nei_detection"] = None
        try:
            holdout_metrics["spearman_uncertainty_vs_error"] = float(pd.Series(test_probs).corr(pd.Series(test_y_err), method="spearman"))
        except Exception:
            holdout_metrics["spearman_uncertainty_vs_error"] = None
        try:
            holdout_metrics["brier_score"] = float(brier_score_loss(test_y_err, test_probs))
        except Exception:
            holdout_metrics["brier_score"] = None

    eval_y_err = test_y_err if holdout_metrics is not None else train_y_err
    eval_probs = test_probs if holdout_metrics is not None else train_probs

    # Bootstrap confidence intervals are reported on the held-out split when available.
    try:
        n_boot = 200  # set to 1000 for more precise CIs; reduce for speed
        bootstrap_metrics = {
            "auroc_error_detection": _bootstrap_ci(eval_y_err, eval_probs, lambda y, p: roc_auc_score(y, p) if len(set(y)) > 1 else None, n_boot=n_boot, seed=args.seed),
            "brier_score": _bootstrap_ci(eval_y_err, eval_probs, lambda y, p: brier_score_loss(y, p), n_boot=n_boot, seed=args.seed),
            "ece": _bootstrap_ci(eval_y_err, eval_probs, lambda y, p: _ece(y, p), n_boot=n_boot, seed=args.seed),
        }
    except Exception:
        bootstrap_metrics = None

    metrics = {
        "num_rows": int(len(df)),
        "num_unique_claims": int(len(unique_claims)),
        "num_valid_uncertainty": int(df["texttree_uncertainty"].notnull().sum()),
        "train_claims": int(len(train_claims)),
        "test_claims": int(len(test_claims)),
        "test_size": float(args.test_size),
        "split_strategy": "stratified_claim_holdout",
        "train_oof": train_metrics,
        "holdout": holdout_metrics,
        "auc_std": train_metrics["auc_std"],
        "calibration_buckets": holdout_metrics["calibration_buckets"] if holdout_metrics is not None else train_metrics["calibration_buckets"],
        "bootstrap": bootstrap_metrics,
    }

    if holdout_metrics is not None:
        metrics["auroc_error_detection"] = holdout_metrics["auroc_error_detection"]
        metrics["ece"] = holdout_metrics["ece"]
        metrics["brier_score"] = holdout_metrics["brier_score"]
        metrics["auroc_nei_detection"] = holdout_metrics["auroc_nei_detection"]
        metrics["spearman_uncertainty_vs_error"] = holdout_metrics["spearman_uncertainty_vs_error"]
    else:
        metrics["auroc_error_detection"] = train_metrics["auroc_error_detection"]
        metrics["ece"] = train_metrics["ece"]
        metrics["brier_score"] = train_metrics["brier_score"]
        metrics["auroc_nei_detection"] = train_metrics["auroc_nei_detection"]
        metrics["spearman_uncertainty_vs_error"] = train_metrics["spearman_uncertainty_vs_error"]

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    run_config = {
        "run_id": run_id,
        "dataset": "FEVER",
        "config": args.config,
        "split": args.split,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "test_size": args.test_size,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_runs": args.num_runs,
        "output_dir": args.output_dir,
        "estimator": "texttree",
        "base_url": "Nebula via nebula_prompting.get_nebula_client",
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    rules_path = os.path.join(run_dir, "texttree_uncertainty_rules.txt")
    rules = _fit_and_export_tree(train_claims, feature_cols, rules_path)
    df.attrs["ranked_features"] = final_model["ranked_features"]
    plot_df = scored_claims.copy()
    plot_df.attrs["ranked_features"] = final_model["ranked_features"]
    _plot_figures(run_dir, plot_df, feature_cols)

    print("\n=== Text-tree rules ===")
    print(rules)
    print(f"\nSaved experiment to: {run_dir}")
    return run_dir


def _save_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Self-contained FEVER text-tree uncertainty benchmark that queries Nebula directly.")
    parser.add_argument("--model", default="FAST.gpt-oss:120b")
    parser.add_argument("--config", default="v1.0")
    parser.add_argument("--split", default="labelled_dev")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=os.path.join(ROOT_DIR, "experiments"))
    parser.add_argument(
        "--system",
        default=(
            "You are a careful fact-checking assistant. Follow the format exactly, "
            "and keep the REASON line short and direct."
        ),
    )
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
