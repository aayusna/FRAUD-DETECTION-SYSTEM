"""
src/models/ensemble.py
───────────────────────
Dual-model ensemble:
  Model A — XGBoost classifier    (optimised for Precision-Recall AUC)
  Model B — Isolation Forest      (anomaly detection for novel fraud)

Final decision: weighted combination of both scores.
Decision thresholds: BLOCK / REVIEW / APPROVE
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    # XGBoost depends on a native library (libxgboost) which in turn requires the OpenMP runtime (libomp).
    # On macOS, missing libomp would otherwise crash imports and prevent the API/Streamlit app from starting.
    from xgboost import XGBClassifier  # type: ignore
    _XGBOOST_AVAILABLE = True
except Exception as e:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False
    _XGBOOST_IMPORT_ERROR = e
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

ROOT        = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR  = os.path.join(ROOT, "data", "processed")
PLOTS_DIR   = os.path.join(ROOT, "data", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

XGB_PATH    = os.path.join(MODELS_DIR, "xgb_model.pkl")
ISO_PATH    = os.path.join(MODELS_DIR, "iso_forest.pkl")
RF_PATH     = os.path.join(MODELS_DIR, "rf_model.pkl")
METRICS_JSON = os.path.join(MODELS_DIR, "metrics.json")

# ─── Decision thresholds ──────────────────────────────────────────────────────
THRESHOLD_BLOCK  = 0.70   # → BLOCK  (automatic decline)
THRESHOLD_REVIEW = 0.40   # → REVIEW (send to human analyst)
#                          # < 0.40  → APPROVE


# ─── XGBoost ──────────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train) -> object:
    if not _XGBOOST_AVAILABLE:
        raise RuntimeError(
            "xgboost failed to import (native dependency missing). "
            "On macOS, install the OpenMP runtime first (e.g. `brew install libomp`) "
            "and ensure it’s on your dynamic linker path, then retry training."
        ) from _XGBOOST_IMPORT_ERROR
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators    = 500,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        scale_pos_weight= spw,
        eval_metric     = "aucpr",
        tree_method     = "hist",
        random_state    = 42,
        n_jobs          = -1,
    )
    model.fit(X_train, y_train, verbose=False)
    joblib.dump(model, XGB_PATH)
    print(f"[xgb]  trained  (scale_pos_weight={spw:.1f}) → {XGB_PATH}")
    return model


# ─── Random Forest (second classifier for ensemble) ───────────────────────────
def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators = 300,
        max_depth    = 12,
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, RF_PATH)
    print(f"[rf]   trained → {RF_PATH}")
    return model


# ─── Isolation Forest ─────────────────────────────────────────────────────────
def train_isolation_forest(X_train, y_train) -> IsolationForest:
    """
    Train on LEGITIMATE transactions only.
    Anything that looks unlike legit transactions → flagged as anomaly.
    """
    X_legit = X_train[y_train == 0]
    fraud_rate = float((y_train == 1).mean())

    model = IsolationForest(
        n_estimators  = 300,
        contamination = max(fraud_rate, 0.005),
        random_state  = 42,
        n_jobs        = -1,
    )
    model.fit(X_legit)
    joblib.dump(model, ISO_PATH)
    print(f"[iso]  trained on {len(X_legit):,} legit transactions → {ISO_PATH}")
    return model


# ─── Ensemble scoring ─────────────────────────────────────────────────────────
def ensemble_score(
    xgb_model,
    iso_model,
    X,
    xgb_weight: float = 0.70,
    iso_weight:  float = 0.30,
) -> np.ndarray:
    """
    Combine XGBoost probability and Isolation Forest anomaly score
    into a single fraud probability [0, 1].

    Isolation Forest returns scores in (−∞, +∞):
      very negative → anomaly   →  maps to high fraud probability
      around 0      → borderline
      positive      → normal    →  maps to low fraud probability
    """
    xgb_prob  = xgb_model.predict_proba(X)[:, 1]

    # Normalise IF score to [0, 1]  (invert sign — more negative = more anomalous)
    iso_raw   = iso_model.decision_function(X)
    iso_score = 1 / (1 + np.exp(iso_raw * 5))   # sigmoid inversion

    combined  = xgb_weight * xgb_prob + iso_weight * iso_score
    return np.clip(combined, 0, 1)


def make_decision(score: float) -> str:
    """Convert fraud probability → 3-way decision."""
    if score >= THRESHOLD_BLOCK:
        return "BLOCK"
    elif score >= THRESHOLD_REVIEW:
        return "REVIEW"
    else:
        return "APPROVE"


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(xgb_model, iso_model, X_test, y_test, feature_names=None) -> dict:
    scores = ensemble_score(xgb_model, iso_model, X_test)
    preds  = (scores >= THRESHOLD_BLOCK).astype(int)

    roc    = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    cm     = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y_test, preds,
                                   target_names=["Legitimate", "Fraud"],
                                   output_dict=True)

    metrics = {
        "roc_auc":   round(roc,    4),
        "pr_auc":    round(pr_auc, 4),
        "f1_fraud":  round(report["Fraud"]["f1-score"], 4),
        "precision_fraud": round(report["Fraud"]["precision"], 4),
        "recall_fraud":    round(report["Fraud"]["recall"],    4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "false_negative_rate": round(fn / max(fn + tp, 1), 4),
        "true_positives":  int(tp),
        "false_negatives": int(fn),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
    }

    # Save metrics
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print
    print("\n" + "="*60)
    print("  ENSEMBLE EVALUATION REPORT")
    print("="*60)
    print(f"  ROC-AUC               : {metrics['roc_auc']}")
    print(f"  Precision-Recall AUC  : {metrics['pr_auc']}  ← primary metric")
    print(f"  F1  (Fraud class)     : {metrics['f1_fraud']}")
    print(f"  Precision (Fraud)     : {metrics['precision_fraud']}")
    print(f"  Recall    (Fraud)     : {metrics['recall_fraud']}")
    print(f"  False Positive Rate   : {metrics['false_positive_rate']*100:.2f}%")
    print(f"  False Negative Rate   : {metrics['false_negative_rate']*100:.2f}%")
    print("="*60)
    print(f"  Fraud caught  : {tp:,} / {tp+fn:,}")
    print(f"  Fraud missed  : {fn:,}")
    print(f"  False alarms  : {fp:,}")
    print("="*60)

    _save_plots(xgb_model, iso_model, X_test, y_test, scores, feature_names)
    return metrics


def _save_plots(xgb_model, iso_model, X_test, y_test, scores, feature_names=None):
    # ROC + PR curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    RocCurveDisplay.from_predictions(y_test, scores, ax=axes[0], name="Ensemble")
    axes[0].set_title("ROC Curve")
    PrecisionRecallDisplay.from_predictions(y_test, scores, ax=axes[1], name="Ensemble")
    axes[1].set_title("Precision-Recall Curve  ← primary metric")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eval_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fraud score distribution
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores[y_test == 0], bins=80, alpha=0.6, label="Legitimate", color="#378ADD")
    ax.hist(scores[y_test == 1], bins=80, alpha=0.8, label="Fraud",      color="#E24B4A")
    ax.axvline(THRESHOLD_BLOCK,  color="black",  linestyle="--", label=f"BLOCK >{THRESHOLD_BLOCK}")
    ax.axvline(THRESHOLD_REVIEW, color="#BA7517", linestyle="--", label=f"REVIEW >{THRESHOLD_REVIEW}")
    ax.set_xlabel("Ensemble fraud score"); ax.set_ylabel("Count")
    ax.set_title("Score distribution: fraud vs legitimate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "score_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance (XGBoost only)
    if feature_names and hasattr(xgb_model, "feature_importances_"):
        imp = pd.Series(xgb_model.feature_importances_, index=feature_names)
        top = imp.nlargest(20).sort_values()
        fig, ax = plt.subplots(figsize=(8, 7))
        top.plot(kind="barh", ax=ax, color="#378ADD")
        ax.set_title("Top 20 feature importances (XGBoost)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[plots] saved to {PLOTS_DIR}/")


# ─── Load helpers ─────────────────────────────────────────────────────────────
_model_cache = {}

def load_models():
    if "xgb" not in _model_cache:
        _model_cache["xgb"] = joblib.load(XGB_PATH)
        _model_cache["iso"] = joblib.load(ISO_PATH)
    return _model_cache["xgb"], _model_cache["iso"]
