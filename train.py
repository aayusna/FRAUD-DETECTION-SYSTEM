"""
train.py
─────────
Orchestrates the full training run:
  1. Preprocess + feature engineering
  2. Train XGBoost + Isolation Forest
  3. Evaluate (PR-AUC, ROC-AUC, confusion matrix)
  4. Save all artefacts

Run:
    python train.py
    python train.py --strategy smote       # SMOTE only
    python train.py --strategy undersample # fast, loses data
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import run_pipeline
from src.models.ensemble import (
    train_xgboost,
    train_isolation_forest,
    evaluate,
    METRICS_JSON,
)


def main(strategy: str = "combined"):
    print("=" * 60)
    print("  FRAUD DETECTION SYSTEM — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Data
    print("\n▶ Phase 1: Data preparation")
    X_train, X_test, y_train, y_test, feature_names = run_pipeline(
        imbalance_strategy=strategy
    )

    # 2. Models
    print("\n▶ Phase 2: Model training")
    xgb = train_xgboost(X_train, y_train)
    iso = train_isolation_forest(X_train, y_train)

    # 3. Evaluate
    print("\n▶ Phase 3: Evaluation")
    metrics = evaluate(xgb, iso, X_test, y_test, feature_names=feature_names)

    print(f"\n✓ Training complete.")
    print(f"  PR-AUC  : {metrics['pr_auc']}")
    print(f"  ROC-AUC : {metrics['roc_auc']}")
    print(f"  Metrics saved → {METRICS_JSON}")
    print(f"\nNext step:  streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["smote", "undersample", "combined"],
        default="combined",
        help="Imbalance handling strategy",
    )
    args = parser.parse_args()
    main(strategy=args.strategy)
