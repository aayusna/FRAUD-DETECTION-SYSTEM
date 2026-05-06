"""
preprocess.py
─────────────
Loads the NeurIPS 2022 BAF dataset (or PaySim as fallback).
Handles extreme class imbalance, cleans data, and prepares
train/test splits for the modelling pipeline.

Dataset sources
───────────────
BAF   : https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1

Usage
─────
    from src.preprocess import run_pipeline
    X_train, X_test, y_train, y_test, feature_names = run_pipeline()
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), "..")
DATA_RAW   = os.path.join(ROOT, "data", "raw")
DATA_PROC  = os.path.join(ROOT, "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)

BAF_CSV    = os.path.join(DATA_RAW, "Base.csv")          # BAF main file
PAYSIM_CSV = os.path.join(DATA_RAW, "PS_20174392719_1491204439457_log.csv")
SCALER_PKL = os.path.join(DATA_PROC, "scaler.pkl")
LABEL_PKL  = os.path.join(DATA_PROC, "label_encoders.pkl")


# ─── Dataset-aware loader ─────────────────────────────────────────────────────
def detect_and_load() -> tuple[pd.DataFrame, str]:
    """Auto-detect which dataset is available and load it."""
    if os.path.exists(BAF_CSV):
        print("[load] Detected: NeurIPS 2022 BAF dataset")
        df = pd.read_csv(BAF_CSV, low_memory=False)
        return df, "baf"
    elif os.path.exists(PAYSIM_CSV):
        print("[load] Detected: PaySim Mobile Money dataset")
        df = pd.read_csv(PAYSIM_CSV, low_memory=False)
        return df, "paysim"
    else:
        print("[load] No dataset found — generating synthetic demo data")
        return _generate_synthetic(), "synthetic"


def _generate_synthetic(n: int = 100_000) -> pd.DataFrame:
    """
    Synthetic dataset that mirrors BAF schema.
    Lets the project run end-to-end before the real dataset is downloaded.
    Replace data/raw/Base.csv with the real BAF CSV for production results.
    """
    rng = np.random.default_rng(42)
    n_fraud = int(n * 0.015)         # 1.5% fraud — realistic

    legit = pd.DataFrame({
        "income":               rng.uniform(0, 1, n),
        "name_email_similarity":rng.uniform(0, 1, n),
        "prev_address_months_count": rng.integers(0, 120, n),
        "current_address_months_count": rng.integers(1, 200, n),
        "customer_age":          rng.integers(18, 80, n),
        "days_since_request":    rng.exponential(10, n),
        "intended_balances_amount": rng.exponential(500, n),
        "payment_type":          rng.choice(["AA","AB","AC","AD","AE"], n),
        "employment_status":     rng.choice(["CA","CB","CC","CD","CE","CF"], n),
        "housing_status":        rng.choice(["BA","BB","BC","BD","BE","BF"], n),
        "source":                rng.choice(["INTERNET","TELEAPP"], n),
        "device_os":             rng.choice(["windows","macOS","linux","x11","other"], n),
        "email_is_free":         rng.integers(0, 2, n),
        "phone_home_valid":      rng.integers(0, 2, n),
        "phone_mobile_valid":    rng.integers(0, 2, n),
        "has_other_cards":       rng.integers(0, 2, n),
        "proposed_credit_limit": rng.choice([200,500,1000,2000,5000], n),
        "foreign_request":       rng.integers(0, 2, n),
        "session_length_in_minutes": rng.exponential(8, n),
        "keep_alive_session":    rng.integers(0, 2, n),
        "device_distinct_emails_8w": rng.integers(0, 10, n),
        "device_fraud_count":    rng.integers(0, 2, n),
        "month":                 rng.integers(0, 7, n),
        "fraud_bool":            0,
    })

    # Fraud rows: inject realistic anomalies
    fraud_idx = rng.choice(n, n_fraud, replace=False)
    legit.loc[fraud_idx, "fraud_bool"] = 1
    legit.loc[fraud_idx, "device_fraud_count"] += rng.integers(1, 5, n_fraud)
    legit.loc[fraud_idx, "days_since_request"]  = rng.exponential(0.5, n_fraud)
    legit.loc[fraud_idx, "name_email_similarity"] = rng.uniform(0, 0.1, n_fraud)
    legit.loc[fraud_idx, "foreign_request"] = 1

    return legit


# ─── Dataset normalisation ────────────────────────────────────────────────────
def normalise_baf(df: pd.DataFrame) -> pd.DataFrame:
    """Map BAF column names → unified schema. Drop leakage columns."""
    df = df.copy()
    df = df.rename(columns={"fraud_bool": "label"})
    # BAF already has good features; just drop any direct ID leakage
    drop_cols = [c for c in ["device_id", "ip_address"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"[baf]   {len(df):,} rows  |  fraud rate: {df['label'].mean()*100:.3f}%")
    return df


def normalise_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """Map PaySim column names → unified schema."""
    df = df.copy()
    df = df.rename(columns={"isFraud": "label"})
    # PaySim columns: step, type, amount, nameOrig, oldbalanceOrg, ...
    df["hour"] = df["step"] % 24
    df["day"]  = df["step"] // 24
    # Encode transaction type
    df["payment_type"] = df["type"].astype("category").cat.codes
    # Balance deviation features
    df["orig_balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_log"]          = np.log1p(df["amount"])
    # Flag mule accounts (destination account receives from many senders)
    dest_counts = df["nameDest"].value_counts()
    df["dest_txn_count"]      = df["nameDest"].map(dest_counts)
    keep = ["hour", "day", "amount", "amount_log", "payment_type",
            "orig_balance_change", "dest_balance_change",
            "dest_txn_count", "label"]
    df = df[keep].copy()
    print(f"[paysim] {len(df):,} rows  |  fraud rate: {df['label'].mean()*100:.3f}%")
    return df


def normalise_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"fraud_bool": "label"})
    print(f"[synth] {len(df):,} rows  |  fraud rate: {df['label'].mean()*100:.3f}%")
    return df


# ─── Feature encoding ─────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Label-encode all object columns. Saves encoders for inference."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}

    if not fit:
        encoders = joblib.load(LABEL_PKL)

    for col in cat_cols:
        le = encoders.get(col, LabelEncoder())
        if fit:
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            df[col] = le.transform(df[col].astype(str))

    if fit and encoders:
        joblib.dump(encoders, LABEL_PKL)

    return df


# ─── Scaling ──────────────────────────────────────────────────────────────────
def scale(df: pd.DataFrame, label_col: str = "label", fit: bool = True):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    scaler = StandardScaler()
    if fit:
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PKL)
    else:
        scaler = joblib.load(SCALER_PKL)
        X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), y


# ─── Imbalance handling ───────────────────────────────────────────────────────
def handle_imbalance(X, y, strategy: str = "smote") -> tuple:
    """
    strategy options:
      'smote'      — oversample minority (best for < 1M rows)
      'undersample'— undersample majority (fast, loses data)
      'combined'   — SMOTE then undersampling (balanced approach)
    """
    print(f"[imbalance] before — {dict(y.value_counts())}")

    if strategy == "smote":
        sampler = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sampler.fit_resample(X, y)

    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=42, sampling_strategy=0.1)
        X_res, y_res = sampler.fit_resample(X, y)

    elif strategy == "combined":
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42, sampling_strategy=0.1)),
            ("under", RandomUnderSampler(random_state=42, sampling_strategy=0.5)),
        ])
        X_res, y_res = pipeline.fit_resample(X, y)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"[imbalance] after  — {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


# ─── Full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    imbalance_strategy: str = "combined",
    test_size: float = 0.2,
    random_state: int = 42,
):
    df_raw, dataset_type = detect_and_load()

    normalise = {"baf": normalise_baf, "paysim": normalise_paysim,
                 "synthetic": normalise_synthetic}[dataset_type]
    df = normalise(df_raw)
    df = encode_categoricals(df, fit=True)
    df = df.dropna()

    X, y = scale(df, label_col="label", fit=True)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, y_train = handle_imbalance(X_train, y_train, strategy=imbalance_strategy)

    # Save processed test set for evaluation scripts
    test_df = X_test.copy()
    test_df["label"] = y_test.values
    test_df.to_parquet(os.path.join(DATA_PROC, "test_set.parquet"), index=False)

    print(f"\n[pipeline] ✓  X_train={X_train.shape}  X_test={X_test.shape}")
    print(f"[pipeline]    features: {len(feature_names)}")
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = run_pipeline()
    print("\nTop 10 features:", features[:10])
