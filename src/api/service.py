"""
src/api/main.py
────────────────
FastAPI endpoint — simulates a production fraud detection API.

"Our system operates on a 100ms budget. When a transaction hits the API,
the system enriches the data with behavioral history, runs it through a
dual-model ensemble, and returns a Block/Review/Approve decision before
the user even sees the loading spinner."

Endpoints
─────────
  POST /predict/transaction   → fraud score + decision
  POST /predict/message       → SMS/email phishing classification
  GET  /health                → service health
  GET  /metrics               → model performance metrics

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/predict/transaction \
      -H "Content-Type: application/json" \
      -d '{"account_id":"ACC001","dest_account":"ACC999","amount":85000,"hour":2}'
"""

import os
import sys
import time
import json
import asyncio
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.features.engineer import enrich_transaction, classify_message
from src.models.ensemble import load_models, ensemble_score, make_decision

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection — 100ms budget",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "metrics.json")


# ─── Request / Response schemas ───────────────────────────────────────────────
class TransactionRequest(BaseModel):
    account_id:   str           = Field(..., example="ACC001")
    dest_account: str           = Field(..., example="ACC999")
    amount:       float         = Field(..., gt=0, example=85000.0)
    hour:         Optional[int] = Field(None, ge=0, le=23, example=2)
    timestamp:    Optional[str] = Field(None, example="2024-03-15T02:14:00")

    # Optional BAF/PaySim features — enriched server-side if not provided
    income:                      Optional[float] = None
    name_email_similarity:       Optional[float] = None
    customer_age:                Optional[int]   = None
    foreign_request:             Optional[int]   = None
    device_fraud_count:          Optional[int]   = None
    session_length_in_minutes:   Optional[float] = None


class TransactionResponse(BaseModel):
    transaction_id:   str
    decision:         str        # BLOCK | REVIEW | APPROVE
    fraud_score:      float      # 0.0 – 1.0
    xgb_score:        float
    anomaly_score:    float
    processing_ms:    float
    risk_flags:       list[str]
    timestamp:        str


class MessageRequest(BaseModel):
    text:       str = Field(..., example="URGENT: Your KYC is blocked. Click here now.")
    account_id: Optional[str] = None


class MessageResponse(BaseModel):
    label:          str          # phishing | suspicious | legitimate
    confidence:     float
    urgency_score:  float
    risk_keywords:  list[str]
    highlighted:    str
    processing_ms:  float


# ─── Inference helpers ────────────────────────────────────────────────────────
_txn_counter = 0

def _build_feature_vector(enriched: dict, feature_names: list) -> pd.DataFrame:
    """Build a single-row DataFrame aligned to the trained model's feature space."""
    row = {f: enriched.get(f, 0.0) for f in feature_names}
    return pd.DataFrame([row])


def _extract_risk_flags(enriched: dict, score: float) -> list[str]:
    flags = []
    if enriched.get("is_night_txn"):
        flags.append(f"Transaction at {enriched.get('hour_of_day', '?')}:00 (high-risk hour)")
    if enriched.get("amount_is_large"):
        flags.append(f"Large amount: ₹{enriched.get('amount', 0):,.0f}")
    if enriched.get("txn_count_1h", 0) > 5:
        flags.append(f"High velocity: {enriched['txn_count_1h']} transactions in last hour")
    if enriched.get("is_new_recipient"):
        flags.append("First-ever transaction to this recipient")
    if enriched.get("mule_score", 0) > 0.5:
        flags.append(f"Destination account may be a mule (score: {enriched['mule_score']:.2f})")
    if enriched.get("amount_zscore", 0) > 3:
        flags.append(f"Amount is {enriched['amount_zscore']:.1f}σ above user average")
    if enriched.get("foreign_request") == 1:
        flags.append("Foreign/international request")
    if score >= 0.70:
        flags.append(f"Ensemble fraud score: {score*100:.1f}%")
    return flags


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/metrics")
async def get_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"message": "Run train.py first to generate model metrics."}


@app.post("/predict/transaction", response_model=TransactionResponse)
async def predict_transaction(req: TransactionRequest):
    global _txn_counter
    t0 = time.perf_counter()

    # 1. Enrich with velocity / behavioral / network features
    txn_dict = req.model_dump()
    if req.timestamp:
        txn_dict["timestamp"] = datetime.fromisoformat(req.timestamp)
    elif req.hour is not None:
        # Construct a timestamp using provided hour
        now = datetime.now()
        txn_dict["timestamp"] = now.replace(hour=req.hour, minute=0, second=0)
    else:
        txn_dict["timestamp"] = datetime.now()

    enriched = enrich_transaction(txn_dict)

    # 2. Load models + score
    try:
        xgb, iso = load_models()
        # Build minimal feature vector from available enriched fields
        feature_cols = [
            "amount", "amount_log", "amount_is_large",
            "txn_count_5m", "txn_count_15m", "txn_count_1h", "txn_count_6h",
            "txn_amount_1h",
            "amount_vs_mean_ratio", "amount_zscore", "user_txn_history_len",
            "dest_unique_senders", "pair_history_count", "is_new_recipient", "mule_score",
            "hour_of_day", "is_night_txn", "is_weekend",
            "income", "customer_age", "foreign_request",
            "device_fraud_count", "session_length_in_minutes",
            "name_email_similarity",
        ]
        row = {f: enriched.get(f, 0.0) for f in feature_cols}
        X = pd.DataFrame([row])

        xgb_score = float(xgb.predict_proba(X)[0, 1])
        iso_raw   = float(iso.decision_function(X)[0])
        iso_score = float(1 / (1 + np.exp(iso_raw * 5)))
        fraud_score = 0.70 * xgb_score + 0.30 * iso_score

    except Exception as e:
        # Fallback rule-based scoring if model not yet trained
        fraud_score = min(
            (enriched.get("is_night_txn", 0)      * 0.25) +
            (enriched.get("amount_is_large", 0)   * 0.20) +
            (enriched.get("is_new_recipient", 0)  * 0.15) +
            (enriched.get("mule_score", 0)        * 0.20) +
            (min(enriched.get("txn_count_1h", 0) / 10, 1) * 0.20),
            0.98
        )
        xgb_score = fraud_score
        iso_score = 0.0

    decision   = make_decision(fraud_score)
    risk_flags = _extract_risk_flags(enriched, fraud_score)

    _txn_counter += 1
    processing_ms = (time.perf_counter() - t0) * 1000

    return TransactionResponse(
        transaction_id  = f"TXN-{_txn_counter:06d}",
        decision        = decision,
        fraud_score     = round(fraud_score, 4),
        xgb_score       = round(xgb_score, 4),
        anomaly_score   = round(iso_score, 4),
        processing_ms   = round(processing_ms, 2),
        risk_flags      = risk_flags,
        timestamp       = datetime.now().isoformat(),
    )


@app.post("/predict/message", response_model=MessageResponse)
async def predict_message(req: MessageRequest):
    t0 = time.perf_counter()
    result = classify_message(req.text)
    return MessageResponse(
        **result,
        processing_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


# ─── Batch simulation endpoint (used by Streamlit live stream) ───────────────
@app.get("/simulate/stream")
async def simulate_stream(n: int = 10):
    """Generate n synthetic transactions for the live dashboard feed."""
    import random

    scenarios = [
        {"amount": 150,    "hour": 14, "account_id": "ACC001", "dest_account": "MER001"},
        {"amount": 85000,  "hour": 2,  "account_id": "ACC002", "dest_account": "ACC999", "foreign_request": 1},
        {"amount": 500,    "hour": 9,  "account_id": "ACC003", "dest_account": "MER002"},
        {"amount": 12000,  "hour": 3,  "account_id": "ACC004", "dest_account": "ACC888"},
        {"amount": 45,     "hour": 12, "account_id": "ACC005", "dest_account": "MER003"},
        {"amount": 250000, "hour": 1,  "account_id": "ACC006", "dest_account": "ACC777", "foreign_request": 1},
        {"amount": 1200,   "hour": 18, "account_id": "ACC007", "dest_account": "MER004"},
        {"amount": 75000,  "hour": 4,  "account_id": "ACC008", "dest_account": "ACC666"},
    ]

    results = []
    for _ in range(n):
        s = random.choice(scenarios).copy()
        # Add slight noise
        s["amount"] = s["amount"] * random.uniform(0.8, 1.2)
        req = TransactionRequest(**s)
        result = await predict_transaction(req)
        results.append(result.model_dump())
        await asyncio.sleep(0.01)

    return {"transactions": results, "count": len(results)}
