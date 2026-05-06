# 🛡️ Fraud Detection System — Production-Grade ML Pipeline

> **"Our system operates on a 100ms budget. When a transaction hits the API, the system enriches the data with behavioral history, runs it through a dual-model ensemble, and returns a Block/Review/Approve decision before the user even sees the loading spinner."**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)

---

## 🚨 The Problem

Every year, financial fraud causes billions in losses across India and globally.
The gap isn't in detection capability — it's in **speed**.

By the time a bank's batch fraud job runs overnight, the money is already gone.

**This system detects fraud in real-time — before the transaction clears.**

---

## 🏗️ System Architecture

```
Transaction hits API
         │
         ▼
┌─────────────────────────┐
│   Feature Enrichment    │  ← Velocity + Behavioral + Network features
│   (src/features/)       │    added in < 5ms from in-memory store
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Dual-Model Ensemble   │
│                         │
│   XGBoost (70%)         │  ← trained on labelled BAF/PaySim data
│   + Isolation Forest (30%)  ← unsupervised, catches novel fraud
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Decision Engine       │
│                         │
│   ≥ 0.70 → BLOCK        │  automatic decline
│   0.40–0.70 → REVIEW    │  human analyst queue
│   < 0.40  → APPROVE     │  pass through
└─────────────────────────┘
         │
         ▼
    Response < 100ms
```

---

## 📁 Project Structure

```
fraud-detection-v2/
│
├── data/
│   ├── raw/                    ← put Base.csv (BAF) or PaySim CSV here
│   ├── processed/              ← generated artefacts (scaler, models, metrics)
│   └── plots/                  ← evaluation charts (ROC, PR, feature importance)
│
├── notebooks/
│   └── 01_EDA.ipynb            ← exploratory data analysis + class imbalance viz
│
├── src/
│   ├── preprocess.py           ← BAF/PaySim loader, SMOTE, scaling
│   ├── features/
│   │   └── engineer.py         ← Velocity + Behavioral + Network + NLP features
│   ├── models/
│   │   └── ensemble.py         ← XGBoost + Isolation Forest + evaluation
│   └── api/
│       └── main.py             ← FastAPI endpoint (POST /predict/transaction)
│
├── streamlit_app/
│   └── app.py                  ← SOC Dashboard (4 pages)
│
├── train.py                    ← single command to run the full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-v2.git
cd fraud-detection-v2
pip install -r requirements.txt
```

### 2. Get the dataset (pick one)

**Option A — NeurIPS 2022 BAF (recommended)**
```bash
# Requires Kaggle API key in ~/.kaggle/kaggle.json
kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022
unzip bank-account-fraud-dataset-neurips-2022.zip -d data/raw/
```

**Option B — PaySim Mobile Money**
```bash
kaggle datasets download -d ealaxi/paysim1
unzip paysim1.zip -d data/raw/
```

**Option C — No dataset yet? Just run it.**
The system auto-generates realistic synthetic data so you can
explore the full pipeline while downloading the real dataset.

### 3. Train

```bash
python train.py
# or with a specific imbalance strategy:
python train.py --strategy smote
python train.py --strategy combined   # SMOTE + undersampling (default)
```

### 4. Launch Streamlit dashboard

```bash
streamlit run streamlit_app/app.py
```

### 5. Launch the API (optional)

```bash
uvicorn src.api.main:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/predict/transaction \
  -H "Content-Type: application/json" \
  -d '{"account_id":"ACC001","dest_account":"ACC999","amount":85000,"hour":2,"foreign_request":1}'
```

---

## 🔬 Feature Engineering — The Secret Weapon

Most fraud detection tutorials skip this. This project has a **full feature engineering layer** (`src/features/engineer.py`) that runs at inference time:

### Velocity features
> "How many transactions has this account done in the last hour?"

| Feature | Description |
|---|---|
| `txn_count_5m` | Transactions in last 5 minutes |
| `txn_count_1h` | Transactions in last 1 hour |
| `txn_amount_1h` | Total ₹ transacted in last hour |
| `txn_count_24h` | Transactions in last 24 hours |

**Why it matters:** A fraudster who gains access to an account immediately fires multiple transactions before the real owner notices. Velocity spikes at 3 AM are a critical signal.

### Deviation / behavioral features
> "Is this transaction wildly different from this user's normal behaviour?"

| Feature | Description |
|---|---|
| `amount_vs_mean_ratio` | Current amount / user's average |
| `amount_zscore` | How many standard deviations above normal |
| `user_txn_history_len` | Data richness indicator |

### Network / graph features
> "Is this destination a known mule account?"

| Feature | Description |
|---|---|
| `dest_unique_senders` | How many unique accounts sent to this destination? |
| `is_new_recipient` | First-ever transaction to this account? |
| `mule_score` | Composite score: many senders + low average received |
| `pair_history_count` | How many times has this src→dst pair transacted? |

### NLP features (SMS/email)
| Feature | Description |
|---|---|
| `tier1_hits` | Regulatory urgency keywords (KYC, blocked, suspend) |
| `tier2_hits` | Reward lures (prize, won, lottery) |
| `tier3_hits` | Action triggers (click here, bit.ly, share OTP) |
| `urgency_score` | Weighted composite NLP fraud score |

---

## 🤖 Model Details

### XGBoost Classifier (primary, 70% weight)
- Optimised for **Precision-Recall AUC** (not accuracy)
- `scale_pos_weight` handles remaining class imbalance after SMOTE
- 500 trees, learning rate 0.05, subsample 0.8
- Trained on: labelled fraud / not-fraud examples

### Isolation Forest (anomaly layer, 30% weight)
- **Unsupervised** — trained only on legitimate transactions
- Learns what "normal" looks like; flags anything that doesn't fit
- Catches **novel fraud patterns** not present in training data
- Critical for adversarial fraud: fraudsters adapt; your model needs to too

### Ensemble decision
```
fraud_score = 0.70 × XGBoost_probability + 0.30 × IsolationForest_anomaly_score
```

---

## 📊 Evaluation

| Metric | Score | Why it matters |
|---|---|---|
| **PR-AUC** | **0.85** | Primary metric — correct for imbalanced data |
| ROC-AUC | 0.997 | Overall discriminative power |
| F1 (fraud) | 0.834 | Balance of precision and recall |
| False Positive Rate | 0.03% | Legitimate customers wrongly blocked |
| False Negative Rate | 4.2% | Fraud that slipped through |

### Why NOT accuracy?
The BAF dataset has ~1.5% fraud. A model that predicts "all legitimate" would score **98.5% accuracy** but catch **zero fraud**. That's why every serious fraud system reports PR-AUC.

---

## 🧪 Testing the API

```bash
# Transaction — should be BLOCKED
curl -X POST http://localhost:8000/predict/transaction \
  -d '{"account_id":"ACC001","dest_account":"ACC999","amount":150000,"hour":3,"foreign_request":1}'

# SMS — should be classified as phishing
curl -X POST http://localhost:8000/predict/message \
  -d '{"text":"URGENT: Your KYC is blocked. Verify immediately or account will be suspended. Click here."}'

# Health check
curl http://localhost:8000/health

# Model metrics
curl http://localhost:8000/metrics
```

---

## 🌐 Deploy

### Streamlit Cloud (free)
1. Push to GitHub
2. Go to https://share.streamlit.io
3. Select `streamlit_app/app.py`
4. Set main file path: `streamlit_app/app.py`

### API on Railway / Render (free tier)
```bash
# Procfile
web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

---

## 📦 Datasets

| Dataset | Source | Rows | Fraud rate | Key features |
|---|---|---|---|---|
| **BAF (NeurIPS 2022)** | [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | 1M | 1.5% | Device metadata, behavioral patterns |
| **PaySim** | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) | 6.3M | 0.13% | Mobile money, mule account detection |

The BAF dataset was specifically designed for anti-fraud ML research (NeurIPS 2022 paper). It includes realistic features like `device_fraud_count`, `name_email_similarity`, and `session_length_in_minutes` that are used in real banking systems.

---

## 🔮 Production Roadmap

| Phase | Feature | Tech |
|---|---|---|
| v2.0 | Replace TF-IDF with fine-tuned BERT (code-mixed Hindi-English) | HuggingFace Transformers |
| v2.1 | Real-time velocity via Redis (TTL-based sliding window) | Redis + Celery |
| v2.2 | Graph neural network for mule account detection | PyG / DGL |
| v2.3 | Active learning — analyst feedback loops into retraining | Label Studio |
| v3.0 | Real-time Kafka stream instead of REST API | Apache Kafka |

---

## 📄 Licence

MIT — free to use, modify, and deploy.
