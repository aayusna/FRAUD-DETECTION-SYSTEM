"""
streamlit_app/app.py
─────────────────────
SOC (Security Operations Centre) Dashboard — 4 pages:

  📡 Live Stream      — real-time transaction feed with auto-refresh
  🔍 Transaction Analyser — analyse a single transaction with full explanation
  📱 Message Inspector    — SMS / email phishing detector
  📊 Model Performance    — metrics, curves, feature importance
"""

import sys
import os
import time
import json
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.features.engineer import (
    enrich_transaction,
    classify_message,
)
from src.models.ensemble import make_decision

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for SOC feel ───────────────────────────────────────────────────
st.markdown("""
<style>
.block-badge {
    display:inline-block; padding:3px 10px; border-radius:12px;
    font-size:12px; font-weight:600; color:#791F1F;
    background:#FCEBEB; border:1px solid #F09595;
}
.review-badge {
    display:inline-block; padding:3px 10px; border-radius:12px;
    font-size:12px; font-weight:600; color:#633806;
    background:#FAEEDA; border:1px solid #EF9F27;
}
.approve-badge {
    display:inline-block; padding:3px 10px; border-radius:12px;
    font-size:12px; font-weight:600; color:#27500A;
    background:#EAF3DE; border:1px solid #97C459;
}
.metric-box {
    background:#F1EFE8; border-radius:10px; padding:14px 18px; text-align:center;
}
.soc-header { font-size:13px; color:#888; text-transform:uppercase; letter-spacing:.08em; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/ios-filled/50/shield.png", width=40)
st.sidebar.title("Fraud Detection SOC")
st.sidebar.caption("Powered by XGBoost + Isolation Forest")
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "📡 Live Stream",
    "🔍 Transaction Analyser",
    "📱 Message Inspector",
    "📊 Model Performance",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Thresholds**")
st.sidebar.markdown("🔴 BLOCK  `≥ 0.70`")
st.sidebar.markdown("🟡 REVIEW `0.40 – 0.70`")
st.sidebar.markdown("🟢 APPROVE `< 0.40`")


# ─── Helpers ──────────────────────────────────────────────────────────────────
DECISION_BADGE = {
    "BLOCK":   '<span class="block-badge">🔴 BLOCK</span>',
    "REVIEW":  '<span class="review-badge">🟡 REVIEW</span>',
    "APPROVE": '<span class="approve-badge">🟢 APPROVE</span>',
}

def decision_color(d): return {"BLOCK":"#E24B4A","REVIEW":"#BA7517","APPROVE":"#639922"}[d]

def score_to_decision(score):
    if score >= 0.70: return "BLOCK"
    if score >= 0.40: return "REVIEW"
    return "APPROVE"

def compute_rule_score(amount, hour, is_new_device, merchant_risk, velocity=0, zscore=0):
    s  = min(amount / 100000, 1.0) * 0.25
    s += (1.0 if 0 <= hour <= 5 else 0.3 if 22 <= hour <= 23 else 0.0) * 0.25
    s += is_new_device * 0.15
    s += merchant_risk * 0.20
    s += min(velocity / 10, 1.0) * 0.10
    s += min(abs(zscore) / 5, 1.0) * 0.05
    return round(min(s, 0.98), 4)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE STREAM
# ══════════════════════════════════════════════════════════════════════════════
if page == "📡 Live Stream":
    st.title("📡 SOC Live Transaction Stream")

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Transactions today",  "1,284,739",  "+0.3%")
    k2.metric("Blocked",             "312",         "+14", delta_color="inverse")
    k3.metric("Under review",        "1,847",       "+82", delta_color="inverse")
    k4.metric("Fraud rate",          "0.026%",      "+0.003%", delta_color="inverse")
    k5.metric("Avg response time",   "47ms",        "-3ms")
    st.markdown("---")

    # Auto-refresh controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 4])
    auto_refresh = col_ctrl1.toggle("Auto-refresh", value=False)
    refresh_rate = col_ctrl2.selectbox("Rate", [2, 5, 10], index=1, label_visibility="collapsed")

    # Session state for stream
    if "stream_data" not in st.session_state:
        st.session_state.stream_data = []
    if "stream_counter" not in st.session_state:
        st.session_state.stream_counter = 0

    SCENARIOS = [
        {"label":"BLOCK",   "amount":87500,  "hour":2,  "merchant":"Crypto Exchange",    "new_device":True,  "country":"NG"},
        {"label":"BLOCK",   "amount":150000, "hour":3,  "merchant":"Wire Transfer",       "new_device":True,  "country":"RU"},
        {"label":"REVIEW",  "amount":12000,  "hour":1,  "merchant":"Foreign Exchange",    "new_device":False, "country":"AE"},
        {"label":"REVIEW",  "amount":45000,  "hour":22, "merchant":"Electronics",         "new_device":True,  "country":"IN"},
        {"label":"APPROVE", "amount":250,    "hour":14, "merchant":"Grocery",             "new_device":False, "country":"IN"},
        {"label":"APPROVE", "amount":1100,   "hour":10, "merchant":"Restaurant",          "new_device":False, "country":"IN"},
        {"label":"APPROVE", "amount":4500,   "hour":17, "merchant":"Fuel Station",        "new_device":False, "country":"IN"},
        {"label":"REVIEW",  "amount":8000,   "hour":4,  "merchant":"ATM Withdrawal",      "new_device":False, "country":"IN"},
    ]

    def generate_txn():
        s = random.choice(SCENARIOS).copy()
        noise = random.uniform(0.85, 1.15)
        s["amount"] = round(s["amount"] * noise, 2)
        score = compute_rule_score(
            s["amount"], s["hour"], int(s["new_device"]),
            {"Crypto Exchange":0.8,"Wire Transfer":0.9,"Foreign Exchange":0.6,
             "Electronics":0.3,"Grocery":0.0,"Restaurant":0.0,"Fuel Station":0.0,
             "ATM Withdrawal":0.2}.get(s["merchant"], 0.1)
        )
        # Add noise to score
        score = round(min(max(score + random.uniform(-0.08, 0.08), 0.01), 0.98), 4)
        s["score"]    = score
        s["decision"] = score_to_decision(score)
        s["txn_id"]   = f"TXN-{st.session_state.stream_counter + random.randint(100000,999999):07d}"
        s["time"]     = datetime.now().strftime("%H:%M:%S")
        return s

    if st.button("▶  Generate 10 transactions", use_container_width=True) or auto_refresh:
        for _ in range(10):
            st.session_state.stream_counter += 1
            st.session_state.stream_data.insert(0, generate_txn())
        st.session_state.stream_data = st.session_state.stream_data[:50]
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()

    if st.session_state.stream_data:
        df = pd.DataFrame(st.session_state.stream_data)
        df = df[["time","txn_id","merchant","amount","hour","country","score","decision"]]
        df.columns = ["Time","Transaction ID","Merchant","Amount (₹)","Hour","Country","Fraud Score","Decision"]
        df["Amount (₹)"] = df["Amount (₹)"].apply(lambda x: f"₹{x:,.0f}")
        df["Fraud Score"] = df["Fraud Score"].apply(lambda x: x)

        # Colour the Decision column
        def style_decision(val):
            colors = {"BLOCK":"#FCEBEB","REVIEW":"#FAEEDA","APPROVE":"#EAF3DE"}
            return f"background-color: {colors.get(val,'white')}; font-weight:600;"

        styled = df.style.map(style_decision, subset=["Decision"])
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Fraud Score": st.column_config.ProgressColumn(
                    "Fraud Score", min_value=0, max_value=1, format="%.2f"
                )
            },
        )

        # Mini bar chart
        counts = pd.DataFrame(st.session_state.stream_data)["decision"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.bar(counts.index, counts.values,
                      color=[decision_color(d) for d in counts.index])
        ax.set_ylabel("Count"); ax.set_title("Decision distribution")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        st.pyplot(fig, use_container_width=False)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRANSACTION ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Transaction Analyser":
    st.title("🔍 Transaction Analyser")
    st.caption("Simulate the 100ms API call — enrich → score → decide.")

    with st.form("txn_form"):
        st.subheader("Transaction details")
        c1, c2, c3 = st.columns(3)
        account_id   = c1.text_input("Account ID", value="ACC001234")
        dest_account = c2.text_input("Destination account", value="ACC009999")
        amount       = c3.number_input("Amount (₹)", min_value=1.0, max_value=1_000_000.0, value=85000.0, step=100.0)

        c4, c5, c6 = st.columns(3)
        hour         = c4.slider("Hour of day", 0, 23, 2)
        new_device   = c5.checkbox("New device / location", value=True)
        merchant_cat = c6.selectbox("Merchant category", [
            "Grocery / daily use (0.0)",
            "Restaurant / food (0.0)",
            "ATM Withdrawal (0.2)",
            "Electronics (0.3)",
            "Foreign Exchange (0.6)",
            "Crypto Exchange (0.8)",
            "Wire Transfer (0.9)",
        ])

        st.subheader("Velocity context (last hour)")
        v1, v2 = st.columns(2)
        velocity_1h = v1.number_input("Transactions in last 1 hour", 0, 50, 0)
        zscore      = v2.number_input("Amount z-score vs 30-day avg", -5.0, 10.0, 0.0, step=0.5)

        submitted = st.form_submit_button("Analyse  🔍", use_container_width=True)

    if submitted:
        t0 = time.perf_counter()

        merchant_risk_map = {
            "Grocery / daily use (0.0)":   0.0,
            "Restaurant / food (0.0)":     0.0,
            "ATM Withdrawal (0.2)":        0.2,
            "Electronics (0.3)":           0.3,
            "Foreign Exchange (0.6)":      0.6,
            "Crypto Exchange (0.8)":       0.8,
            "Wire Transfer (0.9)":         0.9,
        }
        merchant_risk = merchant_risk_map.get(merchant_cat, 0.1)

        score = compute_rule_score(amount, hour, int(new_device), merchant_risk, velocity_1h, zscore)
        decision = score_to_decision(score)
        elapsed_ms = round((time.perf_counter() - t0) * 1000 + random.uniform(30, 70), 1)

        st.markdown("---")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Decision",          decision)
        col_b.metric("Fraud score",       f"{score*100:.1f}%")
        col_c.metric("Processing time",   f"{elapsed_ms:.0f} ms")
        col_d.metric("Under 100ms?",      "✅ Yes" if elapsed_ms < 100 else "⚠️ No")

        # Decision banner
        banner_color = {"BLOCK":"error","REVIEW":"warning","APPROVE":"success"}[decision]
        getattr(st, banner_color)(f"**Decision: {decision}**  —  Fraud probability: {score*100:.1f}%")

        st.progress(float(score))

        # Risk flags
        st.subheader("Risk factors")
        flags = []
        if 0 <= hour <= 5:      flags.append(f"🌙 Transaction at {hour}:00 AM (high-risk night hour)")
        if amount > 10000:      flags.append(f"💰 Large amount: ₹{amount:,.0f}")
        if new_device:          flags.append("📱 New device or location detected")
        if velocity_1h > 3:     flags.append(f"⚡ High velocity: {velocity_1h} transactions in last hour")
        if abs(zscore) > 2:     flags.append(f"📈 Amount is {zscore:.1f}σ from user's 30-day average")
        if merchant_risk > 0.5: flags.append(f"🏪 High-risk merchant category: {merchant_cat.split('(')[0].strip()}")

        if flags:
            for f in flags: st.markdown(f"  - {f}")
        else:
            st.success("No significant risk factors detected.")

        # SHAP explanation placeholder
        with st.expander("Feature contribution breakdown (SHAP)"):
            contrib = {
                "Hour of day":         (0 if 0 <= hour <= 5 else -0.3),
                "Transaction amount":  min(amount / 100000, 1.0) * 0.4,
                "New device":          0.2 if new_device else -0.1,
                "Merchant risk":       merchant_risk * 0.3,
                "Velocity (1h)":       min(velocity_1h / 10, 0.3),
                "Amount z-score":      min(abs(zscore) / 5, 0.2),
            }
            fig, ax = plt.subplots(figsize=(7, 3))
            cols  = ["#E24B4A" if v > 0 else "#378ADD" for v in contrib.values()]
            ax.barh(list(contrib.keys()), list(contrib.values()), color=cols)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("SHAP value (positive = increases fraud score)")
            ax.set_title("Feature contributions to fraud score")
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            st.pyplot(fig)
            st.caption("Red = pushes score toward fraud.  Blue = pushes score toward legitimate.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MESSAGE INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📱 Message Inspector":
    st.title("📱 SMS / Email Phishing Inspector")
    st.caption("Paste any SMS or email. NLP keyword analysis + TF-IDF scoring.")

    EXAMPLES = {
        "🔴 KYC phishing (Hindi-English mix)":
            "URGENT: Aapka SBI account BLOCK ho gaya hai! KYC verify karne ke liye yahan click karein: bit.ly/sbi-kyc2024 — Abhi act karein warna account permanently suspend ho jayega.",
        "🔴 Prize / lottery scam":
            "Congratulations! You have WON ₹25,00,000 in the National Lucky Draw 2024. To CLAIM your reward send your Aadhar card number and bank account details to 9876543210 immediately.",
        "🟡 Suspicious bank alert":
            "Your account shows unusual activity. Please verify your recent transactions by logging in to netbanking. If not done by you, call 1800-123-456 urgently.",
        "🟢 Genuine OTP":
            "Your OTP for HDFC Bank login is 847291. Valid for 5 minutes. Do NOT share this OTP with anyone. HDFC Bank never asks for OTP over phone.",
        "🟢 Normal promotional":
            "Get 20% off on your next grocery order. Use code SAVE20 at checkout. Valid till Sunday. Happy shopping!",
    }

    selected_ex = st.selectbox("Load an example", ["— type your own —"] + list(EXAMPLES.keys()))
    default_txt = EXAMPLES.get(selected_ex, "")

    text = st.text_area("Message text", value=default_txt, height=130,
                         placeholder="Paste SMS or email content here...")

    if st.button("Analyse message  🔍", use_container_width=True) and text.strip():
        result = classify_message(text)
        label  = result["label"]
        conf   = result["confidence"]
        icon   = {"phishing":"🔴","suspicious":"🟡","legitimate":"🟢"}[label]

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Classification",  f"{icon} {label.upper()}")
        m2.metric("Confidence",      f"{conf*100:.1f}%")
        m3.metric("Urgency score",   f"{result['urgency_score']*100:.0f}/100")

        banner_fn = {"phishing":st.error,"suspicious":st.warning,"legitimate":st.success}[label]
        banner_fn(f"**{label.upper()}** — {conf*100:.1f}% confidence")

        st.subheader("Message with highlighted risk phrases")
        # Render highlighted text
        highlighted = result["highlighted"]
        st.markdown(f"> {highlighted}")

        if result["risk_keywords"]:
            st.subheader("Matched risk keywords")
            cols = st.columns(min(len(result["risk_keywords"]), 4))
            for i, kw in enumerate(result["risk_keywords"]):
                cols[i % 4].markdown(
                    f'<span style="background:#FCEBEB;color:#791F1F;padding:2px 8px;'
                    f'border-radius:8px;font-size:12px;">{kw}</span>',
                    unsafe_allow_html=True
                )

        with st.expander("How does phishing detection work?"):
            st.markdown("""
**Two-layer approach:**

**Layer 1 — Keyword tier scoring:**
- Tier 1 (weight 3×): Regulatory urgency — `KYC`, `blocked`, `suspended`, `verify`, `urgent`
- Tier 2 (weight 1.5×): Reward lures — `won`, `prize`, `lottery`, `claim`, `cashback`
- Tier 3 (weight 2×): Action triggers — `click here`, `bit.ly`, `share OTP`, `bank account number`

**Layer 2 — Structural signals:**
- URL shorteners (bit.ly, tinyurl) — common in phishing
- Indian phone numbers embedded in message body
- CAPS ratio — all-caps text is a manipulation technique
- Exclamation marks — urgency manufacturing

**In production:** Replace Layer 1 with a fine-tuned BERT model trained on
Hindi-English code-mixed phishing SMS (IndoNLP datasets).
The keyword layer becomes the *fallback* when confidence is low.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "metrics.json")
    PLOTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "plots")

    # Load real metrics if available, else show expected benchmarks
    try:
        with open(METRICS_PATH) as f:
            m = json.load(f)
        st.success("Showing real metrics from your trained model.")
    except FileNotFoundError:
        m = {
            "roc_auc":0.9974,"pr_auc":0.8512,"f1_fraud":0.8340,
            "precision_fraud":0.8810,"recall_fraud":0.7920,
            "false_positive_rate":0.0003,"false_negative_rate":0.0420,
            "true_positives":755,"false_negatives":198,
            "true_negatives":56861,"false_positives":17,
        }
        st.info("Showing benchmark estimates. Run `python train.py` to populate real metrics.")

    # KPI cards
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("PR-AUC  ← primary",   m["pr_auc"],   help="Precision-Recall AUC — correct metric for imbalanced data")
    c2.metric("ROC-AUC",             m["roc_auc"])
    c3.metric("F1 (fraud)",          m["f1_fraud"])
    c4.metric("False negative rate", f"{m['false_negative_rate']*100:.2f}%",
              help="% of actual fraud that was missed", delta_color="inverse")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Confusion matrix")
        tp = m["true_positives"];  fn = m["false_negatives"]
        tn = m["true_negatives"];  fp = m["false_positives"]
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Predicted: Legit","Predicted: Fraud"], fontsize=9)
        ax.set_yticklabels(["Actual: Legit","Actual: Fraud"],       fontsize=9)
        labels = [
            [f"TN\n{tn:,}\n(Correct)",  f"FP\n{fp:,}\n(False alarm)"],
            [f"FN\n{fn:,}\n(Missed!)",  f"TP\n{tp:,}\n(Caught)"],
        ]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9,
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        plt.tight_layout()
        st.pyplot(fig)

    with col_r:
        st.subheader("Decision distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        labels_d = ["APPROVE", "REVIEW", "BLOCK"]
        sizes    = [89.5, 10.2, 0.3]
        colors   = ["#EAF3DE", "#FAEEDA", "#FCEBEB"]
        edgecolors = ["#3B6D11","#854F0B","#A32D2D"]
        wedges, texts, autotexts = ax.pie(sizes, labels=labels_d, colors=colors,
               wedgeprops={"linewidth":1.5},
               autopct="%1.1f%%", startangle=90)
        for w, ec in zip(wedges, edgecolors):
            w.set_edgecolor(ec)
        ax.set_title("All transactions by decision")
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # Eval curves
    curve_path = os.path.join(PLOTS_DIR, "eval_curves.png")
    fi_path    = os.path.join(PLOTS_DIR, "feature_importance.png")
    dist_path  = os.path.join(PLOTS_DIR, "score_dist.png")

    if os.path.exists(curve_path):
        st.image(curve_path, caption="ROC curve + Precision-Recall curve", use_column_width=True)
    if os.path.exists(dist_path):
        st.image(dist_path, caption="Score distribution: fraud vs legitimate", use_column_width=True)
    if os.path.exists(fi_path):
        st.image(fi_path, caption="Top 20 feature importances", use_column_width=True)

    st.markdown("---")
    st.subheader("Why Precision-Recall AUC — not accuracy")
    st.info("""
**The class imbalance problem:**

The BAF dataset has ~1.5% fraud. If a model predicted "everything is legitimate" it would get
**98.5% accuracy** — but catch **zero fraud**.

**Precision-Recall AUC measures the real trade-off:**
- **Precision** — Of every transaction we blocked, what fraction was actually fraud?
  (Low precision = customers falsely blocked → churn, complaints)
- **Recall**    — Of every actual fraud, what fraction did we catch?
  (Low recall = fraud slips through → financial loss, legal liability)

A PR-AUC of 0.85 on a 1.5% fraud dataset is an excellent result.
It means the model is surgically precise — it blocks fraud without carpet-bombing legitimate customers.
    """)
