import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from feature_engineering import (
    full_preprocessing_pipeline,
    get_consumption_columns,
    FEATURE_ORDER,
)

st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MAX_ROWS_SHOWN = 80  # kept only as a fallback constant, not used to slice tables anymore


# ---------------------------------------------------------------------------
# Cached heavy computations — without this, Streamlit re-runs the full
# preprocessing + prediction pipeline on EVERY interaction (selectbox change,
# scrolling, etc.), which is what makes the app freeze on large CSVs.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Processing consumer data...")
def run_pipeline(df):
    return full_preprocessing_pipeline(df, threshold=50.0)


@st.cache_data(show_spinner="Running predictions...")
def run_predictions(_model, features_df):
    preds = _model.predict(features_df[FEATURE_ORDER])
    probs = _model.predict_proba(features_df[FEATURE_ORDER])[:, 1]
    return preds, probs

# ---------------------------------------------------------------------------
# Page state: "landing" (Get Started screen) -> "app" (main dashboard)
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"


# ---------------------------------------------------------------------------
# LANDING PAGE — dark blue
# ---------------------------------------------------------------------------
if st.session_state.page == "landing":

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent !important; }

    .stApp {
        background: linear-gradient(160deg, #071633 0%, #0b2149 55%, #0f2a5c 100%);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .landing-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 78vh;
        padding-top: 4vh;
    }

    .landing-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #7fb3ff;
        border: 1px solid rgba(127,179,255,0.35);
        padding: 6px 16px;
        border-radius: 20px;
        margin-bottom: 1.8rem;
    }

    .landing-title {
        font-size: 3.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        line-height: 1.1;
        margin-bottom: 0.8rem;
    }

    .landing-title span { color: #4f9dff; }

    .landing-subtitle {
        font-size: 1.05rem;
        color: #b7c8ea;
        max-width: 560px;
        margin-bottom: 2.6rem;
        line-height: 1.6;
    }

    /* Style the Streamlit button to fit the dark hero */
    div[data-testid="stButton"] {
        display: flex;
        justify-content: center;
    }

    div[data-testid="stButton"] > button {
        background: #2563eb;
        color: #ffffff;
        border: none;
        padding: 0.85rem 2.6rem;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        box-shadow: 0 8px 24px rgba(37,99,235,0.45);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(37,99,235,0.6);
        color: #ffffff;
        border: none;
    }

    .landing-footer {
        margin-top: 3rem;
        color: rgba(183,200,234,0.4);
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.1em;
    }
    </style>

    <div class="landing-wrap">
        <div class="landing-badge">Smart Grid Analytics</div>
        <div class="landing-title">Grid<span>Guard</span></div>
        <div class="landing-subtitle">
            AI-powered electricity theft detection. Upload consumer consumption data
            and instantly flag suspicious usage patterns across your grid.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Get Started →", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()

    st.markdown('<div class="landing-footer" style="text-align:center">GRIDGUARD · ELECTRICITY THEFT INTELLIGENCE</div>', unsafe_allow_html=True)

    st.stop()


# ---------------------------------------------------------------------------
# MAIN APP PAGE — light blue
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp {
    background: linear-gradient(180deg, #eaf3ff 0%, #dceeff 100%);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #0f172a;
}

.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(15,60,120,0.18);
    margin-bottom: 2rem;
}

.top-logo {
    font-size: 1.3rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
}

.top-logo span { color: #2563eb; }

.section-label {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    color: #3d5a80;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(15,60,120,0.15);
}

.stat-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(15,60,120,0.12);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #0f172a;
    line-height: 1;
}

.stat-number.danger { color: #dc2626; }
.stat-number.safe   { color: #16a34a; }
.stat-number.info   { color: #2563eb; }

.stat-label {
    font-size: 0.72rem;
    color: #3d5a80;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

.result-pill-safe {
    display: inline-block;
    background: rgba(22,163,74,0.12);
    border: 1px solid rgba(22,163,74,0.4);
    color: #15803d;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
}

.result-pill-danger {
    display: inline-block;
    background: rgba(220,38,38,0.1);
    border: 1px solid rgba(220,38,38,0.35);
    color: #dc2626;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }
</style>

<div class="top-bar">
    <div class="top-logo">Grid<span>Guard</span></div>
</div>
""", unsafe_allow_html=True)

if not os.path.exists("electricity_theft_model.joblib"):
    st.error("Model file not found! Please run train_model.py first.")
    st.stop()

model = joblib.load("electricity_theft_model.joblib")

st.markdown('<div class="section-label">01 — Upload Consumer Data</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])

if uploaded_file is None:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.9);border:1px dashed rgba(37,99,235,0.3);
                border-radius:12px;padding:2rem;text-align:center;
                color:#3d5a80;font-size:0.9rem;margin-top:1rem;">
        Upload <strong style="color:#2563eb">data.csv</strong> to begin analysis<br>
        <small>Format: CONS_NO · FLAG · Daily consumption columns</small>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(uploaded_file)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">02 — Raw Data Preview</div>', unsafe_allow_html=True)
st.dataframe(df, use_container_width=True, height=350)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-number info">{df.shape[0]:,}</div><div class="stat-label">Total Consumers</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{df.shape[1]:,}</div><div class="stat-label">Total Columns</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-number danger">{int(df.isnull().sum().sum()):,}</div><div class="stat-label">Missing Values</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">03 — Data Cleaning (Removing Consumers With &gt;50% Missing Data)</div>', unsafe_allow_html=True)

df_clean, removed_df, missing_percentage, features_df = run_pipeline(df)

# Reset row numbers so the cleaned preview shows 0, 1, 2... instead of the
# original file's row indices (which "skip" over the rows that were dropped).
df_clean = df_clean.reset_index(drop=True)
features_df = features_df.reset_index(drop=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{df.shape[0]:,}</div><div class="stat-label">Original Consumers</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-number danger">{removed_df.shape[0]:,}</div><div class="stat-label">Removed (&gt;50% Missing)</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-number safe">{df_clean.shape[0]:,}</div><div class="stat-label">Remaining Consumers</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Cleaned Data Preview**")
st.dataframe(df_clean, use_container_width=True, height=350)

predictions, probabilities = run_predictions(model, features_df)

total = len(predictions)
theft_count = int(sum(predictions))
normal_count = total - theft_count
theft_pct = theft_count / total * 100 if total > 0 else 0

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">04 — Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-number info">{total:,}</div><div class="stat-label">Total Consumers</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-number safe">{normal_count:,}</div><div class="stat-label">Normal</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-number danger">{theft_count:,}</div><div class="stat-label">Suspected Theft</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-card"><div class="stat-number danger">{theft_pct:.1f}%</div><div class="stat-label">Theft Rate</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_chart, col_table = st.columns([1, 2])

with col_chart:
    st.markdown('<div class="section-label">05 — Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
    ax.pie(
        [normal_count, theft_count],
        labels=["Normal", "Theft"],
        colors=["#16a34a", "#dc2626"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2),
        textprops=dict(color="#334155", fontsize=10)
    )
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    st.pyplot(fig)

with col_table:
    st.markdown('<div class="section-label">06 — All Results</div>', unsafe_allow_html=True)
    results_df = pd.DataFrame()
    if "CONS_NO" in df_clean.columns:
        results_df["Consumer ID"] = df_clean["CONS_NO"].values
    results_df["Status"] = ["🔴 Theft" if p == 1 else "🟢 Normal" for p in predictions]
    results_df["Theft Risk %"] = [f"{p*100:.1f}%" for p in probabilities]
    results_df["Avg kWh/day"] = features_df["avg_consumption"].round(2).values
    results_df["Zero Days"] = features_df["zero_days"].astype(int).values
    st.dataframe(results_df, use_container_width=True, height=300)

st.markdown("<hr style='border:none;border-top:1px solid rgba(15,60,120,0.15);margin:2rem 0'>", unsafe_allow_html=True)
st.markdown('<div class="section-label">07 — Individual Consumer</div>', unsafe_allow_html=True)

if "CONS_NO" in df_clean.columns:
    id_list = df_clean["CONS_NO"].astype(str).tolist()
    selected = st.selectbox("Select Consumer ID", id_list)
    pos = id_list.index(selected)
else:
    pos = st.number_input("Select row", min_value=0, max_value=len(df_clean) - 1, value=0)
    selected = f"Row {pos}"

row_index = df_clean.index[pos]
pred_label = predictions[pos]
pred_prob = probabilities[pos]

pill = (
    f'<span class="result-pill-danger">🔴 Suspected Theft — {pred_prob*100:.1f}% risk</span>'
    if pred_label == 1
    else
    f'<span class="result-pill-safe">🟢 Normal Consumer — {pred_prob*100:.1f}% risk</span>'
)
st.markdown(f"**Consumer:** `{selected}` &nbsp;&nbsp; {pill}", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

consumption_cols = get_consumption_columns(df_clean)
consumer_data = df_clean.loc[row_index, consumption_cols].values.astype(float)
color = "#dc2626" if pred_label == 1 else "#16a34a"

spine_color = (15/255, 60/255, 120/255, 0.25)
grid_color = (15/255, 60/255, 120/255, 0.12)

fig2, ax2 = plt.subplots(figsize=(12, 3.5), facecolor='none')
ax2.fill_between(range(len(consumer_data)), consumer_data, alpha=0.15, color=color)
ax2.plot(consumer_data, color=color, linewidth=1.5)
ax2.set_facecolor('none')
fig2.patch.set_alpha(0)
ax2.tick_params(colors='#334155', labelsize=8)
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax2.spines[spine].set_color(spine_color)
ax2.set_xlabel("Day", color='#334155', fontsize=9)
ax2.set_ylabel("kWh", color='#334155', fontsize=9)
ax2.set_title(f"Daily Consumption — {selected}", color='#0f172a', fontsize=10, pad=12)
ax2.grid(axis='y', color=grid_color, linewidth=0.5)
st.pyplot(fig2)

f1, f2, f3, f4 = st.columns(4)
row = features_df.loc[row_index]
for col, label, val in [
    (f1, "Avg kWh/day", f"{row['avg_consumption']:.2f}"),
    (f2, "Peak kWh", f"{row['max_consumption']:.2f}"),
    (f3, "Zero Days", f"{int(row['zero_days'])}"),
    (f4, "Missing Days", f"{int(row['missing_days'])}"),
]:
    col.markdown(f'<div class="stat-card" style="margin-top:1rem"><div class="stat-number" style="font-size:1.4rem">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prediction Explanation — compares this consumer's feature values against
# the dataset's own distribution (percentile rank within features_df) rather
# than any fixed/invented threshold. This keeps the logic valid no matter
# what dataset gets uploaded, since "high" or "low" is always relative to
# that dataset's own mean, median, and quartiles.
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border:none;border-top:1px solid rgba(15,60,120,0.15);margin:2rem 0'>", unsafe_allow_html=True)
st.markdown('<div class="section-label">08 — Prediction Explanation</div>', unsafe_allow_html=True)
st.markdown(f"**Why was `{selected}` predicted as {'Theft' if pred_label == 1 else 'Normal'}?**")

# percentile rank of this consumer's value for each feature, within the
# full cleaned dataset (0 = lowest in dataset, 100 = highest in dataset)
percentiles = {
    feat: (features_df[feat] < row[feat]).mean() * 100
    for feat in FEATURE_ORDER
}

# feature -> (display name, value formatter, short interpretive tag)
explanation_rows = [
    ("std_consumption", "Standard Deviation", lambda v: f"{v:.1f} kWh"),
    ("missing_days", "Missing Days", lambda v: f"{int(v)}"),
    ("range_consumption", "Consumption Range", lambda v: f"{v:.1f} kWh"),
    ("peak_to_avg_ratio", "Peak-to-Average Ratio", lambda v: f"{v:.1f}"),
    ("zero_days", "Zero Days", lambda v: f"{int(v)}"),
]

table_data = []
for feat, display_name, fmt in explanation_rows:
    if feat not in percentiles:
        continue
    val = row[feat]
    pct = percentiles[feat]
    if pct >= 50:
        interpretation = f"Higher than {pct:.0f}% of consumers"
    else:
        interpretation = f"Lower than {100 - pct:.0f}% of consumers"
    table_data.append({
        "Feature": display_name,
        "Value": fmt(val),
        "Interpretation": interpretation
    })

explanation_table = pd.DataFrame(table_data)
st.dataframe(explanation_table, use_container_width=True, hide_index=True)

st.caption(
    "Note: these observations describe how this consumer's feature values compare to the "
    "dataset's own mean, median, and quartiles — they are not standalone rules. The Random "
    "Forest model makes its final decision by combining all engineered features together, "
    "not from any single feature crossing a fixed threshold."
)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:rgba(15,60,120,0.4);font-size:0.72rem;font-family:monospace">GRIDGUARD · ELECTRICITY THEFT INTELLIGENCE</div>', unsafe_allow_html=True)
