import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import base64

st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
    }

    .stApp {
        background: linear-gradient(160deg, #020818 0%, #0a1628 40%, #0d2144 100%);
        min-height: 100vh;
    }

    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    .nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.6rem 4rem;
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }

    .nav-logo {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f1f5f9;
        letter-spacing: -0.02em;
    }

    .nav-logo span { color: #3b82f6; }

    .nav-tag {
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(59,130,246,0.25);
        color: #60a5fa;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero {
        min-height: 82vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 3rem 2rem;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: #93c5fd;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 2.2rem;
        backdrop-filter: blur(8px);
    }

    .hero-title {
        font-size: clamp(3rem, 8vw, 6.5rem);
        font-weight: 900;
        color: #ffffff;
        line-height: 1.0;
        letter-spacing: -0.04em;
        margin: 0 0 0.4rem 0;
    }

    .hero-title .blue { color: #3b82f6; }

    .hero-sub {
        font-size: 1.1rem;
        color: #64748b;
        max-width: 480px;
        margin: 1.4rem auto 2.8rem auto;
        line-height: 1.75;
        font-weight: 400;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        color: white !important;
        border: none !important;
        padding: 0.9rem 3rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border-radius: 50px !important;
        box-shadow: 0 8px 30px rgba(59,130,246,0.4) !important;
        letter-spacing: 0.02em !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stButton > button:hover {
        box-shadow: 0 12px 40px rgba(59,130,246,0.6) !important;
        transform: translateY(-2px) !important;
    }

    .stats-strip {
        display: flex;
        justify-content: center;
        gap: 0;
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.6rem 3rem;
        flex-wrap: wrap;
        gap: 2rem;
    }

    .s-item {
        text-align: center;
        padding: 0 2rem;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .s-item:last-child { border-right: none; }

    .s-val {
        font-size: 2rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1;
        letter-spacing: -0.03em;
    }

    .s-lbl {
        font-size: 0.7rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 5px;
        font-weight: 500;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
    </style>

    <div class="nav">
        <div class="nav-logo">Grid<span>Guard</span></div>
        <div class="nav-tag">⚡ AI Powered</div>
    </div>

    <div class="hero">
        <div class="hero-eyebrow">🔒 Smart Grid Security System</div>
        <div class="hero-title">
            Electricity<br>
            <span class="blue">Theft Detection</span>
        </div>
        <p class="hero-sub">
            Upload consumer meter data and let our trained AI model
            instantly identify suspicious usage patterns and flag potential theft.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2.2, 1, 2.2])
    with col2:
        if st.button("Get Started →", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()

    st.markdown("""
    <div style="display:flex;justify-content:center;padding:0 2rem 4rem 2rem;">
        <div class="stats-strip">
            <div class="s-item"><div class="s-val">42K+</div><div class="s-lbl">Consumers Analysed</div></div>
            <div class="s-item"><div class="s-val">84%</div><div class="s-lbl">Model Accuracy</div></div>
            <div class="s-item"><div class="s-val">0.73</div><div class="s-lbl">ROC-AUC Score</div></div>
            <div class="s-item"><div class="s-val">9</div><div class="s-lbl">AI Features</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "app":

    if os.path.exists("bg.webp"):
        img_b64 = get_base64_image("bg.webp")
        bg_css = f"""
        .stApp {{
            background-image: url('data:image/webp;base64,{img_b64}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        bg_css = ".stApp { background: #aed6e8; }"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }}

    {bg_css}

    .block-container {{
        padding: 1.5rem 3rem !important;
        max-width: 1400px !important;
    }}

    .top-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(30,58,95,0.2);
        margin-bottom: 2rem;
    }}

    .top-logo {{
        font-size: 1.3rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
    }}

    .top-logo span {{ color: #2563eb; }}

    .section-label {{
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(30,58,95,0.15);
    }}

    .stat-card {{
        background: rgba(255,255,255,0.55);
        border: 1px solid rgba(255,255,255,0.8);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }}

    .stat-number {{
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #0f172a;
        line-height: 1;
    }}

    .stat-number.danger {{ color: #dc2626; }}
    .stat-number.safe   {{ color: #16a34a; }}
    .stat-number.info   {{ color: #2563eb; }}

    .result-pill-safe {{
        display: inline-block;
        background: rgba(22,163,74,0.12);
        border: 1px solid rgba(22,163,74,0.4);
        color: #15803d;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }}

    .result-pill-danger {{
        display: inline-block;
        background: rgba(220,38,38,0.1);
        border: 1px solid rgba(220,38,38,0.35);
        color: #dc2626;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }}

    [data-testid="stFileUploader"] {{
        background: rgba(255,255,255,0.55) !important;
        border: 1px dashed rgba(37,99,235,0.35) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px) !important;
    }}

    [data-testid="stFileUploader"] section {{
        background: transparent !important;
        border: none !important;
    }}

    [data-testid="stFileUploadDropzone"] {{
        background: rgba(255,255,255,0.4) !important;
        border: 1px dashed rgba(37,99,235,0.3) !important;
        border-radius: 8px !important;
    }}

    [data-testid="stFileUploaderFile"] {{
        background: rgba(255,255,255,0.7) !important;
        border: 1px solid rgba(37,99,235,0.2) !important;
        border-radius: 8px !important;
        color: #0f172a !important;
    }}

    [data-testid="stExpander"] {{
        background: rgba(255,255,255,0.5) !important;
        border: 1px solid rgba(255,255,255,0.7) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
    }}

    [data-testid="stSelectbox"] > div > div {{
        background: rgba(255,255,255,0.6) !important;
        border: 1px solid rgba(37,99,235,0.2) !important;
        color: #0f172a !important;
        border-radius: 8px !important;
    }}

    .stButton > button {{
        background: rgba(255,255,255,0.5) !important;
        border: 1px solid rgba(30,58,95,0.2) !important;
        color: #334155 !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        padding: 0.4rem 1rem !important;
        backdrop-filter: blur(8px) !important;
    }}

    .stButton > button:hover {{
        border-color: #2563eb !important;
        color: #2563eb !important;
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="top-bar">
        <div class="top-logo">Grid<span>Guard</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back to Home"):
        st.session_state.page = "landing"
        st.rerun()

    if not os.path.exists("electricity_theft_model.joblib"):
        st.error("Model file not found! Please run eda.py first.")
        st.stop()

    model = joblib.load("electricity_theft_model.joblib")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 — Upload Consumer Data</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])

    if uploaded_file is None:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.45);border:1px dashed rgba(37,99,235,0.3);
                    border-radius:12px;padding:2rem;text-align:center;
                    color:#64748b;font-size:0.9rem;margin-top:1rem;backdrop-filter:blur(8px)">
            Upload <strong style="color:#2563eb">data.csv</strong> to begin analysis<br>
            <small>Format: CONS_NO · FLAG · Daily consumption columns</small>
        </div>
        """, unsafe_allow_html=True)

    else:
        df = pd.read_csv(uploaded_file)

        with st.expander("Raw Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        consumption = df.drop(columns=["CONS_NO", "FLAG"], errors="ignore")

        features_df = pd.concat([
            consumption.mean(axis=1).rename("avg_consumption"),
            consumption.max(axis=1).rename("max_consumption"),
            consumption.min(axis=1).rename("min_consumption"),
            consumption.std(axis=1).rename("std_consumption"),
            consumption.median(axis=1).rename("median_consumption"),
            (consumption == 0).sum(axis=1).rename("zero_days"),
            consumption.isnull().sum(axis=1).rename("missing_days"),
            (consumption.max(axis=1) - consumption.min(axis=1)).rename("range_consumption"),
        ], axis=1)

        features_df["peak_to_avg_ratio"] = (
            features_df["max_consumption"] / (features_df["avg_consumption"] + 1e-5)
        )

        predictions   = model.predict(features_df)
        probabilities = model.predict_proba(features_df)[:, 1]

        total        = len(predictions)
        theft_count  = int(sum(predictions))
        normal_count = total - theft_count
        theft_pct    = theft_count / total * 100 if total > 0 else 0

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">02 — Overview</div>', unsafe_allow_html=True)

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
            st.markdown('<div class="section-label">03 — Distribution</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="section-label">04 — All Results</div>', unsafe_allow_html=True)
            results_df = pd.DataFrame()
            if "CONS_NO" in df.columns:
                results_df["Consumer ID"] = df["CONS_NO"].values
            results_df["Status"]       = ["🔴 Theft" if p == 1 else "🟢 Normal" for p in predictions]
            results_df["Theft Risk %"] = [f"{p*100:.1f}%" for p in probabilities]
            results_df["Avg kWh/day"]  = features_df["avg_consumption"].round(2).values
            results_df["Zero Days"]    = features_df["zero_days"].astype(int).values
            st.dataframe(results_df, use_container_width=True, height=300)

        st.markdown("<hr style='border:none;border-top:1px solid rgba(30,58,95,0.15);margin:2rem 0'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">05 — Individual Consumer</div>', unsafe_allow_html=True)

        if "CONS_NO" in df.columns:
            selected = st.selectbox("Select Consumer ID", df["CONS_NO"].astype(str).tolist())
            idx = df[df["CONS_NO"].astype(str) == selected].index[0]
        else:
            idx = st.number_input("Select row", min_value=0, max_value=len(df)-1, value=0)
            selected = f"Row {idx}"

        pred_label = predictions[idx]
        pred_prob  = probabilities[idx]

        pill = (
            f'<span class="result-pill-danger">🔴 Suspected Theft — {pred_prob*100:.1f}% risk</span>'
            if pred_label == 1
            else
            f'<span class="result-pill-safe">🟢 Normal Consumer — {pred_prob*100:.1f}% risk</span>'
        )
        st.markdown(f"**Consumer:** `{selected}` &nbsp;&nbsp; {pill}", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        consumption_cols = [c for c in df.columns if c not in ["CONS_NO", "FLAG"]]
        consumer_data    = df.loc[idx, consumption_cols].values.astype(float)
        color            = "#dc2626" if pred_label == 1 else "#16a34a"

        # ✅ FIXED: replaced CSS rgba() strings with matplotlib-compatible tuples
        spine_color = (30/255, 58/255, 95/255, 0.2)
        grid_color  = (30/255, 58/255, 95/255, 0.1)

        fig2, ax2 = plt.subplots(figsize=(12, 3.5), facecolor='none')
        ax2.fill_between(range(len(consumer_data)), consumer_data, alpha=0.15, color=color)
        ax2.plot(consumer_data, color=color, linewidth=1.5)
        ax2.set_facecolor('none')
        fig2.patch.set_alpha(0)
        ax2.tick_params(colors='#475569', labelsize=8)
        for spine in ['top', 'right']:
            ax2.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax2.spines[spine].set_color(spine_color)  # ✅ fixed
        ax2.set_xlabel("Day", color='#64748b', fontsize=9)
        ax2.set_ylabel("kWh", color='#64748b', fontsize=9)
        ax2.set_title(f"Daily Consumption — {selected}", color='#334155', fontsize=10, pad=12)
        ax2.grid(axis='y', color=grid_color, linewidth=0.5)  # ✅ fixed
        st.pyplot(fig2)

        f1, f2, f3, f4 = st.columns(4)
        row = features_df.iloc[idx]
        for col, label, val in [
            (f1, "Avg kWh/day",  f"{row['avg_consumption']:.2f}"),
            (f2, "Peak kWh",     f"{row['max_consumption']:.2f}"),
            (f3, "Zero Days",    f"{int(row['zero_days'])}"),
            (f4, "Missing Days", f"{int(row['missing_days'])}"),
        ]:
            col.markdown(f'<div class="stat-card" style="margin-top:1rem"><div class="stat-number" style="font-size:1.4rem">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:rgba(30,58,95,0.3);font-size:0.72rem;font-family:monospace">GRIDGUARD · ELECTRICITY THEFT INTELLIGENCE · 2026</div>', unsafe_allow_html=True)