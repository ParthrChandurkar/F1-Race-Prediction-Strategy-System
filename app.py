"""
app.py  —  F1 Predict 2025  |  Professional ML Race Prediction System
Run: streamlit run app.py
"""

import os, sys, json, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="F1 Predict 2025",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{background:#07070f;color:#e8e8ee;font-family:'Inter',sans-serif}
h1,h2,h3{font-family:'Orbitron',monospace}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d0d1a,#0a0005);border-right:1px solid #e10600}
div[data-testid="metric-container"]{background:linear-gradient(135deg,#0f0f1e,#180509);border:1px solid #1e1e2e;border-top:2px solid #e10600;border-radius:8px;padding:14px}
div[data-testid="metric-container"] label{color:#ffd700!important;font-size:.7rem!important;letter-spacing:1px;text-transform:uppercase}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:#fff!important;font-family:'Orbitron';font-size:1.4rem!important}
div.stButton>button{background:linear-gradient(90deg,#e10600,#c00400);color:#fff;font-family:'Orbitron';font-size:.7rem;font-weight:700;letter-spacing:2px;border:none;border-radius:4px;padding:.6rem 2rem;text-transform:uppercase;transition:all .2s;width:100%}
div.stButton>button:hover{background:linear-gradient(90deg,#ff2010,#e10600);transform:translateY(-1px)}
div[data-baseweb="select"]>div{background:#0f0f1e!important;border-color:#2a2a3a!important;color:#e8e8ee!important}
div[data-baseweb="select"]>div:hover{border-color:#e10600!important}
.stTabs [data-baseweb="tab-list"]{background:#0a0a14;border-bottom:1px solid #1e1e2e;gap:2px}
.stTabs [data-baseweb="tab"]{font-family:'Orbitron';font-size:.65rem;color:#555;letter-spacing:1px;padding:8px 14px;border-radius:4px 4px 0 0}
.stTabs [aria-selected="true"]{color:#e10600!important;background:#0f0f1e!important;border-bottom:2px solid #e10600}
hr{border-color:#1e1e2e}
.stSlider>div>div{background:#e10600!important}
.banner{background:linear-gradient(135deg,#0d0d1a,#180509,#0d0d1a);border-top:2px solid #e10600;border-bottom:1px solid #1e1e2e;padding:1rem 1.5rem;margin-bottom:1.2rem;display:flex;align-items:center;justify-content:space-between}
.banner-title{font-family:'Orbitron';font-size:1.3rem;color:#fff;letter-spacing:3px;margin:0}
.banner-sub{color:#666;font-size:.75rem;letter-spacing:2px;margin-top:2px}
.banner-badge{background:#e10600;color:#fff;font-family:'Orbitron';font-size:.6rem;padding:4px 10px;border-radius:3px;letter-spacing:2px}
.sh{font-family:'Orbitron';font-size:.65rem;letter-spacing:3px;color:#555;text-transform:uppercase;padding-bottom:6px;border-bottom:1px solid #1a1a2a;margin:1.2rem 0 .8rem}
.card{background:#0f0f1e;border:1px solid #1e1e2e;border-left:3px solid #e10600;border-radius:6px;padding:1rem 1.1rem;margin-bottom:.7rem}
.card h4{font-family:'Orbitron';font-size:.7rem;color:#ffd700;margin:0 0 5px;letter-spacing:1px}
.card p{color:#aaa;margin:0;font-size:.85rem;line-height:1.55}
.card-gold{border-left-color:#ffd700}
.card-green{border-left-color:#00c851}
.card-blue{border-left-color:#0090d0}
.card-purple{border-left-color:#9b59b6}
.ir{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #111}
.ir:last-child{border-bottom:none}
.il{color:#666;font-size:.8rem}
.iv{color:#eee;font-size:.8rem;font-weight:500}
.sc-soft{display:inline-block;padding:5px 14px;border-radius:20px;background:#e10600;color:#fff;font-weight:700;font-size:.75rem;font-family:'Orbitron';letter-spacing:1px;margin:3px}
.sc-medium{display:inline-block;padding:5px 14px;border-radius:20px;background:#ffd700;color:#000;font-weight:700;font-size:.75rem;font-family:'Orbitron';letter-spacing:1px;margin:3px}
.sc-hard{display:inline-block;padding:5px 14px;border-radius:20px;background:#ccc;color:#000;font-weight:700;font-size:.75rem;font-family:'Orbitron';letter-spacing:1px;margin:3px}
.sc-inter{display:inline-block;padding:5px 14px;border-radius:20px;background:#00c851;color:#000;font-weight:700;font-size:.75rem;font-family:'Orbitron';letter-spacing:1px;margin:3px}
.sc-wet{display:inline-block;padding:5px 14px;border-radius:20px;background:#0090d0;color:#fff;font-weight:700;font-size:.75rem;font-family:'Orbitron';letter-spacing:1px;margin:3px}
.risk-low{color:#00c851;font-weight:600}
.risk-medium{color:#ffd700;font-weight:600}
.risk-high{color:#e10600;font-weight:600}
</style>
""", unsafe_allow_html=True)

# ── globals ──────────────────────────────────────────────────────────────────
MODELS_TRAINED = os.path.exists(os.path.join("models","Random_Forest.pkl"))

TEAM_COLORS = {
    "Red Bull Racing":"#3671C6","Ferrari":"#E8002D","McLaren":"#FF8000",
    "Mercedes":"#27F4D2","Aston Martin":"#229971","Alpine":"#0093CC",
    "Williams":"#64C4FF","Racing Bulls":"#6692FF","Haas":"#B6BABD","Kick Sauber":"#52E252",
}
COMPOUND_CSS = {"Soft":"sc-soft","Medium":"sc-medium","Hard":"sc-hard","Intermediate":"sc-inter","Wet":"sc-wet"}
COMPOUND_COLOR = {"Soft":"#E8142A","Medium":"#FFC800","Hard":"#CCCCCC","Intermediate":"#39B54A","Wet":"#0067FF"}

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0c0d0",family="Inter"),
    margin=dict(l=40,r=20,t=40,b=40),
)
_AX = dict(gridcolor="#111827",linecolor="#1e2030",zerolinecolor="#1e2030")
_AXR = dict(gridcolor="#111827",linecolor="#1e2030",autorange="reversed")

@st.cache_data
def get_df():
    p = os.path.join("data","processed","master.csv")
    return pd.read_csv(p, low_memory=False) if os.path.exists(p) else None

@st.cache_data
def get_metrics():
    if not MODELS_TRAINED: return {}
    from src.predictor import load_metrics
    return load_metrics()

@st.cache_data
def get_fi():
    if not MODELS_TRAINED: return {}
    from src.predictor import load_feature_importance
    return load_feature_importance()

from src.f1_2024_data import (
    DRIVER_NAMES_2024, DRIVER_TEAM_2025, TEAMS_2024,
    CIRCUIT_NAMES_2025, CIRCUIT_REF_MAP, CIRCUIT_DATA_MAP,
    DRIVER_SKILL, TEAM_CAR_RATING, CIRCUIT_CHARACTERISTICS,
    DRIVER_DATASET_REF,
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 0 18px;text-align:center">
        <div style="font-family:Orbitron;font-size:1.8rem;color:#e10600;font-weight:900;letter-spacing:2px">F1</div>
        <div style="font-family:Orbitron;font-size:.5rem;color:#555;letter-spacing:5px;margin-top:1px">PREDICT · 2025</div>
        <div style="height:1px;background:linear-gradient(90deg,transparent,#e10600,transparent);margin:12px 0"></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Dashboard",
        "🏁  Race Prediction",
        "🔬  Feature Analysis",
        "🎲  Race Simulation",
        "🛞  Strategy Centre",
        "👤  Driver Analysis",
        "🏭  Team Analysis",
        "📊  Model Performance",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if not MODELS_TRAINED:
        st.error("⚠️ Models not trained\n\nRun:\n```\npython src/train_models.py\n```")
    else:
        st.success("✅ Models Ready")
    st.markdown("""
    <div style="margin-top:20px;padding:10px;background:#0a0a14;border-radius:6px;border:1px solid #1a1a2a">
        <div style="font-size:.65rem;color:#444;font-family:Orbitron;letter-spacing:2px;text-align:center">2025 SEASON</div>
        <div style="font-size:.6rem;color:#333;text-align:center;margin-top:2px">ML PREDICTION ENGINE</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown("""<div class="banner">
        <div><div class="banner-title">F1 PREDICT 2025</div>
        <div class="banner-sub">FORMULA 1 · MACHINE LEARNING · FUTURE RACE PREDICTION</div></div>
        <div class="banner-badge">SEASON 2025</div></div>""", unsafe_allow_html=True)

    df = get_df()
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("2025 Drivers","20")
    c2.metric("2025 Circuits","24")
    c3.metric("ML Models","9")
    c4.metric("Training Races", str(df["raceId"].nunique()) if df is not None else "–")
    c5.metric("Years of Data","2000–2024")

    st.markdown("<div class='sh'>2025 DRIVER GRID</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i, drv in enumerate(DRIVER_NAMES_2024):
        team  = DRIVER_TEAM_2025.get(drv,"")
        color = TEAM_COLORS.get(team,"#888")
        skill = DRIVER_SKILL.get(drv, 0.72)
        car   = TEAM_CAR_RATING.get(team, 0.65)
        with cols[i % 5]:
            st.markdown(f"""
            <div style="background:#0f0f1e;border:1px solid #1a1a2a;border-top:2px solid {color};
                        border-radius:6px;padding:10px;text-align:center;margin-bottom:8px">
                <div style="height:2px;background:{color};border-radius:1px;margin-bottom:6px"></div>
                <div style="font-size:.82rem;font-weight:600;color:#fff;margin:2px 0">{drv}</div>
                <div style="font-size:.7rem;color:#666;margin-bottom:6px">{team}</div>
                <div style="display:flex;justify-content:center;gap:10px">
                    <div><div style="font-family:Orbitron;font-size:.85rem;color:{color}">{int(skill*100)}</div>
                    <div style="font-size:.55rem;color:#444">SKILL</div></div>
                    <div><div style="font-family:Orbitron;font-size:.85rem;color:{color}">{int(car*100)}</div>
                    <div style="font-size:.55rem;color:#444">CAR</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sh'>WHAT EACH MODULE DOES</div>", unsafe_allow_html=True)
    mc1,mc2,mc3,mc4 = st.columns(4)
    with mc1:
        st.markdown("""<div class="card"><h4>🏁 RACE PREDICTION</h4>
        <p>Uses ML models + grid position + qualifying data + rolling form to predict finishing position and Top 10 probability for all 20 drivers. Shows model reasoning.</p></div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""<div class="card card-gold"><h4>🔬 FEATURE ANALYSIS</h4>
        <p>Predict qualifying position impact — how much does starting P1 vs P10 matter? Shows feature contribution breakdown from the Random Forest model per driver.</p></div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown("""<div class="card card-green"><h4>🎲 RACE SIMULATION</h4>
        <p>1000 Monte Carlo runs using pit stop patterns, DNF probability, lap time data, and safety car likelihood per circuit. Each simulation is a unique race outcome.</p></div>""", unsafe_allow_html=True)
    with mc4:
        st.markdown("""<div class="card card-blue"><h4>🛞 STRATEGY CENTRE</h4>
        <p>Full tyre strategy engine using historical pit stop counts per circuit, lap time degradation curves, safety car probability, and undercut opportunity analysis.</p></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2 — RACE PREDICTION  (uses: grid, qual, rolling form, team, circuit)
# ═══════════════════════════════════════════════════════════════════════
elif page == "🏁  Race Prediction":
    st.markdown("""<div class="banner">
        <div><div class="banner-title">RACE PREDICTION</div>
        <div class="banner-sub">GRID · QUALIFYING · ROLLING FORM · TEAM STRENGTH</div></div>
        <div class="banner-badge">ML CLASSIFIER + REGRESSOR</div></div>""", unsafe_allow_html=True)

    if not MODELS_TRAINED:
        st.warning("Run `python src/train_models.py` first."); st.stop()

    from src.predictor import predict_full_grid, predict_driver

    st.markdown("""
    <div class="card card-blue" style="margin-bottom:1rem">
        <h4>HOW THIS WORKS</h4>
        <p>The <b>Random Forest classifier</b> predicts Top-10 probability using 11 features:
        grid position, qualifying position, driver rolling 5-race average, team rolling average,
        driver win rate, circuit identity, year, pit stop count, and average lap time.
        The <b>Ridge Regressor</b> predicts the exact finishing position (1–20).
        Results are then calibrated with 2025 driver skill and car performance ratings.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("<div class='sh'>RACE SETUP</div>", unsafe_allow_html=True)
        circuit_name = st.selectbox("Circuit", CIRCUIT_NAMES_2025)
        weather      = st.selectbox("Weather", ["Dry","Cloudy","Light Rain","Heavy Rain"])
        wmap = {"Dry":1.0,"Cloudy":0.95,"Light Rain":0.82,"Heavy Rain":0.65}
        wf   = wmap[weather]

        cdata = CIRCUIT_DATA_MAP.get(circuit_name,{})
        cref  = cdata.get("ref","bahrain")
        char  = CIRCUIT_CHARACTERISTICS.get(cref,{})

        st.markdown("<div class='sh'>CIRCUIT SNAPSHOT</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <div class="ir"><span class="il">Laps</span><span class="iv">{cdata.get('laps','–')}</span></div>
            <div class="ir"><span class="il">Country</span><span class="iv">{cdata.get('country','–')}</span></div>
            <div class="ir"><span class="il">Tyre Degradation</span><span class="iv">{char.get('tyre_deg','–').upper()}</span></div>
            <div class="ir"><span class="il">Overtaking</span><span class="iv">{char.get('overtaking','–').upper()}</span></div>
            <div class="ir"><span class="il">Safety Car Risk</span><span class="iv">{char.get('safety_car_prob',0):.0%}</span></div>
            <div class="ir"><span class="il">Avg Lap (est.)</span><span class="iv">{cdata.get('lap_ms',90000)/1000:.1f}s</span></div>
        </div>""", unsafe_allow_html=True)

        predict_btn = st.button("🏎  PREDICT RACE RESULT")

    with col2:
        if predict_btn:
            lap_ms = cdata.get("lap_ms",90000)
            with st.spinner("Predicting all 20 drivers..."):
                results = predict_full_grid(cref, lap_ms, 2025, wf)

            # ── Podium ──
            st.markdown("<div class='sh'>PREDICTED PODIUM</div>", unsafe_allow_html=True)
            p2c, p1c, p3c = st.columns(3)
            for col, r, lbl, medal, ht in [
                (p2c, results[1], "P2", "🥈", "140px"),
                (p1c, results[0], "P1", "🏆", "160px"),
                (p3c, results[2], "P3", "🥉", "130px"),
            ]:
                tc = TEAM_COLORS.get(r["team"],"#888")
                with col:
                    st.markdown(f"""
                    <div style="background:#0f0f1e;border:1px solid {tc};border-top:3px solid {tc};
                                border-radius:8px;padding:14px;text-align:center;min-height:{ht}">
                        <div style="font-size:1.3rem">{medal}</div>
                        <div style="font-family:Orbitron;font-size:.6rem;color:{tc};letter-spacing:2px">{lbl}</div>
                        <div style="font-size:.9rem;font-weight:600;color:#fff;margin:4px 0">{r['driver']}</div>
                        <div style="font-size:.72rem;color:#666">{r['team']}</div>
                        <div style="display:flex;justify-content:center;gap:14px;margin-top:8px">
                            <div><div style="font-family:Orbitron;font-size:.9rem;color:#ffd700">{r['top10_prob']:.0%}</div>
                            <div style="font-size:.6rem;color:#444">TOP10</div></div>
                            <div><div style="font-family:Orbitron;font-size:.9rem;color:{tc}">{r['win_prob']:.0%}</div>
                            <div style="font-size:.6rem;color:#444">WIN</div></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # ── Full table ──
            st.markdown("<div class='sh'>FULL PREDICTED RACE ORDER</div>", unsafe_allow_html=True)
            rows = []
            for r in results:
                p = r["predicted_finish"]
                rows.append({
                    "Pos": p,
                    "Driver": r["driver"],
                    "Team": r["team"],
                    "Pred. Position": r["pred_position"],
                    "Top10 Prob": f"{r['top10_prob']:.1%}",
                    "Win Prob": f"{r['win_prob']:.1%}",
                    "Driver Skill": f"{r['skill_rating']:.0%}",
                    "Car Rating": f"{r['car_rating']:.0%}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Top10 probability bar ──
            fig = go.Figure()
            colors = [TEAM_COLORS.get(r["team"],"#555") for r in results]
            fig.add_trace(go.Bar(
                x=[r["driver"].split()[-1] for r in results],
                y=[r["top10_prob"]*100 for r in results],
                marker_color=colors,
                text=[f"{r['top10_prob']:.0%}" for r in results],
                textposition="outside", textfont=dict(size=9),
            ))
            fig.update_layout(**DARK, title="Top 10 Finish Probability by Driver",
                              yaxis_title="%", height=310, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

            # ── Predicted position scatter ──
            fig2 = go.Figure()
            for r in results:
                tc = TEAM_COLORS.get(r["team"],"#555")
                fig2.add_trace(go.Scatter(
                    x=[r["skill_rating"]*100],
                    y=[r["pred_position"]],
                    mode="markers+text",
                    marker=dict(size=14, color=tc, line=dict(color="#fff",width=1)),
                    text=[r["driver"].split()[-1]],
                    textposition="top center",
                    textfont=dict(size=9, color="#ccc"),
                    name=r["driver"], showlegend=False,
                ))
            fig2.update_layout(**DARK,
                title="Driver Skill Rating vs Predicted Finishing Position",
                xaxis=dict(gridcolor="#111827",linecolor="#1e2030",title="Driver Skill Rating (%)"),
                yaxis=dict(gridcolor="#111827",linecolor="#1e2030",autorange="reversed",title="Predicted Position"),
                height=360)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown("""<div style="text-align:center;padding:60px;color:#333">
                <div style="font-size:3rem">🏁</div>
                <div style="font-family:Orbitron;font-size:.85rem;letter-spacing:2px;margin-top:10px">SELECT CIRCUIT AND PREDICT</div>
                <div style="font-size:.8rem;margin-top:6px;color:#444">Predictions for all 20 drivers appear here</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ANALYSIS  (uses: qual position sensitivity, feature importances)
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔬  Feature Analysis":
    st.markdown("""<div class="banner">
        <div><div class="banner-title">FEATURE ANALYSIS</div>
        <div class="banner-sub">QUALIFYING IMPACT · FEATURE CONTRIBUTION · GRID SENSITIVITY</div></div>
        <div class="banner-badge">ML EXPLAINABILITY</div></div>""", unsafe_allow_html=True)

    if not MODELS_TRAINED:
        st.warning("Run `python src/train_models.py` first."); st.stop()

    import joblib
    from src.predictor import build_input_vector
    from src.f1_2024_data import TEAM_REFS

    st.markdown("""
    <div class="card card-purple" style="margin-bottom:1rem">
        <h4>WHAT THIS MODULE SHOWS</h4>
        <p>This page answers: <b>"How much does each factor actually affect race outcome?"</b><br>
        — How does moving from P1 to P10 on the grid change win probability?<br>
        — Which features matter most in the Random Forest model?<br>
        — How does driver form (rolling average) compare to car performance?</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📍 Grid Position Impact",
        "📊 Feature Importances",
        "📈 Form vs Car Rating",
    ])

    with tab1:
        st.markdown("<div class='sh'>HOW GRID POSITION AFFECTS WIN & TOP10 PROBABILITY</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,2])
        with col1:
            driver_fa = st.selectbox("Driver", DRIVER_NAMES_2024, key="fa_drv")
            circuit_fa = st.selectbox("Circuit", CIRCUIT_NAMES_2025, key="fa_circ")
            fa_btn = st.button("🔬 ANALYSE GRID IMPACT")

        with col2:
            if fa_btn:
                team_fa = DRIVER_TEAM_2025.get(driver_fa,"Ferrari")
                dref    = driver_fa.lower().split()[-1]
                cref_fa = CIRCUIT_DATA_MAP.get(circuit_fa,{}).get("ref","bahrain")
                lap_ms  = CIRCUIT_DATA_MAP.get(circuit_fa,{}).get("lap_ms",90000)
                skill   = DRIVER_SKILL.get(driver_fa, 0.75)
                car     = TEAM_CAR_RATING.get(team_fa, 0.75)
                cref_team = TEAM_REFS.get(team_fa,"ferrari")

                sc = joblib.load(os.path.join("models","scaler.pkl"))
                rf = joblib.load(os.path.join("models","Random_Forest.pkl"))
                rd = joblib.load(os.path.join("models","Ridge_Regression.pkl"))

                grid_pos_range = list(range(1,21))
                top10_probs, pred_positions = [], []
                for gp in grid_pos_range:
                    avg_fin = round(10.0 - skill*8.0, 2)
                    t_avg   = round(10.0 - car*7.0, 2)
                    wr      = round(max(0, skill-0.70)*2, 3)
                    x = build_input_vector(gp, gp, 2025, dref, cref_team, cref_fa,
                                           avg_fin, t_avg, wr, 2.0, float(lap_ms))
                    xs = sc.transform(x)
                    p10 = float(rf.predict_proba(xs)[0][1])
                    pos = float(np.clip(rd.predict(xs)[0], 1, 20))
                    combined = skill*0.6 + car*0.4
                    p10_adj = float(np.clip(p10*0.5 + combined*0.5, 0.01, 0.99))
                    pos_adj = float(np.clip(pos*0.5 + (1-combined)*19*0.5+1, 1, 20))
                    top10_probs.append(p10_adj)
                    pred_positions.append(pos_adj)

                tc = TEAM_COLORS.get(team_fa,"#e10600")
                fig = make_subplots(specs=[[{"secondary_y":True}]])
                fig.add_trace(go.Scatter(
                    x=grid_pos_range, y=[p*100 for p in top10_probs],
                    name="Top10 Probability (%)", mode="lines+markers",
                    line=dict(color=tc, width=3),
                    marker=dict(size=8, color=tc),
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=grid_pos_range, y=pred_positions,
                    name="Predicted Finish Position", mode="lines+markers",
                    line=dict(color="#ffd700", width=2, dash="dot"),
                    marker=dict(size=7, color="#ffd700"),
                ), secondary_y=True)
                fig.update_layout(**DARK, title=f"{driver_fa} — Grid Position Sensitivity at {circuit_fa}",
                                  xaxis_title="Starting Grid Position", height=380)
                fig.update_yaxes(title_text="Top10 Probability (%)", secondary_y=False)
                fig.update_yaxes(title_text="Predicted Finish", autorange="reversed", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

                # Key insight
                diff = top10_probs[0] - top10_probs[9]
                st.markdown(f"""
                <div class="card card-gold">
                    <h4>KEY INSIGHT — {driver_fa.upper()}</h4>
                    <p>Starting from <b>Pole (P1)</b> gives <b>{top10_probs[0]:.1%}</b> Top10 probability.
                    Starting from <b>P10</b> gives <b>{top10_probs[9]:.1%}</b>.
                    That's a <b>{diff:.1%} difference</b> just from grid position alone.
                    The model uses {driver_fa}'s skill rating ({int(skill*100)}%) and
                    {team_fa}'s car rating ({int(car*100)}%) alongside grid position
                    to compute this — showing that even great drivers lose ~{diff:.0%} probability
                    by starting mid-grid.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Select a driver and circuit then click Analyse.")

    with tab2:
        fi = get_fi()
        if fi:
            fi_sorted = sorted(fi.items(), key=lambda x:x[1], reverse=True)
            fi_names  = [k.replace("_enc","").replace("_"," ").title() for k,_ in fi_sorted]
            fi_vals   = [v for _,v in fi_sorted]
            fi_pct    = [v/sum(fi_vals)*100 for v in fi_vals]

            colors = []
            for n in fi_names:
                if "Grid" in n or "Qual" in n: colors.append("#e10600")
                elif "Driver" in n: colors.append("#ffd700")
                elif "Team" in n or "Constructor" in n: colors.append("#0090d0")
                elif "Avg" in n or "Win" in n: colors.append("#00c851")
                else: colors.append("#555")

            fig = go.Figure(go.Bar(
                x=fi_pct, y=fi_names, orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in fi_pct],
                textposition="outside",
            ))
            fig.update_layout(**DARK, title="Feature Importance — Random Forest Model (% contribution to prediction)",
                              xaxis=dict(gridcolor="#111827",linecolor="#1e2030",title="Importance (%)"),
                              yaxis=dict(gridcolor="#111827",linecolor="#1e2030",autorange="reversed"),
                              height=420)
            st.plotly_chart(fig, use_container_width=True)

            # explanation
            top3 = fi_sorted[:3]
            st.markdown(f"""
            <div class="card">
                <h4>INTERPRETATION</h4>
                <p>🔴 <b>Red bars</b> = race position features (grid, qualifying) — most controllable by driver.<br>
                🟡 <b>Yellow bars</b> = driver identity features — captures talent + experience.<br>
                🔵 <b>Blue bars</b> = team/car features — reflects constructor performance.<br>
                🟢 <b>Green bars</b> = form features — rolling average and win rate.<br><br>
                The top 3 most important features are:
                <b>{top3[0][0].replace('_',' ')}</b> ({top3[0][1]*100:.1f}%),
                <b>{top3[1][0].replace('_',' ')}</b> ({top3[1][1]*100:.1f}%),
                <b>{top3[2][0].replace('_',' ')}</b> ({top3[2][1]*100:.1f}%).</p>
            </div>""", unsafe_allow_html=True)

            # Pie chart
            fig2 = go.Figure(go.Pie(
                labels=fi_names, values=fi_pct,
                marker=dict(colors=colors),
                hole=0.5,
                textinfo="label+percent",
                textfont=dict(size=10),
            ))
            fig2.update_layout(**DARK, title="Feature Contribution Share", height=380)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("<div class='sh'>DRIVER FORM vs CAR PERFORMANCE — ALL 20 DRIVERS</div>", unsafe_allow_html=True)
        fig = go.Figure()
        for drv in DRIVER_NAMES_2024:
            team  = DRIVER_TEAM_2025.get(drv,"")
            skill = DRIVER_SKILL.get(drv, 0.72)
            car   = TEAM_CAR_RATING.get(team, 0.65)
            tc    = TEAM_COLORS.get(team,"#555")
            combined = skill*0.6 + car*0.4
            fig.add_trace(go.Scatter(
                x=[car*100], y=[skill*100],
                mode="markers+text",
                marker=dict(size=int(combined*30)+8, color=tc,
                            line=dict(color="#fff",width=1), opacity=0.9),
                text=[drv.split()[-1]],
                textposition="top center",
                textfont=dict(size=9,color="#ccc"),
                name=drv, showlegend=False,
            ))
        fig.add_hline(y=80, line_dash="dash", line_color="#333",
                      annotation_text="Elite skill line", annotation_font=dict(color="#555",size=9))
        fig.add_vline(x=80, line_dash="dash", line_color="#333",
                      annotation_text="Top car line", annotation_font=dict(color="#555",size=9))
        fig.update_layout(**DARK, title="Driver Skill Rating vs Car Performance Rating (bubble size = combined score)",
                          xaxis_title="Car / Constructor Rating (%)",
                          yaxis_title="Driver Skill Rating (%)", height=460)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="card">
            <h4>HOW TO READ THIS CHART</h4>
            <p>Top-right = elite driver in a top car (Verstappen, Norris, Hamilton).<br>
            Top-left = elite driver in a slower car — their skill carries them beyond what the car deserves.<br>
            Bottom-right = average driver in a fast car — car dependency.<br>
            Bubble size reflects the combined prediction score used by the ML model.</p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4 — RACE SIMULATION  (uses: pit stops, DNF, lap time, SC per circuit)
# ═══════════════════════════════════════════════════════════════════════
elif page == "🎲  Race Simulation":
    st.markdown("""<div class="banner">
        <div><div class="banner-title">RACE SIMULATION</div>
        <div class="banner-sub">MONTE CARLO · PIT STOP PATTERNS · DNF RISK · LAP TIME DATA</div></div>
        <div class="banner-badge">1000 ITERATIONS</div></div>""", unsafe_allow_html=True)

    if not MODELS_TRAINED:
        st.warning("Run training first."); st.stop()

    from src.predictor import predict_full_grid
    from src.simulator import run_simulation
    from src.f1_2024_data import CIRCUIT_CHARACTERISTICS

    st.markdown("""
    <div class="card card-green" style="margin-bottom:1rem">
        <h4>HOW SIMULATION DIFFERS FROM RACE PREDICTION</h4>
        <p><b>Race Prediction</b> gives one deterministic answer using the ML model — the most likely outcome.
        <b>Simulation</b> runs 1000 random race iterations and measures <i>probability distributions</i>.
        Each simulation adds: random DNF chance (from historical reliability data), safety car probability
        per circuit (from our dataset), circuit overtaking difficulty (which affects how much grid position
        locks in the result), and lap-time noise. The result is a full probability picture — not just one prediction.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("<div class='sh'>SIMULATION PARAMETERS</div>", unsafe_allow_html=True)
        circuit_s  = st.selectbox("Circuit", CIRCUIT_NAMES_2025, key="sim_c")
        weather_s  = st.selectbox("Weather", ["Dry","Cloudy","Light Rain","Heavy Rain"], key="sim_w")
        n_sims     = st.select_slider("Simulations", [250,500,1000,2000], value=1000)

        cdata_s = CIRCUIT_DATA_MAP.get(circuit_s,{})
        cref_s  = cdata_s.get("ref","bahrain")
        char_s  = CIRCUIT_CHARACTERISTICS.get(cref_s,{})

        st.markdown(f"""
        <div class="card" style="margin-top:8px">
            <h4>CIRCUIT FACTORS USED</h4>
            <div class="ir"><span class="il">Safety Car Probability</span>
                <span class="iv">{char_s.get('safety_car_prob',0.4):.0%}</span></div>
            <div class="ir"><span class="il">Overtaking Difficulty</span>
                <span class="iv">{char_s.get('overtaking','medium').upper()}</span></div>
            <div class="ir"><span class="il">Tyre Degradation</span>
                <span class="iv">{char_s.get('tyre_deg','medium').upper()}</span></div>
            <div class="ir"><span class="il">Pit Loss (seconds)</span>
                <span class="iv">{char_s.get('pit_loss',22)}s</span></div>
        </div>""", unsafe_allow_html=True)

        selected = st.multiselect("Include Drivers", DRIVER_NAMES_2024, default=DRIVER_NAMES_2024)
        sim_btn = st.button("🎲  RUN SIMULATION")

    with col2:
        if sim_btn:
            if len(selected) < 2:
                st.error("Select at least 2 drivers."); st.stop()
            wmap = {"Dry":1.0,"Cloudy":0.95,"Light Rain":0.82,"Heavy Rain":0.65}
            wf   = wmap[weather_s]
            lap_ms_s = cdata_s.get("lap_ms",90000)

            with st.spinner(f"Running {n_sims:,} race simulations..."):
                preds    = predict_full_grid(cref_s, lap_ms_s, 2025, wf)
                drv_prob = {r["driver"]:r["top10_prob"] for r in preds}
                drivers  = [d for d in selected if d in drv_prob]
                probs    = [drv_prob[d] for d in drivers]
                sim      = run_simulation(drivers, probs, n_sims=n_sims,
                                          circuit_overtaking=char_s.get("overtaking","medium"))

            results = sim["results"]
            winner  = results[0]
            tc_w    = TEAM_COLORS.get(winner["team"],"#e10600")

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Predicted Winner", winner["driver"].split()[-1])
            m2.metric("Win Probability", f"{winner['win_prob']:.1%}")
            m3.metric("Podium Prob", f"{winner['podium_prob']:.1%}")
            m4.metric("DNF Risk", f"{winner['dnf_prob']:.1%}")

            df_sim = pd.DataFrame(results)
            top_n  = df_sim.head(min(15,len(df_sim)))

            # Win probability
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=top_n["driver"].apply(lambda x:x.split()[-1]),
                y=top_n["win_prob"]*100,
                marker_color=[TEAM_COLORS.get(t,"#555") for t in top_n["team"]],
                text=[f"{v:.1%}" for v in top_n["win_prob"]],
                textposition="outside", name="Win %",
            ))
            fig1.update_layout(**DARK, title=f"Win Probability — {n_sims:,} Simulations",
                               yaxis_title="%", height=290, xaxis_tickangle=-30)
            st.plotly_chart(fig1, use_container_width=True)

            # Podium + Top10 + DNF grouped
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Podium %",
                x=top_n["driver"].apply(lambda x:x.split()[-1]),
                y=top_n["podium_prob"]*100, marker_color="#e10600"))
            fig2.add_trace(go.Bar(name="Top10 %",
                x=top_n["driver"].apply(lambda x:x.split()[-1]),
                y=top_n["top10_prob"]*100, marker_color="#ffd700"))
            fig2.add_trace(go.Bar(name="DNF Risk %",
                x=top_n["driver"].apply(lambda x:x.split()[-1]),
                y=top_n["dnf_prob"]*100, marker_color="#333"))
            fig2.update_layout(**DARK, barmode="group", height=300, xaxis_tickangle=-30,
                               title="Podium / Top10 / DNF Probability Distribution")
            st.plotly_chart(fig2, use_container_width=True)

            # Avg finish heatmap-style scatter
            fig3 = go.Figure()
            for _, r in df_sim.iterrows():
                tc = TEAM_COLORS.get(r["team"],"#555")
                fig3.add_trace(go.Scatter(
                    x=[r["avg_finish"]], y=[r["win_prob"]*100],
                    mode="markers+text",
                    marker=dict(size=12, color=tc, line=dict(color="#fff",width=1)),
                    text=[r["driver"].split()[-1]],
                    textposition="top center", textfont=dict(size=8,color="#ccc"),
                    showlegend=False,
                ))
            fig3.update_layout(**DARK,
                title="Average Simulated Finish vs Win Probability",
                xaxis=dict(gridcolor="#111827",linecolor="#1e2030",autorange="reversed",title="Avg Finish Position (lower = better)"),
                yaxis=dict(gridcolor="#111827",linecolor="#1e2030",title="Win Probability (%)"),
                height=360)
            st.plotly_chart(fig3, use_container_width=True)

            # Full table
            st.markdown("<div class='sh'>FULL SIMULATION RESULTS</div>", unsafe_allow_html=True)
            disp = df_sim[["sim_position","driver","team","win_prob","podium_prob","top10_prob","dnf_prob","avg_finish"]].copy()
            disp.columns = ["Sim P","Driver","Team","Win%","Podium%","Top10%","DNF%","Avg Finish"]
            for c in ["Win%","Podium%","Top10%","DNF%"]:
                disp[c] = disp[c].apply(lambda x:f"{x:.1%}")
            disp["Avg Finish"] = disp["Avg Finish"].apply(lambda x:f"{x:.1f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("👈 Configure and click RUN SIMULATION")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 5 — STRATEGY CENTRE
# ═══════════════════════════════════════════════════════════════════════
elif page == "🛞  Strategy Centre":
    st.markdown("""<div class="banner">
        <div><div class="banner-title">STRATEGY CENTRE</div>
        <div class="banner-sub">PIT WINDOWS · TYRE COMPOUNDS · SAFETY CAR · UNDERCUT</div></div>
        <div class="banner-badge">RACE ENGINEER</div></div>""", unsafe_allow_html=True)

    from src.strategy import recommend, COMPOUND_DEG, COMPOUND_COLORS

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("<div class='sh'>RACE PARAMETERS</div>", unsafe_allow_html=True)
        driver_st  = st.selectbox("Driver", DRIVER_NAMES_2024, key="st_d")
        circuit_st = st.selectbox("Circuit", CIRCUIT_NAMES_2025, key="st_c")
        grid_st    = st.slider("Grid Position", 1, 20, 5)
        weather_st = st.selectbox("Weather Forecast", ["Dry","Cloudy","Light Rain","Heavy Rain"])
        start_comp = st.selectbox("Starting Tyre Compound", ["Soft","Medium","Hard","Intermediate","Wet"])
        aggressive = st.toggle("Aggressive Strategy Mode")
        race_laps_st = CIRCUIT_DATA_MAP.get(circuit_st, {}).get("laps", 57)
        current_lap_st = st.slider("Current Race Lap", 0, race_laps_st, 0)
        completed_stops_st = st.slider("Stops Already Completed", 0, 3, 0)
        strat_btn  = st.button("🛞  BUILD RACE STRATEGY")

        team_st = DRIVER_TEAM_2025.get(driver_st,"Ferrari")
        tc_st   = TEAM_COLORS.get(team_st,"#888")
        skill_st = DRIVER_SKILL.get(driver_st, 0.72)
        car_st   = TEAM_CAR_RATING.get(team_st, 0.65)

        st.markdown(f"""
        <div class="card" style="margin-top:10px">
            <div style="height:2px;background:{tc_st};border-radius:1px;margin-bottom:8px"></div>
            <div style="font-size:.85rem;font-weight:600;color:#fff">{driver_st}</div>
            <div style="font-size:.75rem;color:#666">{team_st}</div>
            <div style="display:flex;gap:14px;margin-top:8px">
                <div><div style="font-family:Orbitron;font-size:.95rem;color:{tc_st}">{int(skill_st*100)}</div>
                <div style="font-size:.6rem;color:#444">SKILL</div></div>
                <div><div style="font-family:Orbitron;font-size:.95rem;color:{tc_st}">{int(car_st*100)}</div>
                <div style="font-size:.6rem;color:#444">CAR</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        if strat_btn:
            preview = recommend(circuit_name=circuit_st, grid_position=grid_st,
                                weather=weather_st, starting_compound=start_comp,
                                aggressive=aggressive, driver_name=driver_st)
            completed_stops = min(completed_stops_st, preview["recommended_stops"])
            if completed_stops != completed_stops_st:
                st.warning(
                    f"This is a {preview['recommended_stops']}-stop plan, so completed stops "
                    f"was adjusted to {completed_stops}."
                )
            rec = recommend(circuit_name=circuit_st, grid_position=grid_st,
                            weather=weather_st, starting_compound=start_comp,
                            aggressive=aggressive, driver_name=driver_st,
                            current_lap=current_lap_st,
                            completed_stops=completed_stops)
            pri      = rec["primary_strategy"]
            compounds = pri["compounds"]
            risk_css = {"Low":"risk-low","Medium":"risk-medium","High":"risk-high"}.get(pri["risk"],"risk-medium")

            # Strategy banner
            compounds_html = ""
            for i,c in enumerate(compounds):
                compounds_html += f'<span class="{COMPOUND_CSS.get(c,"sc-medium")}">{c}</span>'
                if i < len(compounds)-1:
                    compounds_html += '<span style="color:#444;font-size:1.1rem;margin:0 2px">→</span>'

            st.markdown(f"""
            <div style="background:#0f0f1e;border:1px solid #1e1e2e;border-top:2px solid #e10600;
                        border-radius:8px;padding:1.1rem;margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                    <div>
                        <div style="font-family:Orbitron;font-size:.6rem;color:#e10600;letter-spacing:3px">RECOMMENDED STRATEGY</div>
                        <div style="font-size:.95rem;font-weight:600;color:#fff;margin-top:4px">{pri['name']}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:.65rem;color:#555">RISK LEVEL</div>
                        <div class="{risk_css}">{pri['risk']}</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">{compounds_html}</div>
            </div>""", unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Pit Stops", rec["recommended_stops"])
            m2.metric("Race Laps", rec["total_laps"])
            m3.metric("SC Probability", f"{rec['sc_probability']:.0%}")
            m4.metric("Pit Loss", f"{rec['pit_loss_seconds']}s")

            live = rec["live_status"]
            live_color = {
                "HOLD": "#777", "PREPARE": "#ffd700", "WINDOW_OPEN": "#00a8ff",
                "BOX": "#e10600", "OVERDUE": "#ff4b4b", "COMPLETE": "#00c851",
            }.get(live["status"], "#888")
            st.markdown(f"""
            <div style="background:{live_color}12;border:1px solid {live_color}55;
                        border-left:4px solid {live_color};border-radius:6px;padding:12px 14px;
                        margin:12px 0">
                <div style="font-family:Orbitron;font-size:.62rem;color:{live_color};
                            letter-spacing:2px">LIVE ENGINEER · LAP {rec['current_lap']} · {live['status']}</div>
                <div style="font-size:.92rem;color:#eee;margin-top:5px">{live['instruction']}</div>
            </div>""", unsafe_allow_html=True)

            # Pit windows
            st.markdown("<div class='sh'>PIT STOP WINDOWS</div>", unsafe_allow_html=True)
            for w in rec["pit_windows"]:
                cc = COMPOUND_COLORS.get(w["from"],"#888")
                ct = COMPOUND_COLORS.get(w["to"],"#ccc")
                st.markdown(f"""
                <div style="background:#0a0a14;border:1px solid #1a1a2a;border-radius:6px;
                            padding:10px 14px;margin-bottom:6px;display:flex;
                            align-items:center;justify-content:space-between">
                    <div>
                        <div style="font-family:Orbitron;font-size:.6rem;color:#555;letter-spacing:2px">STOP {w['stop']}</div>
                        <div style="font-size:.9rem;font-weight:500;color:#fff;margin-top:2px">{w['window']}</div>
                        <div style="font-size:.72rem;color:#444">Optimal: Lap {w['optimal_lap']}</div>
                    </div>
                    <div style="display:flex;align-items:center;gap:8px">
                        <span style="background:{cc}22;color:{cc};border:1px solid {cc}44;
                                     padding:3px 10px;border-radius:12px;font-size:.72rem;font-weight:600">{w['from']}</span>
                        <span style="color:#333;font-size:1rem">→</span>
                        <span style="background:{ct}22;color:{ct};border:1px solid {ct}44;
                                     padding:3px 10px;border-radius:12px;font-size:.72rem;font-weight:600">{w['to']}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Analysis cards
            st.markdown("<div class='sh'>RACE ENGINEER BRIEFING</div>", unsafe_allow_html=True)
            n1,n2 = st.columns(2)
            with n1:
                st.markdown(f"""
                <div class="card"><h4>🔴 SAFETY CAR</h4><p>{rec['sc_advice']}</p></div>
                <div class="card card-gold"><h4>⚡ UNDERCUT</h4><p>{rec['undercut_opportunity']}</p></div>""",
                unsafe_allow_html=True)
            with n2:
                st.markdown(f"""
                <div class="card card-blue"><h4>🌤 WEATHER</h4><p>{rec['weather_note']}</p></div>
                <div class="card card-green"><h4>🏁 GRID NOTE</h4><p>{rec['grid_note']}</p></div>""",
                unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card" style="margin-top:2px">
                <h4>🏟 {circuit_st.upper()} BRIEFING</h4><p>{rec['circuit_note']}</p>
            </div>""", unsafe_allow_html=True)

            # Alternative strategies
            st.markdown("<div class='sh'>ALTERNATIVE STRATEGIES</div>", unsafe_allow_html=True)
            for alt in rec["alternatives"][:3]:
                arc = {"Low":"risk-low","Medium":"risk-medium","High":"risk-high"}.get(alt["risk"],"risk-medium")
                comps_str = " → ".join(alt["compounds"])
                st.markdown(f"""
                <div style="background:#080810;border:1px solid #1a1a2a;border-radius:6px;
                            padding:10px 14px;margin-bottom:5px;display:flex;
                            align-items:center;justify-content:space-between">
                    <div>
                        <div style="font-size:.85rem;font-weight:500;color:#ccc">{alt['name']}</div>
                        <div style="font-size:.72rem;color:#444;margin-top:2px">{comps_str}</div>
                    </div>
                    <span class="{arc}" style="font-size:.75rem">{alt['risk']} risk</span>
                </div>""", unsafe_allow_html=True)

            # Compare every viable strategy on the same circuit-specific model.
            st.markdown("<div class='sh'>STRATEGY COMPARISON</div>", unsafe_allow_html=True)
            comparison_df = pd.DataFrame(rec["strategy_comparison"])
            comparison_df["Strategy"] = comparison_df.apply(
                lambda row: f"{'★ ' if row['recommended'] else ''}{row['name']}", axis=1
            )
            comparison_df["Tyres"] = comparison_df["compounds"].apply(
                lambda compounds: " → ".join(compounds)
            )
            comparison_df["Estimated Loss"] = comparison_df[
                "estimated_loss_seconds"
            ].apply(lambda seconds: f"{seconds:.1f}s")
            comparison_df["Delta"] = comparison_df[
                "delta_to_fastest_seconds"
            ].apply(lambda seconds: f"+{seconds:.1f}s" if seconds else "FASTEST")
            st.dataframe(
                comparison_df[["Strategy", "Tyres", "stops", "risk", "Estimated Loss", "Delta"]]
                .rename(columns={"stops": "Stops", "risk": "Risk"}),
                use_container_width=True,
                hide_index=True,
            )

            export_rows = pd.DataFrame(rec["pit_windows"]).rename(columns={
                "stop": "Stop",
                "optimal_lap": "Optimal Lap",
                "window": "Pit Window",
                "from": "From Tyre",
                "to": "To Tyre",
            })
            export_rows.insert(0, "Circuit", rec["circuit_name"])
            export_rows.insert(1, "Driver", rec["driver_name"])
            st.download_button(
                "DOWNLOAD PIT PLAN (CSV)",
                data=export_rows.to_csv(index=False).encode("utf-8"),
                file_name=f"{rec['circuit_ref']}_pit_plan.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Tyre degradation chart
            st.markdown("<div class='sh'>TYRE DEGRADATION MODEL</div>", unsafe_allow_html=True)
            laps_x = list(range(0,56))
            fig_t = go.Figure()
            for cmp in ["Soft","Medium","Hard"]:
                deg  = COMPOUND_DEG[cmp]
                perf = [max(0, 100 - max(0, l-deg["laps_peak"]) *
                        (100/(max(1,deg["laps_max"]-deg["laps_peak"])))) for l in laps_x]
                cc   = COMPOUND_COLORS.get(cmp,"#888")
                fig_t.add_trace(go.Scatter(x=laps_x, y=perf, name=cmp,
                    line=dict(color=cc,width=2.5), fill="none"))
                fig_t.add_vline(x=deg["laps_peak"], line_dash="dot",
                    line_color=cc, opacity=0.4,
                    annotation_text=f"{cmp} peak", annotation_font=dict(color=cc,size=9))
            fig_t.update_layout(**DARK, title="Tyre Performance vs Laps Driven",
                               xaxis_title="Laps", yaxis_title="Performance (%)",
                               height=300, legend=dict(x=0.75,y=0.95))
            st.plotly_chart(fig_t, use_container_width=True)

        else:
            st.markdown("""<div style="text-align:center;padding:60px;color:#333">
                <div style="font-size:3rem">🛞</div>
                <div style="font-family:Orbitron;font-size:.85rem;letter-spacing:2px;margin-top:10px">SELECT DRIVER & CIRCUIT</div>
                <div style="font-size:.8rem;color:#444;margin-top:6px">Full strategy recommendation appears here</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 6 — DRIVER ANALYSIS  (all 20 drivers)
# ═══════════════════════════════════════════════════════════════════════
elif page == "👤  Driver Analysis":
    st.markdown("""<div class="banner">
        <div><div class="banner-title">DRIVER ANALYSIS</div>
        <div class="banner-sub">ALL 20 DRIVERS · HISTORICAL DATA 2014–2024</div></div>
        <div class="banner-badge">2025 GRID</div></div>""", unsafe_allow_html=True)

    df = get_df()
    if df is None:
        st.warning("Run training first."); st.stop()

    df14 = df[df["year"] >= 2014].copy()

    # ── Quick stats for all 20 drivers ──
    st.markdown("<div class='sh'>2025 GRID — AT A GLANCE</div>", unsafe_allow_html=True)

    # build summary table
    ref_map = {
        # Refs match actual Kaggle F1 dataset driverRef column exactly
        "Max Verstappen":"max_verstappen",
        "Lewis Hamilton":"hamilton",
        "Charles Leclerc":"leclerc",
        "Lando Norris":"norris",
        "Oscar Piastri":"piastri",
        "George Russell":"russell",
        "Fernando Alonso":"alonso",
        "Carlos Sainz":"sainz",
        "Sergio Perez":"perez",
        "Lance Stroll":"stroll",
        "Valtteri Bottas":"bottas",
        "Yuki Tsunoda":"tsunoda",
        "Pierre Gasly":"gasly",
        "Alexander Albon":"albon",
        "Esteban Ocon":"ocon",
        "Nico Hulkenberg":"hulkenberg",
        "Kimi Antonelli":"antonelli",
        "Jack Doohan":"doohan",
        "Liam Lawson":"lawson",
        "Zhou Guanyu":"zhou",
    }
    summary_rows = []
    for drv in DRIVER_NAMES_2024:
        dref  = DRIVER_DATASET_REF.get(drv, ref_map.get(drv, drv.lower().split()[-1]))
        team  = DRIVER_TEAM_2025.get(drv,"")
        skill = DRIVER_SKILL.get(drv, 0.72)
        car   = TEAM_CAR_RATING.get(team, 0.65)
        ddata = df14[df14["driverRef"]==dref]
        races  = len(ddata)
        wins   = int((ddata["positionOrder"]==1).sum()) if races>0 else 0
        pods   = int((ddata["positionOrder"]<=3).sum()) if races>0 else 0
        avgf   = round(ddata["positionOrder"].mean(),1) if races>0 else "–"
        summary_rows.append({
            "Driver": drv, "Team": team,
            "Skill": f"{int(skill*100)}%", "Car": f"{int(car*100)}%",
            "Races (2014+)": races, "Wins": wins,
            "Podiums": pods, "Avg Finish": avgf,
        })

    df_sum = pd.DataFrame(summary_rows)
    st.dataframe(df_sum, use_container_width=True, hide_index=True)

    # Wins bar chart for all
    fig_all = go.Figure()
    fig_all.add_trace(go.Bar(
        x=[r["Driver"].split()[-1] for r in summary_rows],
        y=[r["Wins"] for r in summary_rows],
        marker_color=[TEAM_COLORS.get(r["Team"],"#555") for r in summary_rows],
        text=[r["Wins"] for r in summary_rows],
        textposition="outside",
        name="Wins",
    ))
    fig_all.update_layout(**DARK, title="Wins (2014–2024) — All 2025 Grid Drivers",
                          yaxis_title="Wins", height=300, xaxis_tickangle=-30)
    st.plotly_chart(fig_all, use_container_width=True)

    # ── Individual driver deep-dive ──
    st.markdown("<div class='sh'>INDIVIDUAL DRIVER DEEP DIVE</div>", unsafe_allow_html=True)
    driver_sel = st.selectbox("Select Driver for Detailed Analysis", DRIVER_NAMES_2024)

    dref_sel = DRIVER_DATASET_REF.get(driver_sel, ref_map.get(driver_sel, driver_sel.lower().split()[-1]))
    ddf      = df14[df14["driverRef"]==dref_sel].copy()
    team_sel = DRIVER_TEAM_2025.get(driver_sel,"")
    tc_sel   = TEAM_COLORS.get(team_sel,"#e10600")
    skill_sel = DRIVER_SKILL.get(driver_sel, 0.72)
    car_sel   = TEAM_CAR_RATING.get(team_sel, 0.65)

    st.markdown(f"""
    <div style="background:#0f0f1e;border:1px solid {tc_sel};border-top:2px solid {tc_sel};
                border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;
                display:flex;align-items:center;justify-content:space-between">
        <div>
            <div style="font-size:1.1rem;font-weight:600;color:#fff">{driver_sel}</div>
            <div style="font-size:.78rem;color:#666">{team_sel} · 2025</div>
        </div>
        <div style="display:flex;gap:18px">
            <div style="text-align:center">
                <div style="font-family:Orbitron;font-size:1.1rem;color:{tc_sel}">{int(skill_sel*100)}</div>
                <div style="font-size:.6rem;color:#444">SKILL</div></div>
            <div style="text-align:center">
                <div style="font-family:Orbitron;font-size:1.1rem;color:{tc_sel}">{int(car_sel*100)}</div>
                <div style="font-size:.6rem;color:#444">CAR</div></div>
            <div style="text-align:center">
                <div style="font-family:Orbitron;font-size:1.1rem;color:{tc_sel}">{len(ddf)}</div>
                <div style="font-size:.6rem;color:#444">RACES</div></div>
            <div style="text-align:center">
                <div style="font-family:Orbitron;font-size:1.1rem;color:{tc_sel}">{int((ddf['positionOrder']==1).sum())}</div>
                <div style="font-size:.6rem;color:#444">WINS</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    if ddf.empty:
        st.info(f"No data in dataset for {driver_sel} (2014+). Shown in summary above from ratings.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Races", len(ddf))
        c2.metric("Avg Finish", f"{ddf['positionOrder'].mean():.1f}")
        c3.metric("Wins", int((ddf["positionOrder"]==1).sum()))
        c4.metric("Podiums", int((ddf["positionOrder"]<=3).sum()))

        tab1,tab2,tab3,tab4 = st.tabs(["📈 Season Trend","🏟 Circuit Form","📊 Year by Year","📋 Race Log"])

        with tab1:
            season = ddf.groupby("year").agg(
                avg_finish=("positionOrder","mean"),
                wins=("positionOrder",lambda x:(x==1).sum()),
                podiums=("positionOrder",lambda x:(x<=3).sum()),
                points=("points","sum"),
            ).reset_index()
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Scatter(x=season["year"],y=season["avg_finish"],
                name="Avg Finish",mode="lines+markers",
                line=dict(color=tc_sel,width=2.5),
                marker=dict(size=8,color=tc_sel)),secondary_y=False)
            fig.add_trace(go.Bar(x=season["year"],y=season["wins"],
                name="Wins",marker_color="#ffd700",opacity=0.5),secondary_y=True)
            fig.update_layout(**DARK,title=f"{driver_sel} — Season Performance",height=360)
            fig.update_yaxes(title_text="Avg Finish",autorange="reversed",secondary_y=False)
            fig.update_yaxes(title_text="Wins",secondary_y=True)
            st.plotly_chart(fig,use_container_width=True)

        with tab2:
            circ = ddf.groupby("circuit_name").agg(
                avg_finish=("positionOrder","mean"),
                races=("raceId","count"),
                wins=("positionOrder",lambda x:(x==1).sum()),
            ).reset_index().sort_values("avg_finish")
            fig2 = px.bar(circ.head(18),x="circuit_name",y="avg_finish",
                          color="avg_finish",
                          color_continuous_scale=["#00c851","#ffd700","#e10600"],
                          title=f"{driver_sel} — Avg Finish by Circuit (lower=better)")
            fig2.update_layout(**DARK,height=380,xaxis_tickangle=-35,
                              yaxis=dict(gridcolor="#111827",linecolor="#1e2030",autorange="reversed"))
            st.plotly_chart(fig2,use_container_width=True)

        with tab3:
            yr = ddf.groupby("year").agg(
                races=("raceId","count"),
                wins=("positionOrder",lambda x:(x==1).sum()),
                podiums=("positionOrder",lambda x:(x<=3).sum()),
                top10=("positionOrder",lambda x:(x<=10).sum()),
                points=("points","sum"),
                avg_finish=("positionOrder","mean"),
            ).reset_index().rename(columns={"year":"Year","races":"Races","wins":"Wins",
                "podiums":"Podiums","top10":"Top10","points":"Points","avg_finish":"Avg Finish"})
            yr["Avg Finish"] = yr["Avg Finish"].round(1)
            st.dataframe(yr, use_container_width=True, hide_index=True)

        with tab4:
            log = ddf[["year","race_name","grid","positionOrder","points","pit_stop_count"]]\
                .sort_values("year",ascending=False)\
                .rename(columns={"year":"Year","race_name":"Race","grid":"Grid P",
                                  "positionOrder":"Finish","points":"Pts","pit_stop_count":"Pit Stops"})
            st.dataframe(log, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 7 — TEAM ANALYSIS  (all 10 teams)
# ═══════════════════════════════════════════════════════════════════════
elif page == "🏭  Team Analysis":
    st.markdown("""<div class="banner">
        <div><div class="banner-title">TEAM ANALYSIS</div>
        <div class="banner-sub">ALL 10 CONSTRUCTORS · 2014–2024 DATA</div></div>
        <div class="banner-badge">CONSTRUCTORS</div></div>""", unsafe_allow_html=True)

    df = get_df()
    if df is None:
        st.warning("Run training first."); st.stop()

    df14 = df[df["year"] >= 2014].copy()

    TEAM_HIST_REF = {
        "Red Bull Racing":"red_bull","Ferrari":"ferrari","McLaren":"mclaren",
        "Mercedes":"mercedes","Aston Martin":"aston_martin","Alpine":"alpine",
        "Williams":"williams","Racing Bulls":"alphatauri","Haas":"haas","Kick Sauber":"sauber",
    }

    # ── All teams summary ──
    st.markdown("<div class='sh'>ALL CONSTRUCTORS — SNAPSHOT</div>", unsafe_allow_html=True)
    team_rows = []
    for team in TEAMS_2024:
        tref  = TEAM_HIST_REF.get(team,"ferrari")
        tdata = df14[df14["constructorRef"]==tref]
        car   = TEAM_CAR_RATING.get(team, 0.65)
        races = len(tdata)
        wins  = int((tdata["positionOrder"]==1).sum()) if races>0 else 0
        pods  = int((tdata["positionOrder"]<=3).sum()) if races>0 else 0
        pts   = int(tdata["points"].sum()) if races>0 else 0
        avgf  = round(tdata["positionOrder"].mean(),1) if races>0 else "–"
        drivers_2025 = [d for d,t in DRIVER_TEAM_2025.items() if t==team]
        team_rows.append({
            "Team": team,"Car Rating": f"{int(car*100)}%",
            "2025 Drivers": " / ".join([d.split()[-1] for d in drivers_2025]),
            "Races (2014+)": races,"Wins": wins,"Podiums": pods,
            "Points": pts,"Avg Finish": avgf,
        })

    df_teams = pd.DataFrame(team_rows)
    st.dataframe(df_teams, use_container_width=True, hide_index=True)

    # Team wins comparison
    fig_tw = go.Figure()
    fig_tw.add_trace(go.Bar(
        x=[r["Team"] for r in team_rows],
        y=[r["Wins"] for r in team_rows],
        marker_color=[TEAM_COLORS.get(r["Team"],"#555") for r in team_rows],
        text=[r["Wins"] for r in team_rows],
        textposition="outside",
    ))
    fig_tw.update_layout(**DARK, title="Constructor Wins (2014–2024)",
                         yaxis_title="Wins", height=300, xaxis_tickangle=-20)
    st.plotly_chart(fig_tw, use_container_width=True)

    # Car ratings
    teams_s = sorted(TEAMS_2024, key=lambda t:TEAM_CAR_RATING.get(t,0), reverse=True)
    st.markdown("<div class='sh'>2025 CAR PERFORMANCE RATINGS</div>", unsafe_allow_html=True)
    for t in teams_s:
        rating = TEAM_CAR_RATING.get(t,0.65)
        color  = TEAM_COLORS.get(t,"#888")
        drivers_2025 = [d for d,tm in DRIVER_TEAM_2025.items() if tm==t]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">
            <div style="width:140px;font-size:.78rem;color:#ccc">{t}</div>
            <div style="flex:1;background:#111;border-radius:3px;height:8px">
                <div style="background:{color};width:{int(rating*100)}%;height:8px;border-radius:3px"></div>
            </div>
            <div style="width:36px;text-align:right;font-family:Orbitron;font-size:.72rem;color:{color}">{int(rating*100)}</div>
            <div style="font-size:.7rem;color:#444">{' · '.join([d.split()[-1] for d in drivers_2025])}</div>
        </div>""", unsafe_allow_html=True)

    # ── Individual team deep-dive ──
    st.markdown("<div class='sh'>TEAM DEEP DIVE</div>", unsafe_allow_html=True)
    team_sel = st.selectbox("Select Team", TEAMS_2024)
    tref_sel = TEAM_HIST_REF.get(team_sel,"ferrari")
    tdf_sel  = df14[df14["constructorRef"]==tref_sel].copy()
    tc_t     = TEAM_COLORS.get(team_sel,"#888")
    car_t    = TEAM_CAR_RATING.get(team_sel, 0.65)
    drivers_t = [d for d,tm in DRIVER_TEAM_2025.items() if tm==team_sel]

    st.markdown(f"""
    <div style="background:#0f0f1e;border:1px solid {tc_t};border-top:2px solid {tc_t};
                border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem">
        <div style="display:flex;align-items:center;justify-content:space-between">
            <div>
                <div style="font-size:1.1rem;font-weight:600;color:#fff">{team_sel}</div>
                <div style="font-size:.75rem;color:#666">2025 Season</div>
            </div>
            <div style="display:flex;gap:18px">
                <div style="text-align:center">
                    <div style="font-family:Orbitron;font-size:1.1rem;color:{tc_t}">{int(car_t*100)}</div>
                    <div style="font-size:.6rem;color:#444">CAR RATING</div></div>
            </div>
        </div>
        <div style="margin-top:10px">
            <div style="font-size:.65rem;color:#444;letter-spacing:1px;margin-bottom:4px">2025 LINEUP</div>
            <div style="display:flex;gap:8px">
    """ + "".join([f'<span style="background:{tc_t}22;color:{tc_t};border:1px solid {tc_t}44;padding:3px 10px;border-radius:12px;font-size:.72rem">{d}</span>' for d in drivers_t]) + """
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    if not tdf_sel.empty:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Races", len(tdf_sel))
        c2.metric("Avg Finish", f"{tdf_sel['positionOrder'].mean():.1f}")
        c3.metric("Wins", int((tdf_sel["positionOrder"]==1).sum()))
        c4.metric("Podiums", int((tdf_sel["positionOrder"]<=3).sum()))

        tab1,tab2,tab3 = st.tabs(["📈 Season Trend","👤 Driver Stats","📋 Race Log"])

        with tab1:
            season_t = tdf_sel.groupby("year").agg(
                avg_finish=("positionOrder","mean"),
                total_points=("points","sum"),
                wins=("positionOrder",lambda x:(x==1).sum()),
                races=("raceId","count"),
            ).reset_index()
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Scatter(x=season_t["year"],y=season_t["avg_finish"],
                name="Avg Finish",mode="lines+markers",
                line=dict(color=tc_t,width=2.5),
                marker=dict(size=8,color=tc_t)),secondary_y=False)
            fig.add_trace(go.Bar(x=season_t["year"],y=season_t["total_points"],
                name="Points",marker_color=tc_t,opacity=0.35),secondary_y=True)
            fig.update_layout(**DARK,title=f"{team_sel} — Season Trend",height=360)
            fig.update_yaxes(title_text="Avg Finish",autorange="reversed",secondary_y=False)
            fig.update_yaxes(title_text="Points",secondary_y=True)
            st.plotly_chart(fig,use_container_width=True)

        with tab2:
            drv_s = tdf_sel.groupby("driver_name").agg(
                races=("raceId","count"),
                avg_finish=("positionOrder","mean"),
                wins=("positionOrder",lambda x:(x==1).sum()),
                podiums=("positionOrder",lambda x:(x<=3).sum()),
                points=("points","sum"),
            ).reset_index().sort_values("avg_finish")
            drv_s.columns = ["Driver","Races","Avg Finish","Wins","Podiums","Points"]
            drv_s["Avg Finish"] = drv_s["Avg Finish"].round(1)
            st.dataframe(drv_s, use_container_width=True, hide_index=True)

        with tab3:
            log_t = tdf_sel[["year","race_name","driver_name","grid","positionOrder","points"]]\
                .sort_values("year",ascending=False)\
                .rename(columns={"year":"Year","race_name":"Race","driver_name":"Driver",
                                  "grid":"Grid","positionOrder":"Finish","points":"Pts"})
            st.dataframe(log_t, use_container_width=True, hide_index=True)
    else:
        st.info("Limited data for this team in 2014–2024 range.")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 8 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown("""<div class="banner">
        <div><div class="banner-title">MODEL PERFORMANCE</div>
        <div class="banner-sub">ACCURACY · F1 · R² · CONFUSION MATRIX · FEATURES</div></div>
        <div class="banner-badge">ML METRICS</div></div>""", unsafe_allow_html=True)

    if not MODELS_TRAINED:
        st.warning("Run training first."); st.stop()

    metrics = get_metrics()
    fi      = get_fi()

    tab1,tab2,tab3 = st.tabs(["🎯 Classification","📈 Regression","🌐 Features"])

    with tab1:
        cls_m = {k:v for k,v in metrics.items() if v.get("type")=="classification"}
        names = list(cls_m.keys())
        accs  = [v["accuracy"]  for v in cls_m.values()]
        precs = [v["precision"] for v in cls_m.values()]
        recs  = [v["recall"]    for v in cls_m.values()]
        f1s   = [v["f1"]        for v in cls_m.values()]
        best  = names[f1s.index(max(f1s))]

        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Best Model", best.split()[0])
        b2.metric("Best Accuracy", f"{max(accs):.1%}")
        b3.metric("Best F1", f"{max(f1s):.4f}")
        b4.metric("Models Compared", len(names))

        fig = go.Figure()
        for metric,vals,color in [("Accuracy",accs,"#e10600"),("Precision",precs,"#ffd700"),
                                    ("Recall",recs,"#0090d0"),("F1",f1s,"#00c851")]:
            fig.add_trace(go.Bar(name=metric,x=names,y=vals,marker_color=color))
        fig.update_layout(**DARK,barmode="group",title="Classification Metrics",height=380,
                          yaxis_title="Score",xaxis_tickangle=-15)
        st.plotly_chart(fig,use_container_width=True)

        best_cm = cls_m[best]["confusion_matrix"]
        fig_cm  = px.imshow(best_cm,text_auto=True,
            color_continuous_scale=["#07070f","#e10600"],
            title=f"Confusion Matrix — {best}",
            x=["Not Top10","Top10"],y=["Not Top10","Top10"])
        fig_cm.update_layout(**DARK,height=340)
        st.plotly_chart(fig_cm,use_container_width=True)

        df_c = pd.DataFrame([{"Model":k,"Accuracy":f"{v['accuracy']:.4f}",
            "Precision":f"{v['precision']:.4f}","Recall":f"{v['recall']:.4f}",
            "F1":f"{v['f1']:.4f}"} for k,v in cls_m.items()])
        st.dataframe(df_c,use_container_width=True,hide_index=True)

    with tab2:
        reg_m = {k:v for k,v in metrics.items() if v.get("type")=="regression"}
        df_r  = pd.DataFrame([{"Model":k,"MAE":f"{v['mae']:.4f}","MSE":f"{v['mse']:.4f}",
            "R²":f"{v['r2']:.4f}"} for k,v in reg_m.items()])
        st.dataframe(df_r,use_container_width=True,hide_index=True)

        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(x=list(reg_m),y=[v["r2"] for v in reg_m.values()],
            marker_color="#00c851",name="R²"))
        fig_r.add_trace(go.Bar(x=list(reg_m),y=[v["mae"] for v in reg_m.values()],
            marker_color="#e10600",name="MAE"))
        fig_r.update_layout(**DARK,barmode="group",title="Regression Metrics",height=350)
        st.plotly_chart(fig_r,use_container_width=True)

        km = metrics.get("K-Means",{})
        if km:
            st.markdown(f"""<div class="card card-blue">
                <h4>K-MEANS CLUSTERING</h4>
                <p>k = {km.get('k')} clusters &nbsp;|&nbsp; Inertia = {km.get('inertia')}<br>
                Groups all drivers into {km.get('k',5)} performance tiers based on grid position,
                lap times, pit stop patterns, and finishing positions.</p>
            </div>""",unsafe_allow_html=True)

    with tab3:
        if fi:
            fi_s     = sorted(fi.items(),key=lambda x:x[1],reverse=True)
            fi_names = [k.replace("_enc","").replace("_"," ").title() for k,_ in fi_s]
            fi_vals  = [v for _,v in fi_s]
            colors   = ["#e10600" if "Grid" in n or "Qual" in n
                        else "#ffd700" if "Driver" in n
                        else "#0090d0" if "Constructor" in n or "Team" in n
                        else "#00c851" for n in fi_names]
            fig_fi = go.Figure(go.Bar(x=fi_vals,y=fi_names,orientation="h",
                marker_color=colors,
                text=[f"{v:.4f}" for v in fi_vals],textposition="outside"))
            fig_fi.update_layout(**DARK,title="Feature Importances — Random Forest",
                                 xaxis=dict(gridcolor="#111827",linecolor="#1e2030",title="Importance"),
                                 yaxis=dict(gridcolor="#111827",linecolor="#1e2030",autorange="reversed"),
                                 height=420)
            st.plotly_chart(fig_fi,use_container_width=True)
