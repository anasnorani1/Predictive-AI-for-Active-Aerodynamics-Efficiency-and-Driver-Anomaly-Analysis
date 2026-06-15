import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Aero Intelligence",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── F1 Dark Theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global background */
    .stApp { background-color: #1a1a2e; color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #0f3460; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }

    /* Inputs */
    .stSlider > div > div > div { background-color: #e94560 !important; }
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        background-color: #16213e !important;
        color: #ffffff !important;
        border: 1px solid #e94560 !important;
        border-radius: 6px;
    }
    .stRadio label { color: #ffffff !important; }

    /* Cards */
    .card {
        background: #16213e;
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .card-title {
        color: #e94560;
        font-size: 13px;
        font-family: monospace;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .card-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        font-family: monospace;
    }
    .card-sub {
        color: #aaaaaa;
        font-size: 12px;
        font-family: monospace;
        margin-top: 4px;
    }

    /* Prediction badge */
    .badge-xmode {
        background: #e94560;
        color: #ffffff;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 22px;
        font-weight: 700;
        font-family: monospace;
        text-align: center;
        display: block;
    }
    .badge-zmode {
        background: #0f3460;
        color: #ffffff;
        border-radius: 8px;
        border: 2px solid #e94560;
        padding: 10px 20px;
        font-size: 22px;
        font-weight: 700;
        font-family: monospace;
        text-align: center;
        display: block;
    }

    /* Progress bar override */
    .stProgress > div > div > div > div { background-color: #e94560 !important; }

    /* Headers */
    h1, h2, h3 { color: #e94560 !important; font-family: monospace !important; }
    .stMarkdown p { color: #cccccc; font-family: monospace; }

    /* Divider */
    hr { border-color: #e94560 !important; opacity: 0.3; }

    /* Metric */
    [data-testid="stMetricValue"] { color: #e94560 !important; font-family: monospace !important; }
    [data-testid="stMetricLabel"] { color: #aaaaaa !important; font-family: monospace !important; }

    /* Button */
    .stButton button {
        background-color: #e94560 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: monospace !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 12px 30px !important;
        width: 100% !important;
    }
    .stButton button:hover { background-color: #c73652 !important; }

    /* Info box */
    .stInfo { background-color: #16213e !important; border-left: 4px solid #e94560 !important; }

    /* Sidebar label */
    .sidebar-section {
        color: #e94560;
        font-family: monospace;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 8px 0 4px 0;
        border-bottom: 1px solid #e9456040;
        margin-bottom: 8px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MASS_KG = 798
FEATURE_COLS = [
    'Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'Acceleration',
    'Engine_Load', 'Elevation_Delta', 'Tire_Age_Laps', 'Compound_Encoded',
    'Speed_Rolling_Avg', 'Kinetic_Energy_MJ', 'Longitudinal_Force_N',
    'Energy_Efficiency_Ratio', 'High_Speed_Zone', 'Heavy_Braking', 'Gear_Shift_Active',
    'Track_Zone'
]

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    required = {
        'scaler': 'artefacts/standard_scaler.pkl',
        'lr':     'models/logistic_regression.pkl',
        'rf':     'models/random_forest.pkl',
        'iso':    'models/isolation_forest.pkl',
    }
    models, errors = {}, []
    for key, path in required.items():
        try:
            models[key] = joblib.load(path)
        except FileNotFoundError:
            errors.append(path)
            models[key] = None
    return models, errors

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <div style='font-family:monospace; font-size:13px; color:#aaaaaa; letter-spacing:3px;'>CS-245 MACHINE LEARNING — AERO INTELLIGENCE</div>
    <h1 style='font-size:36px; margin:8px 0;'>F1 Active Aerodynamics Predictor</h1>
    <div style='font-family:monospace; font-size:13px; color:#aaaaaa;'>
        2026 Japanese GP &nbsp;|&nbsp; Red Bull Telemetry &nbsp;|&nbsp; Suzuka Circuit
    </div>
    <div style='font-family:monospace; font-size:12px; color:#666; margin-top:6px;'>
        Hanan Majeed &nbsp;·&nbsp; Maha Mohsin &nbsp;·&nbsp; Anas Norani
    </div>
</div>
<hr/>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
models, missing = load_models()
if missing:
    st.warning(
        f"Could not find model files: `{'`, `'.join(missing)}`\n\n"
        "Make sure you have run **01_Data_Preprocessing.ipynb** and **03_Modelling.ipynb** first "
        "so the `artefacts/` and `models/` folders are generated."
    )


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Telemetry Inputs
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='text-align:center; font-family:monospace; font-size:18px; font-weight:700; color:#e94560; padding:10px 0;'>TELEMETRY INPUT</div>", unsafe_allow_html=True)

    # ── Speed & Motion ─────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>Speed & Motion</div>", unsafe_allow_html=True)
    speed        = st.slider("Speed (km/h)",        0,   375,  280, step=1)
    rpm          = st.slider("RPM",                 0, 18000, 11500, step=100)
    n_gear       = st.selectbox("Gear",             list(range(1, 9)), index=5)
    acceleration = st.number_input("Acceleration (km/h/s)", value=5.0, step=0.5, format="%.2f")

    # ── Pedals & Load ──────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>Pedals & Engine</div>", unsafe_allow_html=True)
    throttle    = st.slider("Throttle (%)",    0, 100, 85)
    engine_load = st.slider("Engine Load (%)", 0, 100, 78)
    brake       = st.radio("Brake", ["Off (0)", "On (1)"], horizontal=True)
    brake_val   = 1 if "On" in brake else 0

    # ── Circuit Conditions ─────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>Circuit Conditions</div>", unsafe_allow_html=True)
    elevation_delta = st.number_input("Elevation Delta (m)", value=0.5, step=0.1, format="%.2f")
    high_speed_zone = st.radio("High Speed Zone", ["No (0)", "Yes (1)"], horizontal=True)
    hsz_val         = 1 if "Yes" in high_speed_zone else 0
    track_zone      = st.selectbox(
        "Track Zone (K-Means cluster)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 — Low Speed / Chicane",
            1: "1 — Medium Speed",
            2: "2 — High Speed",
            3: "3 — Braking Zone",
        }[x],
        index=2,
    )

    # ── Tyre & Flags ───────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>Tyre & Flags</div>", unsafe_allow_html=True)
    compound        = st.radio("Compound", ["HARD (0)", "MEDIUM (1)"], horizontal=True)
    compound_val    = 1 if "MEDIUM" in compound else 0
    tire_age        = st.number_input("Tire Age (laps)", min_value=0, max_value=60, value=12, step=1)
    heavy_braking   = st.radio("Heavy Braking", ["No (0)", "Yes (1)"], horizontal=True)
    hb_val          = 1 if "Yes" in heavy_braking else 0
    gear_shift      = st.radio("Gear Shift Active", ["No (0)", "Yes (1)"], horizontal=True)
    gs_val          = 1 if "Yes" in gear_shift else 0

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("RUN PREDICTION")

# ── Derive physics features ────────────────────────────────────────────────────
ke_mj     = (0.5 * MASS_KG * (speed / 3.6) ** 2) / 1_000_000
long_f    = MASS_KG * (acceleration / 3.6)
eer       = speed / (engine_load + 1)
spd_roll  = float(speed)   # best single-point approximation

# ── Assemble feature vector ────────────────────────────────────────────────────
feature_vector = np.array([[
    speed, rpm, n_gear, throttle, brake_val, acceleration,
    engine_load, elevation_delta, tire_age, compound_val,
    spd_roll, ke_mj, long_f, eer, hsz_val, hb_val, gs_val, track_zone
]])

# ════════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ════════════════════════════════════════════════════════════════════════════════

# ── Derived feature display ────────────────────────────────────────────────────
st.markdown("### Derived Physics Features")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class='card'>
        <div class='card-title'>Kinetic Energy</div>
        <div class='card-value'>{ke_mj:.3f} MJ</div>
        <div class='card-sub'>0.5 x 798kg x v²</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='card'>
        <div class='card-title'>Longitudinal Force</div>
        <div class='card-value'>{long_f:,.0f} N</div>
        <div class='card-sub'>798kg x acceleration</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='card'>
        <div class='card-title'>Energy Efficiency Ratio</div>
        <div class='card-value'>{eer:.2f}</div>
        <div class='card-sub'>Speed / (Engine Load + 1)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ── Predictions ────────────────────────────────────────────────────────────────
st.markdown("### Model Predictions")

if predict_btn or True:   # show live (updates on every slider change)
    scaler = models['scaler']
    lr     = models['lr']
    rf     = models['rf']
    iso    = models['iso']

    if scaler is None or lr is None or rf is None:
        st.error("Models not loaded. Please run the notebooks first to generate model files.")
    else:
        # Scaler was fitted on 17 features (Notebook 01).
        # Track_Zone was appended AFTER scaling in Notebook 03, so it must not pass through the scaler.
        X_17      = feature_vector[:, :17]          # first 17 features → scale
        X_tz      = feature_vector[:, 17:]          # Track_Zone → keep raw
        X_scaled  = np.hstack([scaler.transform(X_17), X_tz])

        # Logistic Regression
        lr_pred  = lr.predict(X_scaled)[0]
        lr_proba = lr.predict_proba(X_scaled)[0]   # [P(Z), P(X)]

        # Random Forest
        rf_pred  = rf.predict(X_scaled)[0]
        rf_proba = rf.predict_proba(X_scaled)[0]

        # Isolation Forest anomaly
        # Isolation Forest uses only 6 driver-input features (raw — tree-based, scaling not needed)
        ISO_FEATURES = ['Throttle', 'Brake', 'Acceleration', 'Engine_Load', 'Heavy_Braking', 'Gear_Shift_Active']
        ISO_IDX      = [FEATURE_COLS.index(f) for f in ISO_FEATURES]
        X_iso        = feature_vector[:, ISO_IDX]
        iso_pred     = iso.predict(X_iso)[0]        # -1 = anomaly, 1 = normal
        iso_score    = iso.decision_function(X_iso)[0]
        is_anomaly   = iso_pred == -1

        # ── Model result cards ──────────────────────────────────────────────────
        col_lr, col_rf = st.columns(2)

        def mode_label(pred):
            return "X-MODE (Optimal)" if pred == 1 else "Z-MODE (Suboptimal)"

        def mode_badge(pred):
            cls = "badge-xmode" if pred == 1 else "badge-zmode"
            return f"<span class='{cls}'>{mode_label(pred)}</span>"

        with col_lr:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>Logistic Regression</div>
                {mode_badge(lr_pred)}
                <div style='margin-top:14px;'>
                    <div style='font-family:monospace; font-size:12px; color:#aaaaaa; margin-bottom:4px;'>
                        X-Mode confidence: {lr_proba[1]*100:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(lr_proba[1]))
            st.caption(f"Z-Mode: {lr_proba[0]*100:.1f}%  |  X-Mode: {lr_proba[1]*100:.1f}%")

        with col_rf:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>Random Forest</div>
                {mode_badge(rf_pred)}
                <div style='margin-top:14px;'>
                    <div style='font-family:monospace; font-size:12px; color:#aaaaaa; margin-bottom:4px;'>
                        X-Mode confidence: {rf_proba[1]*100:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(rf_proba[1]))
            st.caption(f"Z-Mode: {rf_proba[0]*100:.1f}%  |  X-Mode: {rf_proba[1]*100:.1f}%")

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Anomaly Detection ───────────────────────────────────────────────────
        st.markdown("### Isolation Forest — Driver Anomaly Detection")
        col_a, col_b = st.columns([1, 2])

        with col_a:
            if is_anomaly:
                st.markdown("""
                <div class='card' style='border-color:#ff4444;'>
                    <div class='card-title' style='color:#ff4444;'>Anomaly Status</div>
                    <div class='card-value' style='color:#ff4444; font-size:22px;'>ANOMALY DETECTED</div>
                    <div class='card-sub'>Driver behaviour deviates from learned telemetry patterns</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='card' style='border-color:#4caf50;'>
                    <div class='card-title' style='color:#4caf50;'>Anomaly Status</div>
                    <div class='card-value' style='color:#4caf50; font-size:22px;'>NORMAL</div>
                    <div class='card-sub'>Telemetry is consistent with normal driving patterns</div>
                </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>Anomaly Score</div>
                <div class='card-value'>{iso_score:.4f}</div>
                <div class='card-sub'>Negative = more anomalous &nbsp;|&nbsp; Positive = more normal</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── K-Means Track Zone Section ──────────────────────────────────────────
        st.markdown("### K-Means Clustering — Track Zone Analysis")

        zone_data = {
            0: {
                "name":  "Low Speed / Chicane",
                "color": "#f5a623",
                "desc":  "Tight corners and chicanes. Car is slow, braking hard, low gear. Z-Mode mandatory — maximum downforce needed for grip.",
                "speed": "< 160 km/h",
                "gear":  "1 — 4",
                "aero":  "Z-Mode",
                "example": "Suzuka Hairpin, Casino Triangle",
            },
            1: {
                "name":  "Medium Speed / Corner",
                "color": "#4fc3f7",
                "desc":  "Flowing medium-speed corners. Partial throttle, transitional braking. Z-Mode preferred — downforce still critical for cornering stability.",
                "speed": "160 — 240 km/h",
                "gear":  "4 — 6",
                "aero":  "Z-Mode",
                "example": "Suzuka Esses, Spoon Curve",
            },
            2: {
                "name":  "High Speed Straight",
                "color": "#e94560",
                "desc":  "Full throttle straights. Maximum speed, highest gear, no braking. X-Mode optimal — low drag configuration delivers maximum straight-line speed.",
                "speed": "> 240 km/h",
                "gear":  "7 — 8",
                "aero":  "X-Mode",
                "example": "Suzuka Back Straight, Start/Finish",
            },
            3: {
                "name":  "Braking Zone",
                "color": "#ab47bc",
                "desc":  "Hard braking events before corners. Rapid deceleration, downshifting. Z-Mode critical — aerodynamic stability during heavy braking prevents oversteer.",
                "speed": "Decelerating",
                "gear":  "Downshifting",
                "aero":  "Z-Mode",
                "example": "Braking for Spoon, Hairpin entry",
            },
        }

        st.markdown("""
        <div class='card' style='margin-bottom:18px;'>
            <div class='card-title'>How K-Means Was Used</div>
            <div style='font-family:monospace; font-size:13px; color:#cccccc; line-height:1.7;'>
                K-Means (k=4) was trained on <b style='color:#e94560;'>Speed, X, Y, Z coordinates</b> from 63,673 telemetry rows.
                It autonomously identified 4 distinct track zones with no human labelling.
                The resulting <b style='color:#e94560;'>Track_Zone (0–3)</b> was appended as the 18th feature
                to every row before training Logistic Regression and Random Forest —
                giving both supervised models spatial context about where on the circuit each reading was taken.
            </div>
        </div>
        """, unsafe_allow_html=True)

        zone_cards_parts = []
        for zone_id, info in zone_data.items():
            is_active = (zone_id == track_zone)
            border_width = "3px" if is_active else "1px"
            bg = "#1e1e3a" if is_active else "#16213e"
            active_tag = f"<div style='color:#e94560;font-family:monospace;font-size:11px;font-weight:700;margin-bottom:6px;'>ACTIVE ZONE</div>" if is_active else ""
            card = (
                f"<div style='flex:1;background:{bg};border:{border_width} solid {info['color']};"
                f"border-radius:10px;padding:16px;'>"
                f"{active_tag}"
                f"<div style='color:{info['color']};font-family:monospace;font-size:11px;letter-spacing:1px;font-weight:700;'>ZONE {zone_id}</div>"
                f"<div style='color:#ffffff;font-family:monospace;font-size:14px;font-weight:700;margin:6px 0;'>{info['name']}</div>"
                f"<div style='color:#aaaaaa;font-family:monospace;font-size:11px;line-height:1.5;margin-bottom:10px;'>{info['desc']}</div>"
                f"<div style='border-top:1px solid #2d2d44;padding-top:8px;'>"
                f"<div style='font-family:monospace;font-size:11px;color:#cccccc;'>Speed: <span style='color:{info['color']};'>{info['speed']}</span></div>"
                f"<div style='font-family:monospace;font-size:11px;color:#cccccc;'>Gear: <span style='color:{info['color']};'>{info['gear']}</span></div>"
                f"<div style='font-family:monospace;font-size:11px;color:#cccccc;'>Aero: <span style='color:{info['color']};font-weight:700;'>{info['aero']}</span></div>"
                f"<div style='font-family:monospace;font-size:10px;color:#666666;margin-top:4px;'>e.g. {info['example']}</div>"
                f"</div></div>"
            )
            zone_cards_parts.append(card)
        zone_cards_html = "<div style='display:flex;gap:12px;margin-bottom:16px;'>" + "".join(zone_cards_parts) + "</div>"
        st.markdown(zone_cards_html, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Active zone summary bar
        az = zone_data[track_zone]
        st.markdown(f"""
        <div style='background:#16213e; border-left:4px solid {az["color"]};
                    border-radius:0 8px 8px 0; padding:14px 20px; font-family:monospace;'>
            <span style='color:{az["color"]}; font-weight:700; font-size:13px;'>
                ACTIVE: Zone {track_zone} — {az["name"]}
            </span>
            <span style='color:#aaaaaa; font-size:12px; margin-left:16px;'>
                K-Means cluster label appended as feature 18 to both LR and RF models
            </span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style='color:#cccccc; font-size:12px;'>
                Recommended Aero: <b style='color:{az["color"]};'>{az["aero"]}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Physics-based rule check ────────────────────────────────────────────
        st.markdown("### Physics Rule Check (Optimal Aero Logic)")

        rules = {
            "Speed > 240 km/h":       speed > 240,
            "Not braking":            brake_val == 0,
            "Gear >= 6":              n_gear >= 6,
            "No heavy braking":       hb_val == 0,
            "High speed zone":        hsz_val == 1,
            "Elevation > -3 m":       elevation_delta > -3,
        }

        cols = st.columns(3)
        for idx, (rule, passed) in enumerate(rules.items()):
            with cols[idx % 3]:
                color  = "#4caf50" if passed else "#e94560"
                symbol = "PASS" if passed else "FAIL"
                st.markdown(f"""
                <div style='background:#16213e; border:1px solid {color}; border-radius:8px;
                            padding:10px 14px; margin-bottom:10px; font-family:monospace;'>
                    <span style='color:{color}; font-weight:700;'>[{symbol}]</span>
                    <span style='color:#cccccc; font-size:13px;'> {rule}</span>
                </div>""", unsafe_allow_html=True)

        physics_optimal = all(rules.values())
        if physics_optimal:
            st.success("All physics conditions met — X-Mode (optimal aerodynamics) is recommended.")
        else:
            failed = [r for r, p in rules.items() if not p]
            st.info(f"Z-Mode recommended. Failing conditions: {', '.join(failed)}.")

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Full feature vector table ────────────────────────────────────────────
        with st.expander("View Full Feature Vector (scaled input sent to models)"):
            df_display = pd.DataFrame({
                'Feature': FEATURE_COLS,
                'Raw Value': feature_vector[0],
                'Scaled Value': X_scaled[0]
            })
            df_display['Raw Value']    = df_display['Raw Value'].round(4)
            df_display['Scaled Value'] = df_display['Scaled Value'].round(4)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            