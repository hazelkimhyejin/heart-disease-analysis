import streamlit as st
import numpy as np
import pickle
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal']

FEATURE_LABELS = {
    'age':      'Age',
    'sex':      'Sex',
    'cp':       'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol':     'Serum Cholesterol',
    'fbs':      'Fasting Blood Sugar > 120 mg/dL',
    'restecg':  'Resting ECG Results',
    'thalach':  'Max Heart Rate Achieved',
    'exang':    'Exercise-Induced Angina',
    'oldpeak':  'ST Depression (Oldpeak)',
    'slope':    'Slope of Peak Exercise ST',
    'ca':       'Major Vessels (Fluoroscopy)',
    'thal':     'Thalassemia',
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --cream:   #F5F0E8;
    --ink:     #1A1208;
    --red:     #C0392B;
    --red-lt:  #E74C3C;
    --green:   #1A6B3C;
    --green-lt:#27AE60;
    --gold:    #B8960C;
    --border:  #D4C9B0;
    --card-bg: #FDFAF4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream) !important;
    color: var(--ink);
}

/* Remove default Streamlit padding */
.block-container { padding-top: 2rem !important; max-width: 1100px; }

/* Header */
.app-header {
    border-bottom: 2px solid var(--ink);
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.1;
    color: var(--ink);
    margin: 0;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #6B5E45;
    margin-top: 0.4rem;
    font-weight: 300;
    letter-spacing: 0.03em;
}
.badge {
    display: inline-block;
    background: var(--ink);
    color: var(--cream);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    margin-bottom: 0.8rem;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #8B7355;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
    margin-top: 1.8rem;
}

/* Result cards */
.result-card {
    border: 2px solid var(--ink);
    border-radius: 4px;
    padding: 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.result-card.disease {
    background: #FDF0EE;
    border-color: var(--red);
}
.result-card.no-disease {
    background: #EEF7F2;
    border-color: var(--green);
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0;
}
.result-label.disease { color: var(--red); }
.result-label.no-disease { color: var(--green); }
.result-confidence {
    font-size: 0.85rem;
    color: #6B5E45;
    margin-top: 0.3rem;
}

/* Progress bar override */
.stProgress > div > div { background-color: var(--ink) !important; }

/* Feature importance bars */
.feat-bar-wrap { margin: 0.5rem 0; }
.feat-bar-label {
    font-size: 0.82rem;
    color: var(--ink);
    margin-bottom: 2px;
    display: flex;
    justify-content: space-between;
}
.feat-bar-track {
    background: var(--border);
    border-radius: 2px;
    height: 8px;
    width: 100%;
}
.feat-bar-fill {
    height: 8px;
    border-radius: 2px;
    transition: width 0.6s ease;
}
.feat-bar-fill.pos { background: var(--red); }
.feat-bar-fill.neg { background: var(--green); }

/* Metric boxes */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    background: var(--card-bg);
    text-align: center;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: var(--ink);
}
.metric-name {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8B7355;
    margin-top: 0.2rem;
}

/* Info box */
.info-box {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    padding: 0.9rem 1.1rem;
    border-radius: 2px;
    font-size: 0.84rem;
    color: #5A4B30;
    margin: 1rem 0;
}

/* Streamlit widget label overrides */
label { font-size: 0.85rem !important; font-weight: 500 !important; color: var(--ink) !important; }
.stSelectbox > div, .stNumberInput > div { border-color: var(--border) !important; }

/* Button */
.stButton > button {
    background: var(--ink) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="badge">Clinical ML · UCI Dataset · 1,025 Patients</div>
    <h1 class="app-title">🫀 Heart Disease<br>Risk Predictor</h1>
    <p class="app-subtitle">
        Logistic regression model · 87.4% recall · Optimised for medical sensitivity<br>
        Built by <strong>Hazel I.</strong> · <a href="https://github.com/hazelkimhyejin/heart-disease-analysis" target="_blank" style="color:#B8960C;">github.com/hazelkimhyejin</a>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-label">Patient Clinical Data</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age      = st.number_input("Age (years)", min_value=20, max_value=85, value=54)
        trestbps = st.number_input("Resting BP (mmHg)", min_value=80, max_value=220, value=131)
        thalach  = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=210, value=150)
        oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        chol     = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=246)
    with c2:
        sex      = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        cp       = st.selectbox("Chest Pain Type", options=[0,1,2,3],
                                format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal",3:"Asymptomatic"}[x])
        exang    = st.selectbox("Exercise Angina", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
        slope    = st.selectbox("ST Slope", options=[0,1,2], format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
        ca       = st.selectbox("Major Vessels (0–3)", options=[0,1,2,3])

    c3, c4 = st.columns(2)
    with c3:
        fbs     = st.selectbox("Fasting BS > 120 mg/dL", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
        restecg = st.selectbox("Resting ECG", options=[0,1,2],
                               format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
    with c4:
        thal    = st.selectbox("Thalassemia", options=[1,2,3],
                               format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x])

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Prediction →")

# ── Right panel ────────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Prediction & Analysis</div>', unsafe_allow_html=True)

    if not run:
        st.markdown("""
        <div class="info-box">
            Enter the patient's clinical values on the left and click <strong>Run Prediction</strong> to generate a risk assessment.
        </div>
        """, unsafe_allow_html=True)

        # Model metrics
        st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">87.4%</div>
                <div class="metric-name">Recall</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">75.6%</div>
                <div class="metric-name">Precision</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">79.5%</div>
                <div class="metric-name">Accuracy</div>
            </div>
        </div>
        <div class="info-box">
            <strong>Why recall is the key metric:</strong> In medical screening, missing a sick patient (false negative) is far more dangerous than a false alarm (false positive). This model is tuned with a 0.40 threshold to maximise recall.
        </div>
        """, unsafe_allow_html=True)

        # Top predictors
        st.markdown('<div class="section-label">Top Clinical Predictors</div>', unsafe_allow_html=True)
        coefs = model.coef_[0]
        feat_names = FEATURES
        pairs = sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:6]
        max_abs = max(abs(c) for _, c in pairs)
        for fname, coef in pairs:
            pct = int(abs(coef) / max_abs * 100)
            direction = "pos" if coef > 0 else "neg"
            arrow = "↑ Risk" if coef > 0 else "↓ Risk"
            label = FEATURE_LABELS.get(fname, fname)
            st.markdown(f"""
            <div class="feat-bar-wrap">
                <div class="feat-bar-label">
                    <span>{label}</span>
                    <span style="color:{'#C0392B' if direction=='pos' else '#1A6B3C'};font-size:0.75rem">{arrow}</span>
                </div>
                <div class="feat-bar-track">
                    <div class="feat-bar-fill {direction}" style="width:{pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── Run inference ──────────────────────────────────────────────────────
        with st.spinner("Analysing patient data…"):
            time.sleep(0.6)

        X_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
        X_scaled = scaler.transform(X_input)

        prob_disease = model.predict_proba(X_scaled)[0][1]
        threshold    = 0.40
        prediction   = int(prob_disease >= threshold)
        confidence   = prob_disease if prediction == 1 else (1 - prob_disease)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card disease">
                <p class="result-label disease">⚠ Elevated Risk Detected</p>
                <p class="result-confidence">Model confidence: {prob_disease*100:.1f}% probability of heart disease</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card no-disease">
                <p class="result-label no-disease">✓ Lower Risk Indicated</p>
                <p class="result-confidence">Model confidence: {(1-prob_disease)*100:.1f}% probability of no disease</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk probability bar
        st.markdown('<div class="section-label">Risk Probability Score</div>', unsafe_allow_html=True)
        st.progress(prob_disease)
        st.caption(f"Disease probability: **{prob_disease*100:.1f}%** (threshold: 40%)")

        # Feature contributions for THIS patient
        st.markdown('<div class="section-label">What Drove This Prediction</div>', unsafe_allow_html=True)
        coefs       = model.coef_[0]
        vals_scaled = X_scaled[0]
        contribs    = coefs * vals_scaled
        pairs       = sorted(zip(FEATURES, contribs), key=lambda x: abs(x[1]), reverse=True)[:6]
        max_abs     = max(abs(c) for _, c in pairs) or 1
        for fname, contrib in pairs:
            pct       = int(abs(contrib) / max_abs * 100)
            direction = "pos" if contrib > 0 else "neg"
            arrow     = "↑ Raises risk" if contrib > 0 else "↓ Lowers risk"
            label     = FEATURE_LABELS.get(fname, fname)
            st.markdown(f"""
            <div class="feat-bar-wrap">
                <div class="feat-bar-label">
                    <span>{label}</span>
                    <span style="color:{'#C0392B' if direction=='pos' else '#1A6B3C'};font-size:0.75rem">{arrow}</span>
                </div>
                <div class="feat-bar-track">
                    <div class="feat-bar-fill {direction}" style="width:{pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:1.2rem">
            ⚕ <strong>Disclaimer:</strong> This tool is for portfolio demonstration only. It is not a medical device and should not be used for clinical decision-making.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;font-size:0.78rem;color:#8B7355;padding:0.5rem 0">
    <span>Built by <strong>Hazel I.</strong> · UCI Cleveland Heart Disease Dataset · 1,025 patients</span>
    <span>
        <a href="https://github.com/hazelkimhyejin/heart-disease-analysis" target="_blank" style="color:#B8960C;text-decoration:none">GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://linkedin.com/in/hazel-ip-jl" target="_blank" style="color:#B8960C;text-decoration:none">LinkedIn</a>
    </span>
</div>
""", unsafe_allow_html=True)
