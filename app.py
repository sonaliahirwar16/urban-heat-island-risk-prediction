import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="Urban Heat Risk Prediction", layout="wide")

st.markdown("""
<style>
.main { background-color: #f7f9fb; }
.block-container { padding-top: 1.2rem; }

.card {
  background: white;
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  border-left: 6px solid #2e7d32;
}

.title-text {
  font-size: 28px;
  font-weight: 800;
  margin: 0;
  color: #1565c0;
}

.waiting-text {
  font-size: 28px;
  font-weight: 800;
  margin: 0;
  color: #ef6c00;
}

.sub { color: #444; margin-top: 6px; margin-bottom: 0; }

.badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 14px;
}

.low { background: #e8f5e9; color: #1b5e20; }
.med { background: #fff8e1; color: #8d6e00; }
.high{ background: #ffebee; color: #b71c1c; }

.small { color: #555; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("uhi_best_model.pkl")
    features = joblib.load("uhi_features.pkl")
    le_target = joblib.load("uhi_target_encoder.pkl")
    le_wd = joblib.load("uhi_wd_encoder.pkl")
    le_station = joblib.load("uhi_station_encoder.pkl")
    return model, features, le_target, le_wd, le_station

model, FEATURES, le_target, le_wd, le_station = load_artifacts()


def safe_encode(label_encoder, value: str):
    classes = list(label_encoder.classes_)
    if value in classes:
        return int(label_encoder.transform([value])[0])
    return int(label_encoder.transform([classes[0]])[0])


def make_recommendations(risk_label: str):
    if risk_label == "High":
        return [
            "Increase green cover (trees/green roofs).",
            "Reduce peak-time emissions.",
            "Improve ventilation corridors.",
            "Add shading solutions.",
            "Run heat-alert actions."
        ]
    if risk_label == "Medium":
        return [
            "Improve urban greening.",
            "Encourage off-peak energy use.",
            "Increase reflective surfaces.",
            "Track pollution spikes."
        ]
    return [
        "Maintain green spaces.",
        "Monitor summer heat conditions.",
        "Promote sustainable transport."
    ]


# ----------------------------
# UI
# ----------------------------
st.title("🌆 Urban Heat Risk Prediction")
st.caption("Predict Low / Medium / High heat risk.")

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown("### 🔧 Inputs")

    c1, c2, c3 = st.columns(3)
    with c1:
        pm25 = st.number_input("PM2.5", 0.0, 1000.0, 50.0)
        so2  = st.number_input("SO2", 0.0, 500.0, 10.0)
        pres = st.number_input("Pressure", 900.0, 1100.0, 1010.0)
        rain = st.number_input("Rain", 0.0, 100.0, 0.0)

    with c2:
        pm10 = st.number_input("PM10", 0.0, 1000.0, 80.0)
        no2  = st.number_input("NO2", 0.0, 500.0, 30.0)
        dewp = st.number_input("Dew Point", -50.0, 50.0, 5.0)
        wspm = st.number_input("Wind Speed", 0.0, 50.0, 2.0)

    with c3:
        co   = st.number_input("CO", 0.0, 50.0, 1.0)
        o3   = st.number_input("O3", 0.0, 500.0, 40.0)
        hour = st.slider("Hour", 0, 23, 12)
        month = st.slider("Month", 1, 12, 7)

    st.markdown("### 🌬 Wind Direction")
    wd_value = st.selectbox("Wind Direction", list(le_wd.classes_))

    is_weekend = st.selectbox("Weekend?", [0,1])

    predict_btn = st.button("🔍 Predict Heat Risk", use_container_width=True)


with right:
    st.markdown("### 📊 Result")
    placeholder = st.empty()
    placeholder.markdown("""
    <div class='card'>
        <p class='waiting-text'>Waiting for input…</p>
        <p class='sub'>Enter values and click Predict.</p>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------
# Prediction
# ----------------------------
if predict_btn:

    wd_enc = safe_encode(le_wd, wd_value)
    default_station = str(list(le_station.classes_)[0])
    station_enc = safe_encode(le_station, default_station)

    # INTERNAL DEFAULT (instead of UI)
    day_of_year = 200

    pollution_index = pm25 + pm10 + so2 + no2 + co + o3
    cooling_index = wspm + rain
    humidity_proxy = dewp

    row = {
        "PM2.5": pm25,
        "PM10": pm10,
        "SO2": so2,
        "NO2": no2,
        "CO": co,
        "O3": o3,
        "PRES": pres,
        "DEWP": dewp,
        "RAIN": rain,
        "WSPM": wspm,
        "pollution_index": pollution_index,
        "cooling_index": cooling_index,
        "humidity_proxy": humidity_proxy,
        "hour": hour,
        "month": month,
        "day_of_year": day_of_year,
        "is_weekend": is_weekend,
        "wd_enc": wd_enc,
        "station_enc": station_enc
    }

    X_in = pd.DataFrame([row])[FEATURES]

    pred_class = int(model.predict(X_in)[0])
    pred_label = le_target.inverse_transform([pred_class])[0]

    css_class = "low" if pred_label=="Low" else "med" if pred_label=="Medium" else "high"

    placeholder.markdown(f"""
    <div class="card">
        <p class="title-text">Heat Risk: <span class="badge {css_class}">{pred_label}</span></p>
        <p class="sub">Prediction using XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✅ Recommendations")
    for r in make_recommendations(pred_label):
        st.write("• " + r)
