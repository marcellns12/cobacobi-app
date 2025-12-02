# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ----------------------------------------
# Load model
# ----------------------------------------
MODEL_PATH = Path("model_health_budget_xgboost.pkl")
if not MODEL_PATH.exists():
    st.error(f"File model tidak ditemukan: {MODEL_PATH.resolve()}\nPastikan 'model_health_budget_xgboost.pkl' ada di folder yang sama dengan app.py")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----------------------------------------
# Page config + CSS (HTML)
# ----------------------------------------
st.set_page_config(page_title="Health Budget Predictor", page_icon="💊", layout="centered")

# Minimal CSS to make things nicer
st.markdown(
    """
    <style>
    .app-header {
        background: linear-gradient(90deg,#4B79A1,#283E51);
        padding: 18px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 18px;
    }
    .card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .predict-btn {
        background: linear-gradient(90deg,#11998e,#38ef7d);
        color: white;
        font-weight: 600;
    }
    .small-muted { color: #6c757d; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-header"><h2>💊 Health Budget Predictor</h2><div class="small-muted">Prediksi biaya medis (charges) menggunakan model XGBoost + feature engineering</div></div>', unsafe_allow_html=True)

# ----------------------------------------
# Inputs (user)
# ----------------------------------------
st.markdown("<div class='card'><h4>Input Data Pasien</h4></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=32, step=1)
    sex = st.selectbox("Sex", options=["male", "female"], index=0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1, format="%.1f")
with col2:
    children = st.number_input("Children (count)", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker", options=["no", "yes"], index=0)
    region = st.selectbox("Region", options=["southeast","southwest","northeast","northwest"], index=0)

st.markdown("<div class='small-muted'>Tips: gunakan nilai yang realistis agar prediksi berguna.</div>", unsafe_allow_html=True)

# ----------------------------------------
# Create derived features (matching training)
# ----------------------------------------
def make_feature_row(age, sex, bmi, children, smoker, region):
    # bmi_category same bins used in training
    bmi_cat = pd.cut(
        [bmi],
        bins=[0,18.5,25,30,100],
        labels=["underweight","normal","overweight","obese"]
    )[0]

    age_bmi = age * bmi
    smoker_bmi = bmi * (1 if smoker == "yes" else 0)
    age_children = age * children
    is_obese = 1 if bmi >= 30 else 0

    # Build DataFrame with same column names as training X
    row = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "bmi_category": str(bmi_cat),
        "age_bmi": age_bmi,
        "smoker_bmi": smoker_bmi,
        "age_children": age_children,
        "is_obese": is_obese
    }
    return pd.DataFrame([row])

# ----------------------------------------
# Predict button
# ----------------------------------------
if st.button("Predict 💡", key="predict_btn"):
    input_df = make_feature_row(age, sex, bmi, children, smoker, region)

    try:
        pred = model.predict(input_df)
        # model.predict returns array-like
        pred_val = float(pred[0])

        # display nice card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Hasil Prediksi Biaya Kesehatan")
        st.markdown(f"<h2>Rp {pred_val:,.0f}</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Prediksi ini berdasarkan model ML (XGBoost) dan feature engineering. Hasil bukan pengganti nasihat profesional.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Show input summary + optional debugging table
        with st.expander("Lihat input (debug)"):
            st.table(input_df.T.rename(columns={0:"value"}))

    except Exception as e:
        st.error("Terjadi kesalahan saat prediksi: " + str(e))

# ----------------------------------------
# Footer / info
# ----------------------------------------
st.markdown("---")
st.markdown("Made with ❤️ — Masukkan input lalu klik **Predict**. Pastikan file `model_health_budget_xgboost.pkl` ada di folder yang sama.")
