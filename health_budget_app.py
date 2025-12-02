import streamlit as st
import joblib
import pandas as pd
import os

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Health Budget AI Advisor",
    page_icon="🩺",
    layout="centered"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.header-box {
    background: linear-gradient(90deg, #3B82F6, #10B981);
    color: white;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0px 5px 25px rgba(0,0,0,0.15);
}

.card {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.big-money {
    font-size: 26px;
    font-weight: 700;
}

.small-label {
    font-size: 13px;
    color: #6b7280;
}

div.stButton > button:first-child {
    background: linear-gradient(90deg,#10b981,#06b6d4);
    border: none;
    color: white;
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 600;
}
div.stButton > button:first-child:hover {
    filter: brightness(0.95);
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("""
<div class="header-box">
    <h2>🩺 Health Budget AI Advisor</h2>
    <p>Estimasi biaya kesehatan & rekomendasi dana darurat</p>
</div>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL (JOBLIB ONLY)
# =============================
MODEL_PATH = "model_health_budget.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model tidak ditemukan. Pastikan file **model_health_budget.pkl** sudah diupload.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Gagal memuat model!")
    st.code(str(e))
    st.stop()

# =============================
# INPUT FORM
# =============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Input Data Pribadi")

age = st.slider("Usia", 18, 100, 30)
sex = st.selectbox("Jenis Kelamin", ["male", "female"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input("Jumlah Anak Ditanggung", 0, 10, 0)
smoker = st.selectbox("Perokok?", ["yes", "no"])
region = st.selectbox("Wilayah", ["southeast", "southwest", "northwest", "northeast"])

st.markdown('</div>', unsafe_allow_html=True)

# =============================
# PREDIKSI
# =============================
if st.button("Prediksi Biaya Kesehatan"):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    try:
        prediksi = model.predict(input_data)[0]
        bulanan = prediksi / 12
        dana_darurat = bulanan * 6

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Hasil Prediksi")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='small-label'>Tahunan</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-money'>Rp{int(prediksi):,}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='small-label'>Bulanan</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-money'>Rp{int(bulanan):,}</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='small-label'>Dana Darurat (6 bulan)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-money'>Rp{int(dana_darurat):,}</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Gagal melakukan prediksi!")
        st.code(str(e))
