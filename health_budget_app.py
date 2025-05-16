
import streamlit as st
import pickle
import pandas as pd

# Load model
with open("model_health_budget.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🩺 Health Budget AI Advisor")
st.write("Prediksi biaya kesehatan berdasarkan data pribadi")

# Input dari pengguna
age = st.slider("Usia", 18, 100, 30)
sex = st.selectbox("Jenis Kelamin", ["male", "female"])
bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
children = st.number_input("Jumlah Anak yang Ditanggung", 0, 10, 0)
smoker = st.selectbox("Perokok?", ["yes", "no"])
region = st.selectbox("Wilayah", ["southwest", "southeast", "northwest", "northeast"])

# Prediksi ketika tombol diklik
if st.button("Prediksi Biaya Kesehatan"):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # Prediksi biaya
    prediksi = model.predict(input_data)[0]
    bulanan = prediksi / 12
    dana_darurat = bulanan * 6

    st.success(f"💰 Perkiraan biaya kesehatan tahunan: Rp{int(prediksi):,}")
    st.info(f"🛟 Rekomendasi dana darurat (6 bulan): Rp{int(dana_darurat):,}")
    st.write(f"📅 Estimasi pengeluaran bulanan: Rp{int(bulanan):,}")
