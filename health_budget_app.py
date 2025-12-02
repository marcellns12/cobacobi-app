# health_budget_app.py
import streamlit as st
import joblib
import pandas as pd
import os

# ----------------------------
# Helper: format angka ke Rupiah
# ----------------------------
def format_rupiah(amount):
    try:
        amount = int(round(amount))
        return "Rp{:,.0f}".format(amount).replace(",", ".")
    except Exception:
        return f"Rp{amount}"

# ----------------------------
# CSS & HTML styling
# ----------------------------
st.set_page_config(
    page_title="Health Budget AI Advisor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS (card, buttons, fonts)
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .app-header {
        background: linear-gradient(90deg, #6EE7B7 0%, #3B82F6 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(59,130,246,0.15);
        margin-bottom: 18px;
    }

    .app-sub {
        color: rgba(255,255,255,0.95);
        font-size: 14px;
        margin-top: 6px;
    }

    .card {
        background: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        margin-bottom: 14px;
    }

    .muted {
        color: #6b7280;
    }

    .big-money {
        font-size: 22px;
        font-weight: 700;
    }

    .small-label {
        font-size: 12px;
        color: #6b7280;
    }

    /* style for st.button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg,#10b981,#06b6d4);
        color: white;
        border: none;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 600;
    }
    div.stButton > button:first-child:hover {
        filter: brightness(0.95);
    }

    /* responsive tweaks */
    @media (max-width: 600px) {
        .app-header { padding: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with HTML
st.markdown(
    """
    <div class="app-header">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="font-size:28px;">🩺</div>
            <div>
                <div style="font-size:20px; font-weight:700;">Health Budget AI Advisor</div>
                <div class="app-sub">Perkiraan biaya kesehatan & rekomendasi dana darurat</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Load model (joblib)
# ----------------------------
MODEL_PATH = "model_health_budget.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"File model tidak ditemukan di `{MODEL_PATH}`. Pastikan file model hasil `joblib.dump()` sudah diupload.\n\n"
        "Tips: saat menyimpan model gunakan `joblib.dump(model, 'model_health_budget.pkl')` dan pastikan pipeline/preprocessing sudah termasuk."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info(
        "Jika model dibuat dengan scikit-learn, disarankan menyimpan Pipeline (preprocessing + estimator) "
        "dengan versi scikit-learn yang sama di environment deployment."
    )
    st.stop()

# ----------------------------
# Input form (dalam card)
# ----------------------------
with st.form(key="input_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input Data Pribadi")
    col1, col2 = st.columns([2, 2])
    with col1:
        age = st.slider("Usia", 18, 100, 30)
        sex = st.selectbox("Jenis Kelamin", ["male", "female"])
        children = st.number_input("Jumlah Anak yang Ditanggung", 0, 10, 0, step=1)
    with col2:
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 25.0, step=0.1, format="%.1f")
        smoker = st.selectbox("Perokok?", ["no", "yes"])
        region = st.selectbox("Wilayah", ["southwest", "southeast", "northwest", "northeast"])
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Prediksi Biaya Kesehatan")

# ----------------------------
# Prediction & Output (styled)
# ----------------------------
if submitted:
    # Prepare input dataframe (sesuaikan dengan fitur training)
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    try:
        prediction = model.predict(input_df)[0]
        monthly = prediction / 12.0
        emergency = monthly * 6.0

        # Output card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Hasil Prediksi")
        st.write(f"<div class='small-label'>Input:</div>", unsafe_allow_html=True)
        st.write(input_df.T.rename(columns={0: "Value"}))
        st.markdown("---", unsafe_allow_html=True)

        # Money display with nice layout
        col_a, col_b, col_c = st.columns([1.8, 1.8, 1.6])
        with col_a:
            st.markdown("<div class='small-label'>Tahunan</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-money'>{format_rupiah(prediction)}</div>", unsafe_allow_html=True)
            st.caption("Perkiraan total biaya per tahun")

        with col_b:
            st.markdown("<div class='small-label'>Bulanan (rata-rata)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-money'>{format_rupiah(monthly)}</div>", unsafe_allow_html=True)
            st.caption("Rata-rata per bulan (perkiraan)")

        with col_c:
            st.markdown("<div class='small-label'>Dana Darurat (6 bulan)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-money'>{format_rupiah(emergency)}</div>", unsafe_allow_html=True)
            st.caption("Rekomendasi dana darurat (6 bulan)")

        st.markdown('</div>', unsafe_allow_html=True)

        # Optional tips
        st.info(
            "Tips: Kalau model kamu adalah estimator (bukan pipeline), pastikan urutan kolom dan tipe datanya "
            "sama seperti saat training. Untuk hasil lebih stabil, simpan pipeline yang menangani encoding & scaling."
        )

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        st.exception(e)

# ----------------------------
# Footer: how to save model with joblib & requirements
# ----------------------------
st.markdown("---")
with st.expander("Petunjuk: Menyimpan model (joblib) & contoh requirements.txt"):
    st.markdown(
        """
**Simpan model (di script training):**
```python
import joblib

# misal: `pipeline` adalah sklearn Pipeline yang berisi preprocessing + estimator
joblib.dump(pipeline, "model_health_budget.pkl")
