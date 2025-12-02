import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================================================
# LOAD MODEL
# =====================================================
with open("model_health_budget_xgboost.pkl", "rb") as f:
    model = pickle.load(f)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Health Cost Prediction",
    page_icon="💰",
    layout="centered"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

.header-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.form-box {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 25px;
    border-left: 6px solid #4CAF50;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.09);
}

h2 {
    color: #2e7d32;
    font-size: 36px;
    font-weight: 800;
}

.small-muted {
    font-size: 13px;
    color: #777;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.markdown("<h1 class='header-title'>💵 Health Budget Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Estimate medical insurance cost using Machine Learning (XGBoost)</p>", unsafe_allow_html=True)

# =====================================================
# FEATURE INPUT FORM
# =====================================================
st.markdown("<div class='form-box'>", unsafe_allow_html=True)

age = st.number_input("Age", min_value=18, max_value=100, step=1)

sex = st.selectbox("Sex", ["male", "female"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)

children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)

smoker = st.selectbox("Smoker?", ["yes", "no"])

region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FEATURE ENGINEERING (SAMA DENGAN TRAINING)
# =====================================================

def make_feature_row(age, sex, bmi, children, smoker, region):

    df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])

    # BMI category
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"]
    )

    # Interaction Features
    df["age_bmi"] = df["age"] * df["bmi"]
    df["smoker_bmi"] = df["bmi"] * (df["smoker"] == "yes").astype(int)
    df["age_children"] = df["age"] * df["children"]
    df["is_obese"] = (df["bmi"] >= 30).astype(int)

    return df


# =====================================================
# PREDICT BUTTON
# =====================================================
if st.button("Predict 💡"):

    input_df = make_feature_row(age, sex, bmi, children, smoker, region)

    try:
        pred = model.predict(input_df)
        pred_val = float(pred[0])

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Estimated Health Cost")

        # HASIL DALAM DOLLAR
        st.markdown(f"<h2>$ {pred_val:,.0f}</h2>", unsafe_allow_html=True)

        st.markdown(
            "<div class='small-muted'>This prediction is calculated using an XGBoost regression model.</div>",
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
