
# diabetes_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap  


import shap
import matplotlib.pyplot as plt

from streamlit.components.v1 import html  # ✅ Needed for st_shap()

def st_shap(plot, height=None):  # ✅ Define this function
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    html(shap_html, height=height)


# Load and prepare data for SHAP explainer (used for background data)
df = pd.read_excel('t2dm.data.xlsx')

# Replace 0s in specific columns with NaN, then fill with mean
invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[invalid_cols] = df[invalid_cols].replace(0, np.nan)
for col in invalid_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale features
scaler = joblib.load('scaler.pkl')  
X_scaled = scaler.transform(X)


# Load saved model and scaler
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit Page Config
st.set_page_config(page_title="Diabetes Risk Prediction Dashboard", layout="centered")

# App Title
st.title("🔬 Diabetes Risk Prediction Dashboard")
st.markdown("""
This application predicts **diabetes risk** using clinical input data  
and explains how each feature contributes to the prediction using **SHAP**.
""")

# User Input Form
st.sidebar.header("📝 Input Clinical Data")
def user_input():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose", min_value=50, max_value=200, value=120)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=30, max_value=130, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=70.0, value=28.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=33)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Show input
st.subheader("🧾 User Input Data")
st.write(input_df)

# Predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0][1]

# Display Result
st.subheader("🎯 Prediction Result")
st.write("**Diabetes Risk:**", "🟥 Positive" if prediction == 1 else "🟩 Negative")
st.write(f"**Probability of Diabetes:** {prediction_proba:.2%}")

if prediction == 1:
    st.info("This result suggests a higher risk. Factors like high glucose, BMI, or age might have contributed.")
else:
    st.success("This result suggests a lower diabetes risk. Maintain healthy lifestyle habits.")


