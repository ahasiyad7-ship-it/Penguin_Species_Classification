import streamlit as st
import pandas as pd
import joblib

st.title("🐧 Penguin Species Classification")

model = joblib.load("penguin_model.pkl")

st.write("Enter Penguin Measurements")

bill_length = st.number_input("Bill Length (mm)", value=39.1)
bill_depth = st.number_input("Bill Depth (mm)", value=18.7)
flipper_length = st.number_input("Flipper Length (mm)", value=181)
body_mass = st.number_input("Body Mass (g)", value=3750)

if st.button("Predict"):
    sample = pd.DataFrame([{
        "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper_length,
        "body_mass_g": body_mass
    }])

    result = model.predict(sample)
    st.success("Predicted Species: " + result[0])