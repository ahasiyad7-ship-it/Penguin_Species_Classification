import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

st.title("Penguin Species Classification")

# Load dataset directly (pkl vendaam)
df = sns.load_dataset('penguins')
df = df.dropna(subset=['species'])

X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# Imputer - puthu sklearn ku compatible
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_imputed, y)

# Inputs
bill_length = st.number_input("Bill Length (mm)", 30.0, 60.0, 43.0)
bill_depth = st.number_input("Bill Depth (mm)", 13.0, 22.0, 17.0)
flipper = st.number_input("Flipper Length (mm)", 170.0, 235.0, 200.0)
body_mass = st.number_input("Body Mass (g)", 2700.0, 6300.0, 4200.0)

if st.button("Predict"):
    sample = [[bill_length, bill_depth, flipper, body_mass]]
    sample_imputed = imputer.transform(sample)
    result = model.predict(sample_imputed)
    st.success(f"Predicted Species: {result[0]}")
