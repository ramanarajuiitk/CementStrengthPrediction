import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    url = "Concrete_Data.xls"
    return pd.read_excel(url)

# Train model
@st.cache_resource
def train_model(data):
    X = data.drop("Concrete compressive strength(MPa, megapascals) ", axis=1)
    y = data["Concrete compressive strength(MPa, megapascals) "]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

# UI layout
st.title("Machine Learning Based Concrete Strength Predictor Application")
st.markdown("Predict the compressive strength of concrete based on its ingredients.")

data = load_data()
model, feature_names = train_model(data)

# Sidebar input
st.sidebar.header("Input Concrete Mix Parameters")

user_input = {}
for feature in feature_names:
    val = st.sidebar.slider(
        label=feature,
        min_value=float(data[feature].min()),
        max_value=float(data[feature].max()),
        value=float(data[feature].mean()),
        step=0.1
    )
    user_input[feature] = val

input_df = pd.DataFrame([user_input])

# Prediction
prediction = model.predict(input_df)[0]
st.subheader("Predicted Compressive Strength (MPa):")
st.success(f"{prediction:.2f} MPa")

# Visuals
if st.checkbox("Show Data and Feature Info"):
    st.write("### Sample Data", data.head())
    st.write("### Feature Ranges")
    st.write(data.describe())

st.markdown("Reference")
st.markdown("""
This app uses the **Concrete Compressive Strength Dataset** 
om the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
Citation:
Yeh, I-C. (1998),Modeling of strength of high-performance concrete using artificial neural networks,Cement and Concrete Research, 28(12),1797-1808


""")

