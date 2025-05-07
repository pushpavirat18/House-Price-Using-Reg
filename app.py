import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", layout="centered")

model = joblib.load('models/ridge_model.pkl')
top_features = joblib.load('models/top_features.pkl')
all_features = joblib.load('models/all_features.pkl')

label_map = {
    'OverallQual': "Overall Quality (1-10)",
    'TotalSF': "Total Square Feet",
    'GarageFinish_None': "Garage Finish: Is it Missing?",
    'ExterQual_TA': "Exterior Quality = Typical/Average?",
    'KitchenQual_TA': "Kitchen Quality = Typical/Average?",
    'CentralAir_Y': "Has Central Air Conditioning?",
    'GarageCars': "Garage Capacity (Cars)",
    'KitchenAbvGr': "Kitchens Above Ground",
    'TotalBath': "Total Bathrooms",
    'MSZoning_RM': "Zoning: Residential Medium Density?",
    'YearBuilt': "Year Built",
    'GarageType_None': "Garage Type: Is it Missing?",
    'PoolQC_None': "Pool Quality: Is it Missing?",
    'FireplaceQu_None': "Fireplace Quality: Is it Missing?",
    'GrLivArea': "Above Ground Living Area (sq ft)"
}

reverse_label_map = {v: k for k, v in label_map.items()}

st.title("House Price Predictor")
st.markdown("*Forecast house prices with our smart regression model!*")

st.sidebar.header("Enter House Features")

user_pretty_input = {}
for label in [label_map[feat] for feat in top_features]:
    key = reverse_label_map[label]
    if "Missing" in label or "?" in label or key.endswith('_Y') or key.endswith('_TA') or key.endswith('_RM') or key.endswith('_None'):
        user_pretty_input[key] = st.sidebar.selectbox(f"{label}", ['Yes', 'No']) == 'Yes'
    elif "Year" in label:
        user_pretty_input[key] = st.sidebar.number_input(f"{label}", min_value=1900, max_value=2025, value=2000)
    else:
        user_pretty_input[key] = st.sidebar.number_input(f"{label}", min_value=0.0, value=1.0, step=1.0)

input_df = pd.DataFrame([user_pretty_input])
for col in all_features:
    if col not in input_df.columns:
        input_df[col] = 0.0
input_df = input_df[all_features]

if st.button("Predict Sale Price"):
    log_pred = model.predict(input_df)[0]
    price = np.expm1(log_pred)
    st.success(f"Predicted Sale Price: **${price:,.2f}**")
    st.balloons()
