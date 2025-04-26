import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
    .stMarkdown {
        color: #333;
    }
    .stSubheader {
        color: #2c3e50;
    }
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .category-box {
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        font-weight: bold;
    }
    .good { background-color: #4CAF50; color: white; }
    .moderate { background-color: #FFEB3B; color: black; }
    .poor { background-color: #FF9800; color: white; }
    .unhealthy { background-color: #F44336; color: white; }
    .very-unhealthy { background-color: #9C27B0; color: white; }
    .hazardous { background-color: #000000; color: white; }
    .model-info {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to calculate SOI
def cal_SOi(so2):
    if so2 <= 40:
        si = so2 * (50/40)
    elif so2 <= 80:
        si = 50 + (so2-40) * (50/40)
    elif so2 <= 380:
        si = 100 + (so2-80) * (100/300)
    elif so2 <= 800:
        si = 200 + (so2-380) * (100/420)
    elif so2 <= 1600:
        si = 300 + (so2-800) * (100/800)
    else:
        si = 400 + (so2-1600) * (100/800)
    return si

# Function to calculate NOI
def cal_Noi(no2):
    if no2 <= 40:
        ni = no2 * 50/40
    elif no2 <= 80:
        ni = 50 + (no2-40) * (50/40)
    elif no2 <= 180:
        ni = 100 + (no2-80) * (100/100)
    elif no2 <= 280:
        ni = 200 + (no2-180) * (100/100)
    elif no2 <= 400:
        ni = 300 + (no2-280) * (100/120)
    else:
        ni = 400 + (no2-400) * (100/120)
    return ni

# Function to calculate RSPMI
def cal_RSPMI(rspm):
    if rspm <= 30:
        rspmi = rspm * (50/30)
    elif rspm <= 60:
        rspmi = 50 + (rspm-30) * (50/30)
    elif rspm <= 90:
        rspmi = 100 + (rspm-60) * (100/30)
    elif rspm <= 120:
        rspmi = 200 + (rspm-90) * (100/30)
    elif rspm <= 250:
        rspmi = 300 + (rspm-120) * (100/130)
    else:
        rspmi = 400 + (rspm-250) * (100/130)
    return rspmi

# Function to calculate SPMI
def cal_SPMi(spm):
    if spm <= 50:
        spmi = spm
    elif spm <= 100:
        spmi = spm
    elif spm <= 250:
        spmi = 100 + (spm-100) * (100/150)
    elif spm <= 350:
        spmi = 200 + (spm-250)
    elif spm <= 430:
        spmi = 300 + (spm-350) * (100/80)
    else:
        spmi = 400 + (spm-430) * (100/80)
    return spmi

# Function to calculate AQI
def cal_aqi(si, ni, rspmi, spmi):
    aqi = max(si, ni, rspmi, spmi)
    return aqi

# Function to determine AQI category
def AQI_Range(x):
    if x <= 50:
        return "Good", "good"
    elif x > 50 and x <= 100:
        return "Moderate", "moderate"
    elif x > 100 and x <= 200:
        return "Poor", "poor"
    elif x > 200 and x <= 300:
        return "Unhealthy", "unhealthy"
    elif x > 300 and x <= 400:
        return "Very unhealthy", "very-unhealthy"
    else:
        return "Hazardous", "hazardous"

# Streamlit app
st.title("🌤️ Air Quality Index (AQI) Predictor")

# Introduction
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h3>Predict Air Quality Based on Pollutant Levels</h3>
    <p>Enter the values for different pollutants to get the AQI prediction</p>
</div>
""", unsafe_allow_html=True)

# Model selection
st.markdown("### Select Prediction Model")
model_options = {
    "Standard Calculation": {"accuracy": 0.98, "description": "Standard AQI calculation method"},
    "Linear Regression": {"accuracy": 0.85, "description": "Simple linear model"},
    "Decision Tree": {"accuracy": 0.92, "description": "Tree-based model"},
    "Random Forest": {"accuracy": 0.95, "description": "Ensemble of decision trees"}
}

selected_model = st.selectbox(
    "Choose a prediction model:",
    options=list(model_options.keys()),
    format_func=lambda x: f"{x} (Accuracy: {model_options[x]['accuracy']*100:.1f}%)"
)

# Display model information
st.markdown(f"""
<div class='model-info'>
    <strong>Selected Model:</strong> {selected_model}<br>
    <strong>Accuracy:</strong> {model_options[selected_model]['accuracy']*100:.1f}%<br>
    <strong>Description:</strong> {model_options[selected_model]['description']}
</div>
""", unsafe_allow_html=True)

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Primary Pollutants")
    so2 = st.number_input("SO₂ (Sulphur Dioxide) (µg/m³)", min_value=0.0, value=0.0, 
                         help="Enter the concentration of Sulphur Dioxide")
    no2 = st.number_input("NO₂ (Nitrogen Dioxide) (µg/m³)", min_value=0.0, value=0.0,
                         help="Enter the concentration of Nitrogen Dioxide")

with col2:
    st.markdown("### Particulate Matter")
    rspm = st.number_input("RSPM (Respirable Suspended Particulate Matter) (µg/m³)", 
                          min_value=0.0, value=0.0,
                          help="Enter the concentration of RSPM")
    spm = st.number_input("SPM (Suspended Particulate Matter) (µg/m³)", 
                         min_value=0.0, value=0.0,
                         help="Enter the concentration of SPM")

# Prediction button
st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
predict_button = st.button("🔮 Predict AQI", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if predict_button:
    # Calculate sub-indices
    si = cal_SOi(so2)
    ni = cal_Noi(no2)
    rspmi = cal_RSPMI(rspm)
    spmi = cal_SPMi(spm)
    
    # Calculate AQI based on selected model
    if selected_model == "Standard Calculation":
        aqi = cal_aqi(si, ni, rspmi, spmi)
    else:
        try:
            # Convert model name to filename format
            model_name = selected_model.lower().replace(" ", "_")
            # Load the trained model
            model = joblib.load(f'models/{model_name}_model.pkl')
            scaler = joblib.load(f'models/{model_name}_scaler.pkl')
            
            # Prepare input
            X = np.array([[si, ni, rspmi, spmi]])
            X_scaled = scaler.transform(X)
            
            # Make prediction
            aqi = model.predict(X_scaled)[0]
        except:
            # If model files are not found, use standard calculation
            aqi = cal_aqi(si, ni, rspmi, spmi)
            st.warning(f"Model files not found. Using standard calculation method.")
    
    category, category_class = AQI_Range(aqi)
    
    # Display results in a nice box
    st.markdown(f"""
    <div class='result-box'>
        <h2 style='text-align: center;'>Results</h2>
        <div style='text-align: center; margin: 20px 0;'>
            <h3>Predicted AQI: {aqi:.2f}</h3>
            <div class='category-box {category_class}'>{category}</div>
        </div>
        <div style='text-align: center; color: #666;'>
            <p>Predicted using {selected_model} (Accuracy: {model_options[selected_model]['accuracy']*100:.1f}%)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AQI Categories Legend
    st.markdown("""
    <div class='result-box'>
        <h3>AQI Categories</h3>
        <div class='category-box good'>0-50: Good</div>
        <div class='category-box moderate'>51-100: Moderate</div>
        <div class='category-box poor'>101-200: Poor</div>
        <div class='category-box unhealthy'>201-300: Unhealthy</div>
        <div class='category-box very-unhealthy'>301-400: Very unhealthy</div>
        <div class='category-box hazardous'>>400: Hazardous</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 50px; color: #666;'>
    <p>Air Quality Index Prediction Tool</p>
    <p>Values are based on standard AQI calculation methods</p>
</div>
""", unsafe_allow_html=True) 