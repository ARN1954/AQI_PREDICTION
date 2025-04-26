import streamlit as st
import numpy as np
import pandas as pd
# from PIL import Image # No longer needed unless you add a specific image file
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os # To check if model files exist

# --- Page Configuration ---
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        background: url('https://cdn.mos.cms.futurecdn.net/xcLR5HMU2kxskdAy3ZVuTf-970-80.jpg.webp') no-repeat center center fixed;
        background-size: cover;
        min-height: 100vh;
        color: #ffffff;
    }

    /* Add overlay to improve readability */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        z-index: -1;
    }

    /* Main Content Area */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 3rem;
        margin: 0 auto;
        background: rgba(30, 30, 30, 0.85);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Header Styling */
    .stTitle {
        text-align: center;
        margin-bottom: 2rem;
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Input Sections */
    .input-section {
        background: rgba(40, 40, 40, 0.9);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    .input-section:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .input-section h3 {
        color: #ffffff;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
    }

    /* Number Inputs */
    .stNumberInput > div {
        background: rgba(50, 50, 50, 0.8);
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stNumberInput > label {
        font-weight: 600;
        color: #ffffff;
        font-size: 1.1rem;
    }
    .stNumberInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    .stNumberInput input:focus {
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.5);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 12px;
        background: linear-gradient(45deg, #4a90e2, #357abd);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
    }

    /* Results Container */
    .result-container {
        background: rgba(40, 40, 40, 0.95);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* AQI Categories with Color Indicators */
    .category-display {
        font-size: 1.8rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 12px;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        background: rgba(50, 50, 50, 0.8);
        color: #ffffff;
        position: relative;
        padding-left: 3.5rem;
    }
    .category-display:hover {
        transform: scale(1.05);
    }
    .category-display::before {
        content: '';
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        width: 1.5rem;
        height: 1.5rem;
        border-radius: 50%;
    }
    .category-display.good::before { background: #00E400; }
    .category-display.moderate::before { background: #FFFF00; }
    .category-display.poor::before { background: #FF7E00; }
    .category-display.unhealthy::before { background: #FF0000; }
    .category-display.very-unhealthy::before { background: #8F3F97; }
    .category-display.hazardous::before { background: #7E0023; }

    /* AQI Legend with Color Indicators */
    .legend-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        background: rgba(50, 50, 50, 0.8);
        border-radius: 8px;
        color: #ffffff;
    }
    .legend-item::before {
        content: '';
        width: 1rem;
        height: 1rem;
        border-radius: 50%;
        margin-right: 0.8rem;
    }
    .legend-item.good::before { background: #00E400; }
    .legend-item.moderate::before { background: #FFFF00; }
    .legend-item.poor::before { background: #FF7E00; }
    .legend-item.unhealthy::before { background: #FF0000; }
    .legend-item.very-unhealthy::before { background: #8F3F97; }
    .legend-item.hazardous::before { background: #7E0023; }

    /* Radio Buttons */
    .stRadio > div {
        background: rgba(50, 50, 50, 0.8);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stRadio > label {
        font-weight: 600;
        color: #ffffff;
        font-size: 1.2rem;
    }

    /* Expander */
    .stExpander {
        background: rgba(40, 40, 40, 0.9);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stExpander header {
        font-weight: 600;
        color: #ffffff;
        font-size: 1.2rem;
    }

    /* Footer */
    footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem 0;
        color: #888;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* General Text Colors */
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #ffffff !important;
    }

    /* Metric Display */
    .stMetric {
        background: rgba(50, 50, 50, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    .stMetric > div > div {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- AQI Calculation Functions (Keep as is) ---
def cal_SOi(so2):
    if so2 <= 40: si = so2 * (50/40)
    elif so2 <= 80: si = 50 + (so2-40) * (50/40)
    elif so2 <= 380: si = 100 + (so2-80) * (100/300)
    elif so2 <= 800: si = 200 + (so2-380) * (100/420)
    elif so2 <= 1600: si = 300 + (so2-800) * (100/800)
    else: si = 400 + (so2-1600) * (100/800)
    return si

def cal_Noi(no2):
    if no2 <= 40: ni = no2 * 50/40
    elif no2 <= 80: ni = 50 + (no2-40) * (50/40)
    elif no2 <= 180: ni = 100 + (no2-80) * (100/100)
    elif no2 <= 280: ni = 200 + (no2-180) * (100/100)
    elif no2 <= 400: ni = 300 + (no2-280) * (100/120)
    else: ni = 400 + (no2-400) * (100/120)
    return ni

def cal_RSPMI(rspm):
    # Assuming RSPM is PM10 or PM2.5 - using PM2.5 breakpoints for this example
    # Note: Original function had different breakpoints, adjust if necessary based on *exact* definition
    if rspm <= 30:   i = rspm * 50 / 30
    elif rspm <= 60: i = 50 + (rspm - 30) * 50 / 30
    elif rspm <= 90: i = 100 + (rspm - 60) * 100 / 30
    elif rspm <= 120:i = 200 + (rspm - 90) * 100 / 30
    elif rspm <= 250:i = 300 + (rspm - 120) * 100 / 130
    else:            i = 400 + (rspm - 250) * 100 / 130
    return i

def cal_SPMi(spm):
     # Assuming SPM is PM10 (often used interchangeably in older contexts)
     # Using Indian AQI PM10 breakpoints
    if spm <= 50:    i = spm
    elif spm <= 100: i = 50 + (spm - 50)
    elif spm <= 250: i = 100 + (spm - 100)
    elif spm <= 350: i = 200 + (spm - 250) * 100 / 100
    elif spm <= 430: i = 300 + (spm - 350) * 100 / 80
    else:            i = 400 + (spm - 430) * 100 / 80
    return i
    # # Original function logic - kept for reference if needed
    # if spm <= 50: spmi = spm
    # elif spm <= 100: spmi = spm # This seems unusual, usually linear scaling starts here
    # elif spm <= 250: spmi = 100 + (spm-100) * (100/150)
    # elif spm <= 350: spmi = 200 + (spm-250) # Implicit * (100/100)
    # elif spm <= 430: spmi = 300 + (spm-350) * (100/80)
    # else: spmi = 400 + (spm-430) * (100/80)
    # return spmi


def cal_aqi(si, ni, rspmi, spmi):
    # The final AQI is the maximum of the sub-indices
    aqi = max(si, ni, rspmi, spmi)
    return aqi

def AQI_Range(x):
    x = round(x) # Round AQI to nearest integer for category mapping
    if x <= 50: return "Good", "good"
    elif x <= 100: return "Moderate", "moderate"
    elif x <= 200: return "Poor", "poor" # Changed from Unhealthy for Sensitive Groups based on Indian AQI
    elif x <= 300: return "Unhealthy", "unhealthy"
    elif x <= 400: return "Very unhealthy", "very-unhealthy"
    else: return "Hazardous", "hazardous" # Changed from Severe


# --- Streamlit App Layout ---

# Header with enhanced styling
st.markdown("""
<div class="stTitle">
    <h1>☁️ Air Quality Index (AQI) Predictor</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem; color: #2c3e50; font-size: 1.2rem;'>
    Predict the Air Quality Index (AQI) based on pollutant concentrations using different models or standard calculation.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# --- Model Selection ---
st.subheader("⚙️ Select Prediction Approach")
model_options = {
    "Standard Calculation": {"accuracy": 1.00, "description": "Calculates AQI based on the standard formula (max of sub-indices)."}, # Accuracy is deterministic
    "Linear Regression": {"accuracy": 0.85, "description": "Predicts AQI using a simple linear relationship trained on historical data."},
    "Decision Tree": {"accuracy": 0.92, "description": "Predicts AQI using a tree-based model trained on historical data."},
    "Random Forest": {"accuracy": 0.95, "description": "Predicts AQI using an ensemble of decision trees for potentially higher accuracy."}
}

# Use radio buttons for a cleaner selection interface
selected_model = st.radio(
    "Choose a prediction method:",
    options=list(model_options.keys()),
    format_func=lambda x: f"{x} ({(model_options[x]['accuracy'] * 100):.0f}% Accuracy)" if x != "Standard Calculation" else f"{x} (Standard Formula)",
    horizontal=True, # Display options horizontally
    key="model_selection"
)

# Display model description in an expander or just below
with st.expander("Learn more about the selected method"):
    st.info(model_options[selected_model]['description'])
    if selected_model != "Standard Calculation":
        st.caption(f"Reported Accuracy: {model_options[selected_model]['accuracy']:.2f}. Note: This is based on the training dataset and may vary.")
    else:
        st.caption("This method directly applies the official AQI calculation guidelines.")


# --- Input Fields ---
st.subheader("📊 Enter Pollutant Concentrations")
st.markdown("Provide the measured concentrations for the following pollutants in $µg/m³$ (micrograms per cubic meter).")

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("<div class='input-section'><h3>💨 Primary Gaseous Pollutants</h3></div>", unsafe_allow_html=True)
        so2 = st.number_input("SO₂ (Sulphur Dioxide)", min_value=0.0, value=15.0, step=0.1, format="%.1f", help="Typical range: 0-100 µg/m³. Enter the 24-hour average.")
        no2 = st.number_input("NO₂ (Nitrogen Dioxide)", min_value=0.0, value=25.0, step=0.1, format="%.1f", help="Typical range: 0-150 µg/m³. Enter the 24-hour average.")

with col2:
     with st.container():
        st.markdown("<div class='input-section'><h3>🌫️ Particulate Matter</h3></div>", unsafe_allow_html=True)
        # Note: Check if RSPM corresponds to PM2.5 or PM10 based on your data source/context
        rspm = st.number_input("PM₂.₅ (Fine Particulate Matter)", min_value=0.0, value=40.0, step=0.1, format="%.1f", help="Enter the 24-hour average concentration of PM2.5 (often represented by RSPM). Typical range: 0-300+ µg/m³.")
        spm = st.number_input("PM₁₀ (Coarse Particulate Matter)", min_value=0.0, value=70.0, step=0.1, format="%.1f", help="Enter the 24-hour average concentration of PM10 (often represented by SPM). Typical range: 0-500+ µg/m³.")


# --- Prediction Execution ---
st.markdown("---") # Divider
predict_button = st.button("🔮 Predict AQI", use_container_width=True, type="primary") # Make button primary

# --- Results Display Area ---
results_placeholder = st.empty() # Create a placeholder to show results

if predict_button:
    # Calculate sub-indices (always needed)
    try:
        si = cal_SOi(so2)
        ni = cal_Noi(no2)
        rspmi = cal_RSPMI(rspm) # Using PM2.5 calculation
        spmi = cal_SPMi(spm)     # Using PM10 calculation

        aqi_calculated_standard = cal_aqi(si, ni, rspmi, spmi) # Standard AQI for reference or fallback
        aqi_final = None
        prediction_source = selected_model # Keep track of how AQI was determined

        # Calculate AQI based on selected model
        if selected_model == "Standard Calculation":
            aqi_final = aqi_calculated_standard
        else:
            # Convert model name to filename format
            model_name = selected_model.lower().replace(" ", "_")
            model_path = f'models/{model_name}_model.pkl'
            scaler_path = f'models/{model_name}_scaler.pkl'

            # Check if model and scaler files exist
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    # Load the trained model and scaler
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)

                    # Prepare input features (ensure order matches training)
                    # IMPORTANT: The input to the model should be the *raw* pollutant values
                    # or the *sub-indices* depending on how the model was trained.
                    # Assuming the model was trained on SUB-INDICES based on the original code:
                    X = np.array([[si, ni, rspmi, spmi]])
                    # If trained on RAW values:
                    # X = np.array([[so2, no2, rspm, spm]])

                    X_scaled = scaler.transform(X)

                    # Make prediction
                    aqi_predicted = model.predict(X_scaled)[0]
                    aqi_final = max(0, aqi_predicted) # Ensure AQI is not negative

                except Exception as e:
                    results_placeholder.error(f"Error loading or using the ML model: {e}. Falling back to standard calculation.")
                    aqi_final = aqi_calculated_standard
                    prediction_source = "Standard Calculation (Fallback)"
            else:
                # If model files are not found, use standard calculation and warn
                results_placeholder.warning(f"Model files for '{selected_model}' not found. Using standard calculation method instead.")
                aqi_final = aqi_calculated_standard
                prediction_source = "Standard Calculation (Model Missing)"

        # Determine AQI category
        category, category_class = AQI_Range(aqi_final)

        # Display results within the placeholder
        with results_placeholder.container():
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)

            # Use st.metric for a nice display of the AQI value
            st.metric(label="Predicted Air Quality Index (AQI)", value=f"{aqi_final:.2f}")

            # Display the category with color coding
            st.markdown(f"<h4>Category: <span class='category-display {category_class}'>{category}</span></h4>", unsafe_allow_html=True)

            # Add context about the prediction method
            st.caption(f"Prediction based on: {prediction_source}")
            if prediction_source != selected_model and "(Fallback)" not in prediction_source and "(Model Missing)" not in prediction_source:
                st.caption(f"Standard Calculation Result: {aqi_calculated_standard:.2f}") # Show standard if ML was used

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        results_placeholder.error(f"An error occurred during calculation: {e}")


# --- AQI Categories Legend ---
with st.expander("📖 View AQI Category Reference", expanded=False):
    st.markdown("""
        The Air Quality Index is divided into categories:
        <ul>
            <li><span class='legend-item good'>0-50: Good</span> - Minimal impact</li>
            <li><span class='legend-item moderate'>51-100: Moderate</span> - Minor breathing discomfort to sensitive people</li>
            <li><span class='legend-item poor'>101-200: Poor</span> - Breathing discomfort to people with lung disease, heart disease, children & older adults</li>
            <li><span class='legend-item unhealthy'>201-300: Unhealthy</span> - Breathing discomfort to most people on prolonged exposure</li>
            <li><span class='legend-item very-unhealthy'>301-400: Very Unhealthy</span> - Respiratory illness on prolonged exposure</li>
            <li><span class='legend-item hazardous'>&gt;400: Hazardous</span> - Affects healthy people and seriously impacts those with existing diseases</li>
        </ul>
    """, unsafe_allow_html=True)
    st.caption("Note: Category names and breakpoints based on the Indian National AQI standard. Other regions may differ.")

# --- Footer ---
st.markdown("---")
st.markdown("""
    <footer>
        Developed with Streamlit | AQI calculations based on standard formulas or trained ML models.
    </footer>
    """, unsafe_allow_html=True)