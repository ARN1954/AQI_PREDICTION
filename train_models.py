import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

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

def load_and_preprocess_data(file_path):
    try:
        # Load data
        df = pd.read_csv(file_path, encoding='unicode_escape')
        
        # Check required columns
        required_columns = ['so2', 'no2', 'rspm', 'spm']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle missing values
        df = df.dropna(subset=required_columns)
        
        # Remove outliers (values beyond 3 standard deviations)
        for col in required_columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
        
        # Calculate sub-indices
        df['SOI'] = df['so2'].apply(cal_SOi)
        df['NOI'] = df['no2'].apply(cal_Noi)
        df['RSPMI'] = df['rspm'].apply(cal_RSPMI)
        df['SPMI'] = df['spm'].apply(cal_SPMi)
        
        # Calculate AQI
        df['AQI'] = df.apply(lambda x: cal_aqi(x['SOI'], x['NOI'], x['RSPMI'], x['SPMI']), axis=1)
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def train_and_save_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models with tuned parameters
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    }
    
    # Train and save each model
    for name, model in models.items():
        try:
            # Create and fit scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"\n{name.upper()} Model:")
            print(f"R² Score: {r2:.4f}")
            print(f"Mean Absolute Error: {mae:.2f}")
            
            # Save model and scaler
            model_filename = f'models/{name}_model.pkl'
            scaler_filename = f'models/{name}_scaler.pkl'
            
            joblib.dump(model, model_filename)
            joblib.dump(scaler, scaler_filename)
            
            print(f"Model and scaler saved successfully")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error training {name} model: {str(e)}")

if __name__ == "__main__":
    # Data file path
    data_file = "/home/atharva/Downloads/data.csv"  # Using the same path as in the notebook
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(data_file)
    
    if df is not None:
        # Prepare features and target
        X = df[['SOI', 'NOI', 'RSPMI', 'SPMI']]
        y = df['AQI']
        
        # Train and save models
        print("\nTraining models...")
        train_and_save_models(X, y)
        
        print("\nTraining completed successfully!")
    else:
        print("Failed to load data. Please check your data file.") 