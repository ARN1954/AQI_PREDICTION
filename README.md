# 🌤️ Air Quality Index (AQI) Prediction

A Streamlit web application that predicts Air Quality Index (AQI) based on pollutant levels using multiple machine learning models.

## 📋 Features

- Predicts AQI using four different models:
  - Standard Calculation (98% accuracy)
  - Linear Regression (85% accuracy)
  - Decision Tree (92% accuracy)
  - Random Forest (99.14% accuracy)
- Real-time AQI category classification
- Beautiful and responsive UI
- Detailed model information and accuracy scores
- AQI categories legend with color coding

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/air-quality-prediction.git
cd air-quality-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data

The models are trained on real-world air quality data containing:
- SO₂ (Sulphur Dioxide) levels
- NO₂ (Nitrogen Dioxide) levels
- RSPM (Respirable Suspended Particulate Matter) levels
- SPM (Suspended Particulate Matter) levels

## 🧠 Models

The application uses three machine learning models with the following performance metrics:

| Model | R² Score | Mean Absolute Error |
|-------|----------|---------------------|
| Linear Regression | 0.98 | 16.94 |
| Decision Tree | 0.85 | 10.00 |
| Random Forest | 0.9914 | 8.45 |

## 🎯 AQI Categories

The application classifies air quality into six categories:

| AQI Range | Category | Color |
|-----------|----------|-------|
| 0-50 | Good | 🟢 Green |
| 51-100 | Moderate | 🟡 Yellow |
| 101-200 | Poor | 🟠 Orange |
| 201-300 | Unhealthy | 🔴 Red |
| 301-400 | Very Unhealthy | 🟣 Purple |
| >400 | Hazardous | ⚫ Black |

## 🖥️ Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter the pollutant levels:
   - SO₂ (Sulphur Dioxide) in µg/m³
   - NO₂ (Nitrogen Dioxide) in µg/m³
   - RSPM (Respirable Suspended Particulate Matter) in µg/m³
   - SPM (Suspended Particulate Matter) in µg/m³

4. Select your preferred prediction model

5. Click "Predict AQI" to get the results

## 📁 Project Structure

```
air-quality-prediction/
├── app.py                 # Streamlit application
├── train_models.py        # Model training script
├── models/                # Trained models directory
│   ├── linear_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── ... (scalers)
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 🔧 Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- joblib

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📫 Contact

For any questions or suggestions, please open an issue in the repository.

---
