from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Dynamic paths
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, 'model_xgbfinalft.pkl')
IMPUTER_PATH = os.path.join(BASE_DIR, 'imputerfinalft.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'dec7.csv')
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'Powerdemand_actual_pridected_2013to2024.csv')

# Load model, imputer, and data
model_xgb = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)
dft = pd.read_csv(CSV_PATH)
prediction_2013_2024 = pd.read_csv(PREDICTIONS_PATH)
dft['DATE'] = pd.to_datetime(dft['DATE'])

# List of features used for prediction
features = [
    'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
    'humidity', 'precip', 'precipprob', 'precipcover', 'windspeed',
    'sealevelpressure', 'Per Capita Income (in Rupees)',
    'Growth Rate of Per Capita Income (%)', 'GSDP (in Crores)',
    'Growth Rate of GSDP (%)', 'Population Estimate',
    'Growth Rate of Population (%)', 'is_holiday', 'is_weekend', 'month',
    'dayofweek', 'dayofyear', 'year-2000', 'weekofyear', 'tempmax_humidity',
    'tempmin_humidity', 'temp_humidity', 'feelslikemax_humidity',
    'feelslikemin_humidity', 'feelslike_humidity', 'temp_range', 'heat_index'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        date = data.get('date')
        
        if pd.to_datetime(date) < pd.to_datetime('2024-12-01'):
            row = prediction_2013_2024[prediction_2013_2024['DATE'] == date]
            if row.empty:
                return jsonify({'error': f'No data available for the date: {date}'})
            actual = row['POWER_DEMAND'].values[0]
            predicted = row['Predicted_POWER_DEMAND'].values[0]
            return jsonify({
                'date': date,
                'predicted_power_demand': float(predicted),
                'actual_power_demand': float(actual)
            })
        else:
            row = dft[dft['DATE'] == date]
            if row.empty:
                return jsonify({'error': f'No data available for the date: {date}'})
            
            row_features = row[features]
            row_imputed = imputer.transform(row_features)
            prediction = model_xgb.predict(row_imputed)[0]
            return jsonify({
                'date': date,
                'predicted_power_demand': float(prediction)
            })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
