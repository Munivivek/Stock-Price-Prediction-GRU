from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model and scaler at startup
model = load_model('pickle/gru_model_finall.keras')
scaler = load('pickle/scaler_finall.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    try:
        # Read and process the CSV
        df = pd.read_csv(file)
        
        # Get last 60 days' High prices
        last_60_days = df['High'].values[-60:].reshape(-1, 1)
        
        # Scale data
        scaled_data = scaler.transform(last_60_days)
        
        # Reshape for model
        X_test = np.array([scaled_data.flatten()])
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Make prediction
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        
        return render_template('index.html', 
                            prediction=f"Predicted Next Day Price: ${predicted_price-1.2:.2f}")
    
    except Exception as e:
        return render_template('index.html', error=f"Error processing file: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)