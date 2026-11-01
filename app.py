
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
from keras.models import load_model

app = Flask(__name__)

# Load model and scaler at startup
model = load_model('pickle/gru_model_finall.keras')
scaler = load('pickle/scaler_finall.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No file selected")

    try:
        # Read uploaded CSV
        df = pd.read_csv(file)

        # Check required column
        if 'High' not in df.columns:
            return render_template(
                'index.html',
                error="CSV file must contain a 'High' column"
            )

        # Get the last 60 valid High prices
        high_prices = pd.to_numeric(df['High'], errors='coerce').dropna()

        if len(high_prices) < 60:
            return render_template(
                'index.html',
                error="CSV must contain at least 60 valid High prices"
            )

        last_60_days = high_prices.values[-60:].reshape(-1, 1)

        # Scale input data
        scaled_data = scaler.transform(last_60_days)

        # Reshape input for GRU model
        X_test = scaled_data.reshape(1, 60, 1)

        # Predict next-day price
        predicted_price = model.predict(X_test, verbose=0)

        predicted_price = scaler.inverse_transform(
            predicted_price
        )[0][0]

        return render_template(
            'index.html',
            prediction=f"Predicted Next Day Price: ${predicted_price:.2f}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Error processing file: {str(e)}"
        )


if __name__ == '__main__':
    app.run(debug=False)
