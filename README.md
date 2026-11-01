# Stock Price Prediction Using GRU

A deep-learning project that forecasts the next day's stock price using historical stock data. It includes a GRU model, preprocessing/scaling artifacts, Jupyter notebooks for model development, and a Flask web interface for uploading a CSV file and requesting a prediction.

## Features

- Loads and explores historical Google (GOOGL) stock data.
- Preprocesses price data and scales values using `MinMaxScaler`.
- Uses a GRU (Gated Recurrent Unit) neural network for time-series forecasting.
- Saves the trained Keras model and fitted scaler.
- Provides a Flask interface where a user can upload a CSV file and request a next-day prediction.

## Technologies

Python, TensorFlow, Keras, NumPy, Pandas, scikit-learn, Flask, HTML, CSS, Jupyter Notebook.

## Project structure

```text
.
├── app.py
├── requirements.txt
├── Installation.md
├── explanation.txt
├── GOOGL_stock_data.csv
├── Main.ipynb
├── Demo.ipynb
├── stock_price_prediction.ipynb
├── gru_model_final.keras
├── scaler_final.joblib
├── pickle/
│   ├── gru_model_finall.keras
│   └── scaler_finall.joblib
├── Sample-datasets/
├── static/
│   └── style.css
└── templates/
    └── index.html
```

## Setup

Use Python 3.12 as specified in the project's installation notes.

1. Clone or download this repository.
2. Create and activate a virtual environment.

   **Windows**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask application.

   ```bash
   python app.py
   ```

5. Open the local URL printed by Flask in your browser.

## Using the web application

1. Prepare a CSV file containing a `High` column and at least 60 rows of historical prices.
2. Open the application in your browser.
3. Upload the CSV file and submit it.
4. The application uses the most recent 60 `High` values to generate a next-day estimate.

## Model workflow

1. Load and inspect historical stock-price data.
2. Clean and scale the price series.
3. Create rolling sequences using the previous 60 time steps.
4. Train a GRU-based neural network.
5. Generate predictions and convert scaled values back to the original price scale.
6. Evaluate model output using regression metrics in the notebooks.

## Notes and limitations

- This is an educational forecasting project, not financial advice or a trading system.
- Predictions depend on the quality, format, and date range of the uploaded CSV data.
- The web app expects a `High` column and at least 60 valid numeric observations.
- The saved model/scaler files are required by the Flask application and are stored under `pickle/`.
- Review and validate model performance on fresh, unseen data before using predictions for any decision.
- The original project archive included a Python installer and notebook checkpoint cache; these have been excluded from this repository package to keep the repository cleaner.
