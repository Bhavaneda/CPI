from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from pymongo import MongoClient
from datetime import datetime, timedelta
from statsmodels.tsa.vector_ar.var_model import VAR

app = Flask(__name__)

# Load VAR models for each ticker
def load_var_models():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Update with your tickers
    models = {}
    for ticker in tickers:
        with open(f'models/{ticker}_var_model.pkl', 'rb') as f:
            model = pickle.load(f)
            models[ticker] = model
    return models

# Function to fetch historical data from MongoDB up to a specified date
def fetch_historical_data(ticker, end_date):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stockdata']
    collection = db['daily_price']
    cursor = collection.find({'Ticker': ticker, 'Date': {'$lte': end_date}})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df

# Function to preprocess data for VAR prediction
def preprocess_data_for_var(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Function to predict stock price for the next day using the VAR model
def predict_stock_price(var_model, latest_data):
    try:
        # Preprocess latest data for VAR prediction
        ts = preprocess_data_for_var(latest_data)
        
        # Make predictions using VAR model
        lag_order = var_model.k_ar
        forecast_input = ts.values[-lag_order:]
        
        # Predict for the next day
        target_date = ts.index[-1] + timedelta(days=1)
        forecast = var_model.forecast(forecast_input, steps=1)
        
        # Extracting the prediction for the next day's Close price
        predicted_close = forecast[-1][3]  # Index 3 corresponds to 'Close' column
        
        return predicted_close, target_date
    
    except Exception as e:
        print(f"Error predicting stock price: {str(e)}")
        return None, None

# Route for predicting stock price for the next day
@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.json['ticker']
    
    try:
        # Fetch historical data up to 2024/06/17 from MongoDB
        historical_data = fetch_historical_data(ticker, datetime(2024, 6, 17))
        
        # Load VAR model for the specified ticker
        models = load_var_models()
        var_model = models[ticker]
        
        # Predict stock price for the next day using VAR model
        predicted_close, target_date = predict_stock_price(var_model, historical_data)
        
        if predicted_close is not None:
            return jsonify({'ticker': ticker, 'predicted_close': predicted_close, 'target_date': target_date.strftime('%Y-%m-%d')})
        else:
            return jsonify({'error': 'Failed to predict stock price.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
