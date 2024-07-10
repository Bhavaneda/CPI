// tests/navbarNavigation.spec.js
const { test, expect } = require('@playwright/test');

test('navigate through navbar menu items', async ({ page }) => {
  // Navigate to the main page with the navbar
  await page.goto('http://localhost:3000');

  // Click on the "Search" button in the navbar
  await Promise.all([
    page.click('text=Search'),
    page.waitForURL('**/search'),
  ]);

  // Verify URL after navigation
  expect(page.url()).toContain('/search');

  // Click on the "Watchlist" button in the navbar
  await Promise.all([
    page.click('text=Watchlist'),
    page.waitForURL('**/watchlist'),
  ]);

  // Verify URL after navigation
  expect(page.url()).toContain('/watchlist');
});




from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pymongo import MongoClient
from datetime import datetime
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Dictionary of best models for each ticker
best_models = {
    'AAPL': 'VAR', 'MSFT': 'VAR', 'GOOGL': 'SARIMA', 'AMZN': 'SARIMAX', 'META': 'VAR', 'F': 'SARIMA',
    'CAT': 'SARIMA', 'TCS.NS': 'VAR', 'WFC': 'SARIMA', 'TATASTEEL.NS': 'SARIMA', 'NFLX': 'VAR', 'RS': 'VAR',
    'JPM': 'SARIMAX', 'TSLA': 'SARIMAX'
}

# Function to load VAR models from pickle files
def load_var_models():
    tickers = [ticker for ticker, model in best_models.items() if model == 'VAR']
    models = {}
    for ticker in tickers:
        try:
            with open(f'var_models/{ticker}_var_model.pkl', 'rb') as f:
                model = pickle.load(f)
                models[ticker] = model
                print(f"Loaded VAR model for {ticker}.")
        except Exception as e:
            print(f"Error loading VAR model for {ticker}: {str(e)}")
    return models

# Function to load SARIMA models from pickle files
def load_sarima_models():
    tickers = [ticker for ticker, model in best_models.items() if model in ['SARIMA', 'SARIMAX']]
    models = {}
    for ticker in tickers:
        try:
            with open(f'sarima_models/{ticker}_sarima_model.pkl', 'rb') as f:
                model = pickle.load(f)
                models[ticker] = model
                print(f"Loaded SARIMA model for {ticker}.")
        except Exception as e:
            print(f"Error loading SARIMA model for {ticker}: {str(e)}")
    return models

# Load VAR and SARIMA models on application startup
var_models = load_var_models()
sarima_models = load_sarima_models()

def connect_to_mongodb():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['stockdata']
        collection = db['predictions']  # Collection for storing predictions
        print(f"MongoDB connected: {client}, {db}, {collection}")
        return client, collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        return None, None


# Function to insert or update predictions in MongoDB
def insert_or_update_predictions(predictions, ticker):
    try:
        client, collection = connect_to_mongodb()
        if client is not None and collection is not None:
            
            # Insert or update predictions
            for date, adj_close in predictions.items():
                filter_query = {'Date': date, 'Ticker': ticker}
                update_query = {'$set': {'Adj Close': adj_close}}
                collection.update_one(filter_query, update_query, upsert=True)
            
            client.close()
            return True
        else:
            return False
    except Exception as e:
        print(f"Error inserting or updating predictions in MongoDB: {str(e)}")
        return False

# Function to fetch historical data from MongoDB
def fetch_historical_data(ticker):
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['stockdata']
        collection = db['daily_price']
        cursor = collection.find({'Ticker': ticker})
        df = pd.DataFrame(list(cursor))
        client.close()
        return df
    except Exception as e:
        print(f"Error fetching data from MongoDB for {ticker}: {str(e)}")
        return None

# Function to preprocess data for VAR model prediction
def preprocess_data_for_var(df):
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    except Exception as e:
        print(f"Error preprocessing data for VAR model: {str(e)}")
        return None

# Function to preprocess data for SARIMA model prediction
def preprocess_data_for_sarima(df):
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df[['Close', 'Adj Close']]  # Using 'Adj Close' price for SARIMA
    except Exception as e:
        print(f"Error preprocessing data for SARIMA model: {str(e)}")
        return None

# Function to predict stock prices using a VAR model
def predict_stock_prices_var(var_model, latest_data, days_ahead):
    try:
        ts = preprocess_data_for_var(latest_data)
        if ts is None:
            return None
        
        lag_order = var_model.k_ar
        forecast_input = ts.values[-lag_order:]
        
        if len(forecast_input) < lag_order:
            return None

        # Number of days to predict to reach the current date and 30 days ahead
        last_date = ts.index[-1]
        current_date = datetime.now().date()
        days_to_current_date = (current_date - last_date.date()).days
        total_days_ahead = days_to_current_date + days_ahead

        forecast_index = pd.date_range(start=ts.index[-1], periods=total_days_ahead + 1, freq='D')[1:]
        
        forecast = var_model.forecast(forecast_input, steps=len(forecast_index))
        predicted_closes = forecast[:, 4]

        target_dates = [forecast_index[i].strftime('%Y-%m-%d') for i in range(len(forecast_index))]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(len(forecast_index))}

        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using VAR model: {str(e)}")
        return None

# Function to predict stock prices using a SARIMA model
def predict_stock_prices_sarima(sarima_model, latest_data, days_ahead):
    try:
        ts = preprocess_data_for_sarima(latest_data)
        if ts is None:
            return None

        # Number of days to predict to reach the current date and 30 days ahead
        last_date = ts.index[-1]
        current_date = datetime.now().date()
        days_to_current_date = (current_date - last_date.date()).days
        total_days_ahead = days_to_current_date + days_ahead

        forecast_index = pd.date_range(start=ts.index[-1], periods=total_days_ahead + 1, freq='D')[1:]
        
        forecast = sarima_model.get_forecast(steps=len(forecast_index))
        predicted_closes = forecast.predicted_mean.values

        target_dates = [forecast_index[i].strftime('%Y-%m-%d') for i in range(len(forecast_index))]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(len(forecast_index))}

        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMA model: {str(e)}")
        return None

@app.route('/predict/next_month', methods=['POST'])
def predict_next_month():
    ticker = request.json['ticker']
    
    try:
        if ticker not in best_models:
            return jsonify({'error': f'Model for {ticker} not found.'}), 404
        
        historical_data = fetch_historical_data(ticker)
        if historical_data is None:
            return jsonify({'error': f'Failed to fetch historical data for {ticker}.'}), 500
        
        model_type = best_models[ticker]
        if model_type == 'VAR':
            var_model = var_models.get(ticker)
            predictions = predict_stock_prices_var(var_model, historical_data, days_ahead=30)
        elif model_type in ['SARIMA', 'SARIMAX']:
            sarima_model = sarima_models.get(ticker)
            predictions = predict_stock_prices_sarima(sarima_model, historical_data, days_ahead=30)
        else:
            return jsonify({'error': f'Unknown model type for {ticker}.'}), 500
        
        if predictions is not None:
            current_date = datetime.now().date().strftime('%Y-%m-%d')
            predictions = {date: value for date, value in predictions.items() if date >= current_date}
            
            # Insert or update predictions in MongoDB
            if not insert_or_update_predictions(predictions, ticker):
                return jsonify({'error': 'Failed to store predictions in MongoDB.'}), 500
            
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
