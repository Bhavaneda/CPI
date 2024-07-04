
import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

def fetch_data_from_mongodb(ticker):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stockdata']
    collection = db['daily_price']
    cursor = collection.find({'Ticker': ticker})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    df['High_lag'] = df['Open'].shift(1)
    df['Close_lag'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    
    return df[['Adj Close', 'High_lag', 'Close_lag']]

def train_sarimax_model(ticker, df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    try:
        model = SARIMAX(df['Adj Close'], exog=df[['High_lag', 'Close_lag']], order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMAX model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMAX model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        train_sarimax_model(ticker, ts)





from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pymongo import MongoClient
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Function to load SARIMAX models from pickle files
def load_sarimax_models():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    models = {}
    for ticker in tickers:
        try:
            with open(f'models/{ticker}_sarimax_model.pkl', 'rb') as f:
                model = pickle.load(f)
                models[ticker] = model
                print(f"Loaded SARIMAX model for {ticker}.")
        except Exception as e:
            print(f"Error loading SARIMAX model for {ticker}: {str(e)}")
    return models

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

# Function to preprocess data for SARIMAX model prediction
def preprocess_data_for_sarimax(df):
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        df['Open_lag'] = df['Open'].shift(1)
        df['Close_lag'] = df['Close'].shift(1)
        df.dropna(inplace=True)
        
        return df[['Adj Close', 'Open_lag', 'Close_lag']]
    except Exception as e:
        print(f"Error preprocessing data for SARIMAX model: {str(e)}")
        return None

# Function to predict stock prices using a SARIMAX model
def predict_stock_prices_sarimax(sarimax_model, latest_data, days_ahead):
    try:
        ts = preprocess_data_for_sarimax(latest_data)
        if ts is None:
            return None

        # Number of days to predict to reach the current date and 30 days ahead
        last_date = ts.index[-1]
        current_date = datetime.now().date()
        days_to_current_date = (current_date - last_date.date()).days
        total_days_ahead = days_to_current_date + days_ahead

        forecast_index = pd.date_range(start=ts.index[-1], periods=total_days_ahead+1, freq='D')[1:]
        last_known_open = ts['Open_lag'].iloc[-1]
        last_known_close = ts['Close_lag'].iloc[-1]

        forecast_exog = pd.DataFrame({
            'Open_lag': [last_known_open] * len(forecast_index),
            'Close_lag': [last_known_close] * len(forecast_index)
        }, index=forecast_index)

        forecast = sarimax_model.get_forecast(steps=len(forecast_index), exog=forecast_exog)
        predicted_closes = forecast.predicted_mean.values

        target_dates = [forecast_index[i].strftime('%Y-%m-%d') for i in range(len(forecast_index))]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(len(forecast_index))}

        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMAX model: {str(e)}")
        return None

# Load SARIMAX models on application startup
sarimax_models = load_sarimax_models()

@app.route('/predict/next_month', methods=['POST'])
def predict_next_month_sarimax():
    ticker = request.json['ticker']
    
    try:
        if ticker not in sarimax_models:
            return jsonify({'error': f'Model for {ticker} not found.'}), 404
        
        historical_data = fetch_historical_data(ticker)
        if historical_data is None:
            return jsonify({'error': f'Failed to fetch historical data for {ticker}.'}), 500
        
        sarimax_model = sarimax_models[ticker]
        predictions = predict_stock_prices_sarimax(sarimax_model, historical_data, days_ahead=30)
        
        if predictions is not None:
            current_date = datetime.now().date().strftime('%Y-%m-%d')
            predictions = {date: value for date, value in predictions.items() if date >= current_date}
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices using SARIMAX model.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)





import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.vector_ar.var_model import VAR
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

def fetch_data_from_mongodb(ticker):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stockdata']
    collection = db['daily_price']
    cursor = collection.find({'Ticker': ticker})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

def train_model(ticker, df, lag_order=8):
    try:
        model = VAR(df)
        fitted_model = model.fit(lag_order)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_var_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"VAR model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training VAR model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN','META','F','CAT','TCS.NS','WFC','TATASTEEL.NS','NFLX','RS','JPM','TSLA']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        train_model(ticker, ts, 8)







from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pymongo import MongoClient
from datetime import datetime, timedelta
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to load VAR models from pickle files
def load_var_models():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    models = {}
    for ticker in tickers:
        try:
            with open(f'models/{ticker}_var_model.pkl', 'rb') as f:
                model = pickle.load(f)
                models[ticker] = model
                print(f"Loaded VAR model for {ticker}.")
        except Exception as e:
            print(f"Error loading VAR model for {ticker}: {str(e)}")
    return models

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

# Function to predict stock prices using a VAR model
def predict_stock_prices(var_model, latest_data, days_ahead):
    try:
        ts = preprocess_data_for_var(latest_data)
        if ts is None:
            print("Preprocessed data is None.")
            return None
        
        lag_order = var_model.k_ar
        forecast_input = ts.values[-lag_order:]
        
        if len(forecast_input) < lag_order:
            print("Not enough data to make predictions.")
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

# Load VAR models on application startup
models = load_var_models()

@app.route('/predict/next_month', methods=['POST'])
def predict_next_month():
    ticker = request.json['ticker']
    
    try:
        if ticker not in models:
            return jsonify({'error': f'Model for {ticker} not found.'}), 404
        
        historical_data = fetch_historical_data(ticker)
        if historical_data is None:
            return jsonify({'error': f'Failed to fetch historical data for {ticker}.'}), 500
        
        var_model = models[ticker]
        predictions = predict_stock_prices(var_model, historical_data, days_ahead=30)
        
        if predictions is not None:
            current_date = datetime.now().date().strftime('%Y-%m-%d')
            predictions = {date: value for date, value in predictions.items() if date >= current_date}
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)







import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

def fetch_data_from_mongodb(ticker):
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stockdata']
    collection = db['daily_price']
    cursor = collection.find({'Ticker': ticker})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df

def preprocess_data(df):
   
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

def train_sarima_model(ticker, df, order=(5, 2, 0), seasonal_order=(0, 0, 0, 7)):
    
    try:
        model = SARIMAX(df['Adj Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarima_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMA model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMA model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        train_sarima_model(ticker, ts)






from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pymongo import MongoClient
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Function to load SARIMA models from pickle files
def load_sarima_models():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    models = {}
    for ticker in tickers:
        try:
            with open(f'models/{ticker}_sarima_model.pkl', 'rb') as f:
                model = pickle.load(f)
                models[ticker] = model
                print(f"Loaded SARIMA model for {ticker}.")
        except Exception as e:
            print(f"Error loading SARIMA model for {ticker}: {str(e)}")
    return models

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

# Function to preprocess data for SARIMA model prediction
def preprocess_data_for_sarima(df):
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df[['Adj Close']]  # Using 'Adj Close' price for SARIMA
    except Exception as e:
        print(f"Error preprocessing data for SARIMA model: {str(e)}")
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

        forecast_index = pd.date_range(start=ts.index[-1], periods=total_days_ahead+1, freq='D')[1:]
        
        forecast = sarima_model.get_forecast(steps=len(forecast_index))
        predicted_closes = forecast.predicted_mean.values

        target_dates = [forecast_index[i].strftime('%Y-%m-%d') for i in range(len(forecast_index))]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(len(forecast_index))}

        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMA model: {str(e)}")
        return None

# Load SARIMA models on application startup
sarima_models = load_sarima_models()

@app.route('/predict/next_month_sarima', methods=['POST'])
def predict_next_month_sarima():
    ticker = request.json['ticker']
    
    try:
        if ticker not in sarima_models:
            return jsonify({'error': f'Model for {ticker} not found.'}), 404
        
        historical_data = fetch_historical_data(ticker)
        if historical_data is None:
            return jsonify({'error': f'Failed to fetch historical data for {ticker}.'}), 500
        
        sarima_model = sarima_models[ticker]
        predictions = predict_stock_prices_sarima(sarima_model, historical_data, days_ahead=30)
        
        if predictions is not None:
            current_date = datetime.now().date().strftime('%Y-%m-%d')
            predictions = {date: value for date, value in predictions.items() if date >= current_date}
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices using SARIMA model.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
