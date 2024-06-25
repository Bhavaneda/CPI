import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os

def fetch_data_from_mongodb(ticker):
    """
    Fetches daily price data from MongoDB for a given stock ticker.
    
    Parameters:
    ticker (str): Ticker symbol of the stock
    
    Returns:
    pandas.DataFrame: DataFrame containing historical price data
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stockdata']
    collection = db['daily_price']
    cursor = collection.find({'Ticker': ticker})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df

def preprocess_data(df):
    """
    Preprocesses historical price data.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing historical price data
    
    Returns:
    pandas.DataFrame: Preprocessed DataFrame with Date as index and selected columns
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

def train_sarima_model(ticker, df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """
    Trains a SARIMA model on preprocessed historical price data and saves the model.
    
    Parameters:
    ticker (str): Ticker symbol of the stock
    df (pandas.DataFrame): Preprocessed DataFrame with Date as index and selected columns
    order (tuple): Order (p, d, q) of the non-seasonal component of SARIMA (default: (1, 1, 1))
    seasonal_order (tuple): Seasonal order (P, D, Q, S) of the SARIMA model (default: (0, 0, 0, 0))
    
    Saves:
    None
    """
    try:
        model = SARIMAX(df['Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarima_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMA model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMA model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
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
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
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
        return df[['Close']]  # Using 'Close' price for SARIMA
    except Exception as e:
        print(f"Error preprocessing data for SARIMA model: {str(e)}")
        return None

# Function to predict stock prices using a SARIMA model
def predict_stock_prices_sarima(sarima_model, latest_data, days_ahead):
    try:
        ts = preprocess_data_for_sarima(latest_data)
        if ts is None:
            return None
        
        forecast = sarima_model.get_forecast(steps=days_ahead)
        predicted_closes = forecast.predicted_mean.values

        today = datetime.now().date()
        target_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(days_ahead)}
        
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
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices using SARIMA model.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)









import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
import pickle
import os

# MongoDB connection setup
client = MongoClient('mongodb://localhost:27017/')
db = client['stockdata']
collection = db['daily_price']

def fetch_data_from_mongodb(ticker):
    cursor = collection.find({'Ticker': ticker})
    df = pd.DataFrame(list(cursor))
    return df

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

def split_train_test(data, split_date):
    split_date = pd.to_datetime(split_date)
    train = data[data.index <= split_date]
    test = data[data.index > split_date]
    return train, test

def train_sarima_model(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    sarima_model = SARIMAX(train_data['Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    sarima_fit = sarima_model.fit(disp=False)
    return sarima_fit

def train_var_model(train_data, lag_order=7):
    var_model = VAR(train_data)
    var_fit = var_model.fit(lag_order)
    return var_fit

def forecast_sarima_model(model, test_data):
    forecast = model.forecast(steps=len(test_data))
    return forecast

def forecast_var_model(model, test_data):
    forecast = model.forecast(test_data.values, steps=len(test_data))
    return forecast[:, 3]

def evaluate_rmse(actual, forecast):
    return mean_squared_error(actual, forecast, squared=False)

def validate_models(ticker, start_date, end_date, split_date, sarima_order=(1, 1, 1), sarima_seasonal_order=(1, 1, 1, 7), var_lag_order=7):
    try:
        df = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(df)
        train, test = split_train_test(ts, split_date)
        
        sarima_model = train_sarima_model(train, sarima_order, sarima_seasonal_order)
        var_model = train_var_model(train, var_lag_order)
        
        sarima_forecast = forecast_sarima_model(sarima_model, test)
        var_forecast = forecast_var_model(var_model, test)
        
        sarima_rmse = evaluate_rmse(test['Close'], sarima_forecast)
        var_rmse = evaluate_rmse(test['Close'], var_forecast)
        
        print(f"Validation results for {ticker}:")
        print(f"SARIMA RMSE: {sarima_rmse}")
        print(f"VAR RMSE: {var_rmse}")
        
        return sarima_rmse, var_rmse
    
    except Exception as e:
        print(f"Error validating models for {ticker}: {str(e)}")
        return None, None

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 14)
    split_date = datetime(2024, 5, 1)
    
    for ticker in tickers:
        print(f"Validating models for {ticker}...")
        validate_models(ticker, start_date, end_date, split_date)
        print()
    
    # Close MongoDB client connection at the end
    client.close()








import pandas as pd
import numpy as np
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def fetch_data_from_mongodb(ticker):
    """
    Fetches daily price data from MongoDB for a given stock ticker.
    
    Parameters:
    ticker (str): Ticker symbol of the stock
    
    Returns:
    pandas.DataFrame: DataFrame containing historical price data
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stockdata']
    collection = db['daily_price']
    cursor = collection.find({'Ticker': ticker})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df

def preprocess_data(df):
    """
    Preprocesses historical price data.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing historical price data
    
    Returns:
    pandas.DataFrame: Preprocessed DataFrame with Date as index and selected columns
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Close']]  # Adjust columns as per your model requirements

def train_sarima_model(ticker, df):
    """
    Trains a SARIMA model on preprocessed historical price data and saves the model.
    
    Parameters:
    ticker (str): Ticker symbol of the stock
    df (pandas.DataFrame): Preprocessed DataFrame with Date as index and selected columns
    
    Saves:
    None
    """
    try:
        model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Example SARIMA order
        fitted_model = model.fit()
        os.makedirs('models', exist_ok=True)
        fitted_model.save(f'models/{ticker}_sarima_model.pkl')
        print(f"SARIMA model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMA model for {ticker}: {str(e)}")

def train_lstm_model(ticker, df):
    """
    Trains an LSTM model on preprocessed historical price data and saves the model.
    
    Parameters:
    ticker (str): Ticker symbol of the stock
    df (pandas.DataFrame): Preprocessed DataFrame with Date as index and selected columns
    
    Saves:
    None
    """
    try:
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df.values)

        # Prepare data for LSTM
        def create_dataset(dataset, time_step=1):
            X, y = [], []
            for i in range(time_step, len(dataset)):
                X.append(dataset[i-time_step:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        time_step = 100  # Example time step
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Define LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=1)

        # Save the model
        os.makedirs('models', exist_ok=True)
        model.save(f'models/{ticker}_lstm_model.h5')
        print(f"LSTM model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training LSTM model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        train_sarima_model(ticker, ts)
        train_lstm_model(ticker, ts)









import pulp as plp

hist_data = yf.download(stock_name, start="2021-04-01", end="2021-05-04")
hist_data = hist_data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
hist_data = hist_data['Close']
hist_data = np.array(hist_data)

preds = []
mse = []

weights_lstm = 0.3
weight_mcmc = 0.4
weight_arima = 0.4

# weights solver
model = plp.LpProblem('Optimal_weights', plp.LpMinimize)
# weights--->variables
weight_lstm = plp.LpVariable("weight_lstm", lowBound = 0, upBound=0.6)
weight_mcmc = plp.LpVariable("weight_mcmc", lowBound = 0, upBound=0.6)
weight_arima = plp.LpVariable("weight_arima", lowBound = 0, upBound=0.6)

for i in range(len(hist_data)):
    preds.append(lstm_pred[i]*weight_lstm + mcmc_pred[i]*weight_mcmc + arima_pred[i]*weight_arima)
