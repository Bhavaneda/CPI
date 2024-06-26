import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.vector_ar.var_model import VAR
import pickle
import os

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

def train_model(ticker, df):
    try:
        model = VAR(df)
        fitted_model = model.fit()
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_var_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"VAR model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training VAR model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        train_model(ticker, ts)











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
        model = SARIMAX(df['Close'], exog=df[['Open', 'High', 'Low', 'Adj Close', 'Volume']], order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
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
