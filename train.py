import yfinance as yf
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os
import pandas as pd

def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='max')  # Fetch maximum available historical data
    return data
def store_data(ticker, data):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stock_data']
    collection = db['daily_prices']

    for date, row in data.iterrows():
        document = row.to_dict()
        document['ticker'] = ticker
        document['date'] = date

        collection.update_one(
            {'ticker': ticker, 'date': date},
            {'$set': document},
            upsert=True
        )
    client.close()
def preprocess_data(ticker):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stock_data']
    collection = db['daily_prices']

    data = list(collection.find({'ticker': ticker}))
    df = pd.DataFrame(data)
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    client.close()
    return df['Close']  # Extracting Close prices for SARIMA
def train_model(ticker, ts):
    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Example SARIMA parameters
    fitted_model = model.fit()
    
    os.makedirs('models', exist_ok=True)
    
    with open(f'models/{ticker}_sarima_model.pkl', 'wb') as f:
        pickle.dump(fitted_model, f)
if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Add more tickers as needed
    for ticker in tickers:
        data = fetch_data(ticker)
        store_data(ticker, data)
        ts = preprocess_data(ticker)
        train_model(ticker, ts)
