import pandas as pd
import numpy as np
from pymongo import MongoClient
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

def fetch_data(ticker):
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
    return df

def create_lagged_features(df, window_sizes):
    lagged_features = pd.DataFrame(index=df.index)
    
    for column, window_size in window_sizes.items():
        lagged_features[f'{column}_MA'] = df[column].rolling(window=window_size).mean()
    
    return lagged_features

def train_sarimax_model(df, exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    try:
        model = SARIMAX(df['Adj Close'], exog=exog, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=0)
        return model_fit
    except Exception as e:
        print(f"Error training SARIMAX model: {str(e)}")
        return None

def save_sarimax_model(model, ticker):
    try:
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved SARIMAX model for {ticker}.")
    except Exception as e:
        print(f"Error saving SARIMAX model for {ticker}: {str(e)}")

def load_sarimax_model(ticker):
    try:
        with open(f'models/{ticker}_sarimax_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading SARIMAX model for {ticker}: {str(e)}")
        return None

def main():
    tickers = ['AAPL']
    window_sizes = {'Open': 5, 'High': 5, 'Low': 5, 'Close': 5}  # Example window sizes
    
    for ticker in tickers:
        print(f"Training SARIMAX model for {ticker}...")
        data = fetch_data(ticker)
        if data is None or data.empty:
            print(f"No data found for {ticker}. Skipping...")
            continue
        
        processed_data = preprocess_data(data)
        if processed_data is None:
            print(f"Error preprocessing data for {ticker}. Skipping...")
            continue
        
        exog_data = create_lagged_features(processed_data, window_sizes)
        exog_data.dropna(inplace=True)
        processed_data = processed_data.loc[exog_data.index]
        
        sarimax_model = train_sarimax_model(processed_data, exog=exog_data)
        if sarimax_model is not None:
            save_sarimax_model(sarimax_model, ticker)
        else:
            print(f"Failed to train SARIMAX model for {ticker}. Skipping...")
        
        print()

    print("SARIMAX model training completed.")

if __name__ == '__main__':
    main()








from pymongo import MongoClient
import pickle
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def fetch_data(ticker):
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
    return df

def train_sarimax_model(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    try:
        sc1 = StandardScaler()
        sc2 = StandardScaler()

        columns = ['Open','High','Low','Close','Volume']
        df[columns] = sc1.fit_transform(df[columns])
        df['Adj Close'] = sc2.fit_transform(df[['Adj Close']])

        model = SARIMAX(df['Adj Close'], exog=df[columns], order=order, seasonal_order=seasonal_order, enforce_invertibility=False, enforce_stationarity=False)
        model_fit = model.fit(disp=0)
        return model_fit
    
    except Exception as e:
        print(f"Error training SARIMAX model: {str(e)}")
        return None

def save_sarimax_model(model, ticker):
    try:
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved SARIMAX model for {ticker}.")
    except Exception as e:
        print(f"Error saving SARIMAX model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL']
    
    for ticker in tickers:
        print(f"Training SARIMAX model for {ticker}...")
        data = fetch_data(ticker)
        if data is None or data.empty:
            print(f"No data found for {ticker}. Skipping...")
            continue
        
        processed_data = preprocess_data(data)
        if processed_data is None:
            print(f"Error preprocessing data for {ticker}. Skipping...")
            continue
        
        sarimax_model = train_sarimax_model(processed_data)
        if sarimax_model is not None:
            save_sarimax_model(sarimax_model, ticker)
        else:
            print(f"Failed to train SARIMAX model for {ticker}. Skipping...")
        
        print()

    print("SARIMAX model training completed.")








from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

def load_sarimax_model(ticker):
    try:
        with open(f'models/{ticker}_sarimax_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading SARIMAX model for {ticker}: {str(e)}")
        return None

def create_lagged_features(df, window_sizes):
    lagged_features = pd.DataFrame(index=df.index)
    
    for column, window_size in window_sizes.items():
        lagged_features[f'{column}_MA'] = df[column].rolling(window=window_size).mean()
    
    return lagged_features

def forecast_adj_close(model, initial_data, window_sizes, steps=30):
    try:
        forecast_values = []
        exog = create_lagged_features(initial_data, window_sizes).dropna()

        for step in range(steps):
            forecast = model.get_forecast(steps=1, exog=exog.tail(1))
            forecast_value = forecast.predicted_mean.iloc[0]
            forecast_values.append(forecast_value)

            # Append forecasted value to the initial data
            next_date = exog.index[-1] + pd.Timedelta(days=1)
            new_row = initial_data.tail(1).copy()
            new_row.index = [next_date]
            new_row['Adj Close'] = forecast_value

            # Append new_row to initial_data and recalculate exog
            initial_data = pd.concat([initial_data, new_row])
            exog = create_lagged_features(initial_data, window_sizes).dropna()

        forecast_index = pd.date_range(start=initial_data.index[-steps], periods=steps, freq='D')
        forecast_df = pd.DataFrame({'Adj Close': forecast_values}, index=forecast_index)
        return forecast_df
    except Exception as e:
        print(f"Error forecasting with SARIMAX model: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker')
    days_ahead = 30  # Fixed number of days for forecasting

    model = load_sarimax_model(ticker)
    if model is None:
        return jsonify({"error": "Model not found for ticker"}), 400

    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['stockdata']
        collection = db['daily_price']
        cursor = collection.find({'Ticker': ticker})
        df = pd.DataFrame(list(cursor))
        client.close()

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        window_sizes = {'Open': 5, 'High': 5, 'Low': 5, 'Close': 5}

        forecast_df = forecast_adj_close(model, df, window_sizes, steps=days_ahead)
        if forecast_df is not None:
            forecast_json = forecast_df.reset_index().to_dict(orient='records')
            return jsonify({"predictions": forecast_json})
        else:
            return jsonify({"error": "Forecasting failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

