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
    
    df['Open_lag'] = df['Open'].shift(1)
    df['Close_lag'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    
    return df[['Adj Close', 'Open_lag', 'Close_lag']]
def create_exog_vars(data):
    exog_vars = data[['Open', 'Close']].copy()
    exog_vars['Open_lag'] = exog_vars['Open'].shift(1)
    exog_vars['Close_lag'] = exog_vars['Close'].shift(1)
    exog_vars = exog_vars.dropna()
    return exog_vars

def adjust_exog_variables(exog_vars, train_data):
    for col in ['Open_lag', 'Close_lag']:
        last_change = train_data[col.replace('_lag', '')].diff().iloc[-1]
        last_value = train_data[col.replace('_lag', '')].iloc[-1]
        adjustment_factor = 1 + (last_change / last_value)
        exog_vars[col] = exog_vars[col] * adjustment_factor
    return exog_vars
def train_sarimax_model(ticker, df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    try:
        train_exog = create_exog_vars(df)
        model = SARIMAX(df['Adj Close'], exog=train_exog[['Open_lag', 'Close_lag']], order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMAX model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMAX model for {ticker}: {str(e)}")
def forecast_sarimax_model(ticker, model, df, forecast_days=30):
    forecast = []
    exog_vars = create_exog_vars(df)
    adjusted_exog_vars = adjust_exog_variables(exog_vars, df.loc[exog_vars.index])

    for day in range(forecast_days):
        forecast_input = adjusted_exog_vars.iloc[-1].values.reshape(1, -1)  # Use the last adjusted exog values
        prediction = model.get_forecast(steps=1, exog=forecast_input)
        forecast_value = prediction.predicted_mean.iloc[0]

        # Save the forecast value
        forecast.append(forecast_value)

        # Update exogenous variables for the next step
        new_row = pd.Series({
            'Open': adjusted_exog_vars['Open'].iloc[-1] * (1 + np.random.normal(0, 0.01)),  # Random walk for Open
            'Close': forecast_value,  # Use the forecast value as Close
            'Open_lag': adjusted_exog_vars['Open'].iloc[-1],
            'Close_lag': adjusted_exog_vars['Close'].iloc[-1]
        }, name=df.index[-1] + pd.Timedelta(days=1))
        
        adjusted_exog_vars = adjusted_exog_vars.append(new_row)

        # Adjust exogenous variables again
        adjusted_exog_vars = adjust_exog_variables(adjusted_exog_vars, df.loc[adjusted_exog_vars.index])
    
    return forecast

# Load the trained model and forecast
if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        train_sarimax_model(ticker, ts)
        
        with open(f'models/{ticker}_sarimax_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        forecasted_values = forecast_sarimax_model(ticker, model, ts)
        print(f"Forecasted values for {ticker}:", forecasted_values)








def adjust_exog_variables(exog_vars, train_data):
    for col in ['Open_lag', 'Close_lag']:
        last_change = train_data[col.replace('_lag', '')].diff().iloc[-1]  # Calculate the last change
        last_value = train_data[col.replace('_lag', '')].iloc[-1]          # Get the last value
        adjustment_factor = 1 + (last_change / last_value)  # Calculate adjustment factor
        exog_vars[col] = exog_vars[col] * adjustment_factor  # Apply adjustment
    return exog_vars





from pymongo import MongoClient
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def fetch_data(ticker):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stockdata']
    collection = db['daily_price']
    cursor = collection.find({'company_name': ticker})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

def train_random_forest_models(df):
    models = {}
    features = ['open', 'high', 'low', 'close', 'volume']
    
    for feature in features:
        X = df.drop(columns=['adjusted_close', feature])
        y = df[feature]
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        models[feature] = model
    
    return models

def train_sarimax_model(df, exog_vars, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    try:
        sc = StandardScaler()
        df[exog_vars] = sc.fit_transform(df[exog_vars])
        df['adjusted_close'] = sc.fit_transform(df[['adjusted_close']])
        
        model = SARIMAX(df['adjusted_close'], exog=df[exog_vars], order=order, seasonal_order=seasonal_order, enforce_invertibility=False, enforce_stationarity=False)
        model_fit = model.fit(disp=0)
        return model_fit
    
    except Exception as e:
        print(f"Error training SARIMAX model: {str(e)}")
        return None

def save_model(model, model_name):
    try:
        with open(f'models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {model_name}.")
    except Exception as e:
        print(f"Error saving {model_name}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL']
    
    for ticker in tickers:
        print(f"Training models for {ticker}...")
        data = fetch_data(ticker)
        if data is None or data.empty:
            print(f"No data found for {ticker}. Skipping...")
            continue
        
        processed_data = preprocess_data(data)
        if processed_data is None:
            print(f"Error preprocessing data for {ticker}. Skipping...")
            continue
        
        # Train Random Forest models for stock attributes
        rf_models = train_random_forest_models(processed_data)
        for feature, model in rf_models.items():
            save_model(model, f'{ticker}_{feature}_rf_model')
        
        # Train SARIMAX model for adjusted_close
        exog_vars = ['open', 'high', 'low', 'close', 'volume']
        sarimax_model = train_sarimax_model(processed_data, exog_vars)
        if sarimax_model is not None:
            save_model(sarimax_model, f'{ticker}_sarimax_model')
        else:
            print(f"Failed to train SARIMAX model for {ticker}. Skipping...")
        
        print()

    print("Model training completed.")








from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

def load_model(model_name):
    try:
        with open(f'models/{model_name}.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        return None

def fetch_data(ticker):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stockdata']
    collection = db['daily_price']
    cursor = collection.find({'company_name': ticker})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker')

    if not ticker:
        return jsonify({"error": "Ticker not provided"}), 400

    # Load models
    rf_models = {}
    for feature in ['open', 'high', 'low', 'close', 'volume']:
        model = load_model(f'{ticker}_{feature}_rf_model')
        if model:
            rf_models[feature] = model
        else:
            return jsonify({"error": f"Model for {feature} not found"}), 400

    sarimax_model = load_model(f'{ticker}_sarimax_model')
    if sarimax_model is None:
        return jsonify({"error": "SARIMAX model not found for ticker"}), 400

    try:
        data = fetch_data(ticker)
        if data is None or data.empty:
            return jsonify({"error": f"No data found for {ticker}"}), 400

        df = preprocess_data(data)

        # Predict today's stock attributes
        X_today = df.drop(columns=['adjusted_close']).iloc[-1].values.reshape(1, -1)
        predictions = {feature: model.predict(X_today)[0] for feature, model in rf_models.items()}

        # Prepare exogenous variables for SARIMAX prediction
        exog_vars = np.array([predictions[feature] for feature in ['open', 'high', 'low', 'close', 'volume']]).reshape(1, -1)

        # Forecast adjusted_close using SARIMAX
        forecast = sarimax_model.get_forecast(steps=1, exog=exog_vars)
        adj_close_forecast = forecast.predicted_mean.iloc[0]

        predictions['adjusted_close'] = adj_close_forecast
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)









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

