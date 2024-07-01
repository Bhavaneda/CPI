# train.py
import pandas as pd
import numpy as np
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

def create_lagged_features(df, lags=5):
    lagged_data = df.copy()
    for lag in range(1, lags + 1):
        for column in df.columns:
            lagged_data[f'{column}_lag{lag}'] = df[column].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

def train_sarimax_model(ticker, df, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    endog = df['Adj Close']
    exog = df.drop(columns=['Adj Close'])  # Drop Adj Close from exogenous variables
    try:
        model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMAX model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMAX model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        ts_lagged = create_lagged_features(ts, lags=5)  # Create lagged features for SARIMAX
        
        train_sarimax_model(ticker, ts_lagged)





# app.py
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
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Use lagged values of these variables for SARIMAX
    except Exception as e:
        print(f"Error preprocessing data for SARIMAX model: {str(e)}")
        return None

# Function to create lagged features for prediction
def create_future_lagged_features(latest_data, days_ahead, lags=5):
    lagged_data = latest_data.copy()
    future_dates = [lagged_data.index[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future_data = pd.DataFrame(index=future_dates, columns=lagged_data.columns)
    
    lagged_data = pd.concat([lagged_data, future_data])

    for lag in range(1, lags + 1):
        for column in lagged_data.columns:
            lagged_data[f'{column}_lag{lag}'] = lagged_data[column].shift(lag)
    
    lagged_data.dropna(inplace=True)
    return lagged_data[-days_ahead:]

# Function to predict stock prices using a SARIMAX model
def predict_stock_prices_sarimax(sarimax_model, latest_data, days_ahead):
    try:
        exog_data = create_future_lagged_features(latest_data, days_ahead, lags=5)
        forecast = sarimax_model.get_forecast(steps=days_ahead, exog=exog_data.drop(columns=['Adj Close']))
        predicted_closes = forecast.predicted_mean.values

        today = datetime.now().date()
        target_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(days_ahead)}
        
        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMAX model: {str(e)}")
        return None

# Load SARIMAX models on application startup
sarimax_models = load_sarimax_models()

@app.route('/predict/next_month_sarimax', methods=['POST'])
def predict_next_month_sarimax():
    ticker = request.json['ticker']
    
    try:
        if ticker not in sarimax_models:
            return jsonify({'error': f'Model for {ticker} not found.'}), 404
        
        historical_data = fetch_historical_data(ticker)
        if historical_data is None:
            return jsonify({'error': f'Failed to fetch historical data for {ticker}.'}), 500
        
        latest_data = preprocess_data_for_sarimax(historical_data)
        sarimax_model = sarimax_models[ticker]
        predictions = predict_stock_prices_sarimax(sarimax_model, latest_data, days_ahead=30)
        
        if predictions is not None:
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices using SARIMAX model.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)









# train.py
import pandas as pd
import numpy as np
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

def create_lagged_features(df, lags=5):
    lagged_data = df.copy()
    for lag in range(1, lags + 1):
        for column in df.columns:
            lagged_data[f'{column}_lag{lag}'] = df[column].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

def train_sarimax_model(ticker, df, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    endog = df['Adj Close']
    exog = df.drop(columns=['Adj Close'])  # Drop Adj Close from exogenous variables
    try:
        model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMAX model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMAX model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        ts_lagged = create_lagged_features(ts, lags=5)  # Create lagged features for SARIMAX
        
        train_sarimax_model(ticker, ts_lagged)








# app.py
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
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Use lagged values of these variables for SARIMAX
    except Exception as e:
        print(f"Error preprocessing data for SARIMAX model: {str(e)}")
        return None

# Function to create lagged features for prediction
def create_future_lagged_features(latest_data, days_ahead, lags=5):
    lagged_data = latest_data.copy()
    future_dates = [latest_data.index[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future_data = pd.DataFrame(index=future_dates, columns=lagged_data.columns)
    
    lagged_data = pd.concat([lagged_data, future_data])

    for lag in range(1, lags + 1):
        for column in lagged_data.columns:
            lagged_data[f'{column}_lag{lag}'] = lagged_data[column].shift(lag)
    
    lagged_data.dropna(inplace=True)
    return lagged_data[-days_ahead:]

# Function to predict stock prices using a SARIMAX model
def predict_stock_prices_sarimax(sarimax_model, latest_data, days_ahead):
    try:
        exog_data = create_future_lagged_features(latest_data, days_ahead, lags=5)
        endog_data = latest_data['Adj Close']
        forecast = sarimax_model.get_forecast(steps=days_ahead, exog=exog_data.drop(columns=['Adj Close']))
        predicted_closes = forecast.predicted_mean.values

        today = datetime.now().date()
        target_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(days_ahead)}
        
        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMAX model: {str(e)}")
        return None

# Load SARIMAX models on application startup
sarimax_models = load_sarimax_models()

@app.route('/predict/next_month_sarimax', methods=['POST'])
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
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices using SARIMAX model.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)














# train.py
import pandas as pd
import numpy as np
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

def create_lagged_features(df, lags=5):
    lagged_data = df.copy()
    for lag in range(1, lags + 1):
        for column in df.columns:
            lagged_data[f'{column}_lag{lag}'] = df[column].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

def train_sarimax_model(ticker, df, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    endog = df['Adj Close']
    exog = df.drop(columns=['Adj Close'])  # Drop Adj Close from exogenous variables
    try:
        model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMAX model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMAX model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        ts_lagged = create_lagged_features(ts, lags=5)  # Create lagged features for SARIMAX
        
        train_sarimax_model(ticker, ts_lagged)










# app.py
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
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Use lagged values of these variables for SARIMAX
    except Exception as e:
        print(f"Error preprocessing data for SARIMAX model: {str(e)}")
        return None

# Function to predict stock prices using a SARIMAX model
def predict_stock_prices_sarimax(sarimax_model, latest_data, days_ahead):
    try:
        exog_data = preprocess_data_for_sarimax(latest_data)
        if exog_data is None:
            return None
        
        forecast = sarimax_model.get_forecast(steps=days_ahead, exog=exog_data)
        predicted_closes = forecast.predicted_mean.values

        today = datetime.now().date()
        target_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(days_ahead)}
        
        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMAX model: {str(e)}")
        return None

# Load SARIMAX models on application startup
sarimax_models = load_sarimax_models()

@app.route('/predict/next_month_sarimax', methods=['POST'])
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
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices using SARIMAX model.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


















# train.py for SARIMAX
import pandas as pd
import numpy as np
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

def create_lagged_features(df, lags=5):
    lagged_data = df.copy()
    for lag in range(1, lags + 1):
        for column in df.columns:
            lagged_data[f'{column}_lag{lag}'] = df[column].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

def train_sarimax_model(ticker, df, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    endog = df['Close']
    exog = df.drop(columns=['Close'])
    try:
        model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMAX model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMAX model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        ts_lagged = create_lagged_features(ts, lags=5)
        
        train_sarimax_model(ticker, ts_lagged)











# app.py for SARIMAX
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
        return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]  # Include all relevant columns
    except Exception as e:
        print(f"Error preprocessing data for SARIMAX model: {str(e)}")
        return None

# Function to predict stock prices using a SARIMAX model
def predict_stock_prices_sarimax(sarimax_model, latest_data, days_ahead):
    try:
        ts = preprocess_data_for_sarimax(latest_data)
        if ts is None:
            return None
        
        forecast = sarimax_model.get_forecast(steps=days_ahead, exog=ts.drop(columns=['Close']))
        predicted_closes = forecast.predicted_mean.values

        today = datetime.now().date()
        target_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(days_ahead)}
        
        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMAX model: {str(e)}")
        return None

# Load SARIMAX models on application startup
sarimax_models = load_sarimax_models()

@app.route('/predict/next_month_sarimax', methods=['POST'])
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
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices using SARIMAX model.'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
















# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from pymongo import MongoClient
import pickle
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

app = Flask(__name__)

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

def create_lagged_features(df, lags=5):
    lagged_data = df.copy()
    for lag in range(1, lags + 1):
        for column in df.columns:
            lagged_data[f'{column}_lag{lag}'] = df[column].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

def predict_future_values(ticker, exog_future, start_date, periods=30):
    try:
        with open(f'models/{ticker}_sarimax_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Create a range of future dates starting from the current date
        future_dates = [start_date + timedelta(days=i) for i in range(periods)]
        
        # Predict the future values
        predictions = model.get_forecast(steps=periods, exog=exog_future).predicted_mean
        
        # Combine dates with corresponding predicted values
        predictions_with_dates = [{'date': date.strftime('%Y-%m-%d'), 'predicted_close': value} for date, value in zip(future_dates, predictions)]
        
        return predictions_with_dates
    except Exception as e:
        print(f"Error predicting future values for {ticker}: {str(e)}")
        return None

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    # Fetch historical data and preprocess
    data = fetch_data_from_mongodb(ticker)
    ts = preprocess_data(data)
    ts_lagged = create_lagged_features(ts, lags=5)
    
    # Calculate exogenous variables (lagged features) from the last available date
    exog_future = ts_lagged.drop(columns=['Close']).tail(1).values
    exog_future = np.repeat(exog_future, 30, axis=0)
    
    # Get the current date
    current_date = ts.index[-1]
    
    # Predict future values from current date to next 30 days
    future_predictions = predict_future_values(ticker, exog_future, current_date, 30)
    
    if future_predictions is not None:
        return jsonify({"ticker": ticker, "predictions": future_predictions}), 200
    else:
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)















from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pymongo import MongoClient
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Function to load SARIMAX models from pickle files
def load_sarimax_models():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
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
        exog_vars = df[['Open', 'High', 'Low', 'Volume']]
        target_var = df[['Adj Close']]  # Using 'Adj Close' price for SARIMAX
        return target_var, exog_vars
    except Exception as e:
        print(f"Error preprocessing data for SARIMAX model: {str(e)}")
        return None, None

# Function to predict stock prices using a SARIMAX model
def predict_stock_prices_sarimax(sarimax_model, target_data, exog_data, days_ahead):
    try:
        ts, exog = preprocess_data_for_sarimax(pd.concat([target_data, exog_data], axis=1))
        if ts is None or exog is None:
            return None

        exog_future = exog[-1:].values
        exog_future = np.tile(exog_future, (days_ahead, 1))
        
        forecast = sarimax_model.get_forecast(steps=days_ahead, exog=exog_future)
        predicted_closes = forecast.predicted_mean.values

        today = datetime.now().date()
        target_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]
        predictions = {target_dates[i]: predicted_closes[i] for i in range(days_ahead)}
        
        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMAX model: {str(e)}")
        return None

# Load SARIMAX models on application startup
sarimax_models = load_sarimax_models()

@app.route('/predict/next_month_sarimax', methods=['POST'])
def predict_next_month_sarimax():
    ticker = request.json['ticker']
    
    try:
        if ticker not in sarimax_models:
            return jsonify({'error': f'Model for {ticker} not found.'}), 404
        
        historical_data = fetch_historical_data(ticker)
        if historical_data is None:
            return jsonify({'error': f'Failed to fetch historical data for {ticker}.'}), 500
        
        sarimax_model = sarimax_models[ticker]
        predictions = predict_stock_prices_sarimax(sarimax_model, historical_data[['Adj Close']], historical_data[['Open', 'High', 'Low', 'Volume']], days_ahead=30)
        
        if predictions is not None:
            return jsonify({'ticker': ticker, 'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to predict stock prices using SARIMAX model.'}), 500
    
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
    exog_vars = df[['Open', 'High', 'Low', 'Volume']]
    target_var = df[['Adj Close']]  # Using 'Adj Close' price for SARIMAX
    return target_var, exog_vars

def train_sarimax_model(ticker, target_var, exog_vars, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    try:
        model = SARIMAX(target_var, exog=exog_vars, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMAX model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMAX model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        target_var, exog_vars = preprocess_data(data)
        train_sarimax_model(ticker, target_var, exog_vars)















# train.py
import pandas as pd
import numpy as np
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

def create_lagged_features(df, lags=5):
    lagged_data = df.copy()
    for lag in range(1, lags + 1):
        for column in df.columns:
            lagged_data[f'{column}_lag{lag}'] = df[column].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

def train_sarimax_model(ticker, df, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    endog = df['Close']
    exog = df.drop(columns=['Close'])
    try:
        model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"SARIMAX model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMAX model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        ts_lagged = create_lagged_features(ts, lags=5)
        
        train_sarimax_model(ticker, ts_lagged)




# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from pymongo import MongoClient
import pickle
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

app = Flask(__name__)

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

def create_lagged_features(df, lags=5):
    lagged_data = df.copy()
    for lag in range(1, lags + 1):
        for column in df.columns:
            lagged_data[f'{column}_lag{lag}'] = df[column].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

def predict_future_values(ticker, exog_future, periods=30):
    try:
        with open(f'models/{ticker}_sarimax_model.pkl', 'rb') as f:
            model = pickle.load(f)
        predictions = model.get_forecast(steps=periods, exog=exog_future).predicted_mean
        return predictions
    except Exception as e:
        print(f"Error predicting future values for {ticker}: {str(e)}")
        return None

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    data = fetch_data_from_mongodb(ticker)
    ts = preprocess_data(data)
    ts_lagged = create_lagged_features(ts, lags=5)
    
    exog_future = ts_lagged.drop(columns=['Close']).tail(1).values
    exog_future = np.repeat(exog_future, 30, axis=0)
    
    future_predictions = predict_future_values(ticker, exog_future, 30)
    if future_predictions is not None:
        last_date = ts.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        predictions_with_dates = [{'date': date.strftime('%Y-%m-%d'), 'predicted_close': value} for date, value in zip(future_dates, future_predictions)]
        return jsonify({"ticker": ticker, "predictions": predictions_with_dates}), 200
    else:
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)

























# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from pymongo import MongoClient
import pickle
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

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

def create_lagged_features(df, lags=5):
    lagged_data = df.copy()
    for lag in range(1, lags + 1):
        for column in df.columns:
            lagged_data[f'{column}_lag{lag}'] = df[column].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

def predict_future_values(ticker, exog_future, periods=30):
    try:
        with open(f'models/{ticker}_sarimax_model.pkl', 'rb') as f:
            model = pickle.load(f)
        predictions = model.get_forecast(steps=periods, exog=exog_future).predicted_mean
        return predictions
    except Exception as e:
        print(f"Error predicting future values for {ticker}: {str(e)}")
        return None

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    data = fetch_data_from_mongodb(ticker)
    ts = preprocess_data(data)
    ts_lagged = create_lagged_features(ts, lags=5)
    
    exog_future = ts_lagged.drop(columns=['Close']).tail(1).values
    exog_future = np.repeat(exog_future, 30, axis=0)
    
    future_predictions = predict_future_values(ticker, exog_future, 30)
    if future_predictions is not None:
        predictions_list = future_predictions.tolist()
        return jsonify({"ticker": ticker, "predictions": predictions_list}), 200
    else:
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)


















import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.vector_ar.var_model import VAR
import pickle
import os

def fetch_data_from_mongodb(ticker):
    """
    Fetch daily price data for a given ticker from MongoDB.
    Assumes MongoDB connection is running locally on default port.
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
    Preprocess DataFrame: Convert Date column to datetime, set as index, and sort by Date.
    Select relevant columns: Open, High, Low, Close, Adj Close, Volume.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

def calculate_information_criteria(data, max_lag=10, ic='aic'):
    """
    Calculate information criteria (AIC, BIC, HQIC) for VAR models with different lag orders.

    Parameters:
    - data: DataFrame containing time series data with columns representing variables.
    - max_lag: Maximum lag order to consider.
    - ic: Information criterion to use ('aic', 'bic', 'hqic').

    Returns:
    - Dictionary containing the calculated information criteria for each lag order.
    """
    results = {}
    for lag in range(1, max_lag + 1):
        model = VAR(data)
        fitted_model = model.fit(lag)
        if ic == 'aic':
            results[lag] = fitted_model.aic
        elif ic == 'bic':
            results[lag] = fitted_model.bic
        elif ic == 'hqic':
            results[lag] = fitted_model.hqic

    return results

def train_model(ticker, df, lag_order):
    """
    Train VAR model for a given ticker with specified lag order.
    Save the trained model using pickle.
    """
    try:
        model = VAR(df)
        fitted_model = model.fit(lag_order)
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_var_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"VAR model trained and saved successfully for {ticker} with lag_order={lag_order}.")
    except Exception as e:
        print(f"Error training VAR model for {ticker}: {str(e)}")

if __name__ == '__main__':
    # Example list of tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']

    # Loop through each ticker
    for ticker in tickers:
        # Fetch data from MongoDB
        data = fetch_data_from_mongodb(ticker)
        
        # Preprocess data
        ts = preprocess_data(data)
        
        # Calculate AIC, BIC, HQIC for lag orders from 1 to 10
        information_criteria_aic = calculate_information_criteria(ts, max_lag=10, ic='aic')
        information_criteria_bic = calculate_information_criteria(ts, max_lag=10, ic='bic')
        information_criteria_hqic = calculate_information_criteria(ts, max_lag=10, ic='hqic')
        
        # Find the lag order with the minimum AIC, BIC, HQIC
        best_lag_aic = min(information_criteria_aic, key=information_criteria_aic.get)
        best_lag_bic = min(information_criteria_bic, key=information_criteria_bic.get)
        best_lag_hqic = min(information_criteria_hqic, key=information_criteria_hqic.get)
        
        # Train model with the best lag order (using AIC, BIC, or HQIC)
        train_model(ticker, ts, best_lag_aic)  # Train with AIC
        # train_model(ticker, ts, best_lag_bic)  # Train with BIC
        # train_model(ticker, ts, best_lag_hqic)  # Train with HQIC














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










import pandas as pd
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Function to fetch data from MongoDB
def fetch_data_from_mongodb(ticker):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stockdata']
    collection = db['daily_price']
    cursor = collection.find({'Ticker': ticker})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df

# Function to preprocess data
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Function to perform grid search for SARIMA parameters
def sarima_grid_search(ts, p_values, d_values, q_values, P_values, D_values, Q_values, S_values):
    best_score, best_params = float('inf'), None
    for p, d, q, P, D, Q, S in itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, S_values):
        try:
            model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, S), enforce_stationarity=False)
            results = model.fit(disp=False)
            if results.aic < best_score:
                best_score, best_params = results.aic, (p, d, q, P, D, Q, S)
        except Exception as e:
            continue
    return best_params

# Function to train SARIMA model
def train_sarima_model(ts, order, seasonal_order):
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    fitted_model = model.fit(disp=False)
    return fitted_model

if __name__ == '__main__':
    ticker = 'AAPL'
    data = fetch_data_from_mongodb(ticker)
    ts = preprocess_data(data)['Close']

    # Define parameter ranges for grid search
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    P_values = range(0, 2)
    D_values = range(0, 2)
    Q_values = range(0, 2)
    S_values = [7]  # Weekly seasonality

    best_params = sarima_grid_search(ts, p_values, d_values, q_values, P_values, D_values, Q_values, S_values)
    print(f'Best parameters: {best_params}')

    # Train the SARIMA model with the best parameters
    if best_params:
        order = best_params[:3]
        seasonal_order = best_params[3:]
        fitted_model = train_sarima_model(ts, order, seasonal_order)
        print(f"Trained SARIMA model with order: {order} and seasonal order: {seasonal_order}")
        # Save the model if needed
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_sarima_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)












import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
import numpy as np
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

def train_sarima_model(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    sarima_model = SARIMAX(train_data['Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    sarima_fit = sarima_model.fit(disp=False)
    return sarima_fit

def train_var_model(train_data, lag_order=7):
    var_model = VAR(train_data)
    var_fit = var_model.fit(lag_order)
    return var_fit

def walk_forward_validation(ts, order, seasonal_order, var_lag_order, n_test):
    sarima_predictions = []
    var_predictions = []

    for i in range(n_test):
        train = ts[:-(n_test - i)]
        test = ts[-(n_test - i):-(n_test - i) + 1]

        sarima_model = train_sarima_model(train, order, seasonal_order)
        var_model = train_var_model(train, var_lag_order)

        sarima_forecast = sarima_model.forecast(steps=1)[0]
        var_forecast = var_model.forecast(train.values, steps=1)[0][3]

        sarima_predictions.append(sarima_forecast)
        var_predictions.append(var_forecast)

    actual = ts[-n_test:]['Close']
    sarima_rmse = mean_squared_error(actual, sarima_predictions, squared=False)
    var_rmse = mean_squared_error(actual, var_predictions, squared=False)

    return sarima_rmse, var_rmse

def validate_models(ticker, sarima_order=(1, 1, 1), sarima_seasonal_order=(1, 1, 1, 7), var_lag_order=7, n_test=30):
    try:
        df = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(df)

        sarima_rmse, var_rmse = walk_forward_validation(ts, sarima_order, sarima_seasonal_order, var_lag_order, n_test)

        print(f"Validation results for {ticker}:")
        print(f"SARIMA RMSE: {sarima_rmse}")
        print(f"VAR RMSE: {var_rmse}")

        return sarima_rmse, var_rmse

    except Exception as e:
        print(f"Error validating models for {ticker}: {str(e)}")
        return None, None

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    n_test = 30  # Number of test observations

    sarima_rmses = []
    var_rmses = []

    for ticker in tickers:
        print(f"Validating models for {ticker}...")
        sarima_rmse, var_rmse = validate_models(ticker, n_test=n_test)
        if sarima_rmse is not None and var_rmse is not None:
            sarima_rmses.append(sarima_rmse)
            var_rmses.append(var_rmse)
        print()

    avg_sarima_rmse = np.mean(sarima_rmses)
    avg_var_rmse = np.mean(var_rmses)

    print(f"Average SARIMA RMSE: {avg_sarima_rmse}")
    print(f"Average VAR RMSE: {avg_var_rmse}")

    # Close MongoDB client connection at the end
    client.close()

