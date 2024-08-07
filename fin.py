import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
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
    return df[[ 'Adj Close']]

def train_sarima_model(ticker, df):
    
    try:
        model = pm.auto_arima(df['Adj Close'],seasonal=True, m=12,  d=None, D=1, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)
       # fitted_model = model.fit(disp=False)
        os.makedirs('sarima_models', exist_ok=True)
        with open(f'sarima_models/{ticker}_sarima_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"SARIMA model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMA model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['MSFT']
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
    tickers = ['MSFT']
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
def preprocess_data(df):
   
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Adj Close']]
# Function to preprocess data for SARIMA model prediction

# Function to predict stock prices using a SARIMA model
def predict_stock_prices_sarima(sarima_model, latest_data, days_ahead):
    try:
        ts = preprocess_data(latest_data)
        if ts is None:
            return None

       

        # Number of days to predict to reach the current date and 30 days ahead
        last_date = ts.index[-1]
        current_date = datetime.now().date()
        days_to_current_date = (current_date - last_date.date()).days
        total_days_ahead = days_to_current_date + days_ahead

        forecast_index = pd.date_range(start=ts.index[-1], periods=total_days_ahead + 1, freq='D')[1:]
        
        forecast = sarima_model.predict(n_periods=len(forecast_index))
        forecast_values = forecast.tolist()

        target_dates = [forecast_index[i].strftime('%Y-%m-%d') for i in range(len(forecast_index))]
        
        predictions = dict(zip(target_dates, forecast_values))
        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMA model: {str(e)}")
        return None

# Load SARIMA models on application startup
sarima_models = load_sarima_models()

@app.route('/predict/next_month', methods=['POST'])
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






import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
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
    return df[[ 'Open','High','Low','Close', 'Adj Close']]

def train_sarima_model(ticker, df):
    
    try:
        model = pm.auto_arima(df['Adj Close'],exog=df[['Open','High','Low','Close']],seasonal=True, m=12,  d=None, D=1, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)
      
        os.makedirs('sarimax_models', exist_ok=True)
        with open(f'sarimax_models/{ticker}_sarimax_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"SARIMAX model trained and saved successfully for {ticker}.")
    except Exception as e:
        print(f"Error training SARIMA model for {ticker}: {str(e)}")

if __name__ == '__main__':
    tickers = ['MSFT']
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
def load_sarimax_models():
    tickers = ['MSFT']
    models = {}
    for ticker in tickers:
        try:
            with open(f'sarimax_models/{ticker}_sarimax_model.pkl', 'rb') as f:
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
def preprocess_data(df):
   
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Open','High','Low','Close']]
# Function to preprocess data for SARIMA model prediction

def create_exog_vars(data):
    try:
        exog_vars = data[['Open','High','Low','Close']].copy()
        exog_vars['Close_lag'] = exog_vars['Close'].shift(1)
        exog_vars['Open_lag'] = exog_vars['Open'].shift(1)
        exog_vars['High_lag'] = exog_vars['High'].shift(1)
        exog_vars['Low_lag'] = exog_vars['Low'].shift(1)
        exog_vars.dropna(inplace=True)
        return exog_vars
    except Exception as e:
        print(f"Error creating exogenous variables: {str(e)}")
        return None

# Function to adjust exogenous variables
def adjust_exog_variables(exog_vars, train_data, window):
    try:
        for col in ['Open_lag','Close_lag','High_lag','Low_lag']:
            last_change = train_data[col.replace('_lag', '')].diff().tail(window).mean()
            last_value = train_data[col.replace('_lag', '')].iloc[-1]
            adjustment_factor = 1 + (last_change / last_value)
            exog_vars[col] = exog_vars[col] * adjustment_factor
        return exog_vars
    except Exception as e:
        print(f"Error adjusting exogenous variables: {str(e)}")
        return None

# Function to predict stock prices using a SARIMA model
def predict_stock_prices_sarima(sarima_model, latest_data, days_ahead):
    try:
        ts = preprocess_data(latest_data)
        if ts is None:
            return None
        ts=create_exog_vars(ts)
        exog=adjust_exog_variables(ts,latest_data,7)
       

        # Number of days to predict to reach the current date and 30 days ahead
        last_date = ts.index[-1]
        current_date = datetime.now().date()
        days_to_current_date = (current_date - last_date.date()).days
        total_days_ahead = days_to_current_date + days_ahead

        forecast_index = pd.date_range(start=ts.index[-1], periods=total_days_ahead + 1, freq='D')[1:]
        exog=exog.iloc[-total_days_ahead:]
        print(exog)
        forecast = sarima_model.predict(n_periods=len(forecast_index),exog=exog)
        forecast_values = forecast.tolist()

        target_dates = [forecast_index[i].strftime('%Y-%m-%d') for i in range(len(forecast_index))]
        
        predictions = dict(zip(target_dates, forecast_values))
        return predictions
    
    except Exception as e:
        print(f"Error predicting stock prices using SARIMA model: {str(e)}")
        return None

# Load SARIMA models on application startup
sarima_models = load_sarimax_models()

@app.route('/predict/next_month', methods=['POST'])
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




import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

def split_train_test(data, split_date):
    split_date = pd.to_datetime(split_date)
    train = data[data.index <= split_date]
    test = data[data.index > split_date]
    return train, test

def create_exog_vars(data):
    exog_vars = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    exog_vars['Open_lag'] = exog_vars['Open'].shift(1)
    exog_vars['Close_lag'] = exog_vars['Close'].shift(1)
    exog_vars = exog_vars.dropna()
    return exog_vars

def train_sarima_model(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    sarima_model = SARIMAX(train_data['Adj Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    sarima_fit = sarima_model.fit(disp=False)
    return sarima_fit

def train_sarimax_model(train_data, exog_vars, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    sarimax_model = SARIMAX(train_data['Adj Close'], exog=exog_vars, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    sarimax_fit = sarimax_model.fit(disp=False)
    return sarimax_fit

def train_var_model(train_data, lag_order=8):
    var_model = VAR(train_data)
    var_fit = var_model.fit(lag_order)
    return var_fit

def forecast_sarima_model(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

def forecast_sarimax_model(model, train_exog, steps):
    forecast = model.get_forecast(steps=steps, exog=train_exog.iloc[-steps:])
    forecasted_mean = forecast.predicted_mean
    return forecasted_mean

def forecast_var_model(model, test_data):
    forecast = model.forecast(test_data.values, steps=len(test_data))
    return forecast[:, 4]

def evaluate_rmse(actual, forecast):
    mask = ~np.isnan(actual) & (actual != 0)
    actual_filtered = actual[mask]
    forecast_filtered = forecast[mask]
    return mean_squared_error(actual_filtered, forecast_filtered, squared=False)

def evaluate_mape(actual, forecast):
    mask = ~np.isnan(actual) & (actual != 0)
    actual_filtered = actual[mask]
    forecast_filtered = forecast[mask]
    if len(actual_filtered) == 0:
        return np.nan
    return (np.abs((actual_filtered - forecast_filtered) / actual_filtered).mean()) * 100

def evaluate_accuracy(actual, forecast):
    mape = evaluate_mape(actual, forecast)
    if np.isnan(mape):
        return np.nan
    return 100 - mape

def validate_models(ticker, split_date, sarima_order=(1, 1, 1), sarima_seasonal_order=(1, 1, 1, 7), sarimax_order=(1, 1, 1), sarimax_seasonal_order=(1, 1, 1, 7), var_lag_order=8):
    try:
        df = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(df)
        
        if 'Close' not in ts.columns:
            raise ValueError(f"Column 'Close' not found in data for {ticker}. Available columns: {ts.columns}")
        
        train_data, test_data = split_train_test(ts, split_date)
        
        # Train and forecast SARIMA model
        sarima_model = train_sarima_model(train_data, sarima_order, sarima_seasonal_order)
        sarima_forecast = forecast_sarima_model(sarima_model, steps=len(test_data))
        
        # Create exogenous variables from training data
        train_exog = create_exog_vars(train_data)
        
        # Train and forecast SARIMAX model using training data exog
        sarimax_model = train_sarimax_model(train_data.loc[train_exog.index], train_exog, sarimax_order, sarimax_seasonal_order)
        sarimax_forecast = forecast_sarimax_model(sarimax_model, train_exog, steps=len(test_data))
        
        # Train and forecast VAR model
        var_model = train_var_model(train_data, var_lag_order)
        var_forecast = forecast_var_model(var_model, test_data)

        # Align forecasts with test data index
        sarima_forecast.index = test_data.index
        sarimax_forecast.index = test_data.index
        var_forecast = pd.Series(var_forecast, index=test_data.index)

        # Print actual and predicted values for the test set
        print(f"Actual and Predicted Values for {ticker}:")
        print("Date\t\tActual\t\tSARIMA\t\tSARIMAX\t\tVAR")
        for idx in range(len(test_data)):
            date = test_data.index[idx]
            actual_value = test_data.iloc[idx]['Adj Close']
            sarima_pred = sarima_forecast[idx]
            sarimax_pred = sarimax_forecast.iloc[idx] if idx < len(sarimax_forecast) else np.nan
            var_pred = var_forecast.iloc[idx]
            print(f"{date}\t{actual_value:.2f}\t{sarima_pred:.2f}\t{sarimax_pred:.2f}\t{var_pred:.2f}")
        
        print()
        # Check indices order
        print("Training set:")
        print(train_data.index.min(), train_data.index.max())
        print("Test set:")
        print(test_data.index.min(), test_data.index.max())

        # Plotting the actual and forecasted values
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data['Adj Close'], label='Actual')
        plt.plot(test_data.index, sarima_forecast, label='SARIMA Forecast')
        plt.plot(test_data.index, sarimax_forecast, label='SARIMAX Forecast')
        plt.plot(test_data.index, var_forecast, label='VAR Forecast')
        plt.title(f"{ticker} - Actual vs Forecasted Prices")
        plt.xlabel('Date')
        plt.ylabel('Adj Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Evaluate models
        sarima_rmse = evaluate_rmse(test_data['Adj Close'], sarima_forecast)
        sarima_mape = evaluate_mape(test_data['Adj Close'], sarima_forecast)
        sarima_accuracy = evaluate_accuracy(test_data['Adj Close'], sarima_forecast)
        
        sarimax_rmse = evaluate_rmse(test_data['Adj Close'], sarimax_forecast)
        sarimax_mape = evaluate_mape(test_data['Adj Close'], sarimax_forecast)
        sarimax_accuracy = evaluate_accuracy(test_data['Adj Close'], sarimax_forecast)
        
        var_rmse = evaluate_rmse(test_data['Adj Close'], var_forecast)
        var_mape = evaluate_mape(test_data['Adj Close'], var_forecast)
        var_accuracy = evaluate_accuracy(test_data['Adj Close'], var_forecast)

        print(f"Validation results for {ticker}:")
        print(f"SARIMA RMSE: {sarima_rmse}")
        print(f"SARIMA MAPE: {sarima_mape}")
        print(f"SARIMA Accuracy: {sarima_accuracy}%")
        print(f"SARIMAX RMSE: {sarimax_rmse}")
        print(f"SARIMAX MAPE: {sarimax_mape}")
        print(f"SARIMAX Accuracy: {sarimax_accuracy}%")
        print(f"VAR RMSE: {var_rmse}")
        print(f"VAR MAPE: {var_mape}")
        print(f"VAR Accuracy: {var_accuracy}%")

    except ValueError as e:
        print(f"ValueError: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    tickers = ['MSFT']
    split_date = datetime(2024, 6, 1)
    
    for ticker in tickers:
        print(f"Validating models for {ticker}...")
        validate_models(ticker, split_date)
        print()
    
    # Close MongoDB client connection at the end
    client.close()
