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

def split_train_test(data, split_date):
    split_date = pd.to_datetime(split_date)
    train = data[data.index <= split_date]
    test = data[data.index > split_date]
    return train, test

def train_sarimax_model(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    exog_vars = train_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    sarimax_model = SARIMAX(train_data['Close'], exog=exog_vars, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    sarimax_fit = sarimax_model.fit(disp=False)
    return sarimax_fit

def train_var_model(train_data, lag_order=7):
    var_model = VAR(train_data)
    var_fit = var_model.fit(lag_order)
    return var_fit

def forecast_sarimax_model(model, test_data):
    exog_vars = test_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    forecast = model.get_forecast(steps=len(test_data), exog=exog_vars)
    return forecast.predicted_mean

def forecast_var_model(model, test_data):
    forecast = model.forecast(test_data.values, steps=len(test_data))
    return forecast[:, 3]

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

def validate_models(ticker, split_date, sarima_order=(1, 1, 1), sarima_seasonal_order=(1, 1, 1, 7), var_lag_order=7):
    try:
        df = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(df)
        train, test = split_train_test(ts, split_date)
        
        # Train and forecast SARIMAX model
        sarimax_model = train_sarimax_model(train, sarima_order, sarima_seasonal_order)
        sarimax_forecast = forecast_sarimax_model(sarimax_model, test)
        
        # Train and forecast VAR model
        var_model = train_var_model(train, var_lag_order)
        var_forecast = forecast_var_model(var_model, test)
        
        # Evaluate SARIMAX model
        sarimax_rmse = evaluate_rmse(test['Close'], sarimax_forecast)
        sarimax_mape = evaluate_mape(test['Close'], sarimax_forecast)
        sarimax_accuracy = evaluate_accuracy(test['Close'], sarimax_forecast)
        
        # Evaluate VAR model
        var_rmse = evaluate_rmse(test['Close'], var_forecast)
        var_mape = evaluate_mape(test['Close'], var_forecast)
        var_accuracy = evaluate_accuracy(test['Close'], var_forecast)
        
        print(f"Validation results for {ticker}:")
        print(f"SARIMAX RMSE: {sarimax_rmse}")
        print(f"SARIMAX MAPE: {sarimax_mape}")
        print(f"SARIMAX Accuracy: {sarimax_accuracy}%")
        print(f"VAR RMSE: {var_rmse}")
        print(f"VAR MAPE: {var_mape}")
        print(f"VAR Accuracy: {var_accuracy}%")
        
        return sarimax_rmse, var_rmse, sarimax_accuracy, var_accuracy
    
    except Exception as e:
        print(f"Error validating models for {ticker}: {str(e)}")
        return None, None, None, None

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    split_date = datetime(2024, 2, 1)
    
    for ticker in tickers:
        print(f"Validating models for {ticker}...")
        validate_models(ticker, split_date)
        print()
    
    # Close MongoDB client connection at the end
    client.close()










import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
import numpy as np
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

def train_sarimax_model(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    exog_vars = train_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    sarimax_model = SARIMAX(train_data['Close'], exog=exog_vars, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    sarimax_fit = sarimax_model.fit(disp=False)
    return sarimax_fit

def train_var_model(train_data, lag_order=7):
    var_model = VAR(train_data)
    var_fit = var_model.fit(lag_order)
    return var_fit

def forecast_sarimax_model(model, test_data):
    exog_vars = test_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    forecast = model.forecast(steps=len(test_data), exog=exog_vars)
    return forecast

def forecast_var_model(model, test_data):
    forecast = model.forecast(test_data.values, steps=len(test_data))
    return forecast[:, 3]

def evaluate_mape(actual, forecast):
    mask = ~np.isnan(actual) & (actual != 0)
    actual = actual[mask]
    forecast = forecast[mask]
    if len(actual) == 0:
        return np.nan
    return (np.abs((actual - forecast) / actual).mean()) * 100

def validate_models(ticker, start_date, end_date, split_date, sarima_order=(1, 1, 1), sarima_seasonal_order=(1, 1, 1, 7), var_lag_order=7):
    try:
        df = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(df)
        train, test = split_train_test(ts, split_date)
        
        # Train and forecast SARIMAX model
        sarimax_model = train_sarimax_model(train, sarima_order, sarima_seasonal_order)
        sarimax_forecast = forecast_sarimax_model(sarimax_model, test)
        
        # Filter out NaN or zero values in actual
        mask = ~np.isnan(test['Close']) & (test['Close'] != 0)
        sarimax_rmse = mean_squared_error(test['Close'][mask], sarimax_forecast[mask], squared=False)
        sarimax_mape = evaluate_mape(test['Close'].values, sarimax_forecast)
        sarimax_accuracy = 100 - sarimax_mape
        
        # Train and forecast VAR model
        var_model = train_var_model(train, var_lag_order)
        var_forecast = forecast_var_model(var_model, test)
        
        # Filter out NaN or zero values in actual
        mask = ~np.isnan(test['Close']) & (test['Close'] != 0)
        var_rmse = mean_squared_error(test['Close'][mask], var_forecast[mask], squared=False)
        var_mape = evaluate_mape(test['Close'].values, var_forecast)
        var_accuracy = 100 - var_mape
        
        print(f"Validation results for {ticker}:")
        print(f"SARIMAX RMSE: {sarimax_rmse}")
        print(f"SARIMAX MAPE: {sarimax_mape}")
        print(f"SARIMAX Accuracy: {sarimax_accuracy}%")
        print(f"VAR RMSE: {var_rmse}")
        print(f"VAR MAPE: {var_mape}")
        print(f"VAR Accuracy: {var_accuracy}%")
        
        return sarimax_rmse, var_rmse, sarimax_accuracy, var_accuracy
    
    except Exception as e:
        print(f"Error validating models for {ticker}: {str(e)}")
        return None, None, None, None

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

def train_sarimax_model(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    exog_vars = train_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    sarimax_model = SARIMAX(train_data['Close'], exog=exog_vars, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    sarimax_fit = sarimax_model.fit(disp=False)
    return sarimax_fit

def train_var_model(train_data, lag_order=7):
    var_model = VAR(train_data)
    var_fit = var_model.fit(lag_order)
    return var_fit

def forecast_sarimax_model(model, test_data):
    exog_vars = test_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    forecast = model.forecast(steps=len(test_data), exog=exog_vars)
    return forecast

def forecast_var_model(model, test_data):
    forecast = model.forecast(test_data.values, steps=len(test_data))
    return forecast[:, 3]

def evaluate_mape(actual, forecast):
    return (abs((actual - forecast) / actual).mean()) * 100

def validate_models(ticker, start_date, end_date, split_date, sarima_order=(1, 1, 1), sarima_seasonal_order=(1, 1, 1, 7), var_lag_order=7):
    try:
        df = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(df)
        train, test = split_train_test(ts, split_date)
        
        # Train and forecast SARIMAX model
        sarimax_model = train_sarimax_model(train, sarima_order, sarima_seasonal_order)
        sarimax_forecast = forecast_sarimax_model(sarimax_model, test)
        sarimax_rmse = mean_squared_error(test['Close'], sarimax_forecast, squared=False)
        sarimax_mape = evaluate_mape(test['Close'], sarimax_forecast)
        sarimax_accuracy = 100 - sarimax_mape
        
        # Train and forecast VAR model
        var_model = train_var_model(train, var_lag_order)
        var_forecast = forecast_var_model(var_model, test)
        var_rmse = mean_squared_error(test['Close'], var_forecast, squared=False)
        var_mape = evaluate_mape(test['Close'], var_forecast)
        var_accuracy = 100 - var_mape
        
        print(f"Validation results for {ticker}:")
        print(f"SARIMAX RMSE: {sarimax_rmse}")
        print(f"SARIMAX MAPE: {sarimax_mape}")
        print(f"SARIMAX Accuracy: {sarimax_accuracy}%")
        print(f"VAR RMSE: {var_rmse}")
        print(f"VAR MAPE: {var_mape}")
        print(f"VAR Accuracy: {var_accuracy}%")
        
        return sarimax_rmse, var_rmse, sarimax_accuracy, var_accuracy
    
    except Exception as e:
        print(f"Error validating models for {ticker}: {str(e)}")
        return None, None, None, None

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

def train_sarimax_model(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    exog_vars = train_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    sarimax_model = SARIMAX(train_data['Close'], exog=exog_vars, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    sarimax_fit = sarimax_model.fit(disp=False)
    return sarimax_fit

def train_var_model(train_data, lag_order=7):
    var_model = VAR(train_data)
    var_fit = var_model.fit(lag_order)
    return var_fit

def forecast_sarimax_model(model, test_data):
    exog_vars = test_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    forecast = model.forecast(steps=len(test_data), exog=exog_vars)
    return forecast

def forecast_var_model(model, test_data):
    forecast = model.forecast(test_data.values, steps=len(test_data))
    return forecast[:, 3]

def evaluate_rmse(actual, forecast):
    return mean_squared_error(actual, forecast, squared=False)

def evaluate_mape(actual, forecast):
    return (abs((actual - forecast) / actual).mean()) * 100

def validate_models(ticker, start_date, end_date, split_date, sarima_order=(1, 1, 1), sarima_seasonal_order=(1, 1, 1, 7), var_lag_order=7):
    try:
        df = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(df)
        train, test = split_train_test(ts, split_date)
        
        sarimax_model = train_sarimax_model(train, sarima_order, sarima_seasonal_order)
        var_model = train_var_model(train, var_lag_order)
        
        sarimax_forecast = forecast_sarimax_model(sarimax_model, test)
        var_forecast = forecast_var_model(var_model, test)
        
        sarimax_rmse = evaluate_rmse(test['Close'], sarimax_forecast)
        var_rmse = evaluate_rmse(test['Close'], var_forecast)
        
        sarimax_mape = evaluate_mape(test['Close'], sarimax_forecast)
        var_mape = evaluate_mape(test['Close'], var_forecast)
        
        print(f"Validation results for {ticker}:")
        print(f"SARIMAX RMSE: {sarimax_rmse}")
        print(f"VAR RMSE: {var_rmse}")
        print(f"SARIMAX MAPE: {sarimax_mape}%")
        print(f"VAR MAPE: {var_mape}%")
        
        return sarimax_rmse, var_rmse, sarimax_mape, var_mape
    
    except Exception as e:
        print(f"Error validating models for {ticker}: {str(e)}")
        return None, None, None, None

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    start_date = datetime(2022, 6, 17)
    end_date = datetime(2024, 6, 14)
    split_date = datetime(2024, 2, 1)
    
    for ticker in tickers:
        print(f"Validating models for {ticker}...")
        validate_models(ticker, start_date, end_date, split_date)
        print()
    
    # Close MongoDB client connection at the end
    client.close()











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
