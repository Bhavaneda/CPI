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

         # Find the model with maximum accuracy
        max_accuracy = max(sarima_accuracy, sarimax_accuracy, var_accuracy)
        if max_accuracy == sarima_accuracy:
            best_model = 'SARIMA'
        elif max_accuracy == sarimax_accuracy:
            best_model = 'SARIMAX'
        else:
            best_model = 'VAR'
        
        return ticker, best_model

    except ValueError as e:
        print(f"ValueError: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    best_models = {}
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'F', 'CAT', 'TCS.NS', 'WFC', 'TATASTEEL.NS', 'NFLX', 'RS', 'JPM', 'TSLA']
    split_date = '2024-06-03'
    for ticker in tickers:
        ticker, best_model = validate_models(ticker, split_date)
        if best_model is not None:
            best_models[ticker] = best_model

# Print the result in the desired dictionary format
print(f"best_models={best_models}")