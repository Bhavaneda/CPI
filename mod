import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.vector_ar.var_model import VAR
import pickle
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
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

def grid_search_var(df, max_lag=12):
    best_model = None
    best_aic = np.inf
    best_lag_order = 1
    
    # Perform grid search over different lag orders
    for p in tqdm(range(1, max_lag + 1)):
        try:
            model = VAR(df)
            fitted_model = model.fit(p)
            aic = fitted_model.aic
            
            # Update best model if found lower AIC
            if aic < best_aic:
                best_aic = aic
                best_model = fitted_model
                best_lag_order = p
        
        except Exception as e:
            print(f"Error in VAR model fitting for lag order {p}: {str(e)}")
            continue
    
    return best_model, best_lag_order

def train_model(ticker, df, max_lag=12):
    try:
        # Perform grid search for best lag order
        best_model, best_lag_order = grid_search_var(df, max_lag=max_lag)
        
        # Save the best model
        os.makedirs('models', exist_ok=True)
        with open(f'models/{ticker}_var_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"Best VAR model (lag order={best_lag_order}) trained and saved successfully for {ticker}.")
        
        return best_lag_order
    
    except Exception as e:
        print(f"Error training VAR model for {ticker}: {str(e)}")
        return None

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN','META','F','CAT','TCS.NS','WFC','TATASTEEL.NS','NFLX','RS','JPM','TSLA']
    for ticker in tickers:
        data = fetch_data_from_mongodb(ticker)
        ts = preprocess_data(data)
        best_lag_order = train_model(ticker, ts, max_lag=12)
        if best_lag_order is not None:
            print(f"Best lag order for {ticker}: {best_lag_order}")



Project Overview
Project Objective: Predicting the Adjusted Close (Adj Close) prices of various company stocks using SARIMA, SARIMAX, and VAR models based on historical data.

Models Used
SARIMA (Seasonal AutoRegressive Integrated Moving Average):

Parameter Selection: Parameters (p, d, q, P, D, Q, m) are chosen using grid search based on AIC (Akaike Information Criterion).
Training: The SARIMA model is trained on historical data to capture both seasonal and non-seasonal trends in the stock prices.
SARIMAX (SARIMA with Exogenous Variables):

Parameter Selection: Experiments are conducted with various combinations of parameters, and the best-performing set is selected based on empirical testing.
Training: SARIMAX incorporates additional exogenous variables, such as the Close price, to enhance forecasting accuracy.
VAR (Vector AutoRegression):

Parameter Selection: Different lag orders are tested randomly, and the lag order that yields the best results is chosen for each stock.
Training: VAR models the interdependencies among multiple variables (e.g., Close, Volume) to predict Adj Close prices over time.
Accuracy Evaluation
Dataset Splitting: Historical data spanning four years is divided into training and testing sets using a specified split date. The training set is used to fit each model, while the testing set evaluates predictive performance.

Metrics: For each model (SARIMA, SARIMAX, VAR), the following metrics are computed on the testing set:

RMSE (Root Mean Squared Error): Measures the average magnitude of the error between predicted and actual Adj Close prices.
MAPE (Mean Absolute Percentage Error): Calculates the average percentage difference between predicted and actual values.
Accuracy: Determines which model provides the most accurate forecasts for each stock based on RMSE and MAPE.
Model Selection: The best-performing model for each stock ticker is identified based on accuracy metrics calculated during the evaluation phase.

Flask Integration
Functionality:

Route 1 (/forecast_adj_close): Predicts the Adj Close price for the current day. The Flask application loads the pre-trained best model (determined by accuracy evaluation) for each stock from pickle files and generates forecasts.
Route 2 (/forecast_adj_close_range): Predicts Adj Close prices from tomorrow to 30 days ahead. Similar to Route 1, this endpoint utilizes the best model for each stock stored in pickle files.
Implementation:

Flask serves as the backend API, handling requests for current and future Adj Close price predictions.
Models are serialized and stored in pickle files after training, allowing quick loading and prediction within the Flask application.
Conclusion
This project leverages SARIMA, SARIMAX, and VAR models to predict Adj Close prices for various company stocks, employing Flask for API integration and model deployment. By evaluating and comparing model accuracies, it ensures robust forecasting capabilities tailored to each stock's historical data patterns. Adjustments and enhancements can be made based on ongoing performance evaluations and market dynamics.
