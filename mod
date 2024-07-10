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
