from flask import Flask, request, jsonify
import yfinance as yf
import pickle
import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Function to load SARIMA model
def load_model(ticker):
    model_path = f'models/{ticker}_sarima_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        return None

# Route to fetch live stock data
@app.route('/api/stocks/live', methods=['GET'])
def get_live_stock_data():
    try:
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required as query parameter'}), 400
        
        stock = yf.Ticker(ticker)
        live_data = stock.history(period='1d',actions=True)

        
        if live_data.empty:
            return jsonify({'error': f'No data found for ticker {ticker}'}), 404
        
        # Extract relevant fields
        live_data = live_data.tail(1).to_dict('records')[0]
        response = {
            'Open': live_data['Open'],
            'High': live_data['High'],
            'Low': live_data['Low'],
            'Close': live_data['Close'],
            'Volume': live_data['Volume']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Route to predict stock using SARIMA model
@app.route('/api/predict', methods=['POST'])
def predict_stock():
    try:
        data = request.json
        ticker = data.get('ticker')
        
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required in JSON payload'}), 400
        
        # Fetch the latest historical data
        stock = yf.Ticker(ticker)
        df = stock.history(period='5d')
        
        if df.empty:
            return jsonify({'error': f'No historical data available for ticker {ticker}'}), 404
        
        # Preprocess data
        ts = df['Close']
        ts.index = pd.to_datetime(ts.index)  # Ensure datetime index
        
        # Load the SARIMA model
        model = load_model(ticker)
        if model is None:
            return jsonify({'error': 'Model not found for the given ticker'}), 404
        
        # Make a prediction
        forecast = model.get_forecast(steps=1)
        prediction = forecast.predicted_mean.values[-1]
        
        return jsonify({'prediction': prediction})
    
    except KeyError as ke:
        return jsonify({'error': f'KeyError: {str(ke)}'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
