from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime, timedelta
import time
from pandas.tseries.offsets import BDay
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Directory setup
CACHE_DIR = "cache"
MODEL_DIR = "models"
SCALER_DIR = "scalers"
for directory in [CACHE_DIR, MODEL_DIR, SCALER_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Supported stock symbols
STOCK_SYMBOLS = ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX", "AMD", "INTC"]
LOOKBACK = 60
MAX_RETRIES = 5
RETRY_DELAY = 15  # Increased delay

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data with retries for rate limiting and fallback to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")
    
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                raise ValueError("No data returned from yfinance")
            df.to_csv(cache_file, index=True)  # Save with Date index
            print(f"✅ Successfully fetched and cached data for {symbol}")
            return df
        except Exception as e:
            print(f"⚠️ Fetch attempt {attempt + 1}/{MAX_RETRIES} failed for {symbol}: {str(e)}")
            if "Too Many Requests" in str(e) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
            continue
    
    # Fallback to cache if exists
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            if not cached_df.empty and 'Close' in cached_df.columns:
                print(f"✅ Loaded cached data for {symbol}")
                return cached_df
            else:
                print(f"❌ Cache for {symbol} is invalid or missing 'Close' column")
        except Exception as e:
            print(f"❌ Error reading cache for {symbol}: {e}")
    print(f"❌ Failed to fetch or load data for {symbol} after {MAX_RETRIES} attempts")
    return None

def load_model_and_scaler(symbol):
    """Load the trained model and scaler for a given stock symbol."""
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.h5")
    scaler_path = os.path.join(SCALER_DIR, f"{symbol}_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"⚠️ Model or scaler not found for {symbol}")
        return None, None
    
    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"❌ Error loading model/scaler for {symbol}: {e}")
        return None, None

def prepare_prediction_data(data, lookback):
    """Prepare data for LSTM prediction."""
    if data is None or len(data) < lookback or 'Close' not in data.columns:
        return None
    recent_data = data['Close'].values[-lookback:]
    return np.reshape(recent_data, (1, lookback, 1))

def predict_stock_prices(symbol, days=10):
    """Predict stock prices for the next 'days' days."""
    model, scaler = load_model_and_scaler(symbol)
    if not model or not scaler:
        return {'success': False, 'error': f'Model or scaler missing for {symbol}'}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 + days + LOOKBACK)  # Reduced buffer
    data = fetch_stock_data(symbol, start_date, end_date)

    if data is None or data.empty or 'Close' not in data.columns:
        return {'success': False, 'error': f'Failed to retrieve data for {symbol}. Please try again later or check your internet connection.'}

    df = data[['Close']].copy()
    scaled_data = scaler.transform(df)

    predictions = []
    current_batch = scaled_data[-LOOKBACK:].flatten()

    for _ in range(days):
        X = np.reshape(current_batch[-LOOKBACK:], (1, LOOKBACK, 1))
        pred = model.predict(X, verbose=0)
        predictions.append(pred[0][0])
        current_batch = np.append(current_batch, pred[0][0])

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    last_date = df.index[-1]
    prediction_dates = [last_date + BDay(i) for i in range(1, days + 1)]
    results = [
        {'day': i + 1, 'date': date.strftime('%Y-%m-%d'), 'price': round(float(price), 2)}
        for i, (date, price) in enumerate(zip(prediction_dates, predictions.flatten()))
    ]

    current_price = float(df['Close'].iloc[-1])
    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'predictions': results,
        'success': True
    }

@app.route('/')
def index():
    """Render the main page with available symbols."""
    return render_template('index.html', symbols=STOCK_SYMBOLS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests via POST."""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        days = int(data.get('days', 10))

        if symbol not in STOCK_SYMBOLS:
            return jsonify({'success': False, 'error': f'Unsupported symbol {symbol}. Available: {", ".join(STOCK_SYMBOLS)}'})
        if not 1 <= days <= 30:
            return jsonify({'success': False, 'error': 'Days must be between 1 and 30'})

        result = predict_stock_prices(symbol, days)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/available_symbols')
def get_available_symbols():
    """Return list of available stock symbols."""
    return jsonify({'symbols': STOCK_SYMBOLS, 'success': True})

if __name__ == '__main__':
    print("Starting application...")
    if not any(os.path.exists(os.path.join(MODEL_DIR, f"{s}_model.h5")) for s in STOCK_SYMBOLS):
        print("⚠️ Warning: No models found in 'models' directory. Run train_stock.py first.")
    if not any(os.path.exists(os.path.join(SCALER_DIR, f"{s}_scaler.pkl")) for s in STOCK_SYMBOLS):
        print("⚠️ Warning: No scalers found in 'scalers' directory. Run train_stock.py first.")
    app.run(debug=True, host='0.0.0.0', port=5000)