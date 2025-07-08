import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os
import time
from datetime import datetime
import warnings
from tensorflow.keras.layers import Dropout
warnings.filterwarnings('ignore')

# Parameters
stock_symbols = ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX", "AMD", "INTC"]
start_date = "2020-01-01"
end_date = "2025-04-01"
lookback = 60
epochs = 50
batch_size = 32

# Create directories for saving models
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

def create_lstm_model(lookback):
    """Create LSTM model architecture"""
    model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(lookback, 1)),
    LSTM(100),
    Dense(1)
])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, lookback):
    """Prepare sequences for LSTM training"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def process_stock(symbol, start_date, end_date, lookback, epochs, batch_size):
    """Process individual stock data and train LSTM model"""
    try:
        print(f"\n{'='*50}")
        print(f"Processing {symbol}")
        print(f"{'='*50}")
        
        print(f"Fetching data for {symbol}...")
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        except Exception as e:
            print(f"⚠️ Error while fetching {symbol}: {e}")
            time.sleep(60)
            return None
        
        if data.empty or 'Close' not in data:
            print(f"❌ No data fetched for {symbol}")
            return None
        
        # Prepare data
        df = data[['Close']].copy()
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Normalize closing prices
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        
        # Prepare sequences
        X, y = prepare_data(scaled_data, lookback)
        
        if len(X) == 0:
            print(f"❌ Not enough data for {symbol} (need at least {lookback} days)")
            return None
        
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Train/test split
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create and train LSTM model
        print("Creating and training LSTM model...")
        model = create_lstm_model(lookback)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Test Loss: {test_loss:.6f}")
        
        # Save model and scaler
        model_path = f'models/{symbol}_model.h5'
        scaler_path = f'scalers/{symbol}_scaler.pkl'
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Scaler saved: {scaler_path}")
        
        return {
            'symbol': symbol,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'data_points': len(df),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model_path': model_path,
            'scaler_path': scaler_path
        }
        
    except Exception as e:
        print(f"❌ Error processing {symbol}: {str(e)}")
        return None

def main():
    """Main function to process all stocks"""
    print("🚀 Starting Multi-Company LSTM Stock Prediction")
    print(f"Processing {len(stock_symbols)} companies: {', '.join(stock_symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Lookback period: {lookback} days")
    
    results = []
    successful_stocks = []
    failed_stocks = []
    
    for symbol in stock_symbols:
        result = process_stock(symbol, start_date, end_date, lookback, epochs, batch_size)
        
        if result:
            results.append(result)
            successful_stocks.append(symbol)
        else:
            failed_stocks.append(symbol)

        # Prevent Yahoo Finance rate limit
        print(f"⏳ Waiting to prevent rate limit...")
        time.sleep(5)  # Wait 5 seconds between requests
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(successful_stocks)} stocks")
    print(f"Failed: {len(failed_stocks)} stocks")
    
    if successful_stocks:
        print(f"\n✅ Successful stocks: {', '.join(successful_stocks)}")
    
    if failed_stocks:
        print(f"\n❌ Failed stocks: {', '.join(failed_stocks)}")
    
    # Results table
    if results:
        print(f"\n{'='*80}")
        print("DETAILED RESULTS")
        print(f"{'='*80}")
        print(f"{'Symbol':<8} {'Train Loss':<12} {'Test Loss':<12} {'Data Points':<12} {'Train Samples':<12}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['symbol']:<8} {result['train_loss']:<12.6f} "
                  f"{result['test_loss']:<12.6f} {result['data_points']:<12} "
                  f"{result['training_samples']:<12}")
    
    print(f"\n🎉 Training completed! Models and scalers saved in 'models/' and 'scalers/' directories.")
    
    return results

# Run the main function
if __name__ == "__main__":
    results = main()
