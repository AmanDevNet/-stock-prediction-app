# -stock-prediction-app
 is a machine learning web app that predicts future stock prices using LSTM neural networks and real-time market data. Powered by Flask and yFinance, it offers intelligent forecasts for the world's top tech stocks.

---

## 📌 Key Features

- 📉 Predict stock prices up to 30 days ahead
- 🚀 Trained LSTM models for top companies (AAPL, TSLA, MSFT, etc.)
- 🧠 Scaled time series data with `MinMaxScaler`
- 🗃️ Automatic data caching and retry logic for API rate limits
- 🌐 Web-based UI using Flask + HTML templates

---

## ⚙️ Tech Stack

| Layer        | Tech                         |
|--------------|------------------------------|
| Backend      | Flask, Python                |
| Machine Learning | TensorFlow/Keras (LSTM), scikit-learn |
| Data         | yFinance, Pandas, NumPy      |
| Serialization| Joblib                       |
| Frontend     | HTML, CSS (Jinja2 templates) |

---

## 📂 Folder Structure
📁 Stock Price Prediction/
├── MODELS/ # LSTM models (.h5)
├── scalers/ # Scaler files (.pkl)
├── cache/ # Cached stock data (.csv)
├── templates/
│ └── index.html # Web UI
├── app.py # Flask application
├── train_stock.py # Model training script
└── README.md

### 1. Clone the Repo

```bash
git clone https://github.com/AmanDevNet/StockCast.git
cd -stock-prediction-app
2. Install Dependencies
bash
pip install -r requirements.txt
3. Train the Models (First Time Only)
bash
Copy
Edit
python train_stock.py
4. Run the App
bash
Copy
Edit
python app.py
Visit: http://localhost:5000

🎯 Supported Stocks
AAPL (Apple)

TSLA (Tesla)

MSFT (Microsoft)

GOOGL (Google)

AMZN, META, NVDA, NFLX, AMD, INTC

You can expand the list in app.py.

👨‍💻 Author
Aman Sharma
linkedin :  http://www.linkedin.com/in/aman-sharma-842b66318
