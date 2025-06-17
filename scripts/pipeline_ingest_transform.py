# pipeline injestion_transform.py
from pathlib import Path
import yfinance as yf
import pandas as pd

# --- Safe path for GitHub or Streamlit deployment ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
DATA_PATH.mkdir(exist_ok=True)


# Step 1: Fetch data from Yahoo Finance 
btc_tick = yf.Ticker("BTC-USD")
hist = btc_tick.history(interval= '1h', period= '7d') # 7days of hourly data

# Convert index from UTC to local time (e.g., US/Pacific)
hist.index = hist.index.tz_convert('US/Pacific').tz_localize(None)

hist.head()


# Reset and rename
hist = hist.reset_index()
btc = hist[["Datetime", "Close"]]
btc.rename(columns={"Datetime": "timestamp", "Close" : "price"}, inplace=True)
btc.set_index("timestamp", inplace=True)

btc.head()

btc.to_csv(DATA_PATH / "btc_hourly_yf.csv", index=True)
print("✅ BTC hourly price saved to data/btc_hourly_yf.csv")

btc.tail()