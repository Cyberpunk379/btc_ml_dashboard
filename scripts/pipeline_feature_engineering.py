from pathlib import Path
import pandas as pd


# --- Safe path for GitHub or Streamlit deployment ---
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_PATH = BASE_DIR / "data"
DATA_PATH.mkdir(exist_ok=True)

# Step 1: Load Cleaned BTC data
df = pd.read_csv(DATA_PATH / "btc_hourly_yf.csv", parse_dates=['timestamp'], index_col='timestamp')


df.head()

# Step 2: Create lag features
df['return_1h'] = df['price'].pct_change()
df['rolling_mean_3h'] = df['price'].rolling(window=3).mean()
df['rolling_mean_6h'] = df['price'].rolling(window=6).mean()
df['rolling_std_3h'] = df['price'].rolling(window=3).std()

# Step 3: Create next hour's price
df['future_price'] = df['price'].shift(-1)

# Step 4: Drop rows with NaNs from rolling calculations 
df.dropna(inplace=True)

# Step 5: Save feature dataset
df.to_csv(DATA_PATH / "btc_feature.csv")
print("✅ Feature set saved to btc_feature.csv")

df.tail()


