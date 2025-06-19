from pathlib import Path
import pandas as pd
import yfinance as yf
from scripts.pipeline_serve import serve_classification, serve_regression
from scripts.feature_engineering import engineer_features
from scripts.ingest_data import load_and_transform

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"
DATA_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)

# Step 0: Pull fresh BTC hourly data
btc_tick = yf.Ticker("BTC-USD")
hist = btc_tick.history(interval='1h', period='7d')

# Ensure index is a timezone-aware DatetimeIndex
# Reset index and prepare final DataFrame
hist.reset_index(inplace=True)
print("✅ Available columns after reset_index:", hist.columns)

# Use correct column name dynamically
datetime_col = "Datetime" if "Datetime" in hist.columns else hist.columns[0]  # Usually 'Date' or 'index'
btc = hist[[datetime_col, "Close"]].copy()
btc.rename(columns={datetime_col: "timestamp", "Close": "price"}, inplace=True)
btc.to_csv(DATA_PATH / "btc_hourly_yf.csv", index=False)

print("✅ BTC hourly data saved.")




# Step 1: Load and transform
df_raw = load_and_transform(DATA_PATH / "btc_hourly_yf.csv")
df_feat = engineer_features(df_raw)

print("✅ df_feat preview:\n", df_feat.head())
print("✅ df_feat shape:", df_feat.shape)
print("✅ Null counts:\n", df_feat.isnull().sum())


# Step 2: Add future target
df_feat["future_price"] = df_feat["price"].shift(-1)
df_feat["target"] = (df_feat["future_price"] > df_feat["price"]).astype(int)

# Step 3: Preserve latest row for prediction
latest_row = df_feat.tail(1).copy()
df_feat.dropna(inplace=True)
df_pred = pd.concat([df_feat, latest_row]).dropna()

# Save features
df_pred.to_csv(DATA_PATH / "btc_feature.csv", index=False)
print("✅ Feature set saved to btc_feature.csv")

print("🔎 df_pred shape:", df_pred.shape)
print("🧪 Null values in df_pred:\n", df_pred.isnull().sum())
print("🔍 Sample rows from df_pred:\n", df_pred.head())

df_feat.dropna(inplace=True)

df_feat.dropna(inplace=True)
if df_feat.empty:
    print("⚠️ No data after dropna — attempting to use latest row only.")
    latest_row = df_raw.tail(1).copy()
    df_feat = engineer_features(latest_row)
    df_feat.dropna(inplace=True)
    if df_feat.empty:
        raise ValueError("❌ Still empty after fallback. Feature engineering issue.")


# Only add latest_row if it has no NaNs
latest_row = df_feat.tail(1).copy()
if latest_row.dropna().shape[0] == 1:
    df_pred = pd.concat([df_feat, latest_row])
else:
    df_pred = df_feat.copy()



# Step 4a: Classification (LogReg, Random Forest, XGBoost)
X_clf = df_pred[['return_1h', 'rolling_mean_3h', 'rolling_mean_6h', 'rolling_std_3h']]
clf_models = ["Logistic Regression", "Random Forest (Tuned)", "XGBoost (Tuned)"]

for clf in clf_models:
    preds = serve_classification(clf, X_clf)
    df_out = df_pred.copy()
    df_out["prediction"] = preds
    filename = f"btc_{clf.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')}_model.csv"
    df_out.to_csv(MODEL_PATH / filename, index=False)
    print(f"✅ {clf} predictions saved.")

# Step 4b: Regression (Ridge + Lasso)
X_reg = df_pred[['price', 'return_1h', 'rolling_mean_3h', 'rolling_mean_6h', 'rolling_std_3h']]
reg_models = ["Ridge Regression (Tuned)", "Lasso Regression"]

for reg in reg_models:
    preds = serve_regression(reg, X_reg)
    df_out = df_pred.copy()
    df_out["predicted_price"] = preds
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"]) + pd.Timedelta(hours=2)
    filename = f"btc_{reg.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')}_model.csv"
    df_out.to_csv(MODEL_PATH / filename, index=False)
    print(f"✅ {reg} predictions saved.")

print("✅ Full pipeline complete: All predictions ready for dashboard.")
