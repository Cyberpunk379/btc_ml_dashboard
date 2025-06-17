# pipeline_serve.py

import pandas as pd
from ingest_data import load_btc_data
from feature_engineering import engineer_features
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# === Step 1: Load raw BTC price data ===
df_raw = load_btc_data("data/btc_hourly_yf.csv")

# === Step 2: Engineer features ===
df_feat = engineer_features(df_raw)

# === Step 3: Define target ===
df_feat['target'] = df_feat['price'].shift(-1)
df_feat.dropna(inplace=True)

# === Step 4: Train/test split ===
X = df_feat[['return_1h', 'rolling_mean_3', 'rolling_std_3']]
y = df_feat['target']
X_train, X_test = X.iloc[:-31], X.iloc[-31:]
y_train, y_test = y.iloc[:-31], y.iloc[-31:]

# === Step 5: Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Step 6: Train model ===
model = Ridge(alpha=100)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# === Step 7: Evaluate ===
print("📉 RMSE:", round(mean_squared_error(y_test, y_pred, squared=False), 2))
print("📈 MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("🔁 R² Score:", round(r2_score(y_test, y_pred), 2))

# === Step 8: Save results ===
df_result = df_feat.iloc[-31:].copy()
df_result['predicted_price'] = y_pred
df_result.to_csv("output_models/btc_model_output_regr_tuned.csv", index=False)

# === Optional: Save model and scaler ===
joblib.dump(model, "models/final_ridge_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Pipeline complete.")
