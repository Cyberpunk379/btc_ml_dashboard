from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib


# --- Safe path for GitHub or Streamlit deployment ---
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust `.parent` depending on notebook location
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"
DATA_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)

# Step 1: Load Features
df = pd.read_csv(DATA_PATH/"btc_feature.csv", parse_dates=['timestamp'])

# Define target: Future Price
df["target"] = df["future_price"]

df.tail()

# Step 2: Define target: Predict next hour's price
df['target'] = df['price'].shift(-1)
df.dropna(inplace=True)

# Step 3: Feature and Target split
X = df.drop(columns=['timestamp', 'target', 'future_price']) # Drop timestamp and target-related
y = df['target']

# Step 4: Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, ridge_preds)
mae = mean_absolute_error(y_test, ridge_preds)
r2 = r2_score(y_test, ridge_preds)

print(f"✅ Ridge Regression Model Trained.")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 MAE: {mae:.2f}")
print(f"🔁 R² Score: {r2:.2f}")

# Save output for dashboard
df_result = df.iloc[len(df) - len(y_test):].copy()
df_result['timestamp'] = df_result['timestamp'] + pd.Timedelta(hours=2)
df_result['predicted_price'] = ridge_preds
df_result.to_csv(MODEL_PATH/"btc_ridge_model.csv", index=False)

# RIDGE FINE-TUNING
# Define model and hyperparameter grid
ridge = Ridge()
params = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(ridge, params, cv=5)
ridge_grid.fit(X, y)

# Best model
best_ridge = ridge_grid.best_estimator_
y_pred = best_ridge.predict(X)

# Evaluation
rmse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("✅ Tuned Ridge Regression")
print("Best Params:", ridge_grid.best_params_)
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 MAE: {mae:.2f}")
print(f"🔁 R² Score: {r2:.2f}")

# Save output for dashboard
df_result = df.iloc[len(df) - len(y_test):].copy()
df_result['timestamp'] = df_result['timestamp'] + pd.Timedelta(hours=2)
df_result['predicted_price'] = y_pred[-len(y_test):]
df_result.to_csv(MODEL_PATH/"btc_ridge_tuned_model.csv", index=False)

# Save best ridge model
joblib.dump(best_ridge, MODEL_PATH / "btc_ridge_tuned_model.pkl")

# Train Lasso Regressor (for evaluation)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, lasso_preds)
mae = mean_absolute_error(y_test, lasso_preds)
r2 = r2_score(y_test, lasso_preds)

print("✅ Lasso Regression Model Trained.")
print(f"📉 RMSE (test): {mse**0.5:.2f}")
print(f"📈 MAE (test): {mae:.2f}")
print(f"🔁 R² Score (test): {r2:.2f}")

# Refit Lasso on full data for inference
lasso_full = Lasso(alpha=1.0)
lasso_full.fit(X, y)
lasso_preds_full = lasso_full.predict(X)

# Slice to match test set size for dashboard compatibility
df_result = df.iloc[len(df) - len(y_test):].copy()
df_result['timestamp'] = df_result['timestamp'] + pd.Timedelta(hours=2)
df_result['predicted_price'] = lasso_preds_full[-len(y_test):]

# Keep consistent structure for dashboard
df_result = df_result.reset_index()[[
    'timestamp', 'price', 'return_1h', 'rolling_mean_3h',
    'rolling_mean_6h', 'rolling_std_3h', 'predicted_price'
]]

# Save to CSV
df_result.to_csv(MODEL_PATH / "btc_lasso_model.csv", index=False)

# Save model for dashboard use
joblib.dump(lasso_full, MODEL_PATH / "btc_lasso_model.pkl")
