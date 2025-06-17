# btc_dashboard.py

import json
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import subprocess
import time


st.set_page_config(page_title="Bitcoin Classifier & Price Predictor", layout="wide")

# Auto-refresh every 5 minutes (300,000 ms)
st_autorefresh(interval=60000, key="refresh_dashboard")

# --- Sidebar Inputs ---
st.sidebar.title("🧠 Classification Model Options")
model_option = st.sidebar.selectbox("Select Classification Model", [
    "XGBoost (Tuned)", 
    "Random Forest (Tuned)", 
    "Logistic Regression"
])

st.sidebar.title("📈 Regression Model Options")
reg_model_option = st.sidebar.selectbox("Select Regression Model", [
    "Ridge Regression (Tuned)",
    "Lasso Regression"
])

# --- Define Paths ---
clf_path = {
    "XGBoost (Tuned)": "models/btc_xgb_tuned_model.csv",
    "Random Forest (Tuned)": "models/btc_rf_tuned_model.csv",
    "Logistic Regression": "models/btc_logreg_model.csv"
}

reg_path = {
    "Ridge Regression (Tuned)": "models/btc_ridge_tuned_model.csv",
    "Lasso Regression": "models/btc_lasso_model.csv"
}

# --- Load & Cache Data ---
@st.cache_data
def load_data(model_option, reg_model_option):
    df_clf_ = pd.read_csv(clf_path[model_option])
    if 'timestamp' in df_clf_.columns:
        df_clf_['timestamp'] = pd.to_datetime(df_clf_['timestamp'])

    df_reg_ = pd.read_csv(reg_path[reg_model_option])
    if 'timestamp' in df_reg_.columns:
        df_reg_['timestamp'] = pd.to_datetime(df_reg_['timestamp'])
    df_reg_ = df_reg_.rename(columns={"price": "price_reg"})

    return df_clf_, df_reg_

# --- Refresh Pipeline Trigger ---
st.sidebar.title("🔁 Refresh Data")
if st.sidebar.button("Run Inference Pipeline"):
    with st.spinner("Running full pipeline... please wait ⏳"):
        result = subprocess.run(
            ["python", "main_pipeline.py"],
            capture_output=True,
            text=True
        )
        time.sleep(2)

        if result.returncode != 0:
            st.error(f"Pipeline failed:\n{result.stderr}")
        else:
            st.success("✅ Predictions updated successfully!")
            st.cache_data.clear()

# --- Load data ---
df_clf, df_reg = load_data(model_option, reg_model_option)

# --- Align prediction signal by merging on timestamp ---
latest_timestamp = df_clf['timestamp'].max()

merged_latest = pd.merge(
    df_clf[df_clf['timestamp'] == latest_timestamp],
    df_reg[df_reg['timestamp'] == latest_timestamp],
    on='timestamp',
    suffixes=('_clf', '_reg')
)

if not merged_latest.empty:
    latest_row = merged_latest.iloc[0]
    if latest_row['prediction'] == 1:
        st.success(
            f"📢 Last Signal: **BUY** at "
            f"Predicted ${latest_row['predicted_price']:,.2f} | "
            f"Actual ${latest_row['price_reg']:,.2f} (UTC {latest_row['timestamp']})"
        )
    else:
        st.warning(
            f"📢 Last Signal: **HOLD** at "
            f"Predicted ${latest_row['predicted_price']:,.2f} | "
            f"Actual ${latest_row['price_reg']:,.2f} (UTC {latest_row['timestamp']})"
        )
else:
    st.info("No matching timestamps between classification and regression outputs.")


# --- Date Range Filter ---
st.sidebar.title("📅 Date Range")
min_date = df_clf['timestamp'].min().date()
max_date = df_clf['timestamp'].max().date()
start_date, end_date = st.sidebar.date_input("Filter Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if start_date and end_date:
    df_clf = df_clf[(df_clf['timestamp'].dt.date >= start_date) & (df_clf['timestamp'].dt.date <= end_date)]
    df_reg = df_reg[(df_reg['timestamp'].dt.date >= start_date) & (df_reg['timestamp'].dt.date <= end_date)]

# --- Classification Section ---
st.title("🔮 Bitcoin Price Trend Classifier")
st.caption(f"Built with {model_option} | Hourly BTC price with predicted uptrends")

model_metrics = {
    "XGBoost (Tuned)": {"accuracy": "50%", "f1": "0.49", "precision": "38%"},
    "Random Forest (Tuned)": {"accuracy": "54%", "f1": "0.45", "precision": "33%"},
    "Logistic Regression": {"accuracy": "45%", "f1": "0.43", "precision": "40%"}
}

col1, col2, col3 = st.columns(3)
col1.metric("📊 Accuracy", model_metrics[model_option]['accuracy'])
col2.metric("🎯 F1 Score", model_metrics[model_option]['f1'])
col3.metric("📈 Precision (Buy)", model_metrics[model_option]['precision'])

if st.checkbox("🔍 Show Raw Classification Data"):
    st.dataframe(df_clf.tail(50))

st.subheader("📉 Bitcoin Price with Uptrend Predictions")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_clf['timestamp'], df_clf['price'], label='BTC Price', color='black', linewidth=1.5)
buy_signals = df_clf[df_clf['prediction'] == 1]
false_positives = df_clf[(df_clf['prediction'] == 1) & (df_clf['target'] == 0)]
ax.scatter(buy_signals['timestamp'], buy_signals['price'], label='Predicted Up', color='green', marker='^')
ax.scatter(false_positives['timestamp'], false_positives['price'], label='False Positive', color='red', marker='x')
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("BTC Price (USD)")
ax.grid(True)
st.pyplot(fig)

# Save classification chart to PNG
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf.getvalue()

st.download_button("📸 Download Classification Chart", fig_to_bytes(fig), file_name="btc_classification_chart.png")
csv_clf = df_clf.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Classification CSV", data=csv_clf, file_name=f"btc_predictions_{datetime.now().date()}.csv", mime='text/csv')

# --- Regression Section ---
st.subheader("📈 Next-Hour Price Prediction (Regression)")
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(df_reg['timestamp'], df_reg['price_reg'], label='Actual Price', color='black', linewidth=1.5)
ax2.plot(df_reg['timestamp'], df_reg['predicted_price'], label='Predicted Price', color='blue', linestyle='--')
ax2.set_xlabel("Time")
ax2.set_ylabel("BTC Price (USD)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

if st.checkbox("🔍 Show Raw Regression Data"):
    st.dataframe(df_reg.tail(50))

st.download_button("📸 Download Regression Chart", fig_to_bytes(fig2), file_name="btc_regression_chart.png")

reg_r2 = r2_score(df_reg['price_reg'], df_reg['predicted_price'])
reg_rmse = mean_squared_error(df_reg['price_reg'], df_reg['predicted_price']) ** 0.5
reg_mae = mean_absolute_error(df_reg['price_reg'], df_reg['predicted_price'])

colr1, colr2, colr3 = st.columns(3)
colr1.metric("🔁 R² Score", f"{reg_r2:.2f}")
colr2.metric("📉 RMSE", f"{reg_rmse:.2f}")
colr3.metric("📈 MAE", f"{reg_mae:.2f}")

csv_reg = df_reg.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Regression CSV", data=csv_reg, file_name=f"btc_price_predictions_{datetime.now().date()}.csv", mime='text/csv')
