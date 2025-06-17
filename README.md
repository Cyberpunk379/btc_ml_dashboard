# 🧠 Bitcoin ML Dashboard

A real-time Bitcoin trend classifier and price predictor powered by machine learning. Built using Streamlit, this dashboard shows hourly BTC price data, predicts short-term trends (Buy/Hold), and forecasts next-hour prices using trained classification and regression models.

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-orange)

---

## 🚀 Try it Live

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/Cyberpunk379/btc_ml_dashboard/main/app/btc_dashboard.py)

---

## 📸 Demo

![Demo](demo.gif)

> The dashboard lets you:
> - Run fresh inference on hourly BTC data
> - Visualize predicted uptrends vs. false positives
> - Track next-hour price forecasts from Ridge/Lasso
> - Export charts and raw CSVs for further analysis

---

## 🧰 ML Models Used

- **Classification**  
  - Logistic Regression  
  - Random Forest (Tuned)  
  - XGBoost (Tuned)

- **Regression**  
  - Ridge Regression (Tuned)  
  - Lasso Regression  

All models are trained on engineered BTC features such as hourly returns, rolling means, and rolling standard deviations. The pipeline supports inference on the latest market data.

---

## 📂 Project Structure
```
btc_ml_dashboard/
├── app/ # Streamlit dashboard UI
│ └── btc_dashboard.py
├── data/ # Raw + feature engineered CSVs
├── models/ # .pkl files + prediction CSV outputs
├── notebooks/ # Optional: Jupyter explorations
├── scripts/ # Model training & pipeline logic
│ ├── pipeline_serve.py
│ ├── pipeline_classification_model.py
│ └── pipeline_regression_model.py
├── main_pipeline.py # One-click inference runner
├── run_all.sh # (Optional) Shell runner
├── requirements.txt # Python dependencies
└── README.md

```
---

## 🧪 Running Locally

Clone the repo:

```bash
git clone https://github.com/Cyberpunk379/btc_ml_dashboard.git
cd btc_ml_dashboard
```
Create a virtual environment (e.g., with conda or venv) and install dependencies:
```
pip install -r requirements.txt
```

Launch the dashboard:
```
streamlit run app/btc_dashboard.py
```

To run the full pipeline on latest data:
```
python main_pipeline.py
```

📜 License

MIT © 2025 Cyberpunk379

Built with ⚡ by Cyberpunk379 – feel free to fork, star, or contribute!

