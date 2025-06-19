# 🧠 Bitcoin ML Dashboard

A real-time Bitcoin trend classifier and price predictor powered by machine learning. Built using Streamlit, this dashboard shows hourly BTC price data, predicts short-term trends (Buy/Hold), and forecasts next-hour prices using trained classification and regression models.

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-orange)
![Model Accuracy](https://img.shields.io/badge/accuracy-92%25-brightgreen)

---

## 🚀 Try it Live

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://btcmldashboard-9ozcscbhkbzm2xifxgyxnk.streamlit.app)

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
├── app/                  # Streamlit dashboard UI
│   └── btc_dashboard.py
├── data/                # Raw + feature engineered CSVs
├── models/              # .pkl files + prediction CSV outputs
├── notebooks/           # Optional: Jupyter explorations
├── scripts/             # Model training & pipeline logic
│   ├── pipeline_serve.py
│   ├── pipeline_classification_model.py
│   └── pipeline_regression_model.py
├── main_pipeline.py     # One-click inference runner
├── run_all.sh           # (Optional) Shell runner
├── requirements.txt     # Python dependencies
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
```bash
pip install -r requirements.txt
```

To launch the dashboard:
```bash
streamlit run app/btc_dashboard.py
```

To run the end-to-end ML pipeline on fresh BTC data:
```bash
python main_pipeline.py
```

Output files:
- `data/btc_hourly_yf.csv` – raw BTC data (hourly)
- `data/btc_feature.csv` – feature engineered dataset
- `models/btc_*_model.csv` – model predictions (classification + regression)

---

## 🛠 Troubleshooting

**KeyError: 'Datetime' not in index**  
➡️ Ensure you're running `main_pipeline.py` which uses dynamic column detection after resetting the index.

**ValueError: Found array with 0 sample(s)**  
➡️ Check for missing values or logic errors in `feature_engineering.py`. The pipeline will fail if the dataset is empty after preprocessing.

**Unpickling model error (scikit-learn mismatch)**  
➡️ Ensure your environment matches the version used for model training:
```bash
pip install scikit-learn==1.6.1
```
Or retrain models locally to match your installed version.

**Streamlit Cloud errors**  
➡️ Make sure your project paths are relative and all required files are committed. Avoid absolute `~/Users/...` references for deployment.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes
4. Push to your fork and submit a PR

We welcome contributions, feature requests, and issue reports!

📜 License

MIT © 2025 Cyberpunk379

Built with ⚡ by Cyberpunk379 – feel free to fork, star, or contribute!
