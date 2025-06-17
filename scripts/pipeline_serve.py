import joblib
import pandas as pd
from pathlib import Path

def serve_classification(model_name, features_df):
    """
    Load classification model and return predictions
    """
    MODEL_PATH = Path(__file__).resolve().parent.parent / "models"
    model_files = {
        "Logistic Regression": "btc_logreg_model.pkl",
        "Random Forest (Tuned)": "btc_rf_tuned_model.pkl",
        "XGBoost (Tuned)": "btc_xgb_tuned_model.pkl"
    }

    model = joblib.load(MODEL_PATH / model_files[model_name])
    preds = model.predict(features_df)
    return preds

def serve_regression(model_name, features_df):
    """
    Load regression model and return predictions
    """
    MODEL_PATH = Path(__file__).resolve().parent.parent / "models"
    model_files = {
        "Ridge Regression (Tuned)": "btc_ridge_tuned_model.pkl",
        "Lasso Regression": "btc_lasso_model.pkl"
    }

    model = joblib.load(MODEL_PATH / model_files[model_name])
    preds = model.predict(features_df)
    return preds

# Example usage (comment out in production)
#if __name__ == "__main__":
    DATA_PATH = Path(__file__).resolve().parent / "data"
    df = pd.read_csv(DATA_PATH / "btc_feature.csv")

    # Drop unnecessary columns for serving
    X = df.drop(columns=['timestamp', 'future_price', 'target'], errors='ignore')

    # Classification example
    clf_preds = serve_classification("Logistic Regression", X)
    print("Classification Sample:", clf_preds[:5])

    # Regression example
    reg_preds = serve_regression("Ridge Regression (Tuned)", X)
    print("Regression Sample:", reg_preds[:5])
