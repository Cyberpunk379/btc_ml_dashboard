from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust if needed
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"
DATA_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)

# --- Step 1: Load Feature Data ---
df = pd.read_csv(DATA_PATH / "btc_feature.csv", parse_dates=['timestamp'])

# --- Step 2: Define Classification Target ---
df['target'] = (df['future_price'] > df['price']).astype(int)

# --- Step 3: Define Features ---
features = ['return_1h', 'rolling_mean_3h', 'rolling_mean_6h', 'rolling_std_3h']
X = df[features]
y = df['target']

# --- Step 4: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# --- Step 5: Train Model ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Step 6: Predict on Full Data ---
df['prediction'] = model.predict(X)

# --- Step 7: Save CSV Output for Dashboard ---
df_out = df[['timestamp', 'price', 'return_1h', 'rolling_mean_3h', 
             'rolling_mean_6h', 'rolling_std_3h', 'target', 'prediction']].copy()


df_out.to_csv(MODEL_PATH / "btc_logreg_model.csv", index=False)


# --- Step 8: Save Model ---
joblib.dump(model, MODEL_PATH / "btc_logreg_model.pkl")

# --- Step 9: Evaluate ---
print("✅ Logistic Regression Model Trained.")
print(classification_report(y_test, model.predict(X_test)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))



# Step 1: Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Step 2: Predict and store
df['prediction'] = model.predict(X)

# Step 3: Save predictions
df.to_csv(MODEL_PATH/"btc_rf_model.csv")
joblib.dump(model, MODEL_PATH/'btc_rf_model.pkl')

# Step 4: Evaluation
print("✅ Random Forest Model Trained. Evaluation:")
print(classification_report(y_test, model.predict(X_test)))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))

# FINETUNE RANDOM FOREST
# Step 1: Define grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

# Step 2: Grid search
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

# Step 3: Best model
model = grid.best_estimator_
df['prediction'] = model.predict(X)

# Step 4: Save results
df.to_csv(MODEL_PATH/"btc_rf_tuned_model.csv")
joblib.dump(model, MODEL_PATH/'btc_rf_tuned_model.pkl')

# Step 5: Evaluation
print("✅ Tuned RF Model. Best Params:")
print(grid.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))

# Step 1: Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Step 2: Predict
df['prediction'] = model.predict(X)

# Step 3: Save output
df.to_csv(MODEL_PATH/"btc_xgb_model.csv")
joblib.dump(model, MODEL_PATH/'btc_xgb_model.pkl')

# Step 4: Evaluate
print("✅ XGBoost Model Trained.")
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))

# FINE-TUNING XGBOOST
# Step 1: Grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'scale_pos_weight': [1, float(sum(y == 0)) / sum(y == 1)]
}

grid = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# Step 4: Best model
model = grid.best_estimator_
df['prediction'] = model.predict(X)

# Step 5: Save
df.to_csv(MODEL_PATH/"btc_xgb_tuned_model.csv")
joblib.dump(model, MODEL_PATH/"btc_xgb_tuned_model.pkl")

# Step 6: Evaluate
print("✅ Tuned XGBoost Model")
print("Best Params:", grid.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))


with open(MODEL_PATH / 'xgb_best_params.json', 'w') as f:
    json.dump(grid.best_params_, f)
