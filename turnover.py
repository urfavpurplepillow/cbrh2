import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("turnoverBata.csv")

# Clean column names (remove special characters for LightGBM)
df.columns = (
    df.columns
    .str.replace('[^A-Za-z0-9_]+', '_', regex=True)  # Replace bad chars with underscore
    .str.strip('_')  # Remove leading/trailing underscores
)

# Drop unnecessary columns
drop_cols = ["Employee_ID", "FirstName", "LastName", "StartDate", "ExitDate",
             "ADEmail", "Survey_Date"]  # Adjust names after cleaning
df = df.drop(columns=drop_cols, errors='ignore')

# Remove rows where target is NaN
df = df.dropna(subset=["TurnoverScore"])

# Fill missing values in features (LightGBM can handle NaN but metrics can't)
df = df.fillna(0)

# Define features and target
X = df.drop(columns=["TurnoverScore"])
y = df["TurnoverScore"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Replace any NaN predictions with 0 (safe for metrics)
y_pred = np.nan_to_num(y_pred)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Visualization Section
metrics = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
for i, (k, v) in enumerate(metrics.items()):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.show()

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel("Actual Turnover Score")
plt.ylabel("Predicted Turnover Score")
plt.title("Actual vs. Predicted Turnover Scores")
plt.grid(True)
plt.tight_layout()
plt.show()

# Prediction Error Line Plot (first 250 samples)
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:250], label='Actual', marker='o')
plt.plot(y_pred[:250], label='Predicted', marker='x')
plt.title("Actual vs. Predicted Turnover Scores (First 250 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Turnover Score")
plt.legend()
plt.tight_layout()
plt.show()
