import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb



# Load your dataset
df = pd.read_csv("turnover_data.csv")

# Drop unnecessary columns
drop_cols = ["Employee ID", "FirstName", "LastName", "StartDate", "ExitDate", 
             "ADEmail", "TerminationDescription", "Supervisor"]
df = df.drop(columns=drop_cols, errors='ignore')

# Define features and target
X = df.drop(columns=["TurnoverScore"])
y = df["TurnoverScore"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")





# 📊 Visualization Section
# -----------------------
metrics = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
for i, (k, v) in enumerate(metrics.items()):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.show()

# # 1. Feature Importance
# lgb.plot_importance(model, max_num_features=15)
# plt.title("Top 15 Feature Importances (LightGBM)")
# plt.tight_layout()
# plt.show()

# 2. Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel("Actual Turnover Score")
plt.ylabel("Predicted Turnover Score")
plt.title("Actual vs. Predicted Turnover Scores")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Residuals Plot
# residuals = y_test - y_pred
# plt.figure(figsize=(8, 6))
# plt.hist(residuals, bins=30, edgecolor='k')
# plt.title("Residuals Distribution")
# plt.xlabel("Residual (Actual - Predicted)")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# 4. Prediction Error Line Plot (first 100 samples)
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:250], label='Actual', marker='o')
plt.plot(y_pred[:250], label='Predicted', marker='x')
plt.title("Actual vs. Predicted Turnover Scores (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Turnover Score")
plt.legend()
plt.tight_layout()
plt.show()