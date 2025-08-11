import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import joblib  # Import joblib for saving the model


# Load your dataset
df = pd.read_csv("turnoverBata.csv")

# Sort by EmployeeID then SurveyDate
df = df.sort_values(['Employee ID', 'Survey Date']).reset_index(drop=True)

# Rule 1: If ExitDate is set → TurnoverScore = 100
df.loc[df['ExitDate'].notna(), 'TurnoverScore'] = 1

# Rule 2: Look ahead to next survey for same EmployeeID
for i in range(len(df) - 1):
    current_emp = df.loc[i, 'Employee ID']
    next_emp = df.loc[i + 1, 'Employee ID']
    
    if current_emp == next_emp:  # same employee
        if pd.notna(df.loc[i + 1, 'ExitDate']):  # next survey has ExitDate
            if df.loc[i, 'TurnoverScore'] < 70:
                df.loc[i, 'TurnoverScore'] = (random.uniform(70, 100))/100

df['TurnoverScore'] = df['TurnoverScore'].round(2)


# Clean column names (remove special characters for LightGBM)
df.columns = (
    df.columns
    .str.replace('[^A-Za-z0-9_]+', '_', regex=True)  # Replace bad chars with underscore
    .str.strip('_')  # Remove leading/trailing underscores
)

# Drop unnecessary columns
drop_cols = ['Unnamed_0', 'Employee_ID_1',"Employee_ID", "FirstName", "LastName", "StartDate", "ExitDate","Survey_Date",
             "ADEmail", "Survey Date",'TerminationType_x', 'TerminationDescription_x', 'TerminationType_y', 'TerminationDescription_y']  # Adjust names after cleaning
df = df.drop(columns=drop_cols, errors='ignore')

df.rename(columns={
    'Engagement_Score': 'Engagement Score',
    'Satisfaction_Score': 'Satisfaction Score',
    'Work_Life_Balance_Score': 'Work-Life Balance Score',
    'Current_Employee_Rating': 'Current Employee Rating'
}, inplace=True)

df.to_csv("output1.csv", index=False)
with open("columns.txt", "w") as f:
    for col in df.columns:
        f.write(col + "\n")

print("✅ Data saved to output1.csv and columns saved to columns.txt")
# Remove rows where target is NaN
df = df.dropna(subset=["TurnoverScore"])

# Fill missing values in features (LightGBM can handle NaN but metrics can't)
df = df.fillna(0)

# Define features and target
X = df.drop(columns=["TurnoverScore"])
y = df["TurnoverScore"]
y = np.asarray(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'turnover_model.pkl')
print("✅ Model saved to 'turnover_model.pkl'")

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

with open("note.txt", "a") as f:  # "a" means append, not overwrite
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

# Visualization Section
metrics = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
for i, (k, v) in enumerate(metrics.items()):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()

# Save the plot in the same directory as the script
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "model_metrics.png")
plt.savefig(save_path)

print(f"✅ Chart saved to {save_path}")

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

# Save in same folder as script
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "actual_vs_predicted.png")
plt.savefig(save_path)

plt.show()
print(f"✅ Plot saved to {save_path}")

# Prediction Error Line Plot (first 250 samples)
plt.figure(figsize=(10, 5))
plt.plot(y_test[:250], label='Actual', marker='o')
plt.plot(y_pred[:250], label='Predicted', marker='x')
plt.title("Actual vs. Predicted Turnover Scores (First 250 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Turnover Score")
plt.legend()
plt.tight_layout()
plt.show()
