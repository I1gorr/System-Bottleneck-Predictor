import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import early_stopping, log_evaluation

# Load data
data = pd.read_csv('system_usage_1year.csv')
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data['CPU'] = data['CPU'].astype(float)
data['MemoryUsage'] = data['MemoryUsage'].str.replace(' kB', '', regex=False).astype(int)
data['Date'] = pd.to_datetime(data['Date'])

# Compute 75th percentile of CPU usage per day (over all timestamps that day)
daily_agg = data.groupby('Date').agg({
    'CPU': lambda x: np.percentile(x, 75),
    'MemoryUsage': 'sum'
}).reset_index()

daily_agg.rename(columns={'CPU': 'CPU_75th'}, inplace=True)

# Feature engineering
daily_agg['CPU_diff'] = daily_agg['CPU_75th'].diff()
daily_agg['CPU_pct_change'] = daily_agg['CPU_75th'].pct_change()
daily_agg['Memory_diff'] = daily_agg['MemoryUsage'].diff()
daily_agg['CPU_rolling_mean_3'] = daily_agg['CPU_75th'].rolling(window=3).mean()
daily_agg['CPU_rolling_std_3'] = daily_agg['CPU_75th'].rolling(window=3).std()
daily_agg.dropna(inplace=True)

# Prepare training data
X = daily_agg[['CPU_rolling_mean_3', 'CPU_rolling_std_3', 'CPU_diff', 'CPU_pct_change']]
y = daily_agg['CPU_75th']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'lambda_l2': 0.5,
    'max_depth': 50,
    'min_data_in_leaf': 20,
    'verbose': -1
}

# Manual K-Fold Cross Validation
print("\n📊 Running manual 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_list = []
all_preds = []
all_truth = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=0)  # Set to 0 or a number to control logging
        ]
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    all_preds.extend(y_pred)
    all_truth.extend(y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_list.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

print(f"\n✅ Average RMSE: {np.mean(rmse_list):.4f}")

# --- Scatter Plot: Actual vs Predicted ---
plt.figure(figsize=(12, 6))
plt.scatter(range(len(all_truth)), all_truth, label='Actual CPU (75th)', alpha=0.6, color='green')
plt.scatter(range(len(all_preds)), all_preds, label='Predicted CPU (75th)', alpha=0.6, color='blue')
plt.axhline(y=90, color='red', linestyle='--', label='Bottleneck Threshold')
plt.title('Scatter Plot: Actual vs Predicted CPU Usage (75th Percentile Daily)')
plt.xlabel('Sample Index')
plt.ylabel('CPU Usage (%)')
plt.legend()
plt.tight_layout()
plt.show()

# Get the last row of data for prediction
last_row = daily_agg.iloc[-1][['CPU_rolling_mean_3', 'CPU_rolling_std_3', 'CPU_diff', 'CPU_pct_change', 'CPU_75th']]

# User input: Number of days to predict
# Number of days you want to predict
num_days_to_predict = int(input("Enter the number of days to predict: "))

# Initialize the list for predictions and update the features accordingly
predictions = []
last_row = daily_agg.iloc[-1].copy()  # Use .copy() to create a new object for modification

# Define the threshold for considering predictions to be static
change_threshold = 0.1  # Percentage threshold for change
consecutive_static_days = 3  # Number of consecutive days with small change before predicting static

# Variable to track consecutive days with no significant change
static_counter = 0

# Iterate over the number of days to predict
for day in range(num_days_to_predict):
    # Prepare features for the next day's prediction
    features = [
        last_row['CPU_rolling_mean_3'],
        last_row['CPU_rolling_std_3'],
        last_row['CPU_diff'],
        last_row['CPU_pct_change']
    ]

    # Make the prediction for the next day
    next_pred = model.predict([features])
    predictions.append(next_pred[0])

    # Check if the prediction is stable (i.e., doesn't change much from the last day)
    if day > 0:
        change = abs(predictions[-1] - predictions[-2])  # Difference between current and previous prediction
        if change < change_threshold:
            static_counter += 1
        else:
            static_counter = 0  # Reset counter if there is a significant change

    # Update the features for the next prediction
    # Update rolling mean and std with the new prediction
    last_row['CPU_rolling_mean_3'] = np.mean([last_row['CPU_rolling_mean_3'], next_pred[0]])
    last_row['CPU_rolling_std_3'] = np.std([last_row['CPU_rolling_mean_3'], next_pred[0]])
    last_row['CPU_diff'] = next_pred[0] - last_row['CPU_75th']
    last_row['CPU_pct_change'] = (next_pred[0] - last_row['CPU_75th']) / last_row['CPU_75th']
    
    # Update the 'CPU_75th' with the predicted value for the next day
    last_row['CPU_75th'] = next_pred[0]

    # Stop predicting if predictions become static for too long
    if static_counter >= consecutive_static_days:
        print(f"\n⚠️ Predictions have become stable after Day {day}. Further predictions may not be accurate.")
        break

# Print the predicted values
print(f"\nPredicted CPU usage (75th percentile) for the next {num_days_to_predict} days:")
for i, pred in enumerate(predictions, 1):
    print(f"Day {i}: {pred:.2f}%")
