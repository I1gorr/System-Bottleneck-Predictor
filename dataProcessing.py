import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('system_usage_30days.csv')
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data['MemoryUsage'] = data['MemoryUsage'].str.replace(' kB', '', regex=False).astype(int)
data['CPU'] = data['CPU'].astype(float)

# Aggregate per timestamp
agg_data = data.groupby('DateTime').agg({
    'CPU': 'sum',               # Total CPU usage
    'MemoryUsage': 'sum'        # Total memory usage
}).reset_index()

# Add trend-based features
agg_data['CPU_rolling_mean_3'] = agg_data['CPU'].rolling(window=3).mean()
agg_data['CPU_rolling_std_3'] = agg_data['CPU'].rolling(window=3).std()
agg_data['CPU_diff'] = agg_data['CPU'].diff()
agg_data['CPU_pct_change'] = agg_data['CPU'].pct_change()
agg_data['Memory_diff'] = agg_data['MemoryUsage'].diff()

# Drop rows with NaNs (due to rolling and diff)
agg_data.dropna(inplace=True)

# Prepare features and target
X = agg_data[['CPU_rolling_mean_3', 'CPU_rolling_std_3', 'CPU_diff', 'CPU_pct_change', 'MemoryUsage', 'Memory_diff']]
y = agg_data['CPU']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Model params
params = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'lambda_l2': 0.5,
    'max_depth': 6,
    'min_data_in_leaf': 30,
    'verbose': -1
}

# Callbacks
early_stopping_callback = lgb.early_stopping(stopping_rounds=50)
log_eval_callback = lgb.log_evaluation(period=100)

# Train model
evals_result = {}
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    num_boost_round=1000,
    callbacks=[early_stopping_callback, log_eval_callback, lgb.record_evaluation(evals_result)]
)

# Predict on test set
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# --- Predict future usage ---

# Get user input
hours_to_predict = int(input("Enter the number of future hours to predict (e.g., 100): "))

# Simulate future data points with mean/last-known trend
last_row = agg_data.iloc[-1].copy()
future_rows = []
for i in range(hours_to_predict):
    new_row = last_row.copy()
    new_row['DateTime'] = last_row['DateTime'] + pd.Timedelta(hours=1)
    
    # For simulation, we assume CPU continues to follow last known diff
    new_row['CPU'] = last_row['CPU'] + last_row['CPU_diff']
    new_row['MemoryUsage'] = last_row['MemoryUsage'] + (last_row['Memory_diff'] if not np.isnan(last_row['Memory_diff']) else 0)
    
    # Update derived features
    cpu_window = agg_data['CPU'].iloc[-2:].tolist() + [new_row['CPU']]
    new_row['CPU_rolling_mean_3'] = np.mean(cpu_window)
    new_row['CPU_rolling_std_3'] = np.std(cpu_window)
    new_row['CPU_diff'] = new_row['CPU'] - last_row['CPU']
    new_row['CPU_pct_change'] = new_row['CPU_diff'] / last_row['CPU'] if last_row['CPU'] != 0 else 0
    new_row['Memory_diff'] = new_row['MemoryUsage'] - last_row['MemoryUsage']
    
    # Append and update
    future_rows.append(new_row)
    last_row = new_row

future_df = pd.DataFrame(future_rows)

# Prepare future input
X_future = future_df[['CPU_rolling_mean_3', 'CPU_rolling_std_3', 'CPU_diff', 'CPU_pct_change', 'MemoryUsage', 'Memory_diff']]
X_future_scaled = scaler.transform(X_future)

# Predict
future_pred = model.predict(X_future_scaled, num_iteration=model.best_iteration)

# Detect bottlenecks
bottleneck_indices = np.where(future_pred > 90)[0]
if len(bottleneck_indices) > 0:
    print("\n⚠️  Bottleneck predicted at:")
    for i in bottleneck_indices:
        print(future_df.iloc[i]['DateTime'].strftime('%Y-%m-%d %H:%M'))
else:
    print(f"\n✅ No bottleneck predicted in the next {hours_to_predict} hours.")

# --- Plots ---

# Future CPU usage plot
plt.figure(figsize=(12, 6))
plt.plot(future_df['DateTime'], future_pred, label='Predicted CPU Usage', color='blue')
plt.axhline(y=90, color='r', linestyle='--', label='Bottleneck Threshold')
plt.legend()
plt.title(f'Predicted CPU Usage for Next {hours_to_predict} Hours')
plt.xlabel('Time')
plt.ylabel('CPU Usage (%)')
plt.xticks(rotation=45)
plt.tight_layout()

# Actual vs predicted plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual CPU Usage', color='orange')
plt.plot(y_pred, label='Predicted CPU Usage', color='blue')
plt.axhline(y=90, color='r', linestyle='--', label='Bottleneck Threshold')
plt.legend()
plt.title('Actual vs Predicted CPU Usage on Test Data')
plt.xlabel('Sample Index')
plt.ylabel('CPU Usage (%)')
plt.tight_layout()

plt.show()
