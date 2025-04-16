import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('system_usage_30days.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['MemoryUsage'] = data['MemoryUsage'].str.replace(' kB', '', regex=False).astype(int)
data['CPU'] = data['CPU'].astype(float)

# Feature engineering: Add time-based features
data['Hour'] = data['Time'].apply(lambda x: int(x.split(':')[0]))
data['Minute'] = data['Time'].apply(lambda x: int(x.split(':')[1]))
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)

# Drop constant or near-constant features
constant_features = data.columns[data.nunique() <= 1]
data = data.drop(columns=constant_features)

# Prepare features and target
X = data[['MemoryUsage', 'Hour_sin', 'Hour_cos', 'Minute']]
y = data['CPU']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    'num_leaves': 50,  # Increased for better capture of complex patterns
    'learning_rate': 0.01,  # Smaller learning rate to avoid overfitting
    'feature_fraction': 0.8,  # Fraction of features to use
    'lambda_l2': 0.5,  # Regularization parameter
    'max_depth': 6,  # Slightly deeper trees
    'min_data_in_leaf': 30,  # Decreased to allow more flexibility with small dataset
    'verbose': -1  # Suppress output to keep it clean
}

# Early stopping callback
early_stopping_callback = lgb.callback.early_stopping(stopping_rounds=50)
log_eval_callback = lgb.callback.log_evaluation(period=100)

# Train the LightGBM model
evals_result = {}
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    num_boost_round=1000,
    callbacks=[early_stopping_callback, log_eval_callback, lgb.callback.record_evaluation(evals_result)]
)

# Predict on the test data
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nMean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Get user input for hours to predict
hours_to_predict = int(input("Enter the number of hours you want to predict (e.g., 100): "))

# Generate future dates with all time points from original data
unique_times = data['Time'].unique()
last_date = data['Date'].max()
start_date = last_date + pd.DateOffset(days=1)
future_dates = pd.date_range(start=start_date, periods=hours_to_predict, freq='H').date

future_data = []
for date in future_dates:
    for time in unique_times:
        future_data.append({'Date': date, 'Time': time})

future_df = pd.DataFrame(future_data)

# Extract Hour and Minute from Time
future_df['Hour'] = future_df['Time'].apply(lambda x: int(x.split(':')[0]))
future_df['Minute'] = future_df['Time'].apply(lambda x: int(x.split(':')[1]))
future_df['Hour_sin'] = np.sin(2 * np.pi * future_df['Hour'] / 24)
future_df['Hour_cos'] = np.cos(2 * np.pi * future_df['Hour'] / 24)

# Use mean MemoryUsage (adjust if you have trend data)
future_df['MemoryUsage'] = data['MemoryUsage'].mean()

# Prepare features for prediction
X_future = future_df[['MemoryUsage', 'Hour_sin', 'Hour_cos', 'Minute']].values
X_future_scaled = scaler.transform(X_future)

# Predict future CPU usage
future_pred = model.predict(X_future_scaled, num_iteration=model.best_iteration)

# Create datetime for future predictions
future_df['DateTime'] = pd.to_datetime(future_df['Date'].astype(str) + ' ' + future_df['Time'])

# Check for bottleneck (CPU > 90%)
bottleneck_indices = np.where(future_pred > 90)[0]
if len(bottleneck_indices) > 0:
    bottleneck_dates = future_df.iloc[bottleneck_indices]['DateTime']
    print("\nBottleneck predicted at the following dates/times:")
    for date in bottleneck_dates:
        print(date.strftime('%Y-%m-%d %H:%M'))
else:
    print("\nNo bottleneck predicted in the next {} hours.".format(hours_to_predict))


# Plot 1: Predicted CPU Usage for the Next 'hours_to_predict' Hours
plt.figure(figsize=(12, 6))
plt.plot(future_df['DateTime'][:hours_to_predict], future_pred[:hours_to_predict], label='Predicted CPU Usage', color='blue')
plt.axhline(y=90, color='r', linestyle='--', label='Bottleneck Threshold')
plt.legend()
plt.title(f'Predicted CPU Usage for the Next {hours_to_predict} Hours')
plt.ylabel('CPU Usage (%)')
plt.xlabel('Date and Time')
plt.xticks(rotation=45)
plt.tight_layout()

# Plot 2: Actual vs Predicted CPU Usage on Test Data
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual CPU Usage', color='orange')  # Actual test data
plt.plot(y_pred, label='Predicted CPU Usage', color='blue')  # Predicted values
plt.axhline(y=90, color='r', linestyle='--', label='Bottleneck Threshold')
plt.legend()
plt.title('Actual vs Predicted CPU Usage on Test Data')
plt.ylabel('CPU Usage (%)')
plt.xlabel('Sample Index')
plt.tight_layout()

# Show both graphs
plt.show()