import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import logging
import numpy as np
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
from scipy.signal import savgol_filter

data = pd.read_csv('system_usage.csv')
frame = pd.DataFrame(data)

frame['Memory'] = frame['Memory'].str.replace(' kB', '', regex=False)
frame['Memory'] = pd.to_numeric(frame['Memory'])
frame['Date'] = pd.to_datetime(frame['Date'])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
groupDataFrame = frame.sort_values(["ProcessName", "Date"])

print(groupDataFrame)

fig, ax1 = plt.subplots()

for process in groupDataFrame['ProcessName'].unique():
    subset = groupDataFrame[groupDataFrame["ProcessName"] == process]
    ax1.plot(subset["Date"], subset["CPU"], marker="o", linestyle="--", label=f"{process} CPU")

ax1.set_xlabel("Date")
ax1.set_ylabel("CPU Usage (%)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_title("CPU Usage Over Time by Process")
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.legend(loc="upper left")

fig2, ax2 = plt.subplots()

for process in groupDataFrame['ProcessName'].unique():
    subset = groupDataFrame[groupDataFrame["ProcessName"] == process]
    ax2.plot(subset["Date"], subset["Memory"], marker="s", linestyle="--", label=f"{process} Memory")

ax2.set_xlabel("Date")
ax2.set_ylabel("Memory Usage (kB)", color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_title("Memory Usage Over Time by Process")
ax2.grid(axis="y", linestyle="--", alpha=0.7)
ax2.legend(loc="upper left")

data_to = {}
for process in groupDataFrame['ProcessName'].unique():
    subsets = groupDataFrame[groupDataFrame["ProcessName"] == process]
    subsets.loc[:, 'Memory'] = savgol_filter(subsets['Memory'], 11, 3).round().astype('int64')
    subsets = subsets.rename(columns={'Date': 'ds', 'Memory': 'y'})
    data_to[process] = subsets

plt.figure(figsize=(10,6))

colors = [
    'blue', 'green', 'red', 'purple', 'orange',
    'cyan', 'magenta', 'yellow', 'brown', 'pink',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

for i, (process, subset) in enumerate(data_to.items()):
    model = Prophet()
    model.fit(subset)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast['yhat'] = np.maximum(forecast['yhat'], 0)  # Ensure values do not go below zero
    plt.plot(forecast['ds'], forecast['yhat'], color=colors[i % len(colors)], label=process)

plt.xlabel('Date')
plt.ylabel('Forecast Value')
plt.title('Forecast Lines for Multiple Processes')
plt.legend()
plt.show()
