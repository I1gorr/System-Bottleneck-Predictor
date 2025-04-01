import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import logging
import numpy as np
from scipy.signal import savgol_filter

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Load dataset
data = pd.read_csv('system_usage.csv')
frame = pd.DataFrame(data)


frame['Memory'] = frame['Memory'].str.replace(' kB', '', regex=False).astype(int)
frame['CPU'] = frame['CPU'].astype(float)
frame['Date'] = pd.to_datetime(frame['Date'])
groupDataFrame = frame.sort_values(["ProcessName", "Date"])


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


totalCPU = groupDataFrame.groupby("Date")["CPU"].sum().reset_index()


with open('/proc/stat', 'r') as f:
    first_line = f.readline()
cpu_times = list(map(int, first_line.split()[1:])) 
total_cpu_time = sum(cpu_times) 

# Normalize CPU usage
totalCPU["CPU"] = (totalCPU["CPU"] / total_cpu_time) * 100  # Scale CPU usage

# Ensure values stay within 0-100%
totalCPU["CPU"] = totalCPU["CPU"].clip(0, 100)

# **🔹 Plot Overall CPU Usage**
fig3, ax3 = plt.subplots()
ax3.plot(totalCPU["Date"], totalCPU["CPU"], marker="*", linestyle="--", color="green", label="Total CPU Usage")

ax3.set_xlabel("Date")
ax3.set_ylabel("Total CPU Usage (%)")
ax3.set_title("Total CPU Usage Over Time")
ax3.grid(axis="y", linestyle="--", alpha=0.7)
ax3.legend()


forecast_data = totalCPU.rename(columns={"Date": "ds", "CPU": "y"})

model = Prophet()
model.fit(forecast_data)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

plt.figure(figsize=(10, 6))
plt.plot(forecast["ds"], forecast["yhat"], label="Predicted CPU Usage", color="blue")
plt.xlabel("Date")
plt.ylabel("CPU Usage (%)")
plt.title("CPU Usage Forecast")
plt.legend()


plt.figure(figsize=(10, 6))
colors = [
    'blue', 'green', 'red', 'purple', 'orange',
    'cyan', 'magenta', 'yellow', 'brown', 'pink'
]

for i, process in enumerate(groupDataFrame['ProcessName'].unique()):
    subset = groupDataFrame[groupDataFrame["ProcessName"] == process].copy()
    
    # Apply smoothing
    subset.loc[:, 'CPU'] = savgol_filter(subset['CPU'], 5, 2).round(2)
    
    subset = subset.rename(columns={'Date': 'ds', 'CPU': 'y'})
    
    model = Prophet()
    model.fit(subset)
    
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    plt.plot(forecast['ds'], forecast['yhat'], color=colors[i % len(colors)], label=f"{process} Prediction")

plt.xlabel("Date")
plt.ylabel("Predicted CPU Usage (%)")
plt.title("Per-Process CPU Usage Forecast")
plt.legend()
plt.show()
