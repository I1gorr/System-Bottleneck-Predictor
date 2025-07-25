# System Resource Forecasting & Monitoring Tool 📈🖥️

This project is a simulation-based system resource analysis and forecasting tool. It generates realistic CPU and memory usage logs over one year, identifies high CPU-consuming processes, and predicts future CPU bottlenecks using LightGBM.

---

## 🚀 Features

- 🧠 Predicts 75th percentile daily CPU usage using LightGBM
- 🔍 Scans for high CPU usage processes using `/proc` in C++
- 🧪 Simulates 1 year of realistic Linux process activity
- 📊 Performs feature engineering and rolling statistics
- 📉 Forecasts CPU usage trends with early-stopping logic

---

## 🧰 Tech Stack

- Python (pandas, NumPy, LightGBM, matplotlib, scikit-learn)
- C++ (for scanning and logging Linux processes)
- CSV for data interchange

---

## 📂 File Structure

```
├── dataGenerator.py         # Simulates realistic process logs
├── dataProcessing.py        # Loads data, trains LightGBM, predicts future CPU
├── findProcesses.cpp        # Scans /proc for high CPU usage
├── HighCPUProcesses.h       # Header for struct definitions in C++
├── processes.csv            # Output from C++ scanner
├── system_usage_1year.csv   # Generated dataset from simulator
└── README.md                # This file
```

---

## 📦 How to Use

### 1. Generate Synthetic System Usage Logs
```bash
python dataGenerator.py
```
Generates `system_usage_1year.csv` simulating Linux process behavior over one year.

### 2. Identify High CPU Usage Processes (Linux Only)
```bash
g++ findProcesses.cpp -o findProcesses
./findProcesses
```
Produces `processes.csv` with high CPU usage processes based on `/proc`.

### 3. Train Model and Predict CPU Usage
```bash
python dataProcessing.py
```
- Performs feature engineering (diff, % change, rolling mean/std)
- Trains LightGBM with cross-validation
- Predicts user-defined number of future days
- Warns if predictions become stable (flat CPU usage)
- Shows actual vs predicted scatter plot

---

## 🧠 Sample Prediction Output
```
Predicted CPU usage (75th percentile) for the next 7 days:
Day 1: 83.21%
Day 2: 84.07%
Day 3: 84.16%
...
```

---

## 🛠️ Requirements

Install dependencies:
```bash
pip install pandas numpy lightgbm matplotlib scikit-learn
```

---

## 📌 Notes

- The prediction logic uses early stopping and rolling features to avoid overfitting.
- Predictions stop automatically if the trend becomes stable.
- Simulated data includes realistic software patterns (e.g., IDEs, games, browsers).

---

## 📜 License

This project is intended for academic and personal use.
