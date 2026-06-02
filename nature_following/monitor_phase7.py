import os
import pandas as pd
import time

def monitor():
    print("=== Phase 7: Training Monitor ===")
    
    if os.path.exists("reports/FailFast_Report.md"):
        print("\n[CRITICAL] Fail Fast Triggered! Check reports/FailFast_Report.md")
        with open("reports/FailFast_Report.md", "r") as f:
            print(f.read())
            
    if os.path.exists("metrics/anomaly_log.csv"):
        print("\n--- Anomalies ---")
        try:
            df = pd.read_csv("metrics/anomaly_log.csv")
            print(df.tail(5))
        except Exception as e:
            print(f"Could not read anomaly_log.csv: {e}")
            
    if os.path.exists("metrics/epoch_summary.csv"):
        print("\n--- Training Progress ---")
        try:
            df = pd.read_csv("metrics/epoch_summary.csv")
            latest = df.groupby('model').last()
            print(latest[['epoch', 'reward', 'max_temp', 'cooling_energy']])
        except Exception as e:
            print(f"Could not read epoch_summary.csv: {e}")
            
    if os.path.exists("metrics/storage_usage.csv"):
        print("\n--- Storage Usage ---")
        try:
            df = pd.read_csv("metrics/storage_usage.csv")
            print(f"Current Disk Usage: {df['results_mb'].iloc[-1]:.2f} MB")
        except Exception as e:
            print(f"Could not read storage_usage.csv: {e}")

if __name__ == "__main__":
    monitor()
