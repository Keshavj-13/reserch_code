import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Starting verification plotting...")
times = [1, 2, 3]
max_temps = [30, 31, 32]
mean_temps = [25, 26, 27]

plt.figure(figsize=(12, 6))
plt.plot(times, max_temps, label="Max Temp", color='red')
plt.plot(times, mean_temps, label="Mean Temp", color='blue')
plt.axhline(35.0, color='orange', linestyle='--', label="Setpoint (35C)")
plt.axhline(40.0, color='black', linestyle=':', label="Safety Limit (40C)")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (C)")
plt.title("Thermal Replay Trajectory")
plt.legend()
plt.grid(True)
print("Before savefig")
plt.savefig("test_verification.png")
print("After savefig")
