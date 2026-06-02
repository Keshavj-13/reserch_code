import numpy as np
import time
from scipy.integrate import solve_ivp

def battery_thermal_ode(t, y):
    return np.zeros(24) # Placeholder for speed test

# Benchmarking ODE latency
start = time.time()
for _ in range(100):
    solve_ivp(lambda t, y: battery_thermal_ode(t, y), [0, 1.0], np.zeros(24))
end = time.time()
latency_per_step = (end - start) / 100

HORIZON = 1800
EPOCHS = 50
MODELS = 4
BASELINES = 6

total_steps = (MODELS * EPOCHS * HORIZON) + (BASELINES * HORIZON)
total_hours = (total_steps * latency_per_step) / 3600

# Disk estimation (per step)
# Traj (10 floats), Latent (448), Emb (128), Hidden (256), Value (1)
bytes_per_step = (10 + 448 + 128 + 256 + 1) * 4 
total_gb = (total_steps * bytes_per_step) / (1024**3)

print(f"--- RESOURCE ESTIMATION ---")
print(f"Total Steps: {total_steps:,}")
print(f"Estimated Runtime: {total_hours:.2f} hours")
print(f"Estimated Disk Usage: {total_gb:.2f} GB")
print(f"ODE Latency: {latency_per_step:.4f} s/step")
