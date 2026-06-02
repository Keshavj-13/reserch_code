import numpy as np
import pandas as pd
import os

NUM_ZONES = 12
PACK_NOMINAL_VOLTAGE = 44.4
PACK_MAX_POWER = 26600.0
CELLS_IN_PARALLEL = 3

# Environment parameters
EVAL_HORIZON = 100
t_arr = np.arange(EVAL_HORIZON)
raw_speed = 15.0 + 10.0 * np.sin(2 * np.pi * t_arr / 300)
speed_profile = pd.Series(raw_speed).ewm(alpha=0.25).mean().values
accel = np.zeros_like(speed_profile)
accel[1:] = (speed_profile[1:] - speed_profile[:-1])
F_total = 0.5 * 1.225 * 0.24 * 2.34 * speed_profile**2 + 2200.0 * 9.81 * 0.008 + 2200.0 * accel
P_mech = F_total * speed_profile
P_mech[P_mech >= 0] /= 0.92
P_mech[P_mech < 0] *= 0.6
power_profile = np.clip(P_mech, 0, PACK_MAX_POWER)

print(f"1. Duration: {EVAL_HORIZON} seconds")
print(f"2. Drive Cycle: Smoothed Sine (15 +/- 10 m/s)")
print(f"3. Peak Power: {power_profile.max():.2f} W")
print(f"4. Mean Power: {power_profile.mean():.2f} W")

I_pack = power_profile / PACK_NOMINAL_VOLTAGE
I_pack_clipped = np.clip(I_pack, -1.5 * CELLS_IN_PARALLEL, 6.0 * CELLS_IN_PARALLEL)
print(f"5. Peak Current (Unclipped): {I_pack.max():.2f} A")
print(f"5b. Peak Current (Clipped): {I_pack_clipped.max():.2f} A")

# Load one result
df = pd.read_csv("results/iteration_2/flight_recorder/Proposed_Full/trajectory.csv")
print(f"6. Max Temp Rise: {df['max_temp'].max() - 25.0:.4f} °C")
print(f"7. Total Cooling Energy (Pump): {df['energy'].sum():.2f} J (approx)")
print(f"8. Actions Issued: {len(df)}")
