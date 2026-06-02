import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

# System Constants
NUM_ZONES = 12
PACK_NOMINAL_VOLTAGE = 44.4
CELLS_IN_PARALLEL = 3
INITIAL_TEMP = 25.0
TEMP_MAX = 40.0
PACK_MAX_POWER = 26600.0

# Mock Actions (Zero cooling to see max stress)
zero_actions = np.zeros(NUM_ZONES + 2)

# Load Drive Cycle Data from Canonical (Sine Wave used in Iteration 2)
EVAL_HORIZON = 1800
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

# Physics constants
cell_r_variation = np.random.normal(1.0, 0.05, NUM_ZONES) # Seed was 42 in build_pipeline
zone_resistance = (0.030 * cell_r_variation * 12) / 3
zone_heat_capacity = np.full(NUM_ZONES, 20 * 50.0)

def battery_thermal_ode(t, y, power, actions):
    temps = y[:NUM_ZONES]
    I_pack = power / PACK_NOMINAL_VOLTAGE
    # CLIPPING HERE
    I_pack_clipped = np.clip(I_pack, -1.5 * CELLS_IN_PARALLEL, 6.0 * CELLS_IN_PARALLEL)
    Q_gen = (I_pack_clipped ** 2) * zone_resistance
    dTdt = Q_gen / zone_heat_capacity
    return np.concatenate([dTdt, np.zeros(NUM_ZONES)])

print("--- PHASE 1: POWER FLOW TRACE ---")
thresholds = [1000, 5000, 10000, 15000]
for thr in thresholds:
    idx = np.where(power_profile >= thr)[0]
    if len(idx) > 0:
        t = idx[0]
        p_req = power_profile[t]
        i_calc = p_req / PACK_NOMINAL_VOLTAGE
        i_clip = np.clip(i_calc, -4.5, 18.0)
        q_gen = (i_clip**2) * zone_resistance.mean()
        dt_rise = q_gen / zone_heat_capacity.mean()
        print(f"Step {t}: Req {p_req:.1f}W -> I_calc {i_calc:.2f}A -> I_clip {i_clip:.2f}A -> Q_gen {q_gen:.2f}W -> dT/dt {dt_rise:.4f}C/s")

# Comparison stats
print("\n--- PHASE 3: STRESS STATS ---")
state = np.concatenate([np.full(NUM_ZONES, INITIAL_TEMP), np.zeros(NUM_ZONES)])
for t in range(EVAL_HORIZON):
    sol = solve_ivp(lambda tau, y: battery_thermal_ode(tau, y, power_profile[t], zero_actions), [t, t+1.0], state)
    state = sol.y[:, -1]

print(f"Max temp after {EVAL_HORIZON}s (ZERO COOLING): {state[:NUM_ZONES].max():.4f}C")
print(f"Total rise: {state[:NUM_ZONES].max() - INITIAL_TEMP:.4f}C")
