import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp

# --- PHYSICAL CONSTANTS (PRESERVED) ---
NUM_ZONES = 12
CELLS_IN_SERIES = 12
CELLS_IN_PARALLEL = 3
TOTAL_CELLS = CELLS_IN_SERIES * CELLS_IN_PARALLEL
PACK_NOMINAL_VOLTAGE = 44.4
TEMP_MAX = 40.0
TEMP_MIN = 20.0
COOLANT_TEMP = 25.0
INITIAL_TEMP = 25.0
CELL_HEAT_CAPACITY = 50.0 
CELL_INTERNAL_R = 0.030 

zone_heat_capacity = np.full(NUM_ZONES, (TOTAL_CELLS // NUM_ZONES) * CELL_HEAT_CAPACITY)
zone_resistance = np.full(NUM_ZONES, (CELL_INTERNAL_R * CELLS_IN_SERIES) / CELLS_IN_PARALLEL)
zone_positions = np.linspace(0, 1, NUM_ZONES)
lateral_conductance = 3.0 * (1.0 - 0.4 * np.abs(zone_positions - 0.5) * 2.0)

# ================= DRIVE CYCLE LOADER =================
def load_original_drive_cycles(folder_path="drive_cycles"):
    target_files = ["ftpcol.txt", "hwycol.txt", "j1015col.txt", "sc03col.txt", "uddscol.txt", "us06col.txt"]
    dfs = []
    cumulative_time = 0
    for f in target_files:
        path = os.path.join(folder_path, f)
        if not os.path.exists(path): continue
        df = None
        for enc in ['utf-8', 'ISO-8859-1', 'utf-16']:
            try:
                df_try = pd.read_csv(path, sep=None, engine='python', header=None, comment='#', encoding=enc)
                if df_try.shape[1] >= 2:
                    df = df_try.iloc[:, :2].copy()
                    df.columns = ['Time', 'Speed']
                    break
            except Exception:
                continue
        if df is None: continue
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['Speed'] = pd.to_numeric(df['Speed'], errors='coerce')
        df = df.dropna(subset=['Time', 'Speed']).copy()
        df['Time'] = df['Time'].round().astype(int)
        df['Speed'] = df['Speed'].round(1)
        df['Time'] = df['Time'] + cumulative_time
        cumulative_time = df['Time'].max() + 1
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    
    VEHICLE_MASS = 2200.0
    speed_mps = combined['Speed'].values * 0.44704
    speed_s = pd.Series(speed_mps).ewm(alpha=0.25).mean().values
    accel = np.zeros_like(speed_s)
    accel[1:] = (speed_s[1:] - speed_s[:-1])
    F_total = 0.5 * 1.225 * 0.24 * 2.34 * speed_s**2 + VEHICLE_MASS * 9.81 * 0.008 + VEHICLE_MASS * accel
    P_mech = F_total * speed_s
    P_mech[P_mech >= 0] /= 0.92
    P_mech[P_mech < 0] *= 0.6
    pack_current = P_mech / PACK_NOMINAL_VOLTAGE
    pack_current_clipped = np.clip(pack_current, -1.5 * CELLS_IN_PARALLEL, 6.0 * CELLS_IN_PARALLEL)
    power_profile = pack_current_clipped * PACK_NOMINAL_VOLTAGE
    power_profile[speed_s < 0.5] = 0.0
    return power_profile

# ================= PHYSICS ODE =================
def battery_thermal_ode(t, y, power_profile, actions):
    temps = y[:NUM_ZONES]
    idx = min(int(t), len(power_profile)-1)
    I_pack = power_profile[idx] / PACK_NOMINAL_VOLTAGE
    Q_gen = (I_pack ** 2) * zone_resistance
    pump = actions[NUM_ZONES]
    fan = actions[NUM_ZONES+1]
    UA = 0.5 + 5.0 * pump + 2.0 * fan 
    Q_cool = UA * (temps - COOLANT_TEMP)
    Q_cond = np.zeros(NUM_ZONES)
    for i in range(NUM_ZONES):
        if i > 0: Q_cond[i] += lateral_conductance[i-1] * (temps[i-1] - temps[i])
        if i < NUM_ZONES - 1: Q_cond[i] += lateral_conductance[i] * (temps[i+1] - temps[i])
    dTdt = (Q_gen - Q_cool + Q_cond) / zone_heat_capacity
    return dTdt

# ================= CONTROLLERS =================
def get_action(name, temps):
    tmax = temps.max()
    if name == "PID_Standard":
        val = 0.5 if tmax > 30.0 else 0.0
        return np.full(NUM_ZONES + 2, val)
    if name == "PID_Adaptive":
        val = np.clip((tmax - 28.0) / 10.0, 0, 1)
        return np.full(NUM_ZONES + 2, val)
    if name == "Uniform_Flow":
        return np.full(NUM_ZONES + 2, 1.0)
    if name == "Proportional_Temp":
        val = np.clip((tmax - INITIAL_TEMP) / (TEMP_MAX - INITIAL_TEMP), 0, 1)
        return np.full(NUM_ZONES + 2, val)
    return np.zeros(NUM_ZONES + 2)

def run_phase2():
    print("Starting Phase 2: Physics Verification...")
    power_profile = load_original_drive_cycles()
    horizon = len(power_profile)
    controllers = ["PID_Standard", "PID_Adaptive", "Uniform_Flow", "Proportional_Temp"]
    
    results = []
    
    for name in controllers:
        print(f"  Simulating {name}...")
        state = np.full(NUM_ZONES, INITIAL_TEMP)
        history_tmax = []
        history_tmean = []
        energy_sum = 0
        gradients = []
        
        for t in range(horizon):
            actions = get_action(name, state)
            dTdt = battery_thermal_ode(t, state, power_profile, actions)
            state = state + dTdt * 1.0
            
            history_tmax.append(state.max())
            history_tmean.append(state.mean())
            energy_sum += actions[NUM_ZONES] * 200.0 # W
            gradients.append(np.ptp(state))
            
        results.append({
            'controller': name,
            'tmax': float(max(history_tmax)),
            'tmean': float(np.mean(history_tmean)),
            'cooling_energy': energy_sum / 3600.0, # Wh
            'safety_events': int(sum(1 for t in history_tmax if t > 40.0)),
            'thermal_gradient': float(max(gradients))
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("metrics/thermal_replay.csv", index=False)
    
    with open("reports/Physics_Verification_Report.md", "w") as f:
        f.write("# Phase 2: Physics Verification Report\n\n")
        f.write("| Controller | Tmax (°C) | Tmean (°C) | Energy (Wh) | Safety Events | Max Gradient (K) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in df_res.iterrows():
            f.write(f"| {r['controller']} | {r['tmax']:.2f} | {r['tmean']:.2f} | {r['cooling_energy']:.2f} | {r['safety_events']} | {r['thermal_gradient']:.2f} |\n")
            
    print("Phase 2 Complete.")

if __name__ == "__main__":
    run_phase2()
