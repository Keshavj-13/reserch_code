import time
import nbformat as nbf
import os

with open('Canonical_Manuscript_Master.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

validation_results = []

def check_content(pattern, description):
    found = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and pattern in cell.source:
            found = True
            break
    validation_results.append((description, found))

check_content("IS_TEST_MODE = False", "IS_TEST_MODE is False")
check_content("GraphSAGEEncoder", "GraphSAGE branch exists")
check_content("TemporalEncoder", "LSTM branch exists")
check_content("PPOAgent", "PPO update exists")
check_content("np.save", "Flight recorder (NPY) exists")
check_content("pd.read_csv", "Real drive cycle loading exists")

with open("Pre_Run_Validation_Report.md", "w") as f:
    f.write("# Pre-Run Validation Report\n\n")
    all_pass = True
    for desc, res in validation_results:
        status = "✅ PASS" if res else "❌ FAIL"
        f.write(f"- {status}: {desc}\n")
        if not res: all_pass = False
    
    if all_pass:
        f.write("\n**VERDICT: READY FOR PRODUCTION CAMPAIGN**")
    else:
        f.write("\n**VERDICT: STOP. MISSING PRODUCTION SETTINGS**")

# Actual ODE Benchmark
import numpy as np
from scipy.integrate import solve_ivp
NUM_ZONES = 12
zone_resistance = np.ones(12) * 0.1
zone_heat_capacity = np.ones(12) * 150.0
lateral_conductance = np.ones(12) * 3.0
COOLANT_TEMP = 25.0

def real_ode(t, y, power, actions):
    temps = y[:NUM_ZONES]; flows = y[NUM_ZONES:]
    I_pack = power / 44.4
    Q_gen = (I_pack ** 2) * zone_resistance
    valves = actions[:NUM_ZONES]; pump = actions[12]; fan = actions[13]
    total_flow = pump * 15.0 
    UA = 0.5 + 15.0 * (total_flow * valves / 1.0)**0.8 + 5.0 * (fan * 100.0)**0.8
    Q_cool = UA * (temps - COOLANT_TEMP)
    Q_cond = np.zeros(NUM_ZONES)
    dTdt = (Q_gen - Q_cool + Q_cond) / zone_heat_capacity
    dflowdt = (target_flows - flows) / 2.0 if 'target_flows' in globals() else np.zeros(12)
    return np.concatenate([dTdt, dflowdt])

target_flows = np.ones(12)
start = time.time()
for _ in range(100):
    solve_ivp(lambda t, y: real_ode(t, y, 1000, np.ones(14)), [0, 1.0], np.ones(24))
end = time.time()
real_latency = (end - start) / 100
print(f"Real ODE Latency: {real_latency:.4f} s/step")
