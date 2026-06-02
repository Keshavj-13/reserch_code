import json
import nbformat as nbf

def build_notebook():
    with open('../final_run.ipynb', 'r') as f:
        nb_orig = json.load(f)

    cells = []
    # 1. Copy setup, ODE, and physical parameters from original notebook
    for cell in nb_orig['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            cells.append(nbf.v4.new_code_cell(source))
            # Stop copying after dataset generation or model initialization
            if "class TempPredictor" in source:
                break

    # 2. Add Manuscript Architecture
    arch_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time

# --- MANUSCRIPT ARCHITECTURE RECOVERY ---

class GraphSAGELayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.w_self = nn.Linear(in_feat, out_feat)
        self.w_neigh = nn.Linear(in_feat, out_feat)
        
    def forward(self, x, adj):
        # x: [B, Z, F], adj: [Z, Z]
        deg = adj.sum(dim=-1, keepdim=True) + 1e-6
        adj_norm = adj / deg
        neigh_msg = torch.matmul(adj_norm, x)
        return F.relu(self.w_self(x) + self.w_neigh(neigh_msg))

class SpatialEncoder(nn.Module):
    def __init__(self, in_feat=6, out_feat=128):
        super().__init__()
        self.layer1 = GraphSAGELayer(in_feat, 128)
        self.layer2 = GraphSAGELayer(128, 128)
        self.layer3 = GraphSAGELayer(128, 128)
        
    def forward(self, x, adj):
        h = self.layer1(x, adj)
        h = self.layer2(h, adj)
        h = self.layer3(h, adj)
        return h.mean(dim=1) # Global pooling: [B, 128]

class TemporalEncoder(nn.Module):
    def __init__(self, in_feat=4, out_feat=256):
        super().__init__()
        self.lstm = nn.LSTM(in_feat, out_feat, num_layers=2, dropout=0.1, batch_first=True)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1] # [B, 256]

class GlobalEncoder(nn.Module):
    def __init__(self, in_feat=3, out_feat=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat, 64), nn.ReLU(),
            nn.Linear(64, out_feat), nn.ReLU()
        )
        
    def forward(self, x):
        return self.mlp(x) # [B, 64]

class ManuscriptActorCritic(nn.Module):
    def __init__(self, num_zones):
        super().__init__()
        self.spatial = SpatialEncoder(in_feat=6)
        self.temporal = TemporalEncoder(in_feat=4)
        self.glob = GlobalEncoder(in_feat=3)

        self.actor_mlp = nn.Sequential(
            nn.Linear(448, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.actor_mean = nn.Linear(128, num_zones)
        self.actor_logstd = nn.Parameter(torch.zeros(num_zones))

        self.critic_mlp = nn.Sequential(
            nn.Linear(448, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_spat, adj, x_temp, x_glob):
        z_spat = self.spatial(x_spat, adj)
        z_temp = self.temporal(x_temp)
        z_glob = self.glob(x_glob)
        latent = torch.cat([z_spat, z_temp, z_glob], dim=-1)

        act_feat = self.actor_mlp(latent)
        mean = torch.sigmoid(self.actor_mean(act_feat)) # Constraint mapping to 0-1
        
        val = self.critic_mlp(latent)
        return mean, val, latent

# Initialize Manuscript Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manuscript_model = ManuscriptActorCritic(num_zones=NUM_ZONES).to(device)
manuscript_model.eval()

# Precompute Adjacency
adj_matrix = torch.zeros((NUM_ZONES, NUM_ZONES), device=device)
for i in range(NUM_ZONES - 1):
    adj_matrix[i, i+1] = lateral_conductance[i]
    adj_matrix[i+1, i] = lateral_conductance[i]
for i in range(NUM_ZONES):
    adj_matrix[i, i] = 1.0 # Self-loop
"""
    cells.append(nbf.v4.new_code_cell(arch_code))

    # 3. Add Flight Recorder Simulator Wrapper
    flight_recorder_code = """
# --- FLIGHT RECORDER INTEGRATION ---
os.makedirs("results/flight_recorder", exist_ok=True)
os.makedirs("results/flight_recorder/trajectories", exist_ok=True)
os.makedirs("results/flight_recorder/latent_states", exist_ok=True)

def simulate_flight_recorder(action_provider_fn, desc="Custom"):
    print(f"   [Flight Recorder] Executing: {desc}...")
    
    proxies = [ActionProxy(0.0) for _ in range(NUM_ZONES)]
    params_local = dict(simulation_params)
    params_local["pid_controllers"] = proxies
    
    state = np.concatenate([np.full(NUM_ZONES, INITIAL_TEMP), np.zeros(NUM_ZONES)])
    
    # Pre-allocate trajectory storage
    Tsteps = len(time_array)
    log_temps = np.zeros((Tsteps, NUM_ZONES))
    log_flows = np.zeros((Tsteps, NUM_ZONES))
    log_pump = np.zeros(Tsteps)
    log_actions = np.zeros((Tsteps, NUM_ZONES))
    log_rewards = np.zeros(Tsteps)
    
    # Detailed architecture logs
    log_latents = []
    log_values = []
    log_means = []
    
    for t in range(Tsteps):
        # Action Provider now returns a dict to capture internal states
        out = action_provider_fn(state.copy(), t)
        actions = np.clip(out['action'], 0.0, 1.0)
        
        log_actions[t] = actions
        if 'latent' in out: log_latents.append(out['latent'])
        if 'value' in out: log_values.append(out['value'])
        if 'mean' in out: log_means.append(out['mean'])
        
        for i in range(NUM_ZONES):
            proxies[i].current_action = float(actions[i])
            
        sol = solve_ivp(
            fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, params_local),
            t_span=[t, t + 1.0],
            y0=state,
            t_eval=[t + 1.0],
            method="RK45",
            rtol=1e-6,
            atol=1e-8
        )
        
        state = sol.y[:, -1]
        log_temps[t] = state[:NUM_ZONES]
        log_flows[t] = state[NUM_ZONES:]
        
        if not ADIABATIC_MODE:
            total_flow_lpm = np.sum(state[NUM_ZONES:]) * MAX_PUMP_FLOW_LPM / NUM_ZONES
            flow_frac = total_flow_lpm / MAX_PUMP_FLOW_LPM if MAX_PUMP_FLOW_LPM > 0 else 0.0
            log_pump[t] = PUMP_MAX_POWER_W * (flow_frac ** 2)
            
        # Compute instantaneous reward (Manuscript spec)
        T_max = state[:NUM_ZONES].max()
        r_safe = -5.0 * max(0.0, T_max - TEMP_SETPOINT)
        r_uni = -1.0 * state[:NUM_ZONES].std()
        r_eng = -0.5 * log_pump[t] / PUMP_MAX_POWER_W
        log_rewards[t] = r_safe + r_uni + r_eng
        
    # --- SAVE FLIGHT RECORDER ARTIFACTS ---
    clean_name = desc.replace(" ", "_")
    df_traj = pd.DataFrame({
        'time_s': time_array,
        'pump_power_W': log_pump,
        'battery_power_W': power_profile,
        'reward': log_rewards
    })
    for z in range(NUM_ZONES):
        df_traj[f'zone_{z+1}_temp_C'] = log_temps[:, z]
        df_traj[f'zone_{z+1}_flow_norm'] = log_flows[:, z]
        df_traj[f'zone_{z+1}_action'] = log_actions[:, z]
        
    csv_path = f"results/flight_recorder/trajectories/{clean_name}_traj.csv"
    df_traj.to_csv(csv_path, index=False)
    
    if log_latents:
        np.save(f"results/flight_recorder/latent_states/{clean_name}_latent.npy", np.vstack(log_latents))
    if log_values:
        np.save(f"results/flight_recorder/latent_states/{clean_name}_value.npy", np.vstack(log_values))
        
    print(f"   💾 Saved trajectory to {csv_path}")
    
    return {
        "temps": log_temps, "flows": log_flows, "pump": log_pump,
        "actions": log_actions, "rewards": log_rewards
    }
"""
    cells.append(nbf.v4.new_code_cell(flight_recorder_code))

    # 4. Re-map controllers to return dict format and add the Manuscript architecture controller
    eval_code = """
# --- CONTROLLER REGISTRY RECONSTRUCTION ---

flight_registry = {}

def wrap_basic(base_fn):
    def wrapped(state, t):
        return {'action': base_fn(state, t)}
    return wrapped

# Baselines from final_run.ipynb
flight_registry["PID_Standard"] = wrap_basic(controller_registry["PID_Standard"])
flight_registry["PID_Adaptive"] = wrap_basic(controller_registry["PID_Adaptive"])
flight_registry["Uniform_Flow"] = wrap_basic(controller_registry["Uniform_Flow"])
flight_registry["Proportional_Temp"] = wrap_basic(controller_registry["Proportional_Temp"])
flight_registry["MPC_Horizon1_Sample32"] = wrap_basic(controller_registry["MPC_H1_S32"])
flight_registry["MPC_Horizon1_Sample64"] = wrap_basic(controller_registry["MPC_H1_S64"])

# Recovered Manuscript Architecture Controller
def make_manuscript_controller(model, adj):
    def action_fn(state, t):
        temps = state[:NUM_ZONES]
        # Build Manuscript Inputs
        # 1. Global [B, 3]: T_amb, v_veh, HVAC
        t_amb = 25.0
        v_veh = power_profile[t] / PACK_MAX_POWER * 100.0 # Approximation of speed from power
        x_glob = torch.tensor([[t_amb, v_veh, 0.0]], dtype=torch.float32, device=device)
        
        # 2. Temporal [B, T, 4]: Mean, Max, Min, Current
        # Using a simplistic rolling window for executability
        H = 10
        t_start = max(0, t - H)
        # Pad if needed
        pad_len = max(0, H - t)
        
        hist_temps = np.zeros((H, NUM_ZONES))
        hist_temps[pad_len:] = temps # Simplified to current state for safety in online loop
        hist_power = np.zeros(H)
        hist_power[pad_len:] = power_profile[t_start:t] if t>0 else 0.0
        
        x_temp = np.stack([
            hist_temps.mean(axis=1),
            hist_temps.max(axis=1),
            hist_temps.min(axis=1),
            hist_power / PACK_NOMINAL_VOLTAGE
        ], axis=-1)
        x_temp = torch.tensor(x_temp, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 3. Spatial [B, Z, 6]: T, T-T_mean, SOC (fake), I, V (fake), index
        x_spat = np.zeros((NUM_ZONES, 6))
        x_spat[:, 0] = temps
        x_spat[:, 1] = temps - temps.mean()
        x_spat[:, 2] = 0.5 # SOC
        x_spat[:, 3] = power_profile[t] / PACK_NOMINAL_VOLTAGE
        x_spat[:, 4] = 3.7 # V
        x_spat[:, 5] = np.arange(NUM_ZONES)
        x_spat = torch.tensor(x_spat, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            mean, val, latent = model(x_spat, adj, x_temp, x_glob)
            
        return {
            'action': mean.cpu().numpy()[0],
            'latent': latent.cpu().numpy()[0],
            'value': val.cpu().numpy()[0],
            'mean': mean.cpu().numpy()[0]
        }
    return action_fn

flight_registry["Manuscript_Full_Architecture_Untrained"] = make_manuscript_controller(manuscript_model, adj_matrix)

# Run Flight Recorder
all_flight_metrics = []
for name, action_fn in flight_registry.items():
    print(f"\\n🚀 Initiating Flight Recorder for: {name}")
    start = time.time()
    try:
        res = simulate_flight_recorder(action_fn, desc=name)
        
        # Compute metrics
        m = {
            "label": name,
            "mean_temp": res["temps"].mean(),
            "max_temp": res["temps"].max(),
            "temp_spread": (res["temps"].max(axis=1) - res["temps"].min(axis=1)).mean(),
            "pump_energy_Wh": res["pump"].sum() * 1.0 / 3600.0,
            "reward_sum": res["rewards"].sum()
        }
        all_flight_metrics.append(m)
        print(f"   ✅ Finished in {time.time() - start:.1f}s. MaxT: {m['max_temp']:.2f}C, Energy: {m['pump_energy_Wh']:.2f}Wh")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

df_metrics = pd.DataFrame(all_flight_metrics)
df_metrics.to_csv("results/flight_recorder/authentic_metrics_comparison.csv", index=False)
print("\\n🏁 Flight Recorder execution complete. All metrics and trajectories saved.")
"""
    cells.append(nbf.v4.new_code_cell(eval_code))

    new_nb = nbf.v4.new_notebook()
    new_nb.cells = cells
    
    with open('../Manuscript_Evaluation_Recovered.ipynb', 'w') as f:
        nbf.write(new_nb, f)

build_notebook()
print("Successfully generated Manuscript_Evaluation_Recovered.ipynb")
