import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from collections import deque

from run_phase7 import (
    rl_agents, load_original_drive_cycles, extract_features, 
    battery_thermal_ode, get_reward_components,
    NUM_ZONES, INITIAL_TEMP, TEMP_MAX
)
from run_phase6 import baselines

# Canonical Manuscript Names Mapping
NAME_MAP = {
    'PID_Standard': 'PID',
    'PID_Adaptive': 'AdaptivePID',
    'MPC_H1_S32': 'MPC',
    'Uniform_Flow': 'UniformFlow',
    'Proportional_Temp': 'TempProp',
    'Proposed_Full': 'GS+LSTM+PPO',
    'Ablation_NoTemporal': 'GS+PPO',
    'Ablation_NoSpatial': 'LSTM+PPO',
    'Ablation_MLPOnly': 'MLP+PPO'
}

def run_replay_for_plotting():
    print("Starting Deterministic Replay for Plotting...")
    power_profile, speed_profile = load_original_drive_cycles()
    horizon = len(power_profile)
    
    adj = torch.eye(NUM_ZONES); adj += torch.diag(torch.ones(NUM_ZONES-1), 1); adj += torch.diag(torch.ones(NUM_ZONES-1), -1)
    
    os.makedirs("results/replots_from_csv", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    all_models = {**baselines, **rl_agents}
    comp_results = []
    
    for dev_name, controller in all_models.items():
        ms_name = NAME_MAP[dev_name]
        print(f"  Replaying {dev_name} as {ms_name}...")
        
        if dev_name in rl_agents:
            ckpt_path = f"results/checkpoints/{dev_name}_best.pt"
            if os.path.exists(ckpt_path):
                controller.policy.load_state_dict(torch.load(ckpt_path))
                controller.policy_old.load_state_dict(torch.load(ckpt_path))
            else:
                print(f"WARNING: Checkpoint missing for {dev_name}.")
                
        state = np.full(NUM_ZONES, INITIAL_TEMP)
        prev_a = np.zeros(NUM_ZONES + 2)
        history_q = deque([np.zeros(4) for _ in range(10)], maxlen=10)
        
        traj = []
        ep_rs, ep_rt, ep_re, ep_rsm = 0, 0, 0, 0
        history_tmax, history_tmean, spreads = [], [], []
        energy_sum = 0
        total_drive_energy_Wh = np.sum(power_profile) / 3600.0  
        
        for t in range(horizon):
            if dev_name in rl_agents:
                xs, xt, xg = extract_features(state, power_profile[t], speed_profile[t], history_q)
                a_t, _, _, _, _, _, _ = controller.select_action(xs, adj, xt, xg)
                a = np.clip(a_t.squeeze(0).numpy(), 0.0, 1.0)
            else:
                a = controller.get_action(state, t)

            dTdt = battery_thermal_ode(t, state, power_profile, a)
            state = state + dTdt * 1.0

            pump_power = 200.0 * (a[NUM_ZONES]**2)
            rs, rt, re, rsm = get_reward_components(state, pump_power, prev_a, a)
            
            ep_rs+=rs; ep_rt+=rt; ep_re+=re; ep_rsm+=rsm
            history_tmax.append(state.max())
            history_tmean.append(state.mean())
            energy_sum += pump_power
            spreads.append(state.max() - state.min())
            
            row = {
                'time_s': t,
                'pump_power_W': pump_power
            }
            for i in range(NUM_ZONES):
                row[f'zone_{i}_temp_C'] = state[i]
                row[f'zone_{i}_flow_norm'] = a[i]
                
            traj.append(row)
            prev_a = a
            
        pd.DataFrame(traj).to_csv(f"{ms_name}_run.csv", index=False)
        
        reward = ep_rs + ep_rt + ep_re + ep_rsm
        cooling_energy_Wh = energy_sum / 3600.0
        
        t_means = np.array(history_tmean)
        dt = 1.0
        stress = np.mean(np.abs(np.diff(t_means)) / dt) if len(t_means) > 1 else 0.0
        
        overhead = (cooling_energy_Wh / (total_drive_energy_Wh + 1e-6)) * 100.0 if total_drive_energy_Wh > 0 else 0
        
        comp_results.append({
            'label': ms_name,
            'reward': reward,
            'max_temp': float(max(history_tmax)),
            'mean_temp': float(np.mean(history_tmean)),
            'pump_energy_Wh': cooling_energy_Wh,
            'safety_events': int(sum(1 for t in history_tmax if t > 40.0)),
            'temp_spread_mean': float(np.mean(spreads)),
            'thermal_stress': stress,
            'thermal_stress_mean_absdT': stress,
            'cooling_overhead_pct': overhead
        })
        
    df_comp = pd.DataFrame(comp_results)
    df_comp.to_csv("controller_comparison.csv", index=False)
    df_comp.to_csv("comparison_summary.csv", index=False)
    
    print("Replay Complete. Output artifacts generated in root directory.")

if __name__ == "__main__":
    run_replay_for_plotting()
