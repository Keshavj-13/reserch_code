import os
import numpy as np
import pandas as pd
import torch
from collections import deque
from scipy.integrate import solve_ivp
from run_phase7 import (
    rl_agents, load_original_drive_cycles, extract_features, 
    battery_thermal_ode, get_reward_components,
    NUM_ZONES, INITIAL_TEMP, TEMP_MAX
)

def run_phase8_ablations():
    print("Starting Phase 8: Ablation Study & Statistical Validation...")
    power_profile, speed_profile = load_original_drive_cycles()
    horizon = len(power_profile)
    adj = torch.eye(NUM_ZONES); adj += torch.diag(torch.ones(NUM_ZONES-1), 1); adj += torch.diag(torch.ones(NUM_ZONES-1), -1)
    
    results = []
    
    # We would run multiple seeds for statistical validation
    num_seeds = 3
    
    # 1. Evaluate all models
    model_stats = {name: {'rewards': [], 'temps': [], 'energies': []} for name in rl_agents.keys()}
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for name, agent in rl_agents.items():
            # Load best checkpoint
            ckpt_path = f"results/checkpoints/{name}_best.pt"
            if os.path.exists(ckpt_path):
                agent.policy.load_state_dict(torch.load(ckpt_path))
                agent.policy_old.load_state_dict(torch.load(ckpt_path))
            else:
                print(f"Missing checkpoint for {name}, skipping...")
                continue
                
            state = np.full(NUM_ZONES, INITIAL_TEMP)
            prev_a = np.zeros(NUM_ZONES + 2)
            history_q = deque([np.zeros(4) for _ in range(10)], maxlen=10)
            
            ep_r = 0
            ep_energy = 0
            max_temp = -np.inf
            
            for t in range(horizon):
                xs, xt, xg = extract_features(state, power_profile[t], speed_profile[t], history_q)
                a_t, _, _, _, _, _, _ = agent.select_action(xs, adj, xt, xg)
                a = np.clip(a_t.squeeze(0).numpy(), 0.0, 1.0)
                
                dTdt = battery_thermal_ode(t, state, power_profile, a)
                state = state + dTdt * 1.0
                
                pump_power = 200.0 * (a[NUM_ZONES]**2)
                rs, rt, re, rsm = get_reward_components(state, pump_power, prev_a, a)
                
                ep_r += (rs + rt + re + rsm)
                ep_energy += pump_power
                max_temp = max(max_temp, state.max())
                prev_a = a
                
            model_stats[name]['rewards'].append(ep_r)
            model_stats[name]['temps'].append(max_temp)
            model_stats[name]['energies'].append(ep_energy / 3600.0)
            
    # 2. Calculate Statistics
    stat_rows = []
    for name, stats in model_stats.items():
        if len(stats['rewards']) == 0: continue
        stat_rows.append({
            'model': name,
            'mean_reward': np.mean(stats['rewards']),
            'std_reward': np.std(stats['rewards']),
            'mean_temp': np.mean(stats['temps']),
            'std_temp': np.std(stats['temps']),
            'mean_energy': np.mean(stats['energies']),
            'std_energy': np.std(stats['energies'])
        })
        
    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv("metrics/statistical_validation.csv", index=False)
    
    # 3. Calculate Ablation Deltas
    ablation_rows = []
    if "Proposed_Full" in df_stats['model'].values:
        full_row = df_stats[df_stats['model'] == "Proposed_Full"].iloc[0]
        full_r = full_row['mean_reward']
        full_e = full_row['mean_energy']
        full_t = full_row['mean_temp']
        
        for _, r in df_stats.iterrows():
            ablation_rows.append({
                'model': r['model'],
                'reward': r['mean_reward'],
                'reward_delta_vs_full': r['mean_reward'] - full_r,
                'energy_delta_vs_full': r['mean_energy'] - full_e,
                'temp_delta_vs_full': r['mean_temp'] - full_t
            })
            
        df_abl = pd.DataFrame(ablation_rows)
        df_abl.to_csv("metrics/ablation_results.csv", index=False)
        
    print("Phase 8 Complete.")

if __name__ == "__main__":
    run_phase8_ablations()
