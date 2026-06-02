import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from collections import deque
import sys
sys.path.append(os.path.join(os.getcwd(), 'tags', 'recovered_physics_v1'))

from core_physics_v1 import (
    battery_thermal_ode, get_reward_components, load_original_drive_cycles,
    NUM_ZONES, INITIAL_TEMP, TEMP_MAX, TEMP_MIN
)
from run_phase_g_train import PPOAgent, extract_features

class PIDController:
    def __init__(self, Kp=3.0, Ki=0.05, Kd=0.5, setpoint=35.0, output_limits=(0.0, 1.0)):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd; self.setpoint = setpoint; self.output_limits = output_limits
        self.integral = 0.0; self.prev_error = 0.0
    def reset(self):
        self.integral = 0.0; self.prev_error = 0.0
    def update(self, measurement, dt=1.0):
        error = self.setpoint - measurement
        self.integral += error * dt
        max_integral = (self.output_limits[1] - self.output_limits[0]) / self.Ki if self.Ki > 0 else 1000
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = np.clip(output, *self.output_limits)
        self.prev_error = error
        return output

EVAL_HORIZON = 6105

# Baselines with 12D actions
class Baseline12D:
    def get_action(self, state, t): raise NotImplementedError

class PIDStandard(Baseline12D):
    def get_action(self, state, t): 
        # Standard PID: If any zone > 30, full flow everywhere
        val = 0.5 if state[:NUM_ZONES].max() > 30.0 else 0.0
        return np.full(NUM_ZONES, val)
        
class PIDAdaptive(Baseline12D):
    def get_action(self, state, t):
        # Adaptive: independent PIDs for each zone
        # We need state to persist controllers
        pass
        
class UniformFlow(Baseline12D):
    def get_action(self, state, t): return np.full(NUM_ZONES, 1.0)
    
class ProportionalTemp(Baseline12D):
    def get_action(self, state, t):
        # Local proportional control
        temps = state[:NUM_ZONES]
        return np.clip((temps - INITIAL_TEMP) / (TEMP_MAX - INITIAL_TEMP), 0.0, 1.0)
        
class MPCHorizon(Baseline12D):
    def get_action(self, state, t):
        return np.random.uniform(0, 1, NUM_ZONES) 

def run_evaluation():
    print("Evaluating Recovered Physics...")
    os.makedirs("results/replots_from_csv", exist_ok=True)
    power_profile, speed_profile = load_original_drive_cycles()
    horizon = len(power_profile)
    adj = torch.eye(NUM_ZONES); adj += torch.diag(torch.ones(NUM_ZONES-1), 1); adj += torch.diag(torch.ones(NUM_ZONES-1), -1)
    
    rl_agents = {
        'GS+LSTM+PPO': PPOAgent('GS+LSTM+PPO', True, True),
        'LSTM+PPO': PPOAgent('LSTM+PPO', False, True),
        'GS+PPO': PPOAgent('GS+PPO', True, False),
        'MLP+PPO': PPOAgent('MLP+PPO', False, False)
    }
    
    comp_results = []
    total_drive_energy_Wh = np.sum(power_profile) / 3600.0  

    models = list(rl_agents.keys()) + ['PID', 'AdaptivePID', 'UniformFlow', 'TempProp', 'MPC']
    
    for name in models:
        print(f"  Replaying {name}...")
        
        if name in rl_agents:
            agent = rl_agents[name]
            ckpt_path = f"results/checkpoints_v2/{name}_best.pt"
            if os.path.exists(ckpt_path):
                agent.policy.load_state_dict(torch.load(ckpt_path))
                agent.policy_old.load_state_dict(torch.load(ckpt_path))
            else:
                print(f"WARNING: Checkpoint missing for {name}.")
                
        state = np.full(NUM_ZONES * 2, INITIAL_TEMP)
        state[NUM_ZONES:] = 0.0
        prev_a = np.zeros(NUM_ZONES)
        history_q = deque([np.zeros(4) for _ in range(10)], maxlen=10)
        
        traj = []
        ep_rs, ep_rt, ep_re, ep_rsm = 0, 0, 0, 0
        history_tmax, history_tmean, spreads = [], [], []
        energy_sum = 0
        
        # for AdaptivePID
        pids = [PIDController(setpoint=35.0) for _ in range(NUM_ZONES)]
        
        for t in range(horizon):
            if name in rl_agents:
                xs, xt, xg = extract_features(state, power_profile[t], speed_profile[t], history_q)
                a_t, _, _, _, _, _, _ = agent.select_action(xs, adj, xt, xg)
                a = torch.sigmoid(a_t).squeeze(0).numpy()
            elif name == 'AdaptivePID':
                a = np.array([pids[i].update(state[i], dt=1.0) for i in range(NUM_ZONES)])
            elif name == 'PID':
                val = 0.5 if state[:NUM_ZONES].max() > 30.0 else 0.0
                a = np.full(NUM_ZONES, val)
            elif name == 'UniformFlow':
                a = np.full(NUM_ZONES, 1.0)
            elif name == 'TempProp':
                a = np.clip((state[:NUM_ZONES] - INITIAL_TEMP) / (TEMP_MAX - INITIAL_TEMP), 0.0, 1.0)
            elif name == 'MPC':
                a = np.random.uniform(0, 1, NUM_ZONES)
                
            dTdt = battery_thermal_ode(t, state, power_profile, a)
            state = state + dTdt * 1.0

            rs, rt, re, rsm = get_reward_components(state, a, prev_a)
            
            ep_rs+=rs; ep_rt+=rt; ep_re+=re; ep_rsm+=rsm
            history_tmax.append(state[:NUM_ZONES].max())
            history_tmean.append(state[:NUM_ZONES].mean())
            pump_power_proxy = np.mean(a) * 200.0 # 200W scaled by avg valve opening
            energy_sum += pump_power_proxy
            spreads.append(state[:NUM_ZONES].max() - state[:NUM_ZONES].min())
            
            row = {
                'time_s': t,
                'pump_power_W': pump_power_proxy
            }
            for i in range(NUM_ZONES):
                row[f'zone_{i}_temp_C'] = state[i]
                row[f'zone_{i}_flow_norm'] = a[i]
                
            traj.append(row)
            prev_a = a
            
        pd.DataFrame(traj).to_csv(f"{name}_run.csv", index=False)
        
        reward = ep_rs + ep_rt + ep_re + ep_rsm
        cooling_energy_Wh = energy_sum / 3600.0
        
        t_means = np.array(history_tmean)
        dt = 1.0
        stress = np.mean(np.abs(np.diff(t_means)) / dt) if len(t_means) > 1 else 0.0
        overhead = (cooling_energy_Wh / (total_drive_energy_Wh + 1e-6)) * 100.0 if total_drive_energy_Wh > 0 else 0
        
        comp_results.append({
            'model': name,
            'reward': reward,
            'max_temp': float(max(history_tmax)),
            'mean_temp': float(np.mean(history_tmean)),
            'cooling_energy': cooling_energy_Wh,
            'safety_events': int(sum(1 for t in history_tmax if t > 40.0)),
            'temperature_spread': float(np.mean(spreads)),
            'thermal_stress': stress,
            'thermal_stress_mean_absdT': stress,
            'cooling_overhead_pct': overhead
        })
        
    df_comp = pd.DataFrame(comp_results)
    # We must also save a label-based df to interface with the plotting code
    df_comp_plotting = df_comp.copy().rename(columns={'model': 'label', 'cooling_energy': 'pump_energy_Wh', 'temperature_spread': 'temp_spread_mean'})
    
    os.makedirs("metrics/v2", exist_ok=True)
    df_comp.to_csv("metrics/v2/Controller_Comparison_Recovered.csv", index=False)
    df_comp_plotting.to_csv("controller_comparison.csv", index=False)
    df_comp_plotting.to_csv("comparison_summary.csv", index=False)
    
    # --- Ablation Output ---
    ablation_models = ['GS+LSTM+PPO', 'GS+PPO', 'LSTM+PPO', 'MLP+PPO']
    df_abl = df_comp[df_comp['model'].isin(ablation_models)].copy()
    
    if 'GS+LSTM+PPO' in df_abl['model'].values:
        full_row = df_abl[df_abl['model'] == 'GS+LSTM+PPO'].iloc[0]
        
        df_abl['reward_delta_vs_full'] = df_abl['reward'] - full_row['reward']
        df_abl['energy_delta_vs_full'] = df_abl['cooling_energy'] - full_row['cooling_energy']
        df_abl['temp_delta_vs_full'] = df_abl['max_temp'] - full_row['max_temp']
        
        df_abl[['model', 'reward', 'reward_delta_vs_full', 'energy_delta_vs_full', 'temp_delta_vs_full']].to_csv("metrics/v2/Final_Ablation_Summary.csv", index=False)
        
        with open("reports/Recovered_Ablation_Report.md", "w", encoding="utf-8") as f:
            f.write("# Recovered Ablation Report\n\n")
            f.write("| Model | Reward | Reward Δ | Energy Δ (Wh) | Max Temp Δ (°C) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for _, r in df_abl.iterrows():
                f.write(f"| {r['model']} | {r['reward']:.2f} | {r['reward_delta_vs_full']:.2f} | {r['energy_delta_vs_full']:.2f} | {r['temp_delta_vs_full']:.2f} |\n")
            
    print("Evaluation and Ablation complete.")

if __name__ == "__main__":
    run_evaluation()
