import os
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.integrate import solve_ivp
from collections import deque

# --- CONSTANTS ---
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

EVAL_HORIZON = 6105
TRAIN_EPOCHS = 50
DECIMATION_FACTOR = 20

# --- UTILS ---
def log_anomaly(model, event_type, details):
    row = pd.DataFrame([{"timestamp": time.time(), "model": model, "event_type": event_type, "details": details}])
    hdr = not os.path.exists("metrics/anomaly_log.csv")
    row.to_csv("metrics/anomaly_log.csv", mode='a', header=hdr, index=False)

def fail_fast(reason, model, ep):
    log_anomaly(model, "FAIL_FAST_TRIGGERED", reason)
    with open("reports/FailFast_Report.md", "w") as f:
        f.write("# Fail Fast Triggered\n\n")
        f.write(f"- **Model**: {model}\n")
        f.write(f"- **Epoch**: {ep}\n")
        f.write(f"- **Reason**: {reason}\n")
    print(f"\n[FAIL FAST] {reason}")
    os._exit(1)

def get_dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

# ================= DRIVE CYCLE LOADER =================
def load_original_drive_cycles():
    target_files = ["ftpcol.txt", "hwycol.txt", "j1015col.txt", "sc03col.txt", "uddscol.txt", "us06col.txt"]
    dfs = []
    cumulative_time = 0
    for f in target_files:
        path = os.path.join("drive_cycles", f)
        if not os.path.exists(path): continue
        df = None
        for enc in ['utf-8', 'ISO-8859-1', 'utf-16']:
            try:
                df_try = pd.read_csv(path, sep=None, engine='python', header=None, comment='#', encoding=enc)
                if df_try.shape[1] >= 2:
                    df = df_try.iloc[:, :2].copy()
                    df.columns = ['Time', 'Speed']
                    break
            except: pass
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
    return power_profile, speed_s

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

def get_reward_components(state, pump_power, prev_action, action):
    temps = state[:NUM_ZONES]
    r_safe = -5.0 * (np.maximum(0, temps.max() - TEMP_MAX) + np.maximum(0, TEMP_MIN - temps.min()))
    r_temp = -1.0 * np.std(temps)
    r_energy = -0.5 * pump_power / 200.0
    r_smooth = -0.1 * np.sum((action - prev_action)**2)
    return r_safe, r_temp, r_energy, r_smooth

# ================= ARCHITECTURE =================
class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels * 2, out_channels)
    def forward(self, x, adj): return F.relu(self.lin(torch.cat([x, torch.matmul(adj, x)], dim=-1)))

class GraphSAGEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(6, 128); self.conv2 = SAGEConv(128, 128); self.conv3 = SAGEConv(128, 128)
    def forward(self, x, adj): return self.conv3(self.conv2(self.conv1(x, adj), adj), adj).mean(dim=1)

class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 256, num_layers=2, batch_first=True, dropout=0.1)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]

class GlobalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
    def forward(self, x): return self.net(x)

class ManuscriptActorCritic(nn.Module):
    def __init__(self, use_spat=True, use_temp=True):
        super().__init__()
        self.use_spat = use_spat; self.use_temp = use_temp
        if use_spat: self.enc_s = GraphSAGEEncoder()
        if use_temp: self.enc_t = TemporalEncoder()
        self.enc_g = GlobalEncoder()
        self.fused_dim = (128 if use_spat else 0) + (256 if use_temp else 0) + 64
        self.actor = nn.Sequential(nn.Linear(self.fused_dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())    
        self.actor_mu = nn.Linear(64, NUM_ZONES+2)
        nn.init.constant_(self.actor_mu.bias, 0.0)
        self.actor_logstd = nn.Parameter(torch.zeros(NUM_ZONES+2))
        self.critic = nn.Sequential(nn.Linear(self.fused_dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_s, adj, x_t, x_g):
        emb_s = self.enc_s(x_s, adj) if self.use_spat else torch.zeros(x_s.shape[0], 0)
        emb_t = self.enc_t(x_t) if self.use_temp else torch.zeros(x_t.shape[0], 0)
        emb_g = self.enc_g(x_g)
        embs = []
        if self.use_spat: embs.append(emb_s)
        if self.use_temp: embs.append(emb_t)
        embs.append(emb_g)
        z = torch.cat(embs, dim=-1)
        val = self.critic(z)
        act_out = self.actor(z)
        mu = torch.sigmoid(self.actor_mu(act_out))
        std = torch.exp(self.actor_logstd)
        return mu, std, val, z, emb_s, emb_t, act_out

class PPOBuffer:
    def __init__(self): self.reset()
    def reset(self): self.x_s, self.x_t, self.x_g, self.a, self.lp, self.r, self.v, self.term = [], [], [], [], [], [], [], []
    def store(self, xs, xt, xg, a, lp, r, v, term):
        self.x_s.append(xs); self.x_t.append(xt); self.x_g.append(xg)
        self.a.append(a); self.lp.append(lp); self.r.append(r); self.v.append(v); self.term.append(term)        

class PPOAgent:
    def __init__(self, name, use_spat=True, use_temp=True):
        self.name = name
        self.policy = ManuscriptActorCritic(use_spat, use_temp)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.policy_old = ManuscriptActorCritic(use_spat, use_temp)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = 0.99; self.lam = 0.95; self.eps_clip = 0.2; self.K_epochs = 4

    def select_action(self, xs, adj, xt, xg):
        with torch.no_grad():
            mu, std, val, z, es, et, ao = self.policy_old(xs, adj, xt, xg)
        dist = Normal(mu, std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(dim=-1), val, z, es, et, ao

    def update(self, buffer, adj):
        xs = torch.cat(buffer.x_s); xt = torch.cat(buffer.x_t); xg = torch.cat(buffer.x_g)
        a = torch.cat(buffer.a); old_lp = torch.cat(buffer.lp)
        v = torch.cat(buffer.v).squeeze(-1); r = buffer.r; term = buffer.term
        advantages = np.zeros(len(r), dtype=np.float32); gae = 0; v_np = v.numpy(); v_np = np.append(v_np, 0)
        for i in reversed(range(len(r))):
            delta = r[i] + self.gamma * v_np[i+1] * (1 - term[i]) - v_np[i]
            gae = delta + self.gamma * self.lam * (1 - term[i]) * gae
            advantages[i] = gae
        returns = torch.tensor(advantages + v_np[:-1], dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        total_a_loss, total_c_loss, total_ent = 0, 0, 0
        for _ in range(self.K_epochs):
            mu, std, val, _, _, _, _ = self.policy(xs, adj, xt, xg)
            dist = Normal(mu, std)
            lp = dist.log_prob(a).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            ratios = torch.exp(lp - old_lp.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            a_loss = -torch.min(surr1, surr2).mean()
            c_loss = 0.5 * F.mse_loss(val.squeeze(-1), returns)
            loss = a_loss + c_loss - 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            total_a_loss += a_loss.item(); total_c_loss += c_loss.item(); total_ent += entropy.mean().item()
        self.policy_old.load_state_dict(self.policy.state_dict())
        buffer.reset()
        return total_a_loss/self.K_epochs, total_c_loss/self.K_epochs, total_ent/self.K_epochs

def extract_features(state, power, speed, history_q):
    temps = state[:NUM_ZONES] / 100.0
    p_norm = power / 26600.0
    s_norm = speed / 50.0
    xs = np.zeros((NUM_ZONES, 6))
    xs[:,0] = temps; xs[:,1] = temps - temps.mean(); xs[:,2] = 0.5; xs[:,3] = p_norm
    xs[:,4] = 3.7 / 4.2; xs[:,5] = np.arange(NUM_ZONES) / 12.0
    history_q.append([temps.mean(), temps.max(), temps.min(), p_norm])
    return torch.tensor(xs, dtype=torch.float32).unsqueeze(0), torch.tensor(np.array(history_q), dtype=torch.float32).unsqueeze(0), torch.tensor(np.array([25.0/50.0, s_norm, 0.0]), dtype=torch.float32).unsqueeze(0)

# ================= RUNNER =================
rl_agents = {
    'Proposed_Full': PPOAgent('Proposed_Full', True, True),
    'Ablation_NoSpatial': PPOAgent('Ablation_NoSpatial', False, True),
    'Ablation_NoTemporal': PPOAgent('Ablation_NoTemporal', True, False),
    'Ablation_MLPOnly': PPOAgent('Ablation_MLPOnly', False, False)
}

def run_phase7():
    print("Initializing Phase 7: Long Training Campaign")
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/flight_recorder", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    power_profile, speed_profile = load_original_drive_cycles()
    adj = torch.eye(NUM_ZONES); adj += torch.diag(torch.ones(NUM_ZONES-1), 1); adj += torch.diag(torch.ones(NUM_ZONES-1), -1)
    
    for name, agent in rl_agents.items():
        print(f"=== Training {name} ===")
        os.makedirs(f"results/flight_recorder/{name}", exist_ok=True)
        best_reward = -np.inf
        
        for ep in range(TRAIN_EPOCHS):
            state = np.full(NUM_ZONES, INITIAL_TEMP)
            prev_a = np.zeros(NUM_ZONES + 2)
            history_q = deque([np.zeros(4) for _ in range(10)], maxlen=10)
            buffer = PPOBuffer()
            ep_rs, ep_rt, ep_re, ep_rsm = 0, 0, 0, 0
            
            traj, lat, em_s, em_t, act_o, vals = [], [], [], [], [], []
            
            for t in range(EVAL_HORIZON):
                # FAIL FAST: NaN Check
                if np.isnan(state).any(): fail_fast("NaN detected in state", name, ep)
                if state.max() > 60.0: fail_fast("Temperature exceeded 60C absolute limit", name, ep)
                
                xs, xt, xg = extract_features(state, power_profile[t], speed_profile[t], history_q)
                a_t, lp, val, z, es, et, ao = agent.select_action(xs, adj, xt, xg)
                a = np.clip(a_t.squeeze(0).numpy(), 0.0, 1.0)
                
                dTdt = battery_thermal_ode(t, state, power_profile, a)
                state = state + dTdt * 1.0
                pump_power = 200.0 * (a[NUM_ZONES]**2)
                rs, rt, re, rsm = get_reward_components(state, pump_power, prev_a, a)
                r = rs + rt + re + rsm
                buffer.store(xs, xt, xg, a_t, lp, r, val, t==EVAL_HORIZON-1)
                
                traj.append({'time': t, 'max_temp': state.max(), 'mean_temp': state.mean(), 'spread': state.max() - state.min(), 'energy': pump_power, 'pump_cmd': a[NUM_ZONES], 'fan_cmd': a[NUM_ZONES+1], 'reward': r, 'safety_violations': (state > TEMP_MAX).sum()})
                if t % DECIMATION_FACTOR == 0:
                    lat.append(z.squeeze(0).numpy()); em_s.append(es.squeeze(0).numpy()); em_t.append(et.squeeze(0).numpy()); act_o.append(ao.squeeze(0).numpy()); vals.append(val.item())
                
                prev_a = a
                ep_rs+=rs; ep_rt+=rt; ep_re+=re; ep_rsm+=rsm
                
            total_r = ep_rs + ep_rt + ep_re + ep_rsm
            a_loss, c_loss, ent = agent.update(buffer, adj)
            
            # FAIL FAST: Critic Explosion / Reward Explosion
            if c_loss > 1e6: fail_fast("Critic loss exceeded 1e6", name, ep)
            if math.isinf(c_loss) or math.isnan(c_loss): fail_fast("Inf/NaN Critic Loss", name, ep)
            
            # Save Flight Recorder
            pd.DataFrame(traj).to_csv(f"results/flight_recorder/{name}/trajectory.csv", index=False)
            np.save(f"results/flight_recorder/{name}/latents.npy", np.array(lat).astype(np.float32))
            np.save(f"results/flight_recorder/{name}/embeddings.npy", np.array(em_s).astype(np.float32))
            np.save(f"results/flight_recorder/{name}/hidden_states.npy", np.array(em_t).astype(np.float32))
            np.save(f"results/flight_recorder/{name}/actor_outputs.npy", np.array(act_o).astype(np.float32))
            np.save(f"results/flight_recorder/{name}/values.npy", np.array(vals).astype(np.float32))
            
            # Summary Tracking
            ep_row = pd.DataFrame([{"model": name, "epoch": ep, "reward": total_r, "max_temp": state.max(), "mean_temp": state.mean(), "cooling_energy": ep_re*-200.0/0.5, "safety_events": traj[-1]['safety_violations']}])
            ep_row.to_csv("metrics/epoch_summary.csv", mode='a', header=not os.path.exists("metrics/epoch_summary.csv"), index=False)
            
            # Checkpoints
            try:
                torch.save(agent.policy.state_dict(), f"results/checkpoints/{name}_latest.pt")
                if total_r > best_reward:
                    best_reward = total_r
                    torch.save(agent.policy.state_dict(), f"results/checkpoints/{name}_best.pt")
            except Exception as e:
                fail_fast(f"Checkpoint save failed: {str(e)}", name, ep)
                
            # Storage Budget
            disk_mb = get_dir_size_mb("results")
            pd.DataFrame([{"epoch": ep, "model": name, "results_mb": disk_mb}]).to_csv("metrics/storage_usage.csv", mode='a', header=not os.path.exists("metrics/storage_usage.csv"), index=False)
            if disk_mb > 2048: fail_fast("Disk usage exceeded 2GB hard limit", name, ep)
            
            print(f"[{name} Ep {ep}] R: {total_r:.2f} | Tmax: {state.max():.2f}C | Critic: {c_loss:.2f} | Disk: {disk_mb:.1f}MB")

if __name__ == "__main__":
    run_phase7()
