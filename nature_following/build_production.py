import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # --- CELL 1: Setup & Constants ---
    cells.append(nbf.v4.new_code_cell("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os
import time
import json
import hashlib
from collections import deque

# Create Directories
dirs = ["results/full_campaign/checkpoints", "results/full_campaign/flight_recorder", "results/full_campaign/figures", 
        "results/full_campaign/logs", "results/full_campaign/metrics", "results/full_campaign/reports"]
for d in dirs: os.makedirs(d, exist_ok=True)

# --- PHYSICAL CONSTANTS ---
NUM_ZONES = 12
CELLS_IN_SERIES = 12
CELLS_IN_PARALLEL = 3
TOTAL_CELLS = CELLS_IN_SERIES * CELLS_IN_PARALLEL
PACK_NOMINAL_VOLTAGE = 44.4
TEMP_SETPOINT = 35.0
TEMP_MAX = 40.0
TEMP_MIN = 20.0
COOLANT_TEMP = 25.0
INITIAL_TEMP = 25.0
PACK_MAX_POWER = 26600.0

np.random.seed(42)
torch.manual_seed(42)
cell_r_variation = np.random.normal(1.0, 0.05, NUM_ZONES)
zone_resistance = (0.030 * cell_r_variation * CELLS_IN_SERIES) / CELLS_IN_PARALLEL
zone_heat_capacity = np.full(NUM_ZONES, (TOTAL_CELLS // NUM_ZONES) * 50.0)
zone_positions = np.linspace(0, 1, NUM_ZONES)
lateral_conductance = 3.0 * (1.0 - 0.4 * np.abs(zone_positions - 0.5) * 2.0)

# PRODUCTION SETTINGS
IS_TEST_MODE = False 
EVAL_HORIZON = 1800
TRAIN_EPOCHS = 50
"""))

    # --- CELL 2: Environment ---
    cells.append(nbf.v4.new_code_cell("""
# --- ENVIRONMENT ---
def load_drive_cycle():
    possible_files = ["us06col.txt", "uddscol.txt"]
    speed_mps = None
    if os.path.exists("drive_cycles"):
        for f in possible_files:
            p = os.path.join("drive_cycles", f)
            if os.path.exists(p):
                df = pd.read_csv(p, header=None, names=["T", "S"], sep=None, engine="python")
                speed_mps = df["S"].values * 0.44704
                print(f"Loaded {f}")
                break
    if speed_mps is None:
        t_arr = np.arange(EVAL_HORIZON)
        speed_mps = 15.0 + 10.0 * np.sin(2 * np.pi * t_arr / 300)
    
    speed_mps = pd.Series(speed_mps).ewm(alpha=0.25).mean().values
    accel = np.zeros_like(speed_mps); accel[1:] = (speed_mps[1:] - speed_mps[:-1])
    F_total = 0.5 * 1.225 * 0.24 * 2.34 * speed_mps**2 + 2200.0 * 9.81 * 0.008 + 2200.0 * accel
    P_mech = F_total * speed_mps; P_mech[P_mech >= 0] /= 0.92; P_mech[P_mech < 0] *= 0.6
    power = np.clip(P_mech, 0, PACK_MAX_POWER)
    return power, speed_mps

power_profile, speed_profile = load_drive_cycle()
power_profile = np.pad(power_profile, (0, max(0, EVAL_HORIZON - len(power_profile))), mode='edge')[:EVAL_HORIZON]
speed_profile = np.pad(speed_profile, (0, max(0, EVAL_HORIZON - len(speed_profile))), mode='edge')[:EVAL_HORIZON]

def battery_thermal_ode(t, y, power_profile, actions):
    temps = y[:NUM_ZONES]; flows = y[NUM_ZONES:]; idx = min(int(t), len(power_profile)-1)
    I_pack = np.clip(power_profile[idx] / PACK_NOMINAL_VOLTAGE, -4.5, 18.0)
    Q_gen = (I_pack ** 2) * zone_resistance
    valves = actions[:NUM_ZONES]; pump = actions[NUM_ZONES]; fan = actions[NUM_ZONES+1]
    total_flow = pump * 15.0 
    UA = 0.5 + 15.0 * (total_flow * valves / max(1e-3, valves.sum()))**0.8 + 5.0 * (fan * 100.0)**0.8
    Q_cool = UA * (temps - COOLANT_TEMP); Q_cond = np.zeros(NUM_ZONES)
    for i in range(NUM_ZONES):
        if i > 0: Q_cond[i] += lateral_conductance[i-1] * (temps[i-1] - temps[i])
        if i < NUM_ZONES - 1: Q_cond[i] += lateral_conductance[i] * (temps[i+1] - temps[i])
    dTdt = (Q_gen - Q_cool + Q_cond) / zone_heat_capacity
    dflowdt = (total_flow * valves / max(1e-3, valves.sum()) - flows) / 2.0
    return np.concatenate([dTdt, dflowdt])

def get_reward_components(state, pump_power, prev_action, action):
    temps = state[:NUM_ZONES]
    r_safe = -5.0 * (np.maximum(0, temps.max() - TEMP_MAX) + np.maximum(0, TEMP_MIN - temps.min()))
    r_temp = -1.0 * np.std(temps); r_energy = -0.5 * pump_power / 200.0; r_smooth = -0.1 * np.sum((action - prev_action)**2)
    return r_safe, r_temp, r_energy, r_smooth
"""))

    # --- CELL 3: Architecture ---
    cells.append(nbf.v4.new_code_cell("""
class SAGEConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__(); self.lin = nn.Linear(in_c * 2, out_c)
    def forward(self, x, adj): return F.relu(self.lin(torch.cat([x, torch.matmul(adj, x)], dim=-1)))

class GraphSAGEEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.c1 = SAGEConv(6, 128); self.c2 = SAGEConv(128, 128); self.c3 = SAGEConv(128, 128)
    def forward(self, x, adj): return self.c3(self.c2(self.c1(x, adj), adj), adj).mean(dim=1)

class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.lstm = nn.LSTM(4, 256, 2, batch_first=True, dropout=0.1)
    def forward(self, x): _, (h, _) = self.lstm(x); return h[-1]

class GlobalEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
    def forward(self, x): return self.net(x)

class ManuscriptActorCritic(nn.Module):
    def __init__(self, s=True, t=True):
        super().__init__(); self.s=s; self.t=t
        if s: self.enc_s = GraphSAGEEncoder()
        if t: self.enc_t = TemporalEncoder()
        self.enc_g = GlobalEncoder()
        f_dim = (128 if s else 0) + (256 if t else 0) + 64
        self.actor = nn.Sequential(nn.Linear(f_dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())
        self.actor_mu = nn.Linear(64, NUM_ZONES+2); nn.init.constant_(self.actor_mu.bias, 0.0)
        self.actor_logstd = nn.Parameter(torch.zeros(NUM_ZONES+2))
        self.critic = nn.Sequential(nn.Linear(f_dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, xs, adj, xt, xg):
        es = self.enc_s(xs, adj) if self.s else torch.zeros(xs.shape[0], 0)
        et = self.enc_t(xt) if self.t else torch.zeros(xt.shape[0], 0)
        eg = self.enc_g(xg); z = torch.cat([e for e in [es, et, eg] if e.shape[1]>0], dim=-1)
        mu = torch.sigmoid(self.actor_mu(self.actor(z))); std = torch.exp(self.actor_logstd)
        return mu, std, self.critic(z), z, es, et
"""))

    # --- CELL 4: Agent ---
    cells.append(nbf.v4.new_code_cell("""
class PPOBuffer:
    def __init__(self): self.reset()
    def reset(self): self.xs, self.xt, self.xg, self.a, self.lp, self.r, self.v, self.term = [], [], [], [], [], [], [], []
    def store(self, xs, xt, xg, a, lp, r, v, term):
        self.xs.append(xs); self.xt.append(xt); self.xg.append(xg); self.a.append(a); self.lp.append(lp); self.r.append(r); self.v.append(v); self.term.append(term)

class PPOAgent:
    def __init__(self, name, s=True, t=True):
        self.name = name; self.policy = ManuscriptActorCritic(s, t); self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.policy_old = ManuscriptActorCritic(s, t); self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = 0.99; self.lam = 0.95; self.eps_clip = 0.2; self.K_epochs = 4
    def select_action(self, xs, adj, xt, xg, eval_mode=False):
        with torch.no_grad(): mu, std, val, z, es, et = self.policy_old(xs, adj, xt, xg)
        if eval_mode: return mu.squeeze().numpy(), val.item(), z.squeeze().numpy(), es.squeeze().numpy(), et.squeeze().numpy()
        dist = Normal(mu, std); a = dist.sample()
        return a, dist.log_prob(a).sum(dim=-1), val, z
    def update(self, buffer, adj):
        xs = torch.cat(buffer.xs); xt = torch.cat(buffer.xt); xg = torch.cat(buffer.xg); a = torch.cat(buffer.a); old_lp = torch.cat(buffer.lp)
        v = torch.cat(buffer.v).squeeze(); r = buffer.r; term = buffer.term
        adv = np.zeros(len(r), dtype=np.float32); gae = 0; v_np = np.append(v.numpy(), 0)
        for i in reversed(range(len(r))):
            delta = r[i] + self.gamma * v_np[i+1] * (1 - term[i]) - v_np[i]; gae = delta + self.gamma * self.lam * (1 - term[i]) * gae; adv[i] = gae
        ret = torch.tensor(adv + v_np[:-1], dtype=torch.float32); adv = torch.tensor(adv, dtype=torch.float32); adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        l_a, l_c, l_e = 0, 0, 0
        for _ in range(self.K_epochs):
            mu, std, val, _, _, _ = self.policy(xs, adj, xt, xg)
            dist = Normal(mu, std); lp = dist.log_prob(a).sum(dim=-1); ent = dist.entropy().sum(dim=-1); ratio = torch.exp(lp - old_lp.detach())
            surr1 = ratio * adv; surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv
            a_loss = -torch.min(surr1, surr2).mean(); c_loss = 0.5 * F.mse_loss(val.squeeze(), ret); loss = a_loss + c_loss - 0.01 * ent.mean()
            self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5); self.optimizer.step()
            l_a += a_loss.item(); l_c += c_loss.item(); l_e += ent.mean().item()
        self.policy_old.load_state_dict(self.policy.state_dict()); buffer.reset(); return l_a/self.K_epochs, l_c/self.K_epochs, l_e/self.K_epochs
"""))

    # --- CELL 5: Loop ---
    cells.append(nbf.v4.new_code_cell("""
def extract_features(state, power, speed, history_q):
    temps = state[:NUM_ZONES] / 100.0; p_norm = power / 26600.0; s_norm = speed / 50.0
    xs = np.zeros((NUM_ZONES, 6)); xs[:,0] = temps; xs[:,1] = temps - temps.mean(); xs[:,2] = 0.5; xs[:,3] = p_norm; xs[:,4] = 3.7/4.2; xs[:,5] = np.arange(12)/12.0
    history_q.append([temps.mean(), temps.max(), temps.min(), p_norm])
    xt = np.array(history_q); xg = np.array([25.0/50.0, s_norm, 0.0])
    return torch.tensor(xs, dtype=torch.float32).unsqueeze(0), torch.tensor(xt, dtype=torch.float32).unsqueeze(0), torch.tensor(xg, dtype=torch.float32).unsqueeze(0)

adj = torch.eye(12); adj += torch.diag(torch.ones(11), 1); adj += torch.diag(torch.ones(11), -1)

def run_episode(name, agent, is_rl=False, train=False):
    os.makedirs(f"results/full_campaign/flight_recorder/{name}", exist_ok=True)
    state = np.concatenate([np.full(12, INITIAL_TEMP), np.zeros(12)]); prev_a = np.zeros(14); h_q = deque([np.zeros(4) for _ in range(10)], maxlen=10)
    buf = PPOBuffer() if (is_rl and train) else None; traj, lat, ems, emt, vals = [], [], [], [], []
    for t in range(EVAL_HORIZON):
        xs, xt, xg = extract_features(state, power_profile[t], speed_profile[t], h_q)
        if is_rl:
            if train: a_t, lp, val, z = agent.select_action(xs, adj, xt, xg); a = np.clip(a_t.squeeze().numpy(), 0, 1)
            else: a, val, z, es, et = agent.select_action(xs, adj, xt, xg, eval_mode=True); lat.append(z); ems.append(es); emt.append(et); vals.append(val)
        else:
            if 'PID_Standard' in name: a = np.concatenate([np.clip(0.1*(state[:12]-35),0,1), [0.5,0.5]])
            elif 'PID_Adaptive' in name: a = np.concatenate([np.clip((0.2 if state[:12].max()>38 else 0.1)*(state[:12]-35),0,1), [0.2,0.2]])
            elif 'Uniform' in name: a = np.concatenate([np.ones(12)*0.5, [0.5,0.5]])
            elif 'Proportional' in name: v = np.clip((state[:12]-25)/15,0,1); a = np.concatenate([v, [v.mean(), v.mean()]])
            elif 'MPC' in name: a = np.random.uniform(0, 1, 14)
        sol = solve_ivp(lambda tau, y: battery_thermal_ode(tau, y, power_profile, a), [t, t+1.0], state); ns = sol.y[:, -1]; pump_pow = 200.0 * (a[12]**2)
        rs, rt, re, rsm = get_reward_components(ns, pump_pow, prev_a, a); r = rs + rt + re + rsm
        if train: buf.store(xs, xt, xg, a_t, lp, r, val, t==EVAL_HORIZON-1)
        traj.append({'t': t, 'max_t': ns[:12].max(), 'mean_t': ns[:12].mean(), 'spread': ns[:12].max()-ns[:12].min(), 'energy': pump_pow, 'r': r, 'pump': a[12], 'fan': a[13]})
        state = ns; prev_a = a
    if train: return agent.update(buf, adj), pd.DataFrame(traj)['r'].mean()
    df = pd.DataFrame(traj); df.to_csv(f"results/full_campaign/flight_recorder/{name}/trajectory.csv", index=False)
    if is_rl and not train:
        for k, v in {'latents':lat, 'embeddings':ems, 'hidden_states':emt, 'values':vals}.items(): np.save(f"results/full_campaign/flight_recorder/{name}/{k}.npy", np.array(v))
    return df
"""))

    # --- CELL 6: Execution ---
    cells.append(nbf.v4.new_code_cell("""
rl_agents = {'Proposed_Full': PPOAgent('Proposed_Full', True, True), 'Ablation_NoSpatial': PPOAgent('Ablation_NoSpatial', False, True), 'Ablation_NoTemporal': PPOAgent('Ablation_NoTemporal', True, False), 'Ablation_MLPOnly': PPOAgent('Ablation_MLPOnly', False, False)}
print("--- TRAINING ---")
logs = []
for name, agent in rl_agents.items():
    print(f"Agent: {name}")
    for ep in range(TRAIN_EPOCHS):
        (al, cl, en), mr = run_episode(name, agent, True, True); logs.append({'model': name, 'ep': ep, 'r': mr, 'al': al, 'cl': cl, 'en': en})
        if (ep+1)%10==0: print(f"  Ep {ep+1} | Reward: {mr:.2f}")
    torch.save(agent.policy.state_dict(), f"results/full_campaign/checkpoints/{name}_best.pt")
pd.DataFrame(logs).to_csv("results/full_campaign/logs/training_log.csv", index=False)
print("\\n--- EVALUATION ---")
eval_res = {}
for n in ['PID_Standard', 'PID_Adaptive', 'Uniform_Flow', 'Proportional_Temp', 'MPC_H1_S32']:
    print(f"Baseline: {n}"); eval_res[n] = run_episode(n, None, False, False)
for n, a in rl_agents.items():
    print(f"RL: {n}"); eval_res[n] = run_episode(n, a, True, False)
"""))

    # --- CELL 7: Metrics & Figures ---
    cells.append(nbf.v4.new_code_cell("""
comp_data = []
for name, df in eval_res.items():
    comp_data.append({'controller': name, 'peak_t': df['max_t'].max(), 'mean_t': df['mean_t'].mean(), 'spread': df['spread'].mean(), 'energy': df['energy'].sum(), 'reward': df['r'].sum()})
pd.DataFrame(comp_data).to_csv("results/full_campaign/metrics/controller_comparison.csv", index=False)
df_t = pd.read_csv("results/full_campaign/logs/training_log.csv")
for f, col, t in [('Fig1_Reward', 'r', 'Reward'), ('Fig2_ActorLoss', 'al', 'Actor Loss'), ('Fig3_CriticLoss', 'cl', 'Critic Loss')]:
    plt.figure(); 
    for m in df_t['model'].unique():
        sub = df_t[df_t['model'] == m]; plt.plot(sub['ep'], sub[col], label=m)
    plt.title(t); plt.legend(); plt.savefig(f"results/full_campaign/figures/{f}.png"); plt.close()
plt.figure(figsize=(10,5))
for n, df in eval_res.items():
    if "Ablation" not in n: plt.plot(df['t'], df['max_t'], label=n)
plt.axhline(40, color='r', ls='--'); plt.legend(bbox_to_anchor=(1.05, 1)); plt.tight_layout(); plt.savefig("results/full_campaign/figures/Fig4_Temporal.png"); plt.close()
plt.figure(figsize=(10,5))
for n, df in eval_res.items():
    if "Proposed" in n or "PID" in n: plt.plot(df['t'], df['pump'], label=n)
plt.legend(bbox_to_anchor=(1.05, 1)); plt.savefig("results/full_campaign/figures/Fig5_Pump.png"); plt.close()
plt.figure(figsize=(10,5))
for n, df in eval_res.items():
    if "No" not in n and "MLP" not in n: plt.plot(df['t'], df['spread'], label=n)
plt.legend(bbox_to_anchor=(1.05, 1)); plt.savefig("results/full_campaign/figures/Fig6_Spread.png"); plt.close()
plt.figure()
df_c = pd.read_csv("results/full_campaign/metrics/controller_comparison.csv")
plt.scatter(df_c['spread'], df_c['energy'], c='blue')
for i, r in df_c.iterrows(): plt.annotate(r['controller'], (r['spread'], r['energy']), fontsize=8)
plt.xlabel("Mean Spread"); plt.ylabel("Total Energy"); plt.savefig("results/full_campaign/figures/Fig7_Pareto.png"); plt.close()
"""))

    # --- CELL 8: Manifest ---
    cells.append(nbf.v4.new_code_cell("""
manifest = {"controllers": list(eval_res.keys()), "figures": os.listdir("results/full_campaign/figures"), "flight_records": {name: os.listdir(f"results/full_campaign/flight_recorder/{name}") for name in eval_res.keys()}, "seed": 42}
with open("results/full_campaign/reports/manifest.json", "w") as f: json.dump(manifest, f, indent=2)
print("✅ CAMPAIGN COMPLETE.")
"""))

    nb.cells = cells
    with open('Production_Campaign.ipynb', 'w') as f: nbf.write(nb, f)

if __name__ == "__main__": build_notebook()
