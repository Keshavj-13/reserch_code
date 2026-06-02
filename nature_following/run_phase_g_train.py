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
from collections import deque
import sys
sys.path.append(os.path.join(os.getcwd(), 'tags', 'recovered_physics_v1'))

from core_physics_v1 import (
    battery_thermal_ode, get_reward_components, load_original_drive_cycles,
    NUM_ZONES, INITIAL_TEMP, TEMP_MAX, TEMP_MIN
)

EVAL_HORIZON = 6105
TRAIN_EPOCHS = 50
DECIMATION_FACTOR = 20

def log_anomaly(model, event_type, details):
    os.makedirs("metrics/v2", exist_ok=True)
    row = pd.DataFrame([{"timestamp": time.time(), "model": model, "event_type": event_type, "details": details}])
    hdr = not os.path.exists("metrics/v2/anomaly_log.csv")
    row.to_csv("metrics/v2/anomaly_log.csv", mode='a', header=hdr, index=False)

def fail_fast(reason, model, ep):
    log_anomaly(model, "FAIL_FAST_TRIGGERED", reason)
    os.makedirs("reports", exist_ok=True)
    with open("reports/FailFast_Report.md", "a") as f:
        f.write(f"\n- **Model**: {model}, **Epoch**: {ep}, **Reason**: {reason}\n")
    print(f"\n[FAIL FAST] {reason}")
    os._exit(1)

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
        # ACTION DIM is NUM_ZONES (12)
        self.actor_mu = nn.Linear(64, NUM_ZONES)
        nn.init.constant_(self.actor_mu.bias, 0.0)
        self.actor_logstd = nn.Parameter(torch.ones(NUM_ZONES) * -1.0)
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
        mu = self.actor_mu(act_out)
        std = torch.exp(self.actor_logstd).expand_as(mu)
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

def run_phase_g():
    print("Phase G: Retraining on Recovered Physics")
    os.makedirs("results/checkpoints_v2", exist_ok=True)
    os.makedirs("metrics/v2", exist_ok=True)
    
    power_profile, speed_profile = load_original_drive_cycles()
    adj = torch.eye(NUM_ZONES); adj += torch.diag(torch.ones(NUM_ZONES-1), 1); adj += torch.diag(torch.ones(NUM_ZONES-1), -1)
    
    rl_agents = {
        'GS+LSTM+PPO': PPOAgent('GS+LSTM+PPO', True, True),
        'LSTM+PPO': PPOAgent('LSTM+PPO', False, True),
        'GS+PPO': PPOAgent('GS+PPO', True, False),
        'MLP+PPO': PPOAgent('MLP+PPO', False, False)
    }
    
    for name, agent in rl_agents.items():
        print(f"=== Training {name} ===")
        best_reward = -np.inf
        
        for ep in range(TRAIN_EPOCHS):
            state = np.full(NUM_ZONES * 2, INITIAL_TEMP)
            state[NUM_ZONES:] = 0.0 # initial flows
            prev_a = np.zeros(NUM_ZONES)
            history_q = deque([np.zeros(4) for _ in range(10)], maxlen=10)
            buffer = PPOBuffer()
            ep_rs, ep_rt, ep_re, ep_rsm = 0, 0, 0, 0
            
            for t in range(EVAL_HORIZON):
                if np.isnan(state).any(): fail_fast("NaN detected in state", name, ep)
                if state[:NUM_ZONES].max() > 60.0: fail_fast("Temperature exceeded 60C absolute limit", name, ep)
                
                xs, xt, xg = extract_features(state, power_profile[t], speed_profile[t], history_q)
                a_t, lp, val, z, es, et, ao = agent.select_action(xs, adj, xt, xg)
                # Apply sigmoid squashing as per final_run
                a = torch.sigmoid(a_t).squeeze(0).numpy()
                
                dTdt = battery_thermal_ode(t, state, power_profile, a)
                state = state + dTdt * 1.0
                
                rs, rt, re, rsm = get_reward_components(state, a, prev_a)
                r = rs + rt + re + rsm
                buffer.store(xs, xt, xg, a_t, lp, r, val, t==EVAL_HORIZON-1)
                
                prev_a = a
                ep_rs+=rs; ep_rt+=rt; ep_re+=re; ep_rsm+=rsm
                
            total_r = ep_rs + ep_rt + ep_re + ep_rsm
            a_loss, c_loss, ent = agent.update(buffer, adj)
            
            if c_loss > 1e6: fail_fast("Critic loss exceeded 1e6", name, ep)
            if math.isinf(c_loss) or math.isnan(c_loss): fail_fast("Inf/NaN Critic Loss", name, ep)
            
            ep_row = pd.DataFrame([{"model": name, "epoch": ep, "reward": total_r, "max_temp": state[:NUM_ZONES].max(), "mean_temp": state[:NUM_ZONES].mean(), "cooling_energy_proxy": ep_re*-2.0, "critic_loss": c_loss, "entropy": ent}])
            ep_row.to_csv("metrics/v2/epoch_summary.csv", mode='a', header=not os.path.exists("metrics/v2/epoch_summary.csv"), index=False)
            
            try:
                torch.save(agent.policy.state_dict(), f"results/checkpoints_v2/{name}_latest.pt")
                if total_r > best_reward:
                    best_reward = total_r
                    torch.save(agent.policy.state_dict(), f"results/checkpoints_v2/{name}_best.pt")
            except Exception as e:
                fail_fast(f"Checkpoint save failed: {str(e)}", name, ep)
                
            print(f"[{name} Ep {ep}] R: {total_r:.2f} | Tmax: {state[:NUM_ZONES].max():.2f}C | Critic: {c_loss:.2f}")

if __name__ == "__main__":
    run_phase_g()