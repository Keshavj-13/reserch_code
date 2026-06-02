import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

NUM_ZONES = 12

# --- ARCHITECTURE ---
class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels * 2, out_channels)
    def forward(self, x, adj):
        return F.relu(self.lin(torch.cat([x, torch.matmul(adj, x)], dim=-1)))

class GraphSAGEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(6, 128); self.conv2 = SAGEConv(128, 128); self.conv3 = SAGEConv(128, 128)
    def forward(self, x, adj):
        return self.conv3(self.conv2(self.conv1(x, adj), adj), adj).mean(dim=1)

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

def run_phase5_verification():
    print("Starting Phase 5: Architecture Verification...")
    
    bs = 8
    x_s = torch.randn(bs, NUM_ZONES, 6)
    adj = torch.ones(bs, NUM_ZONES, NUM_ZONES)
    x_t = torch.randn(bs, 10, 4)
    x_g = torch.randn(bs, 3)
    
    models = {
        "Proposed_Full": ManuscriptActorCritic(True, True),
        "Ablation_NoSpatial": ManuscriptActorCritic(False, True),
        "Ablation_NoTemporal": ManuscriptActorCritic(True, False),
        "Ablation_MLPOnly": ManuscriptActorCritic(False, False)
    }
    
    results = []
    
    # 1. Component Level Verification (using Proposed_Full)
    model = models["Proposed_Full"]
    mu, std, val, z, emb_s, emb_t, act_out = model(x_s, adj, x_t, x_g)
    
    results.append({'layer': 'GraphSAGE', 'expected_dim': 128, 'actual_dim': emb_s.shape[-1], 'status': 'OK' if emb_s.shape[-1] == 128 else 'FAIL'})
    results.append({'layer': 'LSTM', 'expected_dim': 256, 'actual_dim': emb_t.shape[-1], 'status': 'OK' if emb_t.shape[-1] == 256 else 'FAIL'})
    results.append({'layer': 'GlobalMLP', 'expected_dim': 64, 'actual_dim': model.enc_g(x_g).shape[-1], 'status': 'OK' if model.enc_g(x_g).shape[-1] == 64 else 'FAIL'})
    results.append({'layer': 'Fused_Latent', 'expected_dim': 448, 'actual_dim': z.shape[-1], 'status': 'OK' if z.shape[-1] == 448 else 'FAIL'})
    results.append({'layer': 'Actor_Output', 'expected_dim': 14, 'actual_dim': mu.shape[-1], 'status': 'OK' if mu.shape[-1] == 14 else 'FAIL'})
    results.append({'layer': 'Critic_Output', 'expected_dim': 1, 'actual_dim': val.shape[-1], 'status': 'OK' if val.shape[-1] == 1 else 'FAIL'})
    
    # 2. Ablation Models Latent Dimensions
    expected_fused = {"Ablation_NoSpatial": 320, "Ablation_NoTemporal": 192, "Ablation_MLPOnly": 64}
    for m_name, exp_dim in expected_fused.items():
        _, _, _, m_z, _, _, _ = models[m_name](x_s, adj, x_t, x_g)
        results.append({'layer': f'{m_name}_Latent', 'expected_dim': exp_dim, 'actual_dim': m_z.shape[-1], 'status': 'OK' if m_z.shape[-1] == exp_dim else 'FAIL'})
        
    df_res = pd.DataFrame(results)
    
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    df_res.to_csv("metrics/architecture_verification.csv", index=False)
    
    with open("reports/Architecture_Verification.md", "w") as f:
        f.write("# Phase 5: Architecture Verification Report\n\n")
        f.write("Validating the dimensions of the RL neural network components, particularly the 448-dim fused latent vector for the proposed architecture.\n\n")
        f.write("| Component | Expected Dim | Actual Dim | Status |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for _, r in df_res.iterrows():
            f.write(f"| {r['layer']} | {r['expected_dim']} | {r['actual_dim']} | {r['status']} |\n")
            
    print("Phase 5 Complete.")

if __name__ == "__main__":
    run_phase5_verification()
