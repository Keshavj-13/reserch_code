import os
import pandas as pd
import torch

# This script would import the classes from run_phase7.py
# For brevity in this setup, it outlines the resume logic.
from run_phase7 import rl_agents, run_phase7, TRAIN_EPOCHS

def resume_training():
    print("Resuming Phase 7...")
    if not os.path.exists("metrics/epoch_summary.csv"):
        print("No training log found. Starting fresh.")
        run_phase7()
        return
        
    df = pd.read_csv("metrics/epoch_summary.csv")
    
    for name, agent in rl_agents.items():
        sub = df[df['model'] == name]
        if sub.empty:
            start_ep = 0
        else:
            start_ep = sub['epoch'].max() + 1
            
        if start_ep >= TRAIN_EPOCHS:
            print(f"{name} is already complete (Epochs {start_ep}/{TRAIN_EPOCHS}).")
            continue
            
        print(f"Resuming {name} from epoch {start_ep}...")
        
        # Load weights
        latest_pt = f"results/checkpoints/{name}_latest.pt"
        if os.path.exists(latest_pt):
            agent.policy.load_state_dict(torch.load(latest_pt))
            agent.policy_old.load_state_dict(torch.load(latest_pt))
            print(f"  Loaded {latest_pt}")
            
        # The main training loop from run_phase7 would go here, 
        # using `for ep in range(start_ep, TRAIN_EPOCHS):`
        # (Omitted for brevity in this boilerplate).

if __name__ == "__main__":
    resume_training()
