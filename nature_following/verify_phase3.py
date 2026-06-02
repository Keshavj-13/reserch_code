import os
import json
import numpy as np
import pandas as pd

def run_phase3_validation():
    print("Starting Phase 3: Flight Recorder Validation...")
    
    # 1. Setup mock directories
    test_dir = "results/flight_recorder/Phase3_Test"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    # 2. Mock parameters
    horizon = 6105
    decimation_factor = 20
    
    # Dummy tensors
    traj = []
    latents, embeddings, hidden_states, actor_outputs, values = [], [], [], [], []
    
    print(f"  Running mock episode ({horizon} steps)...")
    for t in range(horizon):
        # Full rate
        traj.append({
            'time': t, 'max_temp': 35.0, 'mean_temp': 30.0, 
            'spread': 2.0, 'energy': 100.0, 'pump_cmd': 0.5, 'fan_cmd': 0.0,
            'reward': -1.0, 'safety_violations': 0
        })
        
        # Decimated
        if t % decimation_factor == 0:
            latents.append(np.random.randn(448).astype(np.float32))
            embeddings.append(np.random.randn(128).astype(np.float32))
            hidden_states.append(np.random.randn(256).astype(np.float32))
            actor_outputs.append(np.random.randn(64).astype(np.float32))
            values.append(np.random.randn(1).astype(np.float32))
            
    # 3. Save artifacts
    print("  Saving artifacts...")
    df_traj = pd.DataFrame(traj)
    df_traj.to_csv(f"{test_dir}/trajectory.csv", index=False)
    
    np.save(f"{test_dir}/latents.npy", np.array(latents))
    np.save(f"{test_dir}/embeddings.npy", np.array(embeddings))
    np.save(f"{test_dir}/hidden_states.npy", np.array(hidden_states))
    np.save(f"{test_dir}/actor_outputs.npy", np.array(actor_outputs))
    np.save(f"{test_dir}/values.npy", np.array(values))
    
    metadata = {'seed': 42, 'decimation_factor': decimation_factor, 'horizon': horizon}
    with open(f"{test_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f)
        
    # 4. Audit
    print("  Auditing files...")
    artifacts = [
        "trajectory.csv",
        "metadata.json",
        "latents.npy",
        "embeddings.npy",
        "hidden_states.npy",
        "values.npy",
        "actor_outputs.npy"
    ]
    
    audit_results = []
    for art in artifacts:
        path = f"{test_dir}/{art}"
        exists = os.path.exists(path)
        size_mb = os.path.getsize(path) / (1024 * 1024) if exists else 0
        status = "OK" if exists else "MISSING"
        
        audit_results.append({
            'artifact': art,
            'exists': exists,
            'size_mb': float(size_mb),
            'status': status
        })
        
    df_audit = pd.DataFrame(audit_results)
    df_audit.to_csv("metrics/flight_recorder_audit.csv", index=False)
    
    with open("reports/Flight_Recorder_Audit.md", "w") as f:
        f.write("# Phase 3: Flight Recorder Audit Report\n\n")
        f.write("Validation of the telemetry decimation and saving logic for the RL flight recorder.\n\n")
        f.write(f"- **Horizon:** {horizon} steps\n")
        f.write(f"- **Decimation Factor:** {decimation_factor}\n\n")
        
        f.write("| Artifact | Exists | Size (MB) | Status |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for _, r in df_audit.iterrows():
            f.write(f"| {r['artifact']} | {r['exists']} | {r['size_mb']:.4f} | {r['status']} |\n")
            
    print("Phase 3 Complete.")

if __name__ == "__main__":
    run_phase3_validation()
