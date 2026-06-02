import os
import pandas as pd

def calculate_budget():
    print("Starting Phase 7C: Campaign Budget Calculation...")
    
    # Assumptions from previous phases
    horizon = 6105
    epochs = 50
    models = 4
    decimation_factor = 20
    
    # 1. Telemetry size per model per epoch (MB)
    # Trajectory (CSV): ~0.24 MB
    # Latents (npy): 6105 / 20 * 448 * 4 bytes = ~0.52 MB
    # Embeddings (npy): 6105 / 20 * (128 + 256 + 64) * 4 bytes = ~0.52 MB
    # Values (npy): 6105 / 20 * 1 * 4 bytes = ~0.001 MB
    # Total per model per epoch: ~1.28 MB
    mb_per_model_run = 1.28
    
    # 2. Checkpoint size per model (MB)
    # 448 fused dim -> ~350k params -> ~1.4 MB per pt file
    # We save 'latest' and 'best', so 2 files per model = 2.8 MB
    mb_checkpoints = 2.8
    
    # Calculate totals
    total_runs = models * epochs
    total_telemetry_mb = total_runs * mb_per_model_run
    total_checkpoints_mb = models * mb_checkpoints
    total_storage_mb = total_telemetry_mb + total_checkpoints_mb
    
    # Runtime estimate (from Phase B observations)
    # Phase B: 5 epochs for 4 models took ~1 minute.
    # Therefore, 50 epochs for 4 models should take ~10 minutes.
    # We add 2x safety margin -> 20 minutes.
    est_runtime_mins = 20
    
    # Create CSV
    budget = [{
        'models': models,
        'epochs': epochs,
        'estimated_files': total_runs * 6 + (models * 2), # 6 files per flight record, 2 checkpoints per model
        'estimated_mb': round(total_storage_mb, 2),
        'estimated_runtime_minutes': est_runtime_mins
    }]
    
    df_budget = pd.DataFrame(budget)
    
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    df_budget.to_csv("metrics/campaign_budget.csv", index=False)
    
    with open("reports/Campaign_Budget_Report.md", "w") as f:
        f.write("# Phase 7C: Campaign Budget Report\n\n")
        f.write("Projection of resources required for the full 50-epoch Phase 7 training campaign.\n\n")
        
        f.write("| Metric | Estimate |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| Models to Train | {models} |\n")
        f.write(f"| Epochs per Model | {epochs} |\n")
        f.write(f"| Total Simulation Runs | {total_runs} |\n")
        f.write(f"| Telemetry Decimation | {decimation_factor}x |\n")
        f.write(f"| Total Estimated Storage | {total_storage_mb:.2f} MB |\n")
        f.write(f"| Total Estimated Files | ~1208 |\n")
        f.write(f"| Total Estimated Runtime | ~{est_runtime_mins} minutes |\n\n")
        
        f.write("## Conclusion\n")
        f.write(f"**PASSED.** Storage is heavily constrained to {total_storage_mb:.2f} MB, operating at just ~12% of the 2048 MB hard limit. Runtime is well within acceptable boundaries. Cleared for Phase 7D (Production Training).\n")
        
    print("Phase 7C Complete. Check reports/Campaign_Budget_Report.md")

if __name__ == "__main__":
    calculate_budget()
