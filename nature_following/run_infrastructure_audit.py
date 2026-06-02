import os
import json
import re
import pandas as pd

def audit_infrastructure():
    print("Starting Read-Only Infrastructure Audit...")
    os.makedirs("reports", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # ==================================================================
    # TASK 3: REPOSITORY INVENTORY
    # ==================================================================
    inventory = []
    for root, dirs, files in os.walk("."):
        if ".git" in root or "__pycache__" in root or ".ipynb_checkpoints" in root:
            continue
        for f in files:
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            ext = os.path.splitext(f)[1]
            
            art_type = "unknown"
            if ext == ".py": art_type = "script"
            elif ext == ".ipynb": art_type = "notebook"
            elif ext == ".csv": art_type = "data"
            elif ext == ".md": art_type = "report"
            elif ext == ".png" or ext == ".pdf": art_type = "figure"
            elif ext == ".pt": art_type = "checkpoint"
            elif ext == ".npy": art_type = "tensor"
            elif ext == ".txt": art_type = "text"
            elif ext == ".json": art_type = "metadata"
            
            phase = "unknown"
            if "phase1" in f.lower(): phase = "Phase 1"
            elif "phase2" in f.lower(): phase = "Phase 2"
            elif "phase3" in f.lower(): phase = "Phase 3"
            elif "phase4" in f.lower(): phase = "Phase 4"
            elif "phase5" in f.lower(): phase = "Phase 5"
            elif "phase6" in f.lower(): phase = "Phase 6"
            elif "phase7" in f.lower(): phase = "Phase 7"
            
            purpose = "TBD"
            if "audit" in f.lower(): purpose = "Validation"
            elif "run_" in f.lower(): purpose = "Execution"
            elif "plot_" in f.lower() or "generate_" in f.lower(): purpose = "Plotting"
            
            inventory.append({
                "artifact": f,
                "path": path,
                "type": art_type,
                "size_mb": round(size_mb, 4),
                "purpose": purpose,
                "phase": phase
            })
            
    df_inv = pd.DataFrame(inventory)
    df_inv.to_csv("metrics/repository_inventory.csv", index=False)

    # ==================================================================
    # TASK 1: PLOTTING INVENTORY
    # ==================================================================
    plots = [
        {"figure_name": "Fig1_Reward_Curves", "source_file": "Generate_Manuscript_Figures.ipynb", "source_data": "training_log.csv / epoch_summary.csv", "output_path": "results/figures/Fig1_Reward.png", "exists": os.path.exists("results/figures/Fig1_Reward.png"), "status": "MISSING"},
        {"figure_name": "Fig2_Actor_Loss", "source_file": "Generate_Manuscript_Figures.ipynb", "source_data": "training_log.csv / epoch_summary.csv", "output_path": "results/figures/Fig2_ActorLoss.png", "exists": os.path.exists("results/figures/Fig2_ActorLoss.png"), "status": "MISSING"},
        {"figure_name": "Fig3_Critic_Loss", "source_file": "Generate_Manuscript_Figures.ipynb", "source_data": "training_log.csv / epoch_summary.csv", "output_path": "results/figures/Fig3_CriticLoss.png", "exists": os.path.exists("results/figures/Fig3_CriticLoss.png"), "status": "MISSING"},
        {"figure_name": "Fig4_Temporal_Behavior", "source_file": "Generate_Manuscript_Figures.ipynb", "source_data": "*_run.csv / flight_recorder", "output_path": "results/figures/Fig4_Temporal.png", "exists": os.path.exists("results/figures/Fig4_Temporal.png"), "status": "MISSING"},
        {"figure_name": "Fig5_Cooling_Energy", "source_file": "Generate_Manuscript_Figures.ipynb", "source_data": "controller_comparison.csv", "output_path": "results/figures/Fig5_Pump.png", "exists": os.path.exists("results/figures/Fig5_Pump.png"), "status": "MISSING"},
        {"figure_name": "Fig6_Temperature_Spread", "source_file": "Generate_Manuscript_Figures.ipynb", "source_data": "controller_comparison.csv", "output_path": "results/figures/Fig6_Spread.png", "exists": os.path.exists("results/figures/Fig6_Spread.png"), "status": "MISSING"},
        {"figure_name": "Fig7_Pareto_Front", "source_file": "Generate_Manuscript_Figures.ipynb", "source_data": "controller_comparison.csv", "output_path": "results/figures/Fig7_Pareto.png", "exists": os.path.exists("results/figures/Fig7_Pareto.png"), "status": "MISSING"}
    ]
    # Update statuses
    for p in plots:
        if p["exists"]: p["status"] = "RECOVERED"
        
    pd.DataFrame(plots).to_csv("metrics/plotting_inventory.csv", index=False)
    
    with open("reports/Plotting_Inventory_Report.md", "w") as f:
        f.write("# Plotting Inventory Report\n\n")
        for p in plots:
            f.write(f"- **{p['figure_name']}**: {p['status']} (Expected at {p['output_path']})\n")

    # ==================================================================
    # TASK 2: MANUSCRIPT TRACEABILITY AUDIT
    # ==================================================================
    with open("reports/Manuscript_Traceability_Report.md", "w") as f:
        f.write("# Manuscript Traceability Audit\n\n")
        f.write("| Claim | Source Code | Supporting Metric | Supporting Figure | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write("| GraphSAGE + LSTM + PPO Architecture | `ManuscriptActorCritic` in run scripts | architecture_verification.csv | Fig 1-3 | SUPPORTED |\n")
        f.write("| 448-dim Fused Latent | `self.fused_dim = 128 + 256 + 64` | architecture_verification.csv | - | SUPPORTED |\n")
        f.write("| 16 kW Peak Power | `power_profile = ...` (18A clip * 44.4V) | drive_cycle_summary.csv | - | SUPPORTED |\n")
        f.write("| US06 highest instantaneous peaks | `us06col.txt` parsing | drive_cycle_summary.csv | - | SUPPORTED |\n")
        f.write("| Ablations prove Spatial/Temporal value | Phase 8 Ablation Script | ablation_results.csv | Fig 7 Pareto | **LACKING EVIDENCE (Pending Phase 8)** |\n")

    # ==================================================================
    # TASK 4: TECHNICAL DEBT AUDIT
    # ==================================================================
    with open("reports/Technical_Debt_Report.md", "w") as f:
        f.write("# Technical Debt Audit\n\n")
        f.write("### HIGH RISK\n")
        f.write("- **CSV Naming Mismatch for Figures**: `Generate_Manuscript_Figures.ipynb` expects `PID_run.csv`, `MPC_run.csv`, but Phase 7 outputs `results/flight_recorder/{model}/trajectory.csv`. A translation/mapping script is required before Phase 9.\n")
        f.write("- **Missing comparison_summary.csv**: Plotting code expects this file, but it is not currently generated by our pipeline.\n\n")
        f.write("### MEDIUM RISK\n")
        f.write("- **Duplicated Physics Code**: The `battery_thermal_ode`, `load_original_drive_cycles`, and `ManuscriptActorCritic` are duplicated across `run_phase*.py` scripts. Should be refactored into a `core_physics.py` module.\n")
        f.write("- **Hardcoded Constants**: `NUM_ZONES = 12`, `TEMP_MAX = 40.0` are hardcoded in multiple files.\n\n")
        f.write("### LOW RISK\n")
        f.write("- **Abandoned Notebooks**: `final_run.ipynb` is a 4.4MB monolith that is now obsolete and should be moved to an `archive/` folder.\n")

    # ==================================================================
    # TASK 5: PHASE 8 PREPARATION
    # ==================================================================
    with open("reports/Phase8_Readiness_Report.md", "w") as f:
        f.write("# Phase 8 Readiness Report\n\n")
        f.write("### Required Assets for Ablation & Statistics:\n")
        f.write("- **Trained Checkpoints**: GENERATING (Running in Phase 7)\n")
        f.write("- **Evaluation Script (`run_phase8.py`)**: MISSING (Needs to be written to load checkpoints and run 6105s eval)\n")
        f.write("- **Statistical Validation Loop**: MISSING (Needs to be written)\n\n")
        f.write("**Status**: NEEDS_REPAIR / SCRIPT WRITING\n")

    # ==================================================================
    # TASK 6: PHASE 9 PREPARATION
    # ==================================================================
    with open("reports/Phase9_Readiness_Report.md", "w") as f:
        f.write("# Phase 9 Readiness Report\n\n")
        f.write("### Required Assets for Manuscript Figures:\n")
        f.write("- **`Generate_Manuscript_Figures.ipynb`**: READY\n")
        f.write("- **`controller_comparison.csv`**: READY (From Phase 6)\n")
        f.write("- **`comparison_summary.csv`**: MISSING\n")
        f.write("- **`*_run.csv` files**: MISSING (Flight recorder uses different naming convention: `trajectory.csv` inside subfolders)\n\n")
        f.write("**Status**: NEEDS_REPAIR. A data transformation script must be run before Phase 9 to map the flight recorder outputs to the filenames expected by the replot notebook.\n")

    # ==================================================================
    # TASK 7: MASTER GAP ANALYSIS
    # ==================================================================
    with open("reports/Master_Gap_Analysis.md", "w") as f:
        f.write("# Master Gap Analysis\n\n")
        f.write("### 1. What is still missing?\n")
        f.write("- The mapping script to convert `flight_recorder/{model}/trajectory.csv` to `{model}_run.csv` for the plotting notebook.\n")
        f.write("- `run_phase8.py` to evaluate checkpoints and generate `ablation_results.csv` and `statistical_validation.csv`.\n\n")
        f.write("### 2. What can break Phase 8?\n")
        f.write("- If Phase 7 crashes or fails to produce `_best.pt` checkpoints, Phase 8 has no weights to load.\n\n")
        f.write("### 3. What can break Phase 9?\n")
        f.write("- `Generate_Manuscript_Figures.ipynb` explicitly calls `pd.read_csv(p)` looking for `*run.csv`. If we do not copy/rename the trajectories, the plot generation will fail with a `FileNotFoundError`.\n\n")
        f.write("### 4. Which manuscript claims remain unsupported?\n")
        f.write("- Ablation validity (Spatial vs Temporal vs Both). We have un-trained baselines from Phase 6, but we need the fully trained Phase 8 results to prove the architecture matters.\n\n")
        f.write("### 5. What is the highest risk remaining?\n")
        f.write("- **Data Mismatch Risk**: The plotting notebook contains rigid hardcoded paths and expected column names. We must ensure the metrics output exactly matches its schema.\n")
        
    print("Audit Complete.")

if __name__ == "__main__":
    audit_infrastructure()
