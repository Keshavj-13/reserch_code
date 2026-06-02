import os
import shutil
import pandas as pd

def prepare_phase9_data():
    print("Preparing Phase 9 Data...")
    
    # Generate_Manuscript_Figures.ipynb expects `{model}_run.csv` in the root directory (RESULTS = ROOT)
    source_base = "results/flight_recorder"
    
    if not os.path.exists(source_base):
        print(f"Source directory {source_base} does not exist.")
        return
        
    models = [d for d in os.listdir(source_base) if os.path.isdir(os.path.join(source_base, d))]
    
    copied = 0
    for model in models:
        src = os.path.join(source_base, model, "trajectory.csv")
        dst = f"{model}_run.csv"
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
            copied += 1
        else:
            print(f"Missing {src}")
            
    # Also ensure comparison_summary.csv exists (if plotting code needs it)
    # The plotting code has: df_summary3 = pd.read_csv('comparison_summary.csv') if summary_path.exists() else pd.DataFrame()
    # So it's optional, but we will make an empty one or extract from epoch_summary if needed.
    
    print(f"Prepared {copied} trajectory files for Phase 9.")

if __name__ == "__main__":
    prepare_phase9_data()
