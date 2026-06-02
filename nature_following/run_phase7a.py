import os
import pandas as pd
import numpy as np

def run_phase7a_audit():
    print("Starting Phase 7A: Pre-Campaign Artifact Audit...")
    
    target_files = [
        "metrics/controller_comparison.csv",
        "metrics/training_log.csv",
        "metrics/learning_signal_check.csv"
    ]
    
    results = []
    all_passed = True
    
    for f in target_files:
        exists = os.path.exists(f)
        if not exists:
            results.append({
                'artifact': f, 'exists': False, 'rows': 0, 'columns': 0, 
                'nan_count': 0, 'inf_count': 0, 'status': 'MISSING'
            })
            all_passed = False
            continue
            
        try:
            df = pd.read_csv(f)
            rows, cols = df.shape
            
            # Check for NaNs
            nan_count = int(df.isna().sum().sum())
            
            # Check for Infs (only in numeric columns)
            num_df = df.select_dtypes(include=[np.number])
            inf_count = int(np.isinf(num_df).sum().sum())
            
            status = 'OK'
            if rows == 0:
                status = 'EMPTY'
                all_passed = False
            elif nan_count > 0 or inf_count > 0:
                status = 'CORRUPT'
                all_passed = False
                
            results.append({
                'artifact': f, 'exists': True, 'rows': rows, 'columns': cols,
                'nan_count': nan_count, 'inf_count': inf_count, 'status': status
            })
            
        except Exception as e:
            results.append({
                'artifact': f, 'exists': True, 'rows': 0, 'columns': 0,
                'nan_count': -1, 'inf_count': -1, 'status': f'ERROR: {str(e)}'
            })
            all_passed = False
            
    df_res = pd.DataFrame(results)
    
    os.makedirs("reports", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    df_res.to_csv("metrics/pre_phase7_validation.csv", index=False)
    
    with open("reports/Phase7A_Audit.md", "w") as f:
        f.write("# Phase 7A: Pre-Campaign Artifact Audit\n\n")
        f.write("Validating the integrity of all prerequisites before authorizing the Phase 7 Long Training Campaign.\n\n")
        
        f.write("| Artifact | Exists | Rows | Cols | NaNs | Infs | Status |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in df_res.iterrows():
            f.write(f"| {r['artifact']} | {r['exists']} | {r['rows']} | {r['columns']} | {r['nan_count']} | {r['inf_count']} | {r['status']} |\n")
            
        f.write("\n## Conclusion\n")
        if all_passed:
            f.write("**PASSED.** All critical artifacts are present and structurally sound. Cleared for Phase 7B (Resume Test).\n")
        else:
            f.write("**FAILED.** Corruption or missing files detected. Do not proceed to training.\n")
            
    print("Phase 7A Complete. Check reports/Phase7A_Audit.md")

if __name__ == "__main__":
    run_phase7a_audit()
