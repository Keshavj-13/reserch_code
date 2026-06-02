import json
import os
import re

def analyze_final_run():
    print("Extracting environment from final_run.ipynb...")
    with open('final_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = [ "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code" ]
    
    # Write all code to a temporary text file for manual review if needed
    with open("final_run_extracted_code.py", "w", encoding="utf-8") as f:
        for i, code in enumerate(code_cells):
            f.write(f"# CELL {i}\n")
            f.write(code + "\n\n")
            
    # We will search for specific definitions to map
    mapping = []
    
    for code in code_cells:
        if "def battery_thermal_ode" in code:
            mapping.append({"item": "Thermal ODE & Flow ODE", "code": code})
        elif "cell_r_variation =" in code:
            mapping.append({"item": "Resistance Variation & Init", "code": code})
        elif "class BatteryEnv" in code or "def step(" in code:
            mapping.append({"item": "Environment Step & Reward", "code": code})
        elif "class PID" in code:
            mapping.append({"item": "Baseline Controllers", "code": code})
        elif "class ActorCritic" in code or "class BatteryPPO" in code or "class Manuscript" in code:
            mapping.append({"item": "RL Architecture", "code": code})
            
    with open("reports/Original_Environment_Map.md", "w", encoding="utf-8") as f:
        f.write("# Original Environment Map\n\n")
        f.write("Extracted strictly from `final_run.ipynb`.\n\n")
        for m in mapping:
            f.write(f"## {m['item']}\n```python\n{m['code']}\n```\n\n")
            
    print("Extraction complete. Check reports/Original_Environment_Map.md and final_run_extracted_code.py")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    analyze_final_run()
