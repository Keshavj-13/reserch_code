import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def generate_hotspot_tracking():
    models = ['AdaptivePID', 'GS+LSTM+PPO', 'GS+PPO']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    
    print("=== Hotspot Tracking Statistics ===")
    
    for i, model in enumerate(models):
        df = pd.read_csv(f"{model}_run.csv")
        temp_cols = [c for c in df.columns if c.endswith('_temp_C')]
        
        t_max = df[temp_cols].max(axis=1)
        t_mean = df[temp_cols].mean(axis=1)
        t_min = df[temp_cols].min(axis=1)
        time = df['time_s']
        
        spread = t_max - t_min
        print(f"Model: {model}")
        print(f"  Max Temp Peak:  {t_max.max():.2f} °C")
        print(f"  Max Spread:     {spread.max():.2f} °C")
        print(f"  Mean Spread:    {spread.mean():.2f} °C")
        
        ax = axes[i]
        ax.plot(time, t_max, label='Max Temp (Hotspot)', color='#D6604D', linestyle='-', linewidth=1.5)
        ax.plot(time, t_mean, label='Mean Temp', color='#878787', linestyle='--', linewidth=1.5)
        ax.plot(time, t_min, label='Min Temp (Coldspot)', color='#2166AC', linestyle='-', linewidth=1.5)
        
        # Highlight the spread
        ax.fill_between(time, t_min, t_max, color='#878787', alpha=0.15, label='Spatial Spread')
        
        ax.set_title(f"{model} - Hotspot Tracking", fontweight='bold')
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    
    os.makedirs("replots_from_csv", exist_ok=True)
    plt.savefig("replots_from_csv/08_hotspot_tracking.png", bbox_inches='tight', dpi=300)
    plt.savefig("replots_from_csv/08_hotspot_tracking.pdf", bbox_inches='tight')
    
    with open("reports/Hotspot_Analysis_Report.md", "w", encoding="utf-8") as f:
        f.write("# Hotspot Analysis Report\n\n")
        f.write("Investigating the causal link between GraphSAGE spatial flow redistribution and actual hotspot suppression.\n\n")
        f.write("## Findings\n")
        f.write("See `replots_from_csv/08_hotspot_tracking.pdf` for visual evidence.\n")

if __name__ == "__main__":
    generate_hotspot_tracking()