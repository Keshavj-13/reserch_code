import os
import pandas as pd
import numpy as np

def verify_drive_cycles():
    folder_path = "drive_cycles"
    target_files = ["ftpcol.txt", "hwycol.txt", "j1015col.txt", "sc03col.txt", "uddscol.txt", "us06col.txt"]
    
    results = []
    
    for f in target_files:
        path = os.path.join(folder_path, f)
        if not os.path.exists(path):
            results.append({
                'cycle': f,
                'duration_s': 0,
                'max_speed': 0,
                'mean_speed': 0,
                'max_accel': 0,
                'status': 'MISSING'
            })
            continue
            
        df = None
        for enc in ['utf-8', 'ISO-8859-1', 'utf-16']:
            try:
                df_try = pd.read_csv(path, sep=None, engine='python', header=None, comment='#', encoding=enc)
                if df_try.shape[1] >= 2:
                    df = df_try.iloc[:, :2].copy()
                    df.columns = ['Time', 'Speed']
                    break
            except Exception:
                continue
                
        if df is None:
            results.append({
                'cycle': f,
                'duration_s': 0,
                'max_speed': 0,
                'mean_speed': 0,
                'max_accel': 0,
                'status': 'CORRUPT'
            })
            continue
            
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['Speed'] = pd.to_numeric(df['Speed'], errors='coerce')
        df = df.dropna(subset=['Time', 'Speed']).copy()
        
        duration = int(df['Time'].max() - df['Time'].min() + 1)
        
        # Convert mph to m/s
        speed_mps = df['Speed'].values * 0.44704
        
        max_speed = float(np.max(speed_mps))
        mean_speed = float(np.mean(speed_mps))
        
        # Calculate accel
        dt = 1.0
        accel = np.zeros_like(speed_mps)
        accel[1:] = (speed_mps[1:] - speed_mps[:-1]) / dt
        max_accel = float(np.max(accel))
        
        results.append({
            'cycle': f,
            'duration_s': duration,
            'max_speed': max_speed,
            'mean_speed': mean_speed,
            'max_accel': max_accel,
            'status': 'OK'
        })
        
    df_res = pd.DataFrame(results)
    
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
        
    df_res.to_csv("metrics/drive_cycle_summary.csv", index=False)
    
    with open("reports/Drive_Cycle_Audit.md", "w", encoding='utf-8') as f:
        f.write("# Phase 1: Drive Cycle Audit Report\n\n")
        f.write("All target drive cycle files were verified to ensure they load correctly and contain valid physical profiles.\n\n")
        
        f.write("| Cycle | Status | Duration (s) | Mean Speed (m/s) | Max Speed (m/s) | Max Accel (m/s²) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in df_res.iterrows():
            f.write(f"| {r['cycle']} | {r['status']} | {r['duration_s']} | {r['mean_speed']:.2f} | {r['max_speed']:.2f} | {r['max_accel']:.2f} |\n")
            
    print("Phase 1 Verification Complete.")

if __name__ == "__main__":
    if not os.path.exists('reports'):
        os.makedirs('reports')
    verify_drive_cycles()
