import pandas as pd

def generate_final_comparison():
    print("Generating Final Controller Comparison and Ranking...")
    
    df = pd.read_csv("controller_comparison.csv")
    
    df.to_csv("metrics/Final_Controller_Comparison.csv", index=False)
    
    # Generate ranking based on a combined score (e.g., maximizing reward)
    # The higher the reward, the better the rank.
    df['rank'] = df['reward'].rank(ascending=False, method='min').astype(int)
    
    df_ranked = df.sort_values(by='rank')
    
    # Select columns for ranking report
    cols = ['rank', 'label', 'reward', 'max_temp', 'pump_energy_Wh']
    df_ranked[cols].to_csv("metrics/Final_Controller_Ranking.csv", index=False)
    
    print("Final comparison and ranking complete.")

if __name__ == "__main__":
    generate_final_comparison()