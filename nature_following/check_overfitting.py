import pandas as pd

df = pd.read_csv('metrics/epoch_summary.csv')
for model in df['model'].unique():
    m_df = df[df['model'] == model]
    best_reward = m_df['reward'].max()
    best_epoch = m_df['reward'].idxmax() - m_df.index[0]
    final_reward = m_df['reward'].iloc[-1]
    print(f'{model}: Best Reward {best_reward:.2f} at Epoch {best_epoch}, Final Reward {final_reward:.2f}')
