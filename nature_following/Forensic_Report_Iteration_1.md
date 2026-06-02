# Forensic Report - Iteration 1

## 1. Observation Audit (Temperature Bounds)
| controller                           |   max_temp |   mean_temp |
|:-------------------------------------|-----------:|------------:|
| Ablation_MLPOnly                     |    25.3123 |     25.2861 |
| Ablation_NoSpatial                   |    25.2953 |     25.2668 |
| Ablation_NoTemporal                  |    25.2604 |     25.209  |
| MPC_H1_S32                           |    25.8769 |     25.338  |
| MPC_H1_S64                           |   113.176  |    nan      |
| PID_Adaptive                         |    26.2973 |     26.1795 |
| PID_Standard                         |    25.3643 |     25.3406 |
| Proportional_Temp                    |    26.3455 |     26.2437 |
| Proposed_Ablation_NoSpatial_LSTM_PPO |   384.861  |     66.9995 |
| Proposed_Full                        |    43.8543 |     36.0953 |
| Proposed_Full_GraphSAGE_LSTM_PPO     |   400.35   |     75.4245 |
| Uniform_Flow                         |    25.3344 |     25.3129 |

## 2. Action Audit (Saturation)
| controller                           |     pump_mean |     fan_mean |   sat_pct |
|:-------------------------------------|--------------:|-------------:|----------:|
| Ablation_MLPOnly                     |   0.395171    |   0.575821   |         0 |
| Ablation_NoSpatial                   |   0.412708    |   0.630982   |         0 |
| Ablation_NoTemporal                  |   0.107221    |   0.912108   |         0 |
| MPC_H1_S32                           |   0.15013     |   0.559724   |         2 |
| MPC_H1_S64                           | nan           | nan          |       nan |
| PID_Adaptive                         |   0.1         |   0.1        |         0 |
| PID_Standard                         |   0.5         |   0.5        |         0 |
| Proportional_Temp                    |   0.0820589   |   0.0820589  |         1 |
| Proposed_Ablation_NoSpatial_LSTM_PPO | nan           | nan          |       nan |
| Proposed_Full                        |   0.000554249 |   0.00262248 |        99 |
| Proposed_Full_GraphSAGE_LSTM_PPO     | nan           | nan          |       nan |
| Uniform_Flow                         |   0.5         |   0.5        |         0 |

## 3. Reward Audit
| controller                           |        reward_sum |   reward_mean |
|:-------------------------------------|------------------:|--------------:|
| Ablation_MLPOnly                     |      -9.17127     |    -0.0917127 |
| Ablation_NoSpatial                   |      -9.91009     |    -0.0991009 |
| Ablation_NoTemporal                  |      -2.04846     |    -0.0204846 |
| MPC_H1_S32                           |     -16.847       |    -0.16847   |
| MPC_H1_S64                           | -147665           | -2953.31      |
| PID_Adaptive                         |      -4.12331     |    -0.0412331 |
| PID_Standard                         |     -13.6959      |    -0.136959  |
| Proportional_Temp                    |      -3.94895     |    -0.0394895 |
| Proposed_Ablation_NoSpatial_LSTM_PPO | -921913           | -1843.83      |
| Proposed_Full                        |    -455.295       |    -4.55295   |
| Proposed_Full_GraphSAGE_LSTM_PPO     |      -1.17663e+06 | -2353.26      |
| Uniform_Flow                         |     -13.9062      |    -0.139062  |

## 4. Latent Audit
| controller                           |    z_std |     z_max |
|:-------------------------------------|---------:|----------:|
| Ablation_MLPOnly                     |  2.25118 |   8.57955 |
| Ablation_NoSpatial                   |  1.57618 |  11.7137  |
| Ablation_NoTemporal                  | 67.8685  | 472.963   |
| Proposed_Ablation_NoSpatial_LSTM_PPO |  2.33455 |  21.8202  |
| Proposed_Full                        | 52.8932  | 438.674   |
| Proposed_Full_GraphSAGE_LSTM_PPO     | 23.316   | 236.498   |

## 5. Training Audit (Final Epoch Stats)
| model               |   reward |   actor_loss |   critic_loss |   entropy |
|:--------------------|---------:|-------------:|--------------:|----------:|
| Ablation_MLPOnly    | -79.4356 |  -0.00340401 |       58.5856 |   19.86   |
| Ablation_NoSpatial  | -77.702  |  -0.0128428  |       54.3816 |   19.859  |
| Ablation_NoTemporal | -66.1941 |  -0.0217485  |       19.664  |   19.854  |
| Proposed_Full       | -66.8467 |   0.00605184 |       22.7128 |   19.8735 |
