# Forensic Report - Iteration 2

## 1. Observation Audit (Temperature Bounds)
| controller          |   max_temp |   mean_temp |
|:--------------------|-----------:|------------:|
| Ablation_MLPOnly    |    25.3297 |     25.3082 |
| Ablation_NoSpatial  |    25.332  |     25.3105 |
| Ablation_NoTemporal |    25.307  |     25.2847 |
| MPC_H1_S32          |    25.8769 |     25.338  |
| PID_Adaptive        |    26.2973 |     26.1795 |
| PID_Standard        |    25.3643 |     25.3406 |
| Proportional_Temp   |    26.3455 |     26.2437 |
| Proposed_Full       |    25.3736 |     25.3465 |
| Uniform_Flow        |    25.3344 |     25.3129 |

## 2. Action Audit (Saturation)
| controller          |   pump_mean |   sat_pct |
|:--------------------|------------:|----------:|
| Ablation_MLPOnly    |   0.48992   |         0 |
| Ablation_NoSpatial  |   0.480426  |         0 |
| Ablation_NoTemporal |   0.415042  |         0 |
| MPC_H1_S32          |   0.15013   |         2 |
| PID_Adaptive        |   0.1       |         0 |
| PID_Standard        |   0.5       |         0 |
| Proportional_Temp   |   0.0820589 |         1 |
| Proposed_Full       |   0.389976  |         0 |
| Uniform_Flow        |   0.5       |         0 |

## 3. Reward Audit
| controller          |   reward_sum |
|:--------------------|-------------:|
| Ablation_MLPOnly    |    -13.403   |
| Ablation_NoSpatial  |    -12.948   |
| Ablation_NoTemporal |    -10.0127  |
| MPC_H1_S32          |    -16.847   |
| PID_Adaptive        |     -4.12331 |
| PID_Standard        |    -13.6959  |
| Proportional_Temp   |     -3.94895 |
| Proposed_Full       |     -9.16581 |
| Uniform_Flow        |    -13.9062  |

## 4. Latent Audit
| controller          |    z_std |    z_max |
|:--------------------|---------:|---------:|
| Ablation_MLPOnly    | 0.11572  | 0.556049 |
| Ablation_NoSpatial  | 0.200152 | 0.423468 |
| Ablation_NoTemporal | 0.985011 | 4.28799  |
| Proposed_Full       | 0.809764 | 4.39392  |

## 5. Training Audit
| model               |   reward |   actor_loss |   critic_loss |   entropy |
|:--------------------|---------:|-------------:|--------------:|----------:|
| Ablation_MLPOnly    | -80.7098 | -0.000431037 |       67.5612 |   19.8607 |
| Ablation_NoSpatial  | -78.8894 | -0.000924264 |       64.6247 |   19.8588 |
| Ablation_NoTemporal | -75.3254 |  0.000853609 |       54.2102 |   19.8579 |
| Proposed_Full       | -77.1884 | -0.000591815 |       55.8606 |   19.8624 |
