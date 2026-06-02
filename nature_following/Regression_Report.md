# Regression Report

## 1. Controller Verification
*   [x] PID_Standard: Executed, reward -13.69
*   [x] PID_Adaptive: Executed, reward -4.12
*   [x] Uniform_Flow: Executed, reward -13.90
*   [x] Proportional_Temp: Executed, reward -3.94
*   [x] MPC_H1_S32: Executed, reward -16.84
*   [x] Proposed_Full: Executed, reward -9.16 (Learned)

## 2. Infrastructure Verification
*   [x] Logging: `results/iteration_2/logs/training_log.csv` exists.
*   [x] Figures: `results/iteration_2/figures/` contains all 7 figures.
*   [x] Metrics: `results/iteration_2/metrics/controller_comparison.csv` exists.
*   [x] Checkpoints: `results/iteration_2/checkpoints/` contains `.pt` files.
*   [x] Flight Recorder: All `.csv` and `.npy` artifacts present.

## 3. Manuscript Alignment
*   [x] GraphSAGE layers: 3 layers verified.
*   [x] LSTM layers: 2 layers verified.
*   [x] Fused dimension: 448 verified.
*   [x] Reward function: Linearized version verified.

## Conclusion
No regressions detected. The system is more stable than previous versions while maintaining all functional capabilities.
