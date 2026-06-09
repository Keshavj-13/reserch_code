# Numerical Reconciliation Report

This report summarizes the surgical corrections made to the manuscript text to align with the canonical `recovered_physics_v1` evidence base.

| Text Value | Source Location | Canonical Value | Action Taken |
| :--- | :--- | :--- | :--- |
| 34.737 Wh | Results / Discussion | 158.11 Wh | Corrected to canonical value |
| 35.021 Wh | Results / Discussion | 169.53 Wh | Corrected to canonical value |
| 86.03 Wh | Figure 2 caption | 158.11 Wh | Corrected to canonical value |
| 37.73 C | Figure 2 caption | 35.60 C | Corrected to canonical value |
| >41.3 C | Results / Discussion | 35.66 C | Removed safety violation claim for MLP+PPO |
| 34.852 C | Results | 37.40 C | Corrected MPC peak temperature |
| 39.995 Wh | Results | 169.53 Wh | Corrected MPC energy |
| 30.634 C | Results | 36.46 C | Corrected PID peak temperature |
| 367.121 Wh | Results | 148.67 Wh | Corrected PID energy |
| 33.843 C | Results | 31.35 C | Corrected MLP+PPO mean temperature |
| 32.704 C | Results | 30.86 C | Corrected MPC mean temperature |
| 28.596 C | Results | 31.21 C | Corrected PID mean temperature |
| 32.083 - 35.693 C | Results | 31.21 - 35.60 C | Corrected GS+LSTM+PPO spatial range |
| 31.500 - 33.920 C | Results | 30.86 - 37.40 C | Corrected MPC spatial range |
| 27.852 - 29.402 C | Results | 31.21 - 36.46 C | Corrected PID spatial range |
| 1.070 (std) | Results | 0.55 (mean spread) | Corrected spread metric |
| 1.601 (std) | Results | 2.54 (mean spread) | Corrected spread metric |
| 2.189 (std) | Results | 0.32 (mean spread) | Corrected spread metric |

## Verification of Canonical Figures
- **Figure 1 (Pareto)**: Correctly reflects root `controller_comparison.csv`.
- **Figure 2 (Summary Bars)**: Correctly reflects root `controller_comparison.csv`.
- **Figure 3 (Temporal)**: Correctly reflects root `*_run.csv` files.
- **Figure 4 (Stats)**: Correctly reflects root `controller_comparison.csv`.
- **Figure 5 (Spatial)**: Correctly reflects root `*_run.csv` and `controller_comparison.csv`.
- **Figure 6 (Control)**: Correctly reflects root `*_run.csv` files.
- **Figure 7 (Radar)**: Correctly reflects root `controller_comparison.csv`.
- **Figure 8 (Hotspot)**: Matches the `GS+LSTM+PPO_run.csv` metrics.
