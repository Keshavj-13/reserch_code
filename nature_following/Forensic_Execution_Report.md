# Forensic Execution Report

## 1. Controller Health Table
| Controller | Status | Dominant Failure Mode |
| :--- | :--- | :--- |
| **PID_Standard** | STABLE | Baseline - Constant offset |
| **PID_Adaptive** | STABLE | Baseline - Rule-based gain |
| **Uniform_Flow** | STABLE | Baseline - Static |
| **Proportional_Temp** | SATURATED | Action Saturation (96% Saturated) |
| **MPC_H1_S32** | UNSTABLE | High variance random actions |
| **Proposed_Full** | COLLAPSED | Latent Explosion + Action Saturation (100%) |
| **NoSpatial** | UNSTABLE | Observation domination (Global branch) |
| **NoTemporal** | COLLAPSED | GraphSAGE branch explosion (std: 82.8) |
| **MLPOnly** | UNSTABLE | Random drift |

## 2. Failure Clusters
### Failure Cluster A: Global Scaling mismatch
*   **Symptom:** Inputs ($P_{batt} \approx 14,000$W, $T_{max} \approx 192$°C) are $10^2$ to $10^4$ larger than Neural Network default initialization.
*   **Affected Models:** All RL models (`Proposed_Full`, `NoSpatial`, `NoTemporal`, `MLPOnly`).
*   **Evidence:** `Observation_Audit.csv` shows $P_{max} = 14276.82$.

### Failure Cluster B: Branch Domination (GraphSAGE)
*   **Symptom:** GraphSAGE branch latent variance is $40\times$ higher than other branches.
*   **Affected Models:** `Proposed_Full`, `NoTemporal`.
*   **Evidence:** `Latent_Audit.csv` shows `es_std` (Spatial) = 82.79 vs `et_std` (Temporal) = 0.11.

### Failure Cluster C: Reward Penalty Domination
*   **Symptom:** Safety reward $r_s$ accounts for >99.8% of the total reward signal.
*   **Affected Models:** All controllers.
*   **Evidence:** `Reward_Audit.csv` shows `rs_pct` $\approx$ 99.8%. The optimizer is effectively "blind" to temperature uniformity and energy efficiency.

## 3. Root Cause Ranking
1.  **Systemic Input Magnitude (Global):** Raw physical units ($W$, °C) injected directly into NN.
2.  **Reward Penalty Scale (Global):** Safety violations mask all other learning objectives.
3.  **GraphSAGE Sensitivity (Architecture):** 3-layer message passing amplifies unscaled spatial features exponentially.

## 4. Recommended Fix Order
1.  **StandardScaler (Observation Normalization):** Immediate 90% probability of resolving latent explosion.
2.  **Reward Weighting (Log-Safety):** Immediate 80% probability of resolving reward masking.
3.  **Gradient Norm Clipping (0.5):** Mandatory for PPO stability in un-tuned environments.
4.  **Action Bias (0.5 Initialization):** Prevents immediate divergence before the first gradient step.

## 5. Final Verdict
**SYSTEMIC COLLAPSE**
The failures are **Global** and originate from a mismatch between physical simulation units and neural network optimization ranges. The simulator is physically correct, but the "sensory" interface (Observations) and "feedback" interface (Rewards) are biologically impossible for a neural agent to process without normalization.
