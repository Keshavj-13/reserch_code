# Current Todo List

## 1. Reward Penalty Explosion
*   **Issue:** Reward penalty for safety (`r_safe`) and uniformity (`r_temp`) use `sum()` and `var()`, creating quadratic/multiplicative explosions (-985k penalties).
*   **Evidence:** `Reward_Audit.csv` shows 99.8% reward dominance by `r_safe`.
*   **Source Report:** Forensic Execution Report.
*   **Confidence:** HIGH
*   **Expected Impact:** Resolves astronomical 10^10 Critic Loss.
*   **Rank:** CRITICAL

## 2. Thermal Runaway via Input Magnitude
*   **Issue:** Physics model calculates current without limits, resulting in raw mechanical spikes reaching 192°C thermal runaway.
*   **Evidence:** `Observation_Audit.csv` shows Max Temp of 192.53°C. `Latent_Audit.csv` shows standard deviations > 2000.
*   **Source Report:** Forensic Execution Report & Knowledge Ledger.
*   **Confidence:** HIGH
*   **Expected Impact:** Limits observation magnitudes, resolving Latent Explosion and bounding environmental physics.
*   **Rank:** CRITICAL

## 3. Discontinuous Power Spikes
*   **Issue:** Derivative of noisy velocity creates massive `accel` spikes in power generation.
*   **Evidence:** `Observation_Audit.csv` shows P_max = 14276.82W with extreme step-to-step volatility.
*   **Source Report:** Forensic Knowledge Ledger.
*   **Confidence:** HIGH
*   **Expected Impact:** Stabilizes baseline power profile, further bounding input features.
*   **Rank:** CRITICAL

## 4. Normal vs Clipped Action Distribution
*   **Issue:** PPO calculates log-probabilities on unbounded Normal distributions, but actions are clipped in physics, decoupling learning from reality.
*   **Evidence:** Step 1 divergence into zero-action (Action Bias Sinking).
*   **Source Report:** Collapse Report.
*   **Confidence:** HIGH
*   **Expected Impact:** Resolves PPO Actor collapse.
*   **Rank:** HIGH