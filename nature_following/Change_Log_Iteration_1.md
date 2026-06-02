# Change Log - Iteration 1

## Change 1: Speed EWMA Smoothing
*   **File:** `build_pipeline.py` (Cell 2)
*   **Location:** `speed_profile` generation.
*   **Before:** `speed_profile = 15.0 + 10.0 * np.sin(2 * np.pi * t_arr / 300)`
*   **After:** `speed_profile = pd.Series(raw_speed).ewm(alpha=0.25).mean().values`
*   **Reason:** Raw mechanical physics derivatives amplify noise into massive power spikes (~14kW), destabilizing the un-normalized observation space. Smoothing bounds the initial environment input.

## Change 2: Current Clipping
*   **File:** `build_pipeline.py` (Cell 2)
*   **Location:** `battery_thermal_ode` current calculation.
*   **Before:** `I_pack = power_profile[idx] / PACK_NOMINAL_VOLTAGE`
*   **After:** `I_pack = np.clip(I_pack, -1.5 * CELLS_IN_PARALLEL, 6.0 * CELLS_IN_PARALLEL)`
*   **Reason:** Without bounded current, requested power mathematically allows heat generation beyond cell chemistry limits, leading to 192°C thermal runaway.

## Change 3: Linear Reward Formulation
*   **File:** `build_pipeline.py` (Cell 2)
*   **Location:** `get_reward_components` safety and uniformity calculation.
*   **Before:** `r_safe` used `.sum()` and `r_temp` used `np.var()`.
*   **After:** `r_safe` uses `.max()` / `.min()` and `r_temp` uses `np.std()`.
*   **Reason:** The sum and variance operations created astronomical penalties (-985k), leading to $10^{10}$ Critic Loss. The linear formulation bounds the reward scale significantly.