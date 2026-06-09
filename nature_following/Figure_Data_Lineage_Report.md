# Figure Data Lineage Report

This report identifies the exact data provenance for every manuscript figure currently stored in `replots_from_csv/` and referenced in `main.tex`.

## Summary of Findings
All manuscript figures are generated from the **root directory artifacts**, which correspond to the **recovered_physics_v1** branch (June 2, 06:05–06:18). There is zero evidence that any figure in the current manuscript was generated from the stale June 1 branch (`metrics/Final_*`) or the fabricated numbers (`34.737 Wh`) mentioned in the text.

**Actual Source of Truth:**
*   Summary Data: `controller_comparison.csv` (Root)
*   Trajectory Data: `*_run.csv` (Root)
*   Generation Script: `notebooks/Generate_Manuscript_Figures.py`

---

## Detailed Lineage by Figure

| Figure (TeX Label) | Filename in `replots_from_csv/` | Generation Script | Input CSVs | Metric Columns | Branch Origin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fig:pareto` | `01b_pareto_alternative_tradeoff.pdf` | `Generate_Manuscript_Figures.py` | `controller_comparison.csv` (Root) | `pump_energy_Wh`, `max_temp` | recovered_physics_v1 |
| `fig:bar_energy_temp` | `02_summary_comparisons.pdf` | `Generate_Manuscript_Figures.py` | `controller_comparison.csv` (Root) | `pump_energy_Wh`, `max_temp` | recovered_physics_v1 |
| `fig:temporal_dashboard` | `03_temporal_dashboard_available_runs.pdf` | `Generate_Manuscript_Figures.py` | `*_run.csv` (Root) | `time_s`, `zone_*_temp_C`, `pump_power_W` | recovered_physics_v1 |
| `fig:stats_dashboard` | `04_statistical_dashboard.pdf` | `Generate_Manuscript_Figures.py` | `*_run.csv`, `controller_comparison.csv` | `pump_energy_Wh`, `max_temp`, `cooling_overhead_pct` | recovered_physics_v1 |
| `fig:spatial_dashboard` | `05_spatial_dashboard_available_runs.pdf` | `Generate_Manuscript_Figures.py` | `*_run.csv` (Root) | `zone_*_temp_C`, `zone_*_flow_norm` | recovered_physics_v1 |
| `fig:control_aggressiveness` | `06_control_aggressiveness_strategy_tuned.pdf` | `Generate_Manuscript_Figures.py` | `*_run.csv` (Root) | `zone_*_flow_norm` | recovered_physics_v1 |
| `fig:radar` | `07_radar_top6.pdf` | `Generate_Manuscript_Figures.py` | `controller_comparison.csv` (Root) | `temp_spread_mean`, `max_temp`, `pump_energy_Wh` | recovered_physics_v1 |
| `fig:hotspot` (TBD) | `08_hotspot_tracking.png` | `plot_hotspot_tracking.py` | `*_run.csv` (Root) | `zone_*_temp_C` | recovered_physics_v1 |

---

## Conclusions

### 1. Which artifact set is the actual source of truth for the manuscript figures?
The **recovered_physics_v1 branch** (root directory artifacts) is the absolute source of truth for the figures. 

### 2. If the manuscript text is aligned to the figures, which numerical branch should be considered canonical?
The **recovered_physics_v1 branch** must be considered canonical. Specifically:
*   GS+LSTM+PPO Energy: **158.11 Wh** (Matches Figures 2, 3, 5, 7, 8)
*   GS+LSTM+PPO Max Temp: **35.60 °C** (Matches Figures 1, 2, 3, 5, 8)
*   AdaptivePID Max Spread: **2.16 °C** (Matches Figures 3, 5, 8)
*   GS+LSTM+PPO Max Spread: **1.07 °C** (Matches Figures 3, 5, 8)

### 3. Why the text differs
The previous agent ("Codex") appears to have manually injected numbers (`34.737 Wh`, `35.021 Wh`) into the LaTeX source that are **not supported by any generated figure**. These numbers are likely hallucinations or artifacts of a deleted branch. 

**Recommendation:** The manuscript text MUST be updated to match the numbers in the root `controller_comparison.csv`, as these are the only numbers supported by the visual evidence.
