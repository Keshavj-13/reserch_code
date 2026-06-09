# Artifact Lineage Report

## Purpose

This report resolves the artifact conflict found by the manuscript audits. It does not rerun training, replay controllers, regenerate figures, or modify source code. It only reconstructs which files belong to which result branch and which source set should be treated as canonical for the current repository state.

## Executive Finding

The manuscript currently mixes at least three result branches:

1. **Original notebook physics branch** from `C:\Users\keshav\Documents\reserch_code\final_run.ipynb`.
2. **June 1 faulty/rebuilt summary branch** in `metrics/Final_*` and `metrics/Controller_Comparison_Rebuilt.csv`.
3. **June 2 recovered-physics replay branch** in root `*_run.csv`, root `controller_comparison.csv`, `comparison_summary.csv`, and `metrics/v2/Controller_Comparison_Recovered.csv`.

The Gemini forensic audit correctly found inconsistencies, but some labels such as `FABRICATED` are too strong. A more precise status is **untraced in current data artifacts** unless every historical branch, ignored file, and external notebook output has been exhausted.

## Branch Inventory

| Branch | Main files | Timestamp | Physics status | Spread status | Energy example for GS+LSTM+PPO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original notebook | `../final_run.ipynb`, `final_run_extracted_code.py` | Feb 23 / extracted Jun 1 | Original physics source, correct environment reference | Nonzero physics mechanisms present | Does not contain the full final comparison set cleanly |
| Faulty/rebuilt summaries | `metrics/Final_Controller_Comparison.csv`, `metrics/Final_Controller_Ranking.csv`, `metrics/Controller_Comparison_Rebuilt.csv` | Jun 1 22:18 to 23:02 | Reconstructed/faulty branch | `temp_spread_mean = 0.0` for all controllers | 86.034 Wh, 37.736 C |
| Current recovered replay | root `*_run.csv`, root `controller_comparison.csv`, `comparison_summary.csv`, `metrics/v2/Controller_Comparison_Recovered.csv` | Jun 2 06:05 to 06:06 | Generated through `run_phase_h_i_eval.py` using `tags/recovered_physics_v1/core_physics_v1.py` | Nonzero spread | 158.113 Wh, 35.602 C, mean spread 0.551 C |
| Current figure package | `replots_from_csv/*.pdf`, `replots_from_csv/*.png` | Jun 2 06:07 to 06:18 | Generated after current recovered replay | Mostly depends on root CSVs and run files | Should be treated as visually tied to current recovered replay unless individual figure code proves otherwise |
| Alignment prose branch | `reports/Results_Edit_Proposal.md`, `reports/Discussion_Edit_Proposal.md`, `reports/Final_Alignment_Summary.md` | Jun 2 06:44 | Mixed evidence | Uses hardcoded hotspot spreads plus old ranking values | Introduces manuscript prose mixing branch values |

## Generation Chain

### Current recovered replay branch

`run_phase_h_i_eval.py` imports:

- `battery_thermal_ode`
- `get_reward_components`
- `load_original_drive_cycles`
- constants from `tags/recovered_physics_v1/core_physics_v1.py`

It writes:

- root `{Controller}_run.csv`
- root `controller_comparison.csv`
- root `comparison_summary.csv`
- `metrics/v2/Controller_Comparison_Recovered.csv`
- `metrics/v2/Final_Ablation_Summary.csv`

Evidence:

- `run_phase_h_i_eval.py` lines around the replay loop write each trajectory with `pd.DataFrame(traj).to_csv(f"{name}_run.csv")`.
- The same script computes `cooling_energy_Wh = energy_sum / 3600.0`.
- It then writes `metrics/v2/Controller_Comparison_Recovered.csv`, `controller_comparison.csv`, and `comparison_summary.csv`.

### Faulty/rebuilt summary branch

`generate_comparison.py` reads whatever `controller_comparison.csv` existed at the time and writes:

- `metrics/Final_Controller_Comparison.csv`
- `metrics/Final_Controller_Ranking.csv`

Those files currently contain zero spatial spread and June 1 values, so they are stale relative to the June 2 recovered replay branch.

### Mixed alignment branch

`run_alignment.py` reads:

- `metrics/Final_Controller_Ranking.csv` from the June 1 branch.
- `metrics/v2/Final_Ablation_Summary.csv` from the June 2 branch.
- hardcoded hotspot spread values copied from the hotspot analysis.

This is the most direct cause of the manuscript's mixed-branch narrative: it combines stale energy/temperature rankings with recovered spread evidence.

## Disputed Metric Resolution

### GS+LSTM+PPO energy: 86.03 Wh vs 158.11 Wh

| Value | Source | Meaning | Status |
| :--- | :--- | :--- | :--- |
| 86.034 Wh | `metrics/Final_Controller_Ranking.csv` and `metrics/Final_Controller_Comparison.csv` | Stale June 1 summary energy | Valid within stale branch, not valid for current recovered branch |
| 158.113 Wh | root `GS+LSTM+PPO_run.csv`, root `controller_comparison.csv`, `metrics/v2/Controller_Comparison_Recovered.csv` | Current recovered replay integrated pump proxy energy | Canonical if using recovered-physics replay branch |

Conclusion: this is a **branch conflict**, not proof that either number is fabricated. The current repository state supports 158.113 Wh for the recovered replay branch.

### GS+LSTM+PPO spatial spread

| Value | Source | Status |
| :--- | :--- | :--- |
| mean spread 0.551 C | root `GS+LSTM+PPO_run.csv`, `controller_comparison.csv`, `metrics/v2/Controller_Comparison_Recovered.csv` | Current recovered branch match |
| max spread 1.071 C | direct calculation from root `GS+LSTM+PPO_run.csv`; `Hotspot_Analysis_Report.md` | Current recovered branch match |
| spread 0.0 | `metrics/Final_Controller_Comparison.csv` | Stale faulty branch |

Conclusion: nonzero spatial spread should be taken from the current recovered replay branch. The zero-spread metrics should not be used for spatial claims.

### 34.737 Wh and 35.021 Wh

These values occur in `main.tex` and alignment/proposal prose, but were not found in the current summary CSVs checked during this lineage pass.

Status: **untraced in current repository metric artifacts**.

Recommended wording: do not call these values fabricated unless the full historical workspace and any external notebook outputs are exhaustively searched. For manuscript purposes, remove or replace them because they are not traceable to the selected canonical source set.

### MLP+PPO and LSTM+PPO safety violation contradiction

| Claim | Source branch | Current recovered branch value |
| :--- | :--- | :--- |
| MLP+PPO and LSTM+PPO exceed 41.3 C | `metrics/Final_Controller_Ranking.csv` / stale June 1 branch | Not supported |
| MLP+PPO max temp 35.664 C; LSTM+PPO max temp 35.136 C | root `*_run.csv`, `controller_comparison.csv`, `metrics/v2/Controller_Comparison_Recovered.csv` | Supported |

Conclusion: if the canonical branch is the current recovered replay branch, manuscript claims that MLP+PPO and LSTM+PPO have safety violations must be removed. If the old branch is chosen for ablation safety claims, then its zero-spread physics invalidates the spatial narrative. The manuscript cannot defensibly use both at once.

## Canonical Source Recommendation

### Recommended canonical source set for current manuscript repair

Use only:

- `tags/recovered_physics_v1/core_physics_v1.py`
- `run_phase_g_train.py`
- `run_phase_h_i_eval.py`
- root `*_run.csv`
- root `controller_comparison.csv`
- root `comparison_summary.csv`
- `metrics/v2/Controller_Comparison_Recovered.csv`
- `metrics/v2/Final_Ablation_Summary.csv`
- `replots_from_csv/*.pdf` and `replots_from_csv/*.png` only if confirmed to have been generated after the root CSVs and not from stale `metrics/Final_*` data.

Do not use for final claims:

- `metrics/Final_Controller_Comparison.csv`
- `metrics/Final_Controller_Ranking.csv`
- `metrics/Controller_Comparison_Rebuilt.csv`
- `metrics/Final_Ablation_Summary.csv`
- `reports/Results_Edit_Proposal.md`
- `reports/Discussion_Edit_Proposal.md`
- `reports/Final_Alignment_Summary.md`
- any text claiming 34.737 Wh or 35.021 Wh unless a traceable source is found.

## Consequences for Manuscript Claims

If the recommended canonical source set is accepted, the manuscript must be revised as follows:

1. Replace GS+LSTM+PPO energy `86.03 Wh` with `158.11 Wh`.
2. Replace MPC energy `35.021 Wh` and GS+LSTM+PPO energy `34.737 Wh` with current recovered values or remove those sentences.
3. Remove claims that MLP+PPO and LSTM+PPO have safety violations in the recovered branch.
4. Reframe ablation conclusions:
   - Current recovered branch does not support "GS+LSTM+PPO is strongest learned controller by reward."
   - Current recovered branch does support "GS+LSTM+PPO reduces max spread versus AdaptivePID and GS+PPO."
   - Current recovered branch does not support "all non-spatial learned models are unsafe."
5. Treat hotspot tracking as evidence for the subset it actually compares: AdaptivePID, GS+LSTM+PPO, and GS+PPO.
6. Do not use stale zero-spread `metrics/Final_*` files for any spatial claim.

## Final Recommendation

Canonical Source Set:

`{tags/recovered_physics_v1/core_physics_v1.py, run_phase_g_train.py, run_phase_h_i_eval.py, root *_run.csv, root controller_comparison.csv, root comparison_summary.csv, metrics/v2/Controller_Comparison_Recovered.csv, metrics/v2/Final_Ablation_Summary.csv}`

Final status:

**NOT READY FOR PROFESSOR REVIEW until `main.tex` is rewritten against the canonical source set.**

The central unresolved issue is no longer whether an inconsistency exists. It does. The central decision is whether the current recovered replay branch is accepted as canonical. If yes, the manuscript narrative must become a spatial-uniformity tradeoff paper, not a universal learned-controller superiority paper.

