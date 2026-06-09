# Manuscript Weakness Audit

## 1. Scientific Weaknesses
*   **Numerical Discontinuity**: The single greatest weakness. The manuscript reports different energy values for the same controllers in the same section. This suggests a failure of version control in the data pipeline.
*   **Ablation Integrity**: The "non-spatial" safety violations (>41.3 C) are used to justify GraphSAGE, but the root `MLP+PPO_run.csv` shows 35.66 C (Safe). The claim of safety-dominance for GraphSAGE is contradicted by the raw data in the repository.
*   **Energy Claims**: The claim of 34 Wh for GS+LSTM+PPO is nearly 5x lower than the actual run in the root (158 Wh). This makes the energy efficiency claims appear hyper-inflated.

## 2. Narrative Weaknesses
*   **Figure Narration**: The text is structured as a walkthrough of panels rather than a scientific argument. Example: "Figure 3 uses measured trajectories from MLP+PPO... In panel (a)... In panel (b)..." This should be rewritten to focus on takeaways.
*   **Redundant Methods**: The manuscript contains duplicate sections for Reward Functions and Actor-Critic structures. While intended for clarity, it creates "dead space" that could be used for deeper Results analysis.
*   **Abstract Genericness**: The abstract avoids specific numerical gains, likely because the previous agent could not reconcile the conflicting data branches.

## 3. Reproducibility Weaknesses
*   **Metric Origins**: The "diversity index" and "overhead comparison" numbers appear in the text but are not present in any CSV file in the `metrics/` directory.
*   **Hyperparameter Sync**: While the table exists, the training logs in `metrics/training_log.csv` do not clearly state which set of hyperparameters was used for the "canonical" run.

## 4. Figure Weaknesses
*   **Figure 8 (Hotspot Tracking)**: This is a critical evidence piece for the "spatial" claim but is absent from the TeX.
*   **Radar Plot Consistency**: The radar plot in `replots_from_csv/07_radar_top6.pdf` uses the "34 Wh" data branch, which is inconsistent with the root `_run.csv` artifacts.
*   **Captions**: Captions are purely descriptive ("Temporal dashboard for available measured runs") rather than stating the conclusion ("Predictive control reduces energy while maintaining safety").

## 5. Discussion Weaknesses
*   **Overclaiming**: The discussion claims GS+LSTM+PPO "reduces gradients by >50%", which is true for one data branch but not verified across all evaluated schedules in the root artifacts.
*   **Missing Limitations**: There is no discussion of how the model performs under sensor noise or edge cases, which is a major gap for a "Practical" control claim.
