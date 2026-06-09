# MANUSCRIPT EVIDENCE AUDIT (NO EDITS ALLOWED)

**Date**: June 3, 2026  
**Status**: Detection Only  
**Scope**: `main.tex` vs Workspace Data Artifacts  

---

## 1. EXECUTIVE SUMMARY

The manuscript presents a compelling case for "balanced" predictive control, particularly focusing on spatial uniformity. However, significant internal contradictions exist regarding the control architecture (specifically the action space dimensionality) and the derivation of auxiliary actuator commands. While the primary claim of spread reduction (2.16 °C to 1.07 °C) is well-supported by recovered data artifacts, several secondary claims regarding the "safety" of ablation models and the "superiority" of the learned controller's efficiency are potentially overstated when compared against the full baseline suite (e.g., PID/TempProp). The manuscript is currently in a state that hybridizes two different implementation versions (12-DOF vs 14-DOF), which represents a high risk for reviewer scrutiny.

---

## 2. NOTATION CONSISTENCY FINDINGS

### Location A
`x^{\mathrm{edge}}_{i\to j} = [d_{ij},\ k_{ij}^{\mathrm{cond}}]` (Eq. 4, Page 4)

### Location B
`\sum_{k \in \mathcal{N}(j)} k_{jk}(T_{b,k} - T_{b,j})` (Eq. 26, Page 6)

### Assessment
* **DEFINITELY SAME QUANTITY**

### Suggested Action
A human author should unify the notation. Prefer $k_{jk}^{\mathrm{cond}}$ in the dynamics equation to maintain consistency with the input feature definitions.

---

### Location A
`\alpha_i = \sigma(\tilde \alpha_i)` (Eq. 22, Page 6)

### Location B
"valve opening is $\alpha_j \in [0, 1]$" (Eq. 31, Page 7)

### Assessment
* **DEFINITELY SAME QUANTITY** (Note the index change $i$ vs $j$)

### Suggested Action
Ensure the index ($i$ vs $j$) is consistent throughout the Methods section. $j$ is used for zones in most other equations.

---

## 3. CLAIMS VS EVIDENCE FINDINGS

### Claim
"Across the evaluated schedules, the proposed policy ... reduces the maximum inter-zone temperature spread from 2.16 °C to 1.07 °C" (Abstract; Results 3.2.1, 3.2.3, 3.2.8; Discussion)

### Evidence Used
`Figure_Data_Lineage_Report.md`: "AdaptivePID Max Spread: 2.16 °C", "GS+LSTM+PPO Max Spread: 1.07 °C".  
`Controller_Comparison_Recovered.csv`: Reports mean spread (0.55 for GS+LSTM+PPO), but max spread in `run_alignment.py` matches the manuscript.

### Assessment
* **SUPPORTED**

### Suggested Action
None. This is the strongest and most consistent evidence in the paper.

---

### Claim
"Crucially, all learned models, including the MLP+PPO and LSTM+PPO ablations, maintain temperatures well within the 40 °C safety limit." (Results 3.2.1, Page 8)

### Evidence Used
`Final_Controller_Comparison.csv` (Root): MLP+PPO Max Temp = 41.3 °C; LSTM+PPO Max Temp = 41.6 °C.  
`Controller_Comparison_Recovered.csv` (V2): MLP+PPO Max Temp = 35.66 °C; LSTM+PPO Max Temp = 35.13 °C.

### Assessment
* **PARTIALLY SUPPORTED / OVERSTATED**

### Suggested Rewording
If the root CSV represents the most recent training run, the claim is factually incorrect. If the "Recovered" CSV is intended, the claim stands. However, the phrase "well within" is risky if even one schedule (e.g., US06) approaches the limit. Suggest: "All learned models maintained temperatures below the 40 °C threshold across the majority of test cases, with the proposed integrated architecture providing the most robust safety margins."

---

### Claim
"GS+LSTM+PPO and its ablations anchor the efficiency and pump energy axes" (Results 3.2.7, Page 9)

### Evidence Used
`Controller_Comparison_Recovered.csv`:  
Energy Ranks:  
1. TempProp (143 Wh)  
2. PID (148 Wh)  
3. MLP+PPO (154 Wh)  
4. GS+PPO (154 Wh)  
5. GS+LSTM+PPO (158 Wh)

### Assessment
* **OVERSTATED**

### Suggested Rewording
The proposed controller is *less* efficient than simple PID and TempProp baselines in terms of absolute Wh. It is only "efficient" relative to the brute-force AdaptivePID/UniformFlow. Suggest: "GS+LSTM+PPO provides a significantly more energy-efficient alternative to classical global cooling baselines while matching their peak temperature performance."

---

## 4. SIGMOID BOUNDARY FINDINGS

### Relevant Text
"All actuator outputs are projected into physically admissible ranges by differentiable squashing functions... Valve openings, pump flow rate, and fan speed are each constrained by separate mappings [Eq 22-24] ... Hardware safety checks provide a final clamp after squashing to catch residual violations." (Section 2.2, 2.10)

### Reviewer Concern
The manuscript describes a 14-dimensional action space (12 valves + 1 pump + 1 fan) in Equations 22-24, but Table 1 states: "All controllers output twelve normalized zone valve commands." Furthermore, Section 2.11 states: "pump power is reconstructed from mean valve opening." 

If the pump is reconstructed from valves, then Equation 23 is a ghost equation and the system is not 3-DOF as claimed in Section 2.2. A reviewer will likely ask how the "reconstruction" relates to the "sigmoid mapping" in Eq 23.

### Suggested Clarification
Decide if the pump/fan are independent learned degrees of freedom or derived proxies. If they are derived, remove Eq 23-24 and clearly state the pump/fan power derivation in the Reward Function section.

---

## 5. AI STYLE FINDINGS

### Sentence
"The significance of this work lies in integration." (Intro, Page 3)

### Why It Sounds Generic
"The significance of this work lies in..." is a hallmark of LLM-generated summaries. It lacks the technical specificity of a human-authored contribution statement.

### More Natural Scientific Style
"We demonstrate that the end-to-end integration of spatial and temporal encoders into the policy loop is necessary for anticipatory multi-zone coordination."

---

### Sentence
"Our reading of this gap is straightforward..." (Intro, Page 3)

### Why It Sounds Generic
"Our reading of this [X] is straightforward" is conversational and "coach-like" rather than scientific.

### More Natural Scientific Style
"We hypothesize that the typical separation of prediction and control modules limits the policy's ability to [X]..."

---

## 6. NARRATIVE MISALIGNMENT ON SUPERIORITY

### Text
"Overall, the results indicate that proactive predictive control is a viable and efficient route for real multi zone battery systems." (Abstract)

### Why It Conflicts With Results
The Pareto analysis (Fig 2) shows GS+LSTM+PPO is the 5th most efficient controller. Calling it "the efficient route" ignores that PID is 6% more efficient. The true contribution is the **Spread-Efficiency Tradeoff**, not absolute energy superiority.

### Suggested Positioning
Focus on the **Pareto Optimal Balancing**. A reviewer would prefer: "The proposed architecture achieves a Pareto-optimal balance, providing the highest degree of spatial uniformity among all evaluated methods with competitive energy consumption."

---

## 7. HIGHEST RISK ISSUES BEFORE SUBMISSION

1.  **Action Space Contradiction**: The manuscript vacillates between a 12-action and 14-action model. Eq 22-24 vs Table 1 vs Section 2.11. This is the most likely reason for a "Major Revision" or "Reject".
2.  **Energy Superiority Claim**: The text repeatedly implies the learned model is an energy leader, while the data shows it is mid-tier (Rank 5).
3.  **Ablation Safety**: The claim that MLP+PPO and LSTM+PPO are "well within" 40 °C is contradicted by the `Final_Controller_Comparison.csv` file in the root directory (which shows violations).
4.  **Notation Divergence**: $k_{jk}$ vs $k_{ij}^{\mathrm{cond}}$.

---
**ABSOLUTE RULE CHECK**: No manuscript files were modified. Only this report was generated.
