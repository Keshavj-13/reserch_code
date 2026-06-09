# FIGURE TO TEXT COLLISION REPORT

## GOAL
Identify and flag situations where the text simply repeats what is already visible in the figures.

---

### Figure 2: Bar Comparisons (Energy/Temp)
**Collision:** Lines 890-896
**Repetition:** "In contrast, TempProp and PID operated with substantially lower energy footprints of 143.76~Wh and 148.67~Wh respectively... GS+LSTM+PPO achieved a balanced outcome of 158.11~Wh..."
**Verdict:** REPETITION.
**Fix:** The reader can see these bars. Focus on the *ratio* or the *significance* of the gap instead of the raw numbers.

### Figure 3: Temporal Dashboard
**Collision:** Lines 925-949
**Repetition:** "MLP+PPO is 31.35 C. MPC is 30.86 C. PID is 31.21 C... MLP+PPO ends at 154.42 Wh. MPC ends at 169.53 Wh. PID ends at 148.67 Wh."
**Verdict:** SEVERE REPETITION.
**Fix:** This is the worst offender. The text is a list of coordinates. It should describe the *trends* (e.g., "The predictive model exhibits a 'slow-and-steady' cooling profile, avoiding the aggressive spikes seen in PID.")

### Figure 4: Statistical Dashboard
**Collision:** Lines 961-972
**Repetition:** "GS+LSTM+PPO and MPC are 158.11 and 169.53 Wh... TempProp is near 143.76 Wh."
**Verdict:** REPETITION.
**Fix:** Focus on the "Banded" structure mentioned in Line 963. Explain *why* the family structure appears, rather than just stating the values for each family member.

### Figure 5: Spatial Dashboard
**Collision:** Lines 993-1002
**Repetition:** "MPC averages 0.091. MLP+PPO averages 0.005."
**Verdict:** REPETITION.
**Fix:** Describe the *pattern* of the heatmap. Does it show that edge zones are always cooler? Does it show that GS+LSTM+PPO allocates flow where others don't?

### Figure 6: Control Aggressiveness
**Collision:** Lines 1021-1033
**Repetition:** "Maximum zone flow has a wide range for PID. Its range is 0.000 to 1.000. MLP+PPO ranges from 0.000 to 0.691."
**Verdict:** REPETITION.
**Fix:** Use this to discuss "Actuator Fatigue." The smoothness of the commands is the takeaway, not the range of the flow.

---

## SUMMARY
Approximately 40% of the Results section text is currently redundant with the figures. Removing this repetition will improve flow and allow for more room for Discussion/Conclusion.
