# ActionSpace_Conflict_Report.md

**Date**: June 3, 2026  
**Status**: DRAFT  

---

## 1. CONFLICTING DESCRIPTIONS

### Location 1
Section 2.2 (Environment), Page 7  
**Exact Text**: "Action dimension equals the number of independent zone valve channels. Aggregate pump power is reconstructed from the mean valve opening for energy accounting."  
**Reason It Conflicts**: The verified implementation uses a 14-DOF action space where pump and fan are independent learned outputs, not reconstructed from valves.

### Location 2
Table 1 Caption, Page 7  
**Exact Text**: "All controllers output twelve normalized zone valve commands."  
**Reason It Conflicts**: The controllers output 14 commands (12 valves + 1 pump + 1 fan).

### Location 3
Table 2, Page 9  
**Exact Text**: "Mean valve opening penalty" (as the source for $w_{\mathrm{energy}}$)  
**Reason It Conflicts**: The reward function $r_{\mathrm{energy}}$ is defined as $-\lambda_4 P_{\mathrm{aux}}$, which depends on the learned pump and fan actions, not just mean valve opening.

### Location 4
Section 2.11 (Thermal Dynamics), Page 15  
**Exact Text**: "aggregate pump power is reconstructed from the mean valve opening during post processing."  
**Reason It Conflicts**: In the implementation, pump power is an independent degree of freedom controlled by the actor.

### Location 5
Implementation notes (end of Section 2), Page 16  
**Exact Text**: "Cooling uses lumped conductance tied to zone valve commands; pump power is reconstructed from mean valve opening."  
**Reason It Conflicts**: Conflicts with the 14-DOF implementation where pump and fan are active control variables.

---
