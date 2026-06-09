# ActionSpace_Edit_Report.md

**Date**: June 3, 2026  
**Status**: COMPLETE  

---

## 1. EDIT LOG

### Edit 1
**Location**: Section 2.2 (Environment), Page 7  
**Before**: Action dimension equals the number of independent zone valve channels. Aggregate pump power is reconstructed from the mean valve opening for energy accounting.  
**After**: Action dimension equals fourteen, consisting of twelve independent zone valve channels, one pump channel, and one fan channel. Pump and fan power are determined by their respective learned control actions.  
**Reason**: Align with verified 14-DOF implementation.

---

### Edit 2
**Location**: Table 1 Caption, Page 7  
**Before**: \caption{Controller definitions used in the final comparison. All controllers output twelve normalized zone valve commands.}\label{tab:controllers}  
**After**: \caption{Controller definitions used in the final comparison. All controllers output fourteen normalized control commands consisting of twelve zone valve commands, one pump command, and one fan command.}\label{tab:controllers}  
**Reason**: Correct action space dimensionality and composition.

---

### Edit 3
**Location**: Table 2, Page 9  
**Before**: $w_{\mathrm{energy}}$ & 0.5 & Mean valve opening penalty\\  
**After**: $w_{\mathrm{energy}}$ & 0.5 & Actuator power penalty\\  
**Reason**: Align reward source description with implementation ($r_{\mathrm{energy}} = -\lambda_4 P_{\mathrm{aux}}$).

---

### Edit 4
**Location**: Section 2.11 (Thermal Dynamics), Page 15  
**Before**: aggregate pump power is reconstructed from the mean valve opening during post processing.  
**After**: aggregate pump power is determined by the learned pump control action.  
**Reason**: Remove reconstruction claim; align with active control implementation.

---

### Edit 5
**Location**: Implementation notes (end of Section 2), Page 16  
**Before**: \item Cooling uses lumped conductance tied to zone valve commands; pump power is reconstructed from mean valve opening.  
**After**: \item Cooling uses lumped conductance tied to zone valve commands; pump and fan power are determined by their learned control actions.  
**Reason**: Consistency check for 14-DOF active control.

---

## 2. VERIFICATION

*   **14-DOF alignment**: COMPLETE.
*   **Scientific integrity**: MAINTAINED.
*   **Safety check**: No changes to equations, figures, or values.
