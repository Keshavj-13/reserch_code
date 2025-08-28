# Battery Pack Thermal Simulation - Realistic Scaling
# Designed to work with real EV power datasets and scale properly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
import os
from datetime import datetime


# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

# Set your paths here
DRIVE_CYCLE_FOLDER = "/home/keshav/Documents"
CORE_CSV_PATH = "/home/keshav/Documents/reserch_code/core.csv"
OUTPUT_PATH = "/home/keshav/Documents/reserch_code/simulated_battery_cooling.csv"

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Simulation control
FORCE_REBUILD_CORE = True
ADIABATIC_MODE = False  # Set True to disable cooling


# =============================================================================
# REALISTIC BATTERY PACK CONFIGURATION
# =============================================================================

# Pack architecture - scales everything properly
CELLS_IN_SERIES = 96      # Typical EV pack (96s)
CELLS_IN_PARALLEL = 4     # 4 cells in parallel per series group
TOTAL_CELLS = CELLS_IN_SERIES * CELLS_IN_PARALLEL
NUM_ZONES = 8            # Thermal zones in the pack

# Single cell properties (based on typical 18650/21700 cells)
CELL_NOMINAL_VOLTAGE = 3.7  # V
CELL_CAPACITY = 3.0         # Ah
CELL_INTERNAL_R = 0.030     # Ohms (single cell)
CELL_HEAT_CAPACITY = 50.0   # J/K (single cell thermal mass)
CELL_MAX_DISCHARGE_RATE = 3.0  # C-rate

# Derived pack properties
PACK_NOMINAL_VOLTAGE = CELLS_IN_SERIES * CELL_NOMINAL_VOLTAGE  # ~355V
PACK_CAPACITY = CELLS_IN_PARALLEL * CELL_CAPACITY  # 12 Ah
PACK_ENERGY = PACK_NOMINAL_VOLTAGE * PACK_CAPACITY / 1000  # kWh
PACK_MAX_POWER = PACK_NOMINAL_VOLTAGE * PACK_CAPACITY * CELL_MAX_DISCHARGE_RATE / 1000  # kW

print(f"🔋 BATTERY PACK SPECIFICATIONS:")
print(f"   Configuration: {CELLS_IN_SERIES}s{CELLS_IN_PARALLEL}p = {TOTAL_CELLS} cells")
print(f"   Pack voltage: {PACK_NOMINAL_VOLTAGE:.1f} V")
print(f"   Pack capacity: {PACK_CAPACITY:.1f} Ah")
print(f"   Pack energy: {PACK_ENERGY:.1f} kWh")
print(f"   Max continuous power: {PACK_MAX_POWER:.0f} kW")
print(f"   Thermal zones: {NUM_ZONES}")


# =============================================================================
# ZONE-BASED PACK PROPERTIES WITH REALISTIC VARIATION
# =============================================================================

np.random.seed(42)  # Reproducible results

# Cells per zone
cells_per_zone = TOTAL_CELLS // NUM_ZONES
cells_in_zone = np.full(NUM_ZONES, cells_per_zone)
# Distribute remainder cells
remainder = TOTAL_CELLS % NUM_ZONES
cells_in_zone[:remainder] += 1

# Zone internal resistance (parallel cells reduce resistance)
# Manufacturing variation: ±5% on cell resistance
cell_r_variation = np.random.normal(1.0, 0.05, NUM_ZONES)
zone_resistance = np.zeros(NUM_ZONES)

for i in range(NUM_ZONES):
    # Series resistance of cells in zone, parallel combination with other parallel groups
    cells_in_series_per_zone = CELLS_IN_SERIES
    cells_in_parallel_per_zone = cells_in_zone[i] // CELLS_IN_SERIES
    
    # Resistance: R_cell * n_series / n_parallel
    zone_resistance[i] = (CELL_INTERNAL_R * cell_r_variation[i] * CELLS_IN_SERIES) / CELLS_IN_PARALLEL

# Zone heat capacity (sum of all cells in zone)
zone_heat_capacity = cells_in_zone * CELL_HEAT_CAPACITY

# Zone-to-zone thermal conductance (realistic for battery pack)
# Higher conductance at center zones, lower at edges
zone_positions = np.linspace(0, 1, NUM_ZONES)
lateral_conductance = 15.0 * (1 - 0.3 * np.abs(zone_positions - 0.5) * 2)  # W/K

print(f"📊 ZONE DISTRIBUTION:")
for i in range(NUM_ZONES):
    print(f"   Zone {i+1}: {cells_in_zone[i]} cells, {zone_resistance[i]:.4f}Ω, {zone_heat_capacity[i]:.0f}J/K")


# =============================================================================
# DRIVE CYCLE DATA PROCESSING
# =============================================================================

def load_drive_cycle_files(folder_path):
    """Load and combine drive cycle files"""
    print("🔄 Loading drive cycle data...")
    
    dfs = []
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    for file in txt_files[:5]:  # Limit to first 5 files for testing
        file_path = os.path.join(folder_path, file)
        
        # Try multiple encodings
        df = None
        for encoding in ["utf-16", "ISO-8859-1", "utf-8"]:
            try:
                df = pd.read_csv(file_path, sep="\t", skiprows=1, 
                               names=["Time", "Speed"], encoding=encoding, engine="python")
                break
            except:
                continue
                
        if df is not None:
            df = df.dropna()
            df["File"] = file
            dfs.append(df)
            print(f"   ✅ {file}: {len(df)} rows")
    
    if not dfs:
        raise RuntimeError("No valid drive cycle files found!")
    
    return pd.concat(dfs, ignore_index=True)


def calculate_realistic_ev_power(drive_df):
    """Calculate realistic EV power consumption from speed profile"""
    
    # EV vehicle parameters (Tesla Model S-class)
    VEHICLE_MASS = 2200      # kg
    DRAG_COEFF = 0.24        # Very aerodynamic
    FRONTAL_AREA = 2.34      # m²
    ROLLING_RESISTANCE = 0.008
    AIR_DENSITY = 1.225      # kg/m³
    DRIVETRAIN_EFF = 0.92    # EV drivetrain efficiency
    REGEN_EFF = 0.85         # Regenerative braking efficiency
    
    # Convert and smooth speed
    drive_df["Speed_mps"] = drive_df["Speed"] * 0.44704  # mph to m/s
    smooth_speed = drive_df["Speed_mps"].rolling(7, center=True).mean().fillna(method="bfill").fillna(method="ffill")
    
    # Calculate acceleration
    dt = 1.0  # 1 second timesteps
    drive_df["Acceleration"] = smooth_speed.diff().fillna(0) / dt
    
    # Forces
    F_aero = 0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA * smooth_speed**2
    F_rolling = VEHICLE_MASS * 9.81 * ROLLING_RESISTANCE
    F_inertia = VEHICLE_MASS * drive_df["Acceleration"]
    
    # Total tractive force
    F_total = F_aero + F_rolling + F_inertia
    
    # Mechanical power at wheels
    P_mechanical = F_total * smooth_speed  # Watts
    
    # Electrical power (account for efficiency)
    P_electrical = np.where(
        P_mechanical >= 0,
        P_mechanical / DRIVETRAIN_EFF,  # Motoring (discharge)
        P_mechanical * REGEN_EFF        # Regenerating (negative)
    )
    
    # Only positive power heats the battery significantly
    drive_df["Power_W"] = np.maximum(P_electrical, 0)
    
    return drive_df


# Load or build drive cycle dataset
if not os.path.exists(CORE_CSV_PATH) or FORCE_REBUILD_CORE:
    drive_df = load_drive_cycle_files(DRIVE_CYCLE_FOLDER)
    drive_df = calculate_realistic_ev_power(drive_df)
    drive_df["CumulativeTime"] = np.arange(len(drive_df))
    drive_df.to_csv(CORE_CSV_PATH, index=False)
    print(f"✅ Saved core dataset: {len(drive_df)} rows")
else:
    drive_df = pd.read_csv(CORE_CSV_PATH)
    print(f"📄 Loaded existing dataset: {len(drive_df)} rows")

# Extract simulation arrays
time_array = np.arange(len(drive_df))
power_profile = drive_df["Power_W"].values

print(f"📈 POWER PROFILE STATS:")
print(f"   Duration: {len(time_array)} seconds ({len(time_array)/3600:.1f} hours)")
print(f"   Power range: {np.min(power_profile):.0f} - {np.max(power_profile):.0f} W")
print(f"   Average power: {np.mean(power_profile):.0f} W")
print(f"   Peak vs pack max: {np.max(power_profile)/1000:.1f} kW / {PACK_MAX_POWER:.0f} kW")


# =============================================================================
# COOLING SYSTEM CONFIGURATION
# =============================================================================

# Cooling system sizing (scales with pack size and power)
if not ADIABATIC_MODE:
    # Pump capacity: ~1 L/min per kW of pack power
    MAX_PUMP_FLOW_LPM = max(20.0, PACK_MAX_POWER * 0.8)  # L/min
    
    # Heat transfer: size to handle ~80% of max pack power at 15K temperature difference
    target_cooling_power = PACK_MAX_POWER * 800  # 80% in watts
    target_delta_t = 15.0  # K
    total_UA = target_cooling_power / target_delta_t  # W/K
    
    # Distribute UA across zones based on number of cells
    zone_UA_base = np.zeros(NUM_ZONES)
    for i in range(NUM_ZONES):
        zone_UA_base[i] = total_UA * (cells_in_zone[i] / TOTAL_CELLS)
    
    # Flow enhancement factor
    FLOW_ENHANCEMENT = 1.5  # UA increases 50% at full flow
    
    # Pump power sizing
    PUMP_MAX_POWER_W = max(200.0, PACK_MAX_POWER * 2)  # 2W pump power per kW pack power
    
else:
    # Adiabatic mode
    MAX_PUMP_FLOW_LPM = 0.0
    zone_UA_base = np.zeros(NUM_ZONES)
    FLOW_ENHANCEMENT = 0.0
    PUMP_MAX_POWER_W = 0.0

COOLANT_TEMP = 25.0  # °C
INITIAL_TEMP = 25.0  # °C

print(f"🌊 COOLING SYSTEM:")
print(f"   Mode: {'ADIABATIC' if ADIABATIC_MODE else 'ACTIVE COOLING'}")
if not ADIABATIC_MODE:
    print(f"   Max pump flow: {MAX_PUMP_FLOW_LPM:.1f} L/min")
    print(f"   Total heat transfer capacity: {np.sum(zone_UA_base):.0f} W/K")
    print(f"   Max pump power: {PUMP_MAX_POWER_W:.0f} W")


# =============================================================================
# PID CONTROLLER FOR COOLING
# =============================================================================

class PIDController:
    def __init__(self, Kp=3.0, Ki=0.05, Kd=0.5, setpoint=35.0, output_limits=(0.0, 1.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, measurement, dt=1.0):
        error = self.setpoint - measurement
        
        self.integral += error * dt
        # Integral windup protection
        max_integral = (self.output_limits[1] - self.output_limits[0]) / self.Ki if self.Ki > 0 else 1000
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = np.clip(output, *self.output_limits)
        
        self.prev_error = error
        return output

# Create PID controllers for each zone
TEMP_SETPOINT = 40.0  # °C - start cooling above 40°C
pid_controllers = [PIDController(setpoint=TEMP_SETPOINT) for _ in range(NUM_ZONES)]


# =============================================================================
# BATTERY THERMAL DYNAMICS ODE SYSTEM
# =============================================================================

def battery_thermal_ode(t, state, power_profile, params):
    """ODE system for battery pack thermal dynamics"""
    
    # Unpack state vector
    temperatures = state[:NUM_ZONES]
    coolant_flows = state[NUM_ZONES:]  # Normalized flows (0-1)
    
    # Get current power demand
    time_idx = min(int(t), len(power_profile) - 1)
    power_demand = power_profile[time_idx]  # Watts
    
    # Calculate pack current
    pack_current = power_demand / PACK_NOMINAL_VOLTAGE  # A
    
    # Heat generation per zone (I²R losses)
    heat_generation = np.zeros(NUM_ZONES)
    for i in range(NUM_ZONES):
        heat_generation[i] = (pack_current ** 2) * params['zone_resistance'][i]
    
    # Initialize derivatives
    dT_dt = np.zeros(NUM_ZONES)
    dF_dt = np.zeros(NUM_ZONES)
    
    # Calculate temperature derivatives
    for i in range(NUM_ZONES):
        T_i = temperatures[i]
        
        # Heat input from electrical losses
        Q_generation = heat_generation[i]
        
        # Lateral heat conduction to neighboring zones
        Q_lateral = 0.0
        if i > 0:  # Heat from left neighbor
            Q_lateral += params['lateral_conductance'] * (temperatures[i-1] - T_i)
        if i < NUM_ZONES - 1:  # Heat to right neighbor  
            Q_lateral += params['lateral_conductance'] * (temperatures[i+1] - T_i)
        
        # Convective cooling (depends on coolant flow)
        Q_cooling = 0.0
        if not ADIABATIC_MODE and coolant_flows[i] > 0:
            # Effective heat transfer coefficient increases with flow
            flow_factor = 1.0 + FLOW_ENHANCEMENT * coolant_flows[i]
            UA_effective = params['zone_UA_base'][i] * flow_factor
            Q_cooling = UA_effective * (T_i - COOLANT_TEMP)
        
        # Net heat change
        Q_net = Q_generation + Q_lateral - Q_cooling
        
        # Temperature derivative
        dT_dt[i] = Q_net / params['zone_heat_capacity'][i]
    
    # Coolant flow dynamics (first-order response to PID commands)
    FLOW_TIME_CONSTANT = 3.0  # seconds
    
    for i in range(NUM_ZONES):
        if not ADIABATIC_MODE:
            # PID controller determines target flow
            pid_output = params['pid_controllers'][i].update(temperatures[i], dt=1.0)
            target_flow = pid_output  # Already normalized 0-1
        else:
            target_flow = 0.0
        
        # Flow rate of change
        dF_dt[i] = (target_flow - coolant_flows[i]) / FLOW_TIME_CONSTANT
    
    return np.concatenate([dT_dt, dF_dt])


# Prepare simulation parameters
simulation_params = {
    'zone_resistance': zone_resistance,
    'zone_heat_capacity': zone_heat_capacity,
    'zone_UA_base': zone_UA_base,
    'lateral_conductance': lateral_conductance[0] if len(lateral_conductance) > 0 else 5.0,
    'pid_controllers': pid_controllers
}


# =============================================================================
# RUN THERMAL SIMULATION
# =============================================================================

print("🚀 Running battery thermal simulation...")

# Initial conditions
initial_temperatures = np.full(NUM_ZONES, INITIAL_TEMP)
initial_flows = np.zeros(NUM_ZONES)
initial_state = np.concatenate([initial_temperatures, initial_flows])

# Time integration
t_span = [0, len(time_array) - 1]
t_eval = time_array

# Solve the ODE system
solution = solve_ivp(
    fun=lambda t, y: battery_thermal_ode(t, y, power_profile, simulation_params),
    t_span=t_span,
    y0=initial_state,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-6,
    atol=1e-8
)

if not solution.success:
    print(f"❌ Simulation failed: {solution.message}")
    raise RuntimeError("ODE integration failed")

print("✅ Simulation completed successfully!")

# Extract results
temperatures = solution.y[:NUM_ZONES, :].T
coolant_flows = solution.y[NUM_ZONES:, :].T

print(f"📊 Results shape: {temperatures.shape} temperatures, {coolant_flows.shape} flows")


# =============================================================================
# POST-PROCESSING AND PUMP POWER CALCULATION
# =============================================================================

# Calculate pump power consumption
pump_power = np.zeros(len(time_array))
total_flow_rate = np.zeros(len(time_array))

for t in range(len(time_array)):
    if not ADIABATIC_MODE:
        # Total flow is sum of zone flows times max capacity
        total_flow_lpm = np.sum(coolant_flows[t, :]) * MAX_PUMP_FLOW_LPM / NUM_ZONES
        total_flow_rate[t] = total_flow_lpm
        
        # Pump power scales with flow squared (centrifugal pump characteristic)
        flow_fraction = total_flow_lpm / MAX_PUMP_FLOW_LPM if MAX_PUMP_FLOW_LPM > 0 else 0
        pump_power[t] = PUMP_MAX_POWER_W * (flow_fraction ** 2)
    else:
        pump_power[t] = 0.0
        total_flow_rate[t] = 0.0

# Calculate statistics
temp_stats = {
    'initial': np.mean(temperatures[0, :]),
    'final': np.mean(temperatures[-1, :]),
    'max': np.max(temperatures),
    'min': np.min(temperatures),
    'final_spread': np.max(temperatures[-1, :]) - np.min(temperatures[-1, :])
}

energy_stats = {
    'battery_total_kwh': np.sum(power_profile) / 3600 / 1000,
    'battery_peak_kw': np.max(power_profile) / 1000,
    'pump_total_wh': np.sum(pump_power) / 3600,
    'pump_peak_w': np.max(pump_power)
}


# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Battery Pack Thermal Management Simulation', fontsize=16, fontweight='bold')

# Temperature profiles
ax1 = axes[0, 0]
for i in range(NUM_ZONES):
    ax1.plot(time_array/3600, temperatures[:, i], label=f'Zone {i+1}', linewidth=2)
ax1.axhline(y=TEMP_SETPOINT, color='r', linestyle='--', alpha=0.7, label='PID Setpoint')
ax1.axhline(y=COOLANT_TEMP, color='b', linestyle='--', alpha=0.7, label='Coolant Temp')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Zone Temperatures')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Power profiles
ax2 = axes[0, 1]
ax2.plot(time_array/3600, power_profile/1000, color='orange', label='Battery Power', linewidth=2)
if not ADIABATIC_MODE:
    ax2.plot(time_array/3600, pump_power/1000, color='purple', label='Pump Power', linewidth=2)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Power (kW)')
ax2.set_title('Power Profiles')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Coolant flows
ax3 = axes[0, 2]
if not ADIABATIC_MODE:
    for i in range(NUM_ZONES):
        flow_lpm = coolant_flows[:, i] * MAX_PUMP_FLOW_LPM / NUM_ZONES
        ax3.plot(time_array/3600, flow_lpm, label=f'Zone {i+1}', linewidth=2)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Flow Rate (L/min)')
    ax3.set_title('Coolant Flow Rates')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'ADIABATIC MODE\n(No Cooling)', 
             ha='center', va='center', transform=ax3.transAxes, fontsize=14)
    ax3.set_title('Coolant Flow (Disabled)')

# Temperature statistics
ax4 = axes[1, 0]
temp_mean = np.mean(temperatures, axis=1)
temp_max = np.max(temperatures, axis=1)
temp_min = np.min(temperatures, axis=1)
ax4.plot(time_array/3600, temp_mean, 'k-', label='Mean', linewidth=3)
ax4.fill_between(time_array/3600, temp_min, temp_max, alpha=0.3, label='Min-Max Range')
ax4.axhline(y=TEMP_SETPOINT, color='r', linestyle='--', alpha=0.7, label='Setpoint')
ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Temperature (°C)')
ax4.set_title('Temperature Statistics')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Heat generation
ax5 = axes[1, 1]
pack_current = power_profile / PACK_NOMINAL_VOLTAGE
total_heat = np.zeros(len(time_array))
for i in range(NUM_ZONES):
    zone_heat = (pack_current ** 2) * zone_resistance[i]
    total_heat += zone_heat
ax5.plot(time_array/3600, total_heat/1000, color='red', label='Heat Generation', linewidth=2)
ax5.set_xlabel('Time (hours)')
ax5.set_ylabel('Heat Generation (kW)')
ax5.set_title('Battery Heat Generation')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Energy balance
ax6 = axes[1, 2]
cumulative_battery = np.cumsum(power_profile) / 3600 / 1000  # kWh
cumulative_pump = np.cumsum(pump_power) / 3600 / 1000  # kWh
ax6.plot(time_array/3600, cumulative_battery, color='orange', label='Battery Energy', linewidth=2)
if not ADIABATIC_MODE:
    ax6.plot(time_array/3600, cumulative_pump, color='purple', label='Pump Energy', linewidth=2)
ax6.set_xlabel('Time (hours)')
ax6.set_ylabel('Cumulative Energy (kWh)')
ax6.set_title('Energy Consumption')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# SAVE RESULTS
# =============================================================================

# Create comprehensive results dataframe
results_df = pd.DataFrame({
    'Time_s': time_array,
    'Time_h': time_array / 3600,
    'Battery_Power_W': power_profile,
    'Pump_Power_W': pump_power,
    'Total_Flow_LPM': total_flow_rate,
    'Pack_Current_A': power_profile / PACK_NOMINAL_VOLTAGE,
})

# Add zone temperatures
for i in range(NUM_ZONES):
    results_df[f'Zone_{i+1}_Temp_C'] = temperatures[:, i]

# Add zone flows  
for i in range(NUM_ZONES):
    results_df[f'Zone_{i+1}_Flow_LPM'] = coolant_flows[:, i] * MAX_PUMP_FLOW_LPM / NUM_ZONES

# Add summary statistics
results_df['Mean_Temp_C'] = np.mean(temperatures, axis=1)
results_df['Max_Temp_C'] = np.max(temperatures, axis=1)
results_df['Min_Temp_C'] = np.min(temperatures, axis=1)
results_df['Temp_Spread_K'] = results_df['Max_Temp_C'] - results_df['Min_Temp_C']

# Save results
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Results saved to: {OUTPUT_PATH}")


# =============================================================================
# PERFORMANCE SUMMARY
# =============================================================================

print("\n" + "="*60)
print("🎯 BATTERY PACK THERMAL SIMULATION SUMMARY")
print("="*60)

print(f"📦 PACK CONFIGURATION:")
print(f"   Architecture: {CELLS_IN_SERIES}s{CELLS_IN_PARALLEL}p ({TOTAL_CELLS} cells)")
print(f"   Energy capacity: {PACK_ENERGY:.1f} kWh")
print(f"   Max power: {PACK_MAX_POWER:.0f} kW")
print(f"   Thermal zones: {NUM_ZONES}")

print(f"\n⏱️ SIMULATION:")
print(f"   Duration: {len(time_array)/3600:.1f} hours ({len(time_array)} seconds)")
print(f"   Mode: {'Adiabatic (no cooling)' if ADIABATIC_MODE else 'Active liquid cooling'}")

print(f"\n🌡️ THERMAL PERFORMANCE:")
print(f"   Initial temperature: {temp_stats['initial']:.1f}°C")
print(f"   Final average temperature: {temp_stats['final']:.1f}°C")
print(f"   Maximum temperature: {temp_stats['max']:.1f}°C")
print(f"   Temperature rise: {temp_stats['max'] - temp_stats['initial']:.1f}°C")
print(f"   Final zone spread: {temp_stats['final_spread']:.1f}°C")

print(f"\n⚡ ENERGY ANALYSIS:")
print(f"   Battery energy delivered: {energy_stats['battery_total_kwh']:.2f} kWh")
print(f"   Peak battery power: {energy_stats['battery_peak_kw']:.1f} kW")
print(f"   Battery utilization: {energy_stats['battery_peak_kw']/PACK_MAX_POWER:.1%} of max")

if not ADIABATIC_MODE:
    print(f"   Pump energy consumed: {energy_stats['pump_total_wh']:.1f} Wh")
    print(f"   Peak pump power: {energy_stats['pump_peak_w']:.1f} W")
    print(f"   Cooling overhead: {energy_stats['pump_total_wh']/1000/energy_stats['battery_total_kwh']:.2%}")
    
    # Cooling effectiveness
    if temp_stats['max'] < 50:  # Reasonable operating temp
        print(f"   Cooling status: ✅ EFFECTIVE")
    elif temp_stats['max'] < 60:
        print(f"   Cooling status: ⚠️ MARGINAL")
    else:
        print(f"   Cooling status: ❌ INSUFFICIENT")

print("="*60)
print("🎉 Simulation completed successfully!")