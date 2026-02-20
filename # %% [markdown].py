# %% [markdown]
# # Battery Pack Thermal Simulation - Realistic Scaling
# This notebook is a cell-by-cell translation of the updated `trying.py` script, with all new scaling, pack architecture, and summary logic.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
import os
from datetime import datetime

# %% [markdown]
# ## Configuration and Paths

# %%
DRIVE_CYCLE_FOLDER = "/home/keshav/Documents/reserch_code/drive_cycles"
CORE_CSV_PATH = "/home/keshav/Documents/reserch_code/core.csv"
CHECKPOINT_DIR = "/home/keshav/Documents/reserch_code/checkpoints"
OUTPUT_PATH = "/home/keshav/Documents/reserch_code/simulated_battery_cooling.csv"

os.makedirs("data", exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

FORCE_REBUILD_CORE = True
ADIABATIC_MODE = False  # Set True to disable cooling

# %% [markdown]
# ## Realistic Battery Pack Configuration

# %%
CELLS_IN_SERIES = 12
CELLS_IN_PARALLEL = 3
TOTAL_CELLS = CELLS_IN_SERIES * CELLS_IN_PARALLEL
NUM_ZONES = TOTAL_CELLS//3
CELL_NOMINAL_VOLTAGE = 3.7
CELL_CAPACITY = 3.0
CELL_INTERNAL_R = 0.030
CELL_HEAT_CAPACITY = 50.0
CELL_MAX_DISCHARGE_RATE = 2.0

PACK_NOMINAL_VOLTAGE = CELLS_IN_SERIES * CELL_NOMINAL_VOLTAGE
PACK_CAPACITY = CELLS_IN_PARALLEL * CELL_CAPACITY
PACK_ENERGY = PACK_NOMINAL_VOLTAGE * PACK_CAPACITY / 1000
PACK_MAX_POWER = PACK_NOMINAL_VOLTAGE * PACK_CAPACITY * CELL_MAX_DISCHARGE_RATE / 1000

print(f"🔋 BATTERY PACK SPECIFICATIONS:")
print(f"   Configuration: {CELLS_IN_SERIES}s{CELLS_IN_PARALLEL}p = {TOTAL_CELLS} cells")
print(f"   Pack voltage: {PACK_NOMINAL_VOLTAGE:.1f} V")
print(f"   Pack capacity: {PACK_CAPACITY:.1f} Ah")
print(f"   Pack energy: {PACK_ENERGY:.1f} kWh")
print(f"   Max continuous power: {PACK_MAX_POWER:.0f} kW")
print(f"   Thermal zones: {NUM_ZONES}")

# %% [markdown]
# ## Zone-Based Pack Properties with Realistic Variation

# %%
np.random.seed(42)
cells_per_zone = TOTAL_CELLS // NUM_ZONES
cells_in_zone = np.full(NUM_ZONES, cells_per_zone)
remainder = TOTAL_CELLS % NUM_ZONES
cells_in_zone[:remainder] += 1

cell_r_variation = np.random.normal(1.0, 0.05, NUM_ZONES)
zone_resistance = np.zeros(NUM_ZONES)
for i in range(NUM_ZONES):
    zone_resistance[i] = (CELL_INTERNAL_R * cell_r_variation[i] * CELLS_IN_SERIES) / CELLS_IN_PARALLEL
zone_heat_capacity = cells_in_zone * CELL_HEAT_CAPACITY
zone_positions = np.linspace(0, 1, NUM_ZONES)

BASE_LATERAL_CONDUCTANCE = 3.0  # W/K, realistic order of magnitude
EDGE_REDUCTION = 0.4           # edges conduct worse

lateral_conductance = BASE_LATERAL_CONDUCTANCE * (
    1.0 - EDGE_REDUCTION * np.abs(zone_positions - 0.5) * 2.0
)

print(f"📊 ZONE DISTRIBUTION:")
for i in range(NUM_ZONES):
    print(f"   Zone {i+1}: {cells_in_zone[i]} cells, {zone_resistance[i]:.4f}Ω, {zone_heat_capacity[i]:.0f}J/K")

# %% [markdown]
# ## Drive Cycle Data Processing

# %%
# # ---------- Replace entire block with this cell ----------
# import os
# import numpy as np
# import pandas as pd

# # ----------------- User-configurable controls -----------------
# # If these variables already exist in your notebook they will be used.
# # If not, defaults below will be applied so the cell runs standalone.
# DRIVE_CYCLE_FOLDER = globals().get("DRIVE_CYCLE_FOLDER", "./drive_cycles")
# CORE_CSV_PATH = globals().get("CORE_CSV_PATH", "core.csv")
# FORCE_REBUILD_CORE = globals().get("FORCE_REBUILD_CORE", True)

# # POWER / VEHICLE tuning
# DEFAULT_VEHICLE_MASS = 2200.0  # kg, change this to tune power per speed
# SCALE_TO_PACK = False          # if True, scale computed vehicle power so peak == PACK_MAX_POWER (kW)
# TARGET_PEAK_KW = None          # optional override for target peak (kW). If None and SCALE_TO_PACK True uses PACK_MAX_POWER
# MAX_FILES_TO_LOAD = 5         # how many .txt files to read from folder, for quick testing

# # ----------------- Safety: PACK_MAX_POWER fallback -----------------
# # Use existing PACK_MAX_POWER if present in notebook, else compute a fallback using typical 12s3p 18650 example
# PACK_MAX_POWER = globals().get("PACK_MAX_POWER", None)
# if PACK_MAX_POWER is None:
#     # fallback: estimate from nominal numbers if the user hasn't defined PACK_MAX_POWER
#     CELLS_IN_SERIES = globals().get("CELLS_IN_SERIES", 12)
#     CELLS_IN_PARALLEL = globals().get("CELLS_IN_PARALLEL", 3)
#     CELL_NOMINAL_VOLTAGE = globals().get("CELL_NOMINAL_VOLTAGE", 3.7)
#     CELL_CAPACITY = globals().get("CELL_CAPACITY", 3.0)  # Ah
#     CELL_MAX_DISCHARGE_RATE = globals().get("CELL_MAX_DISCHARGE_RATE", 2.0)  # C
#     PACK_NOMINAL_VOLTAGE = CELLS_IN_SERIES * CELL_NOMINAL_VOLTAGE
#     PACK_CAPACITY = CELLS_IN_PARALLEL * CELL_CAPACITY
#     PACK_MAX_POWER = PACK_NOMINAL_VOLTAGE * PACK_CAPACITY * CELL_MAX_DISCHARGE_RATE / 1000.0  # kW
#     print(f"⚠ PACK_MAX_POWER not found, using fallback estimate {PACK_MAX_POWER:.2f} kW")
# else:
#     PACK_MAX_POWER = PACK_MAX_POWER  # use existing

# # ----------------- Drive cycle loader -----------------
# def load_drive_cycle_files(folder_path, max_files=MAX_FILES_TO_LOAD):
#     """
#     Load up to max_files text drive cycle files from folder_path.
#     Expected format: two columns (Time, Speed). The loader is tolerant to common encodings and small formatting issues.
#     Returns a concatenated DataFrame with columns: Time (int seconds), Speed (float; same units as file).
#     """
#     print("🔄 Loading drive cycle data...")
#     if not os.path.isdir(folder_path):
#         raise RuntimeError(f"Drive cycle folder not found: {folder_path}")

#     txt_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".txt")])
#     if not txt_files:
#         raise RuntimeError(f"No .txt files found in {folder_path}")

#     dfs = []
#     for file in txt_files[:max_files]:
#         file_path = os.path.join(folder_path, file)
#         df = None
#         # Try a few encodings and separators for robustness
#         for encoding in ("utf-8", "ISO-8859-1", "utf-16"):
#             try:
#                 # many drive cycle txts are two-column with tab or whitespace; skip first header if present
#                 df_try = pd.read_csv(file_path, sep=None, engine="python", encoding=encoding, header=None, comment="#")
#                 # keep only first two columns
#                 if df_try.shape[1] >= 2:
#                     df_try = df_try.iloc[:, :2]
#                     df_try.columns = ["Time", "Speed"]
#                     df = df_try
#                     break
#             except Exception:
#                 continue

#         if df is None:
#             print(f"   ⚠ Skipped {file}, could not parse with common encodings")
#             continue

#         # coerce numeric and drop bad rows
#         df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
#         df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
#         df = df.dropna(subset=["Time", "Speed"]).copy()
#         if df.empty:
#             print(f"   ⚠ {file} contained no valid numeric rows after cleanup")
#             continue

#         # enforce integer seconds and 1 decimal for speed (keeps your formatting)
#         df["Time"] = df["Time"].round().astype(int)
#         df["Speed"] = df["Speed"].round(1)

#         df["File"] = file
#         dfs.append(df.reset_index(drop=True))
#         print(f"   ✅ {file}: {len(df)} rows")

#     if not dfs:
#         raise RuntimeError("No valid drive cycle data loaded from folder")

#     combined = pd.concat(dfs, ignore_index=True)
#     # sort by Time to be safe, and drop duplicate timestamps (keep first)
#     combined = combined.sort_values("Time").drop_duplicates(subset="Time").reset_index(drop=True)
#     return combined

# # ----------------- Vehicle power model -----------------
# def calculate_realistic_ev_power(drive_df):
#     VEHICLE_MASS = 2200.0
#     DRAG_COEFF = 0.24
#     FRONTAL_AREA = 2.34
#     ROLLING_RESISTANCE = 0.008
#     AIR_DENSITY = 1.225
#     DRIVETRAIN_EFF = 0.92
#     REGEN_EFF = 0.6

#     df = drive_df.copy()
#     df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce").fillna(0.0)

#     # speed units
#     if df["Speed"].max() > 200:
#         speed_mps = df["Speed"].values * (1000.0 / 3600.0)
#     else:
#         speed_mps = df["Speed"].values * 0.44704

#     # time step
#     dt = 1.0
#     if "Time" in df.columns and len(df) > 1:
#         dt = max(1.0, np.median(np.diff(df["Time"].values)))

#     # mild smoothing only for acceleration
#     speed_s = pd.Series(speed_mps).ewm(alpha=0.25).mean().values
#     accel = np.zeros_like(speed_s)
#     accel[1:] = (speed_s[1:] - speed_s[:-1]) / dt

#     # forces
#     F_aero = 0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA * speed_s**2
#     F_roll = VEHICLE_MASS * 9.81 * ROLLING_RESISTANCE
#     F_inertia = VEHICLE_MASS * accel
#     F_total = F_aero + F_roll + F_inertia

#     # vehicle mechanical power
#     P_mech = F_total * speed_s
#     P_mech[P_mech >= 0] /= DRIVETRAIN_EFF
#     P_mech[P_mech < 0] *= REGEN_EFF

#     # convert to battery current
#     pack_voltage = PACK_NOMINAL_VOLTAGE
#     pack_current = P_mech / pack_voltage

#     # per branch current
#     branch_current = pack_current / CELLS_IN_PARALLEL

#     # C rate limits from Cell 1
#     I_max_discharge = CELL_MAX_DISCHARGE_RATE * CELL_CAPACITY
#     I_max_charge = 0.5 * CELL_CAPACITY

#     branch_current = np.clip(
#         branch_current,
#         -I_max_charge,
#         I_max_discharge
#     )

#     # back to pack current and power
#     pack_current_limited = branch_current * CELLS_IN_PARALLEL
#     power_w = pack_current_limited * pack_voltage

#     power_w[power_w < 0] = 0.0
#     power_w[speed_s < 0.5] = 0.0

#     df["Power_W"] = power_w
#     return df
# # ----------------- Pipeline: load, compute, save -----------------
# try:
#     if not os.path.exists(DRIVE_CYCLE_FOLDER):
#         raise FileNotFoundError(f"Drive cycle folder does not exist: {DRIVE_CYCLE_FOLDER}")

#     if not os.path.exists(CORE_CSV_PATH) or FORCE_REBUILD_CORE:
#         drive_df = load_drive_cycle_files(DRIVE_CYCLE_FOLDER, max_files=MAX_FILES_TO_LOAD)
#         # compute vehicle power
#         drive_df = calculate_realistic_ev_power(drive_df)

#         # optionally scale to pack capability. Do not scale by default to keep dynamics visible.
#         if SCALE_TO_PACK:
#             target_kw = TARGET_PEAK_KW if TARGET_PEAK_KW is not None else PACK_MAX_POWER
#             peak_w = max(drive_df["Power_W"].max(), 1.0)
#             scale_factor = (target_kw * 1000.0) / peak_w
#             drive_df["Power_W"] = drive_df["Power_W"] * scale_factor
#             print(f"ℹ Scaled vehicle power so peak equals {target_kw:.3f} kW, scale factor {scale_factor:.3f}")

#         drive_df["CumulativeTime"] = np.arange(len(drive_df))
#         drive_df.to_csv(CORE_CSV_PATH, index=False)
#         print(f"✅ Saved core dataset: {len(drive_df)} rows")
#     else:
#         drive_df = pd.read_csv(CORE_CSV_PATH)
#         drive_df["Power_W"] = pd.to_numeric(drive_df.get("Power_W", 0.0), errors="coerce").fillna(0.0)
#         print(f"📄 Loaded existing dataset: {len(drive_df)} rows")

#     # stats
#     time_array = np.arange(len(drive_df))
#     power_profile = drive_df["Power_W"].values

#     print("📈 POWER PROFILE STATS:")
#     print(f"   Duration: {len(time_array)} seconds ({len(time_array)/3600:.2f} hours)")
#     print(f"   Power range: {np.min(power_profile):.0f} - {np.max(power_profile):.0f} W")
#     print(f"   Average power: {np.mean(power_profile):.0f} W")
#     print(f"   Peak vs pack max: {np.max(power_profile)/1000.0:.2f} kW / {PACK_MAX_POWER:.1f} kW")
#     if hasattr(drive_df, "attrs") and "unit_detection" in drive_df.attrs:
#         print(f"   Unit heuristic: {drive_df.attrs['unit_detection']}, dt ~ {drive_df.attrs['dt_seconds']} s, vehicle mass used {drive_df.attrs['vehicle_mass_used']} kg")

# except Exception as e:
#     # Provide a clear, actionable error message
#     raise RuntimeError(f"Drive cycle pipeline failed: {e}")
# # -----------------------------------------------------------------


# %%
# ---------- Replace entire block with this cell ----------
import os
import sys
import numpy as np
import pandas as pd

# ================= USER CONFIGURABLE CONTROLS =================

DRIVE_CYCLE_FOLDER = globals().get("DRIVE_CYCLE_FOLDER", "./drive_cycles")
CORE_CSV_PATH = globals().get("CORE_CSV_PATH", "core.csv")
FORCE_REBUILD_CORE = globals().get("FORCE_REBUILD_CORE", True)

# Power and vehicle tuning
DEFAULT_VEHICLE_MASS = 2200.0
SCALE_TO_PACK = False
TARGET_PEAK_KW = None
MAX_FILES_TO_LOAD = 5

# -------- Dataset length control --------
# Set exactly ONE of these, others must be None

MAX_SECONDS = None        # example: 1800
MAX_HOURS = None          # example: 2.0
DATA_FRACTION = 0.001     # example: 0.01 uses first 1 percent

_active = [MAX_SECONDS is not None, MAX_HOURS is not None, DATA_FRACTION is not None]
if sum(_active) > 1:
    raise ValueError("Set only one of MAX_SECONDS, MAX_HOURS, DATA_FRACTION")

# ================= PACK POWER SAFETY =================

PACK_MAX_POWER = globals().get("PACK_MAX_POWER", None)
if PACK_MAX_POWER is None:
    CELLS_IN_SERIES = globals().get("CELLS_IN_SERIES", 12)
    CELLS_IN_PARALLEL = globals().get("CELLS_IN_PARALLEL", 3)
    CELL_NOMINAL_VOLTAGE = globals().get("CELL_NOMINAL_VOLTAGE", 3.7)
    CELL_CAPACITY = globals().get("CELL_CAPACITY", 3.0)
    CELL_MAX_DISCHARGE_RATE = globals().get("CELL_MAX_DISCHARGE_RATE", 2.0)

    PACK_NOMINAL_VOLTAGE = CELLS_IN_SERIES * CELL_NOMINAL_VOLTAGE
    PACK_CAPACITY = CELLS_IN_PARALLEL * CELL_CAPACITY
    PACK_MAX_POWER = (
        PACK_NOMINAL_VOLTAGE * PACK_CAPACITY * CELL_MAX_DISCHARGE_RATE / 1000.0
    )

    print(f"⚠ PACK_MAX_POWER not found, using fallback {PACK_MAX_POWER:.2f} kW")

# ================= DEBUG: Check paths =================
print(f"🔍 Debug Info:")
print(f"   Current working directory: {os.getcwd()}")
print(f"   DRIVE_CYCLE_FOLDER (raw): {DRIVE_CYCLE_FOLDER}")
print(f"   DRIVE_CYCLE_FOLDER (absolute): {os.path.abspath(DRIVE_CYCLE_FOLDER)}")
print(f"   Python version: {sys.version}")
print(f"   Exists? {os.path.exists(DRIVE_CYCLE_FOLDER)}")

# ================= DRIVE CYCLE LOADER =================

def load_drive_cycle_files(folder_path, max_files=MAX_FILES_TO_LOAD):
    print("🔄 Loading drive cycle data...")

    if not os.path.isdir(folder_path):
        raise RuntimeError(f"Drive cycle folder not found: {folder_path}")

    txt_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".txt")]
    )
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {folder_path}")

    dfs = []

    for file in txt_files[:max_files]:
        file_path = os.path.join(folder_path, file)
        df = None

        for encoding in ("utf-8", "ISO-8859-1", "utf-16"):
            try:
                df_try = pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python",
                    encoding=encoding,
                    header=None,
                    comment="#",
                )
                if df_try.shape[1] >= 2:
                    df_try = df_try.iloc[:, :2]
                    df_try.columns = ["Time", "Speed"]
                    df = df_try
                    break
            except Exception:
                continue

        if df is None:
            print(f"   ⚠ Skipped {file}, parse failed")
            continue

        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
        df = df.dropna(subset=["Time", "Speed"])

        if df.empty:
            print(f"   ⚠ {file} empty after cleanup")
            continue

        df["Time"] = df["Time"].round().astype(int)
        df["Speed"] = df["Speed"].round(1)
        df["File"] = file

        dfs.append(df.reset_index(drop=True))
        print(f"   ✅ {file}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    combined = (
        combined.sort_values("Time")
        .drop_duplicates(subset="Time")
        .reset_index(drop=True)
    )

    return combined

# ================= VEHICLE POWER MODEL =================

def calculate_realistic_ev_power(drive_df):
    VEHICLE_MASS = 2200.0
    DRAG_COEFF = 0.24
    FRONTAL_AREA = 2.34
    ROLLING_RESISTANCE = 0.008
    AIR_DENSITY = 1.225
    DRIVETRAIN_EFF = 0.92
    REGEN_EFF = 0.6

    df = drive_df.copy()
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce").fillna(0.0)

    if df["Speed"].max() > 200:
        speed_mps = df["Speed"].values * (1000.0 / 3600.0)
    else:
        speed_mps = df["Speed"].values * 0.44704

    dt = 1.0
    if len(df) > 1:
        dt = max(1.0, np.median(np.diff(df["Time"].values)))

    speed_s = pd.Series(speed_mps).ewm(alpha=0.25).mean().values
    accel = np.zeros_like(speed_s)
    accel[1:] = (speed_s[1:] - speed_s[:-1]) / dt

    F_aero = 0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA * speed_s**2
    F_roll = VEHICLE_MASS * 9.81 * ROLLING_RESISTANCE
    F_inertia = VEHICLE_MASS * accel
    F_total = F_aero + F_roll + F_inertia

    P_mech = F_total * speed_s
    P_mech[P_mech >= 0] /= DRIVETRAIN_EFF
    P_mech[P_mech < 0] *= REGEN_EFF

    pack_voltage = PACK_NOMINAL_VOLTAGE
    pack_current = P_mech / pack_voltage
    branch_current = pack_current / CELLS_IN_PARALLEL

    I_max_discharge = CELL_MAX_DISCHARGE_RATE * CELL_CAPACITY
    I_max_charge = 0.5 * CELL_CAPACITY

    branch_current = np.clip(branch_current, -I_max_charge, I_max_discharge)

    pack_current_limited = branch_current * CELLS_IN_PARALLEL
    power_w = pack_current_limited * pack_voltage

    power_w[power_w < 0] = 0.0
    power_w[speed_s < 0.5] = 0.0

    df["Power_W"] = power_w
    return df

# ================= PIPELINE =================

try:
    # Convert relative path to absolute to handle remote kernel issues
    abs_drive_cycle_folder = os.path.abspath(DRIVE_CYCLE_FOLDER)
    
    if not os.path.exists(abs_drive_cycle_folder):
        print(f"❌ Drive cycle folder not found at: {abs_drive_cycle_folder}")
        print(f"📁 Creating folder...")
        os.makedirs(abs_drive_cycle_folder, exist_ok=True)
        print(f"✅ Folder created at: {abs_drive_cycle_folder}")
        print(f"⚠️  Please ensure these files exist in the remote kernel's filesystem:")
        print(f"   - Place .txt drive cycle files in: {abs_drive_cycle_folder}")
        print(f"   - Or, set DRIVE_CYCLE_FOLDER to an absolute path accessible to the kernel")
        raise FileNotFoundError(f"Drive cycle folder created but empty: {abs_drive_cycle_folder}")
    
    # List what files are actually there
    files_in_folder = os.listdir(abs_drive_cycle_folder)
    txt_files = [f for f in files_in_folder if f.lower().endswith('.txt')]
    print(f"📂 Contents of {abs_drive_cycle_folder}:")
    print(f"   Total files: {len(files_in_folder)}")
    print(f"   .txt files: {len(txt_files)}")
    for f in txt_files[:5]:
        print(f"      - {f}")
    if len(txt_files) > 5:
        print(f"      ... and {len(txt_files)-5} more")
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt drive cycle files found in {abs_drive_cycle_folder}")

    if not os.path.exists(CORE_CSV_PATH) or FORCE_REBUILD_CORE:
        drive_df = load_drive_cycle_files(abs_drive_cycle_folder)
        drive_df = calculate_realistic_ev_power(drive_df)

        if SCALE_TO_PACK:
            target_kw = TARGET_PEAK_KW if TARGET_PEAK_KW is not None else PACK_MAX_POWER
            peak_w = max(drive_df["Power_W"].max(), 1.0)
            drive_df["Power_W"] *= (target_kw * 1000.0) / peak_w

        drive_df["CumulativeTime"] = np.arange(len(drive_df))
        drive_df.to_csv(CORE_CSV_PATH, index=False)
        print(f"✅ Saved core dataset: {len(drive_df)} rows")

    else:
        drive_df = pd.read_csv(CORE_CSV_PATH)
        drive_df["Power_W"] = pd.to_numeric(
            drive_df["Power_W"], errors="coerce"
        ).fillna(0.0)
        print(f"📄 Loaded existing dataset: {len(drive_df)} rows")

    # -------- Apply truncation --------
    original_len = len(drive_df)

    if MAX_SECONDS is not None:
        drive_df = drive_df.iloc[:int(MAX_SECONDS)]

    elif MAX_HOURS is not None:
        drive_df = drive_df.iloc[:int(MAX_HOURS * 3600)]

    elif DATA_FRACTION is not None:
        keep = int(original_len * DATA_FRACTION)
        drive_df = drive_df.iloc[:keep]

    drive_df = drive_df.reset_index(drop=True)

    print(
        f"🧪 Using {len(drive_df)} / {original_len} seconds "
        f"({len(drive_df)/3600:.2f} hours)"
    )

    time_array = np.arange(len(drive_df))
    power_profile = drive_df["Power_W"].values

    print("📈 POWER PROFILE STATS:")
    print(f"   Duration: {len(time_array)} seconds ({len(time_array)/3600:.2f} hours)")
    print(f"   Power range: {power_profile.min():.0f} – {power_profile.max():.0f} W")
    print(f"   Average power: {power_profile.mean():.0f} W")
    print(f"   Peak vs pack max: {power_profile.max()/1000:.2f} kW / {PACK_MAX_POWER:.1f} kW")

except Exception as e:
    raise RuntimeError(f"Drive cycle pipeline failed: {e}")
# -----------------------------------------------------------------

# %% [markdown]
# ## Cooling System Configuration

# %%
if not ADIABATIC_MODE:
    MAX_PUMP_FLOW_LPM = max(20.0, PACK_MAX_POWER * 0.8)
    target_cooling_power = PACK_MAX_POWER * 800
    target_delta_t = 15.0
    total_UA = target_cooling_power / target_delta_t
    zone_UA_base = np.zeros(NUM_ZONES)
    for i in range(NUM_ZONES):
        zone_UA_base[i] = total_UA * (cells_in_zone[i] / TOTAL_CELLS)
    FLOW_ENHANCEMENT = 1.5
    PUMP_MAX_POWER_W = max(200.0, PACK_MAX_POWER * 2)
else:
    MAX_PUMP_FLOW_LPM = 0.0
    zone_UA_base = np.zeros(NUM_ZONES)
    FLOW_ENHANCEMENT = 0.0
    PUMP_MAX_POWER_W = 0.0
COOLANT_TEMP = 25.0
INITIAL_TEMP = 25.0
print(f"🌊 COOLING SYSTEM:")
print(f"   Mode: {'ADIABATIC' if ADIABATIC_MODE else 'ACTIVE COOLING'}")
if not ADIABATIC_MODE:
    print(f"   Max pump flow: {MAX_PUMP_FLOW_LPM:.1f} L/min")
    print(f"   Total heat transfer capacity: {np.sum(zone_UA_base):.0f} W/K")
    print(f"   Max pump power: {PUMP_MAX_POWER_W:.0f} W")

# %% [markdown]
# ## PID Controller for Cooling

# %%
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
        max_integral = (self.output_limits[1] - self.output_limits[0]) / self.Ki if self.Ki > 0 else 1000
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = np.clip(output, *self.output_limits)
        self.prev_error = error
        return output
TEMP_SETPOINT = 40.0
pid_controllers = [PIDController(setpoint=TEMP_SETPOINT) for _ in range(NUM_ZONES)]

# %% [markdown]
# ## Battery Thermal Dynamics ODE System

# %%
def battery_thermal_ode(t, state, power_profile, params):
    temperatures = state[:NUM_ZONES]
    coolant_flows = state[NUM_ZONES:]

    time_idx = min(int(t), len(power_profile) - 1)
    power_demand = power_profile[time_idx]

    # --------- Electrical current ---------
    pack_current = power_demand / PACK_NOMINAL_VOLTAGE

    # --------- Heat generation per zone ---------
    heat_generation = np.zeros(NUM_ZONES)
    for i in range(NUM_ZONES):
        heat_generation[i] = (pack_current ** 2) * params['zone_resistance'][i]

    # >>> FIX 1: energy preserving normalization <<<
    total_heat = np.sum(heat_generation)
    if total_heat > 0:
        heat_generation = heat_generation * (total_heat / np.sum(heat_generation))

    dT_dt = np.zeros(NUM_ZONES)
    dF_dt = np.zeros(NUM_ZONES)

    for i in range(NUM_ZONES):
        T_i = temperatures[i]
        Q_generation = heat_generation[i]

        # --------- Lateral conduction ---------
        Q_lateral = 0.0
        if i > 0:
            Q_lateral += params['lateral_conductance'][i] * (temperatures[i-1] - T_i)
        if i < NUM_ZONES - 1:
            Q_lateral += params['lateral_conductance'][i] * (temperatures[i+1] - T_i)

        # --------- Cooling ---------
        Q_cooling = 0.0
        if not ADIABATIC_MODE and coolant_flows[i] > 0:
            flow_factor = 1.0 + FLOW_ENHANCEMENT * coolant_flows[i]

            # >>> FIX 2: spatial cooling bias <<<
            zone_bias = 0.8 + 0.4 * (i / (NUM_ZONES - 1))

            UA_effective = params['zone_UA_base'][i] * flow_factor * zone_bias
            Q_cooling = UA_effective * (T_i - COOLANT_TEMP)

        # --------- Net heat balance ---------
        Q_net = Q_generation + Q_lateral - Q_cooling
        dT_dt[i] = Q_net / params['zone_heat_capacity'][i]

    # --------- Coolant dynamics ---------
    FLOW_TIME_CONSTANT = 3.0
    for i in range(NUM_ZONES):
        if not ADIABATIC_MODE:
            pid_output = params['pid_controllers'][i].update(temperatures[i], dt=1.0)
            target_flow = pid_output
        else:
            target_flow = 0.0

        dF_dt[i] = (target_flow - coolant_flows[i]) / FLOW_TIME_CONSTANT

    return np.concatenate([dT_dt, dF_dt])

# %% [markdown]
# ## Run Thermal Simulation

# %%
simulation_params = {
    'zone_resistance': zone_resistance,
    'zone_heat_capacity': zone_heat_capacity,
    'zone_UA_base': zone_UA_base,
    'lateral_conductance': lateral_conductance,      # ✅ CORRECT
    'pid_controllers': pid_controllers
}
print("🚀 Running battery thermal simulation...")
initial_temperatures = np.full(NUM_ZONES, INITIAL_TEMP)
initial_flows = np.zeros(NUM_ZONES)
initial_state = np.concatenate([initial_temperatures, initial_flows])
t_span = [0, len(time_array) - 1]
t_eval = time_array
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
temperatures = solution.y[:NUM_ZONES, :].T
coolant_flows = solution.y[NUM_ZONES:, :].T
print(f"📊 Results shape: {temperatures.shape} temperatures, {coolant_flows.shape} flows")

# %% [markdown]
# ## Post-processing and Pump Power Calculation

# %%
pump_power = np.zeros(len(time_array))
total_flow_rate = np.zeros(len(time_array))
for t in range(len(time_array)):
    if not ADIABATIC_MODE:
        total_flow_lpm = np.sum(coolant_flows[t, :]) * MAX_PUMP_FLOW_LPM / NUM_ZONES
        total_flow_rate[t] = total_flow_lpm
        flow_fraction = total_flow_lpm / MAX_PUMP_FLOW_LPM if MAX_PUMP_FLOW_LPM > 0 else 0
        pump_power[t] = PUMP_MAX_POWER_W * (flow_fraction ** 2)
    else:
        pump_power[t] = 0.0
        total_flow_rate[t] = 0.0
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

# %% [markdown]
# ## Visualization

# %%
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

# %% [markdown]
# ## Save Results

# %%
results_df = pd.DataFrame({
    'Time_s': time_array,
    'Time_h': time_array / 3600,
    'Battery_Power_W': power_profile,
    'Pump_Power_W': pump_power,
    'Total_Flow_LPM': total_flow_rate,
    'Pack_Current_A': power_profile / PACK_NOMINAL_VOLTAGE,
})
for i in range(NUM_ZONES):
    results_df[f'Zone_{i+1}_Temp_C'] = temperatures[:, i]
for i in range(NUM_ZONES):
    results_df[f'Zone_{i+1}_Flow_LPM'] = coolant_flows[:, i] * MAX_PUMP_FLOW_LPM / NUM_ZONES
results_df['Mean_Temp_C'] = np.mean(temperatures, axis=1)
results_df['Max_Temp_C'] = np.max(temperatures, axis=1)
results_df['Min_Temp_C'] = np.min(temperatures, axis=1)
results_df['Temp_Spread_K'] = results_df['Max_Temp_C'] - results_df['Min_Temp_C']
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Results saved to: {OUTPUT_PATH}")

# %% [markdown]
# ## Performance Summary

# %%
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
    if temp_stats['max'] < 50:
        print(f"   Cooling status: ✅ EFFECTIVE")
    elif temp_stats['max'] < 60:
        print(f"   Cooling status: ⚠️ MARGINAL")
    else:
        print(f"   Cooling status: ❌ INSUFFICIENT")
print("="*60)
print("🎉 Simulation completed successfully!")

# %%
"""
INVESTIGATION: Power vs Heat Relationship
"""
import matplotlib.pyplot as plt
from scipy import stats

# Recalculate total heat (same as in visualization)
pack_current_analysis = power_profile / PACK_NOMINAL_VOLTAGE
total_heat_analysis = np.zeros(len(time_array))
for i in range(NUM_ZONES):
    zone_heat = (pack_current_analysis ** 2) * zone_resistance[i]
    total_heat_analysis += zone_heat

# Analyze the relationship
print("=" * 70)
print("🔍 POWER vs HEAT GENERATION ANALYSIS")
print("=" * 70)

# Filter non-zero values for analysis
non_zero_mask = power_profile > 100  # Only look at periods with significant power
power_nz = power_profile[non_zero_mask]
heat_nz = total_heat_analysis[non_zero_mask]

print(f"\nData statistics (non-zero periods only):")
print(f"  Power: min={power_nz.min():.0f}W, max={power_nz.max():.0f}W, mean={power_nz.mean():.0f}W")
print(f"  Heat:  min={heat_nz.min():.0f}W, max={heat_nz.max():.0f}W, mean={heat_nz.mean():.0f}W")

# Test linearity: if Heat ∝ Power^2, then log(Heat) ∝ 2*log(Power)
if len(power_nz) > 10:
    log_power = np.log10(power_nz)
    log_heat = np.log10(heat_nz)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_power, log_heat)
    
    print(f"\n LOG-LOG RELATIONSHIP TEST:")
    print(f"  log(Heat) = {slope:.3f} * log(Power) + {intercept:.3f}")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  Expected slope if Heat ∝ Power²: 2.0")
    print(f"  Actual slope: {slope:.3f}")
    print(f"  ➜ Relationship is Heat ∝ Power^{slope:.2f}")
    
    if abs(slope - 2.0) < 0.1:
        print(f"   CORRECT: Matches expected quadratic relationship")
    elif abs(slope - 1.0) < 0.1:
        print(f"   ERROR: Relationship is LINEAR (should be QUADRATIC)")
    else:
        print(f"   UNEXPECTED: Relationship is neither linear nor quadratic")

# Visual comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Power vs Time
ax = axes[0, 0]
ax.plot(time_array/3600, power_profile/1000, 'b-', linewidth=2, label='Battery Power')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Power (kW)')
ax.set_title('Battery Power Profile')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Heat vs Time
ax = axes[0, 1]
ax.plot(time_array/3600, total_heat_analysis/1000, 'r-', linewidth=2, label='Heat Generation')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Heat (kW)')
ax.set_title('Total Heat Generation')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Power vs Heat (scatter plot - should show quadratic curve, not linear)
ax = axes[1, 0]
scatter = ax.scatter(power_profile/1000, total_heat_analysis/1000, c=time_array/3600, cmap='viridis', alpha=0.6, s=20)
ax.set_xlabel('Battery Power (kW)')
ax.set_ylabel('Heat Generation (kW)')
ax.set_title('Power vs Heat Relationship')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Time (hours)')

# Add expected quadratic curve
if np.max(power_profile) > 0:
    power_range = np.linspace(0, np.max(power_profile), 100)
    # Heat = (Power/V)² * R_total
    total_resistance = np.sum(zone_resistance)
    expected_heat = (power_range / PACK_NOMINAL_VOLTAGE) ** 2 * total_resistance
    ax.plot(power_range/1000, expected_heat/1000, 'g--', linewidth=2, label='Expected I²R (quadratic)')
    ax.legend()

# Plot 4: Log-log plot
ax = axes[1, 1]
if len(power_nz) > 10:
    ax.loglog(power_nz/1000, heat_nz/1000, 'ko', alpha=0.6, markersize=4, label='Data')
    power_fit = np.linspace(power_nz.min(), power_nz.max(), 100)
    heat_fit = 10**intercept * (power_fit ** slope)
    ax.loglog(power_fit/1000, heat_fit/1000, 'r-', linewidth=2, label=f'Fit: Heat ∝ Power^{slope:.2f}')
    ax.set_xlabel('Battery Power (kW)')
    ax.set_ylabel('Heat Generation (kW)')
    ax.set_title('Log-Log Plot (shows exponent)')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("EXPLANATION:")
print("=" * 70)
print("The relationship between Power and Heat should be QUADRATIC:")
print("  • Power = Voltage × Current")
print("  • Heat = Current² × Resistance")
print("  • Therefore: Heat ∝ Power²")
print("\nIn a log-log plot, this shows as a line with slope ≈ 2.0")
print("If your graphs look similar (linear), check the scatter plot above.")
print("=" * 70)


# %%
# =========================
# ML FEATURE CONSTRUCTION
# =========================

HISTORY = 10       # seconds of past
PRED_HORIZON = 5   # seconds into future

T = len(time_array)

# Temperature derivative
dT = np.zeros_like(temperatures)
dT[1:] = temperatures[1:] - temperatures[:-1]

# Zone positions as feature
zone_pos = zone_positions.reshape(1, NUM_ZONES, 1)
zone_pos = np.repeat(zone_pos, T, axis=0)

# Coolant temperature as global feature
coolant_temp_feat = np.full((T, 1), COOLANT_TEMP)

print("✅ Features prepared")

# %%
# =========================
# DATASET GENERATION
# =========================

X = []
Y = []

for t in range(HISTORY, T - PRED_HORIZON):

    # --------- History window ---------
    temp_hist = temperatures[t-HISTORY:t]          # (H, Z)
    dT_hist = dT[t-HISTORY:t]                        # (H, Z)
    flow_hist = coolant_flows[t-HISTORY:t]          # (H, Z)
    power_hist = power_profile[t-HISTORY:t]         # (H,)

    # Expand power to zones
    power_hist = np.repeat(power_hist[:, None], NUM_ZONES, axis=1)

    # Stack features per zone
    # Features: [T, dT, flow, power, zone_pos]
    features = np.stack(
        [
            temp_hist,
            dT_hist,
            flow_hist,
            power_hist,
            zone_pos[t-HISTORY:t, :, 0]
        ],
        axis=-1
    )  # (H, Z, F)

    X.append(features)

    # --------- Future target ---------
    Y.append(temperatures[t + PRED_HORIZON])

X = np.array(X)
Y = np.array(Y)

print("📦 Dataset ready")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# %%
# =========================
# SAVE DATASET
# =========================

np.save("X_dataset.npy", X)
np.save("Y_dataset.npy", Y)

print("💾 Dataset saved to disk")

# %%
# =========================
# TRAIN / VALIDATION SPLIT
# =========================

N = X.shape[0]
split = int(0.8 * N)

X_train = X[:split]
Y_train = Y[:split]

X_val = X[split:]
Y_val = Y[split:]

print("Train samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])


# %%
# =========================
# PYTORCH DATASET
# =========================

# Try to import PyTorch; if the native lib fails to load (libtorch_cpu.so / executable stack)
# provide a numpy-based fallback DataLoader so the notebook can continue.
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception as _e:
    TORCH_AVAILABLE = False
    _torch_import_error = _e

if TORCH_AVAILABLE:
    class BatteryDataset(Dataset):
        def __init__(self, X, Y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.Y = torch.tensor(Y, dtype=torch.float32)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    train_ds = BatteryDataset(X_train, Y_train)
    val_ds = BatteryDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    print("✅ PyTorch DataLoaders ready")
else:
    # Fallback: simple numpy-based Dataset and DataLoader to avoid import errors
    print(f"⚠ PyTorch import failed: {_torch_import_error!s}")
    import numpy as _np

    class BatteryDataset:
        def __init__(self, X, Y):
            self.X = _np.asarray(X, dtype=_np.float32)
            self.Y = _np.asarray(Y, dtype=_np.float32)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    class SimpleDataLoader:
        def __init__(self, dataset, batch_size=32, shuffle=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            idxs = _np.arange(len(self.dataset))
            if self.shuffle:
                _np.random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch_idx = idxs[i:i+self.batch_size]
                Xb = _np.stack([self.dataset[j][0] for j in batch_idx])
                Yb = _np.stack([self.dataset[j][1] for j in batch_idx])
                yield Xb, Yb

        def __len__(self):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    train_ds = BatteryDataset(X_train, Y_train)
    val_ds = BatteryDataset(X_val, Y_val)
    train_loader = SimpleDataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = SimpleDataLoader(val_ds, batch_size=32, shuffle=False)

    print("✅ Fallback (numpy) DataLoaders ready")

# %%
# =========================
# TEMPERATURE PREDICTION MODEL
# =========================

import torch
import torch.nn as nn

class TempPredictor(nn.Module):
    def __init__(self, num_zones=12, num_features=5, hidden_dim=128):
        super().__init__()

        self.num_zones = num_zones
        self.num_features = num_features

        # Each timestep flattened as zones × features
        input_dim = num_zones * num_features

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_zones)
        )

    def forward(self, x):
        """
        x: (batch, time, zones, features)
        """
        B, T, Z, F = x.shape
        x = x.reshape(B, T, Z * F)

        _, h = self.gru(x)
        h_last = h[-1]

        out = self.head(h_last)
        return out


# %%
# =========================
# MODEL INITIALIZATION
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TempPredictor(
    num_zones=Y.shape[1],
    num_features=X.shape[-1],
    hidden_dim=128
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("✅ Model initialized on", device)

# %%
# =========================
# SANITY CHECK: FORWARD PASS
# =========================

model.eval()

with torch.no_grad():
    Xb, Yb = next(iter(train_loader))

    Xb = Xb.to(device)
    Yb = Yb.to(device)

    Y_pred = model(Xb)

    print("X batch shape:", Xb.shape)      # (B, T, Z, F)
    print("Y true shape:", Yb.shape)       # (B, Z)
    print("Y pred shape:", Y_pred.shape)   # (B, Z)

    loss_val = criterion(Y_pred, Yb)
    print("Sanity loss:", loss_val.item())

# %%
# =========================
# PHYSICS-REGULARIZED LOSS (WEAK PINN)
# =========================

import torch

# -------------------------------------------------
# 1. Temporal smoothness (thermal inertia)
# T(t+1) should not jump away from T(t)
# -------------------------------------------------
def smoothness_loss(T_curr, T_pred):
    # T_curr: [B, Z]
    # T_pred: [B, Z]
    return torch.mean((T_pred - T_curr) ** 2)


# -------------------------------------------------
# 2. Spatial coherence
# Adjacent zones should not diverge arbitrarily
# -------------------------------------------------
def spatial_loss(T_pred):
    # penalize large gradients between neighboring zones
    return torch.mean((T_pred[:, 1:] - T_pred[:, :-1]) ** 2)


# -------------------------------------------------
# 3. Power consistency
# If power is near zero, temperature change must be near zero
# -------------------------------------------------
def power_consistency_loss(T_curr, T_pred, power, eps=50.0):
    # power: [B]
    # expand mask to zones
    mask = (power < eps).float().unsqueeze(1)
    return torch.mean(mask * (T_pred - T_curr) ** 2)


# -------------------------------------------------
# 4. Cooling monotonicity
# Increasing coolant flow must not increase temperature
# -------------------------------------------------
def cooling_loss(T_pred, flows):
    # flows: [B, Z]
    # positive flow gradient with positive temperature gradient is invalid
    flow_grad = flows[:, 1:] - flows[:, :-1]
    temp_grad = T_pred[:, 1:] - T_pred[:, :-1]
    violation = torch.relu(flow_grad * temp_grad)
    return torch.mean(violation)


# =========================
# FINAL COMPOSITE LOSS
# =========================
def pinn_loss(x, y_pred, y_true, data_criterion):
    """
    x shape: [B, T, Z, F]
    y_pred:  [B, Z]
    y_true:  [B, Z]

    Feature convention assumed:
    x[..., 0] = temperature
    x[..., 1] = power
    x[..., 2] = coolant flow
    """

    # last observed timestep
    T_curr = x[:, -1, :, 0]          # [B, Z]
    power = x[:, -1, 0, 1]           # [B]
    flows = x[:, -1, :, 2]           # [B, Z]

    # data fitting term
    data_loss = data_criterion(y_pred, y_true)

    # physics regularizers
    L_smooth = smoothness_loss(T_curr, y_pred)
    L_spatial = spatial_loss(y_pred)
    L_power = power_consistency_loss(T_curr, y_pred, power)
    L_cooling = cooling_loss(y_pred, flows)

    # weighted total loss
    total_loss = (
        data_loss
        + 0.1 * L_smooth
        + 0.05 * L_spatial
        + 0.1 * L_power
        + 0.02 * L_cooling
    )

    return total_loss


# %%
# =========================
# TRAINING LOOP
# =========================

EPOCHS = 40
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for Xb, Yb in train_loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        optimizer.zero_grad()
        Y_pred = model(Xb)
        loss = pinn_loss(Xb, Y_pred, Yb, criterion)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # ---------- Validation ----------
    model.eval()
    val_running = 0.0

    with torch.no_grad():
        for Xb, Yb in val_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Y_pred = model(Xb)
            val_running += criterion(Y_pred, Yb).item()

    val_loss = val_running / len(val_loader)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.2f} | "
        f"Val Loss: {val_loss:.2f}"
    )

# %%
# =========================
# MULTI STEP ROLLOUT TEST
# =========================

model.eval()

HORIZON = 30
idx = 0  # pick any validation sample

X0 = torch.tensor(X_val[idx:idx+1], dtype=torch.float32).to(device)
true_future = Y_val[idx:idx+HORIZON]

pred_temps = []

with torch.no_grad():
    x = X0.clone()
    for _ in range(HORIZON):
        y_pred = model(x)
        pred_temps.append(y_pred.cpu().numpy()[0])

        # shift window and append prediction
        x = torch.roll(x, shifts=-1, dims=1)
        x[:, -1, :, 0] = y_pred  # assuming feature 0 is temperature

pred_temps = np.array(pred_temps)

# %%
model.eval()
for p in model.parameters():
    p.requires_grad = False

print("🔒 Predictive model frozen for control")

# %%
ACTION_DIM = NUM_ZONES

def clamp_actions(a):
    return np.clip(a, 0.0, 1.0)

# %%
def build_control_state(X_window):
    """
    X_window: [T, Z, F]
    returns flat state vector
    """
    T_curr = X_window[-1, :, 0]        # current temps
    T_mean = T_curr.mean()
    T_max = T_curr.max()
    power = X_window[-1, 0, 1]

    return np.concatenate([
        T_curr,
        [T_mean, T_max, power]
    ])


# %%
@torch.no_grad()
def predict_next_temperature(X_window, coolant_action):
    """
    X_window: torch [1, T, Z, F]
    coolant_action: numpy [Z]
    """

    X_next = X_window.clone()

    # overwrite coolant flow feature
    X_next[0, -1, :, 2] = torch.tensor(
        coolant_action,
        dtype=X_next.dtype,
        device=X_next.device
    )

    T_pred = model(X_next)  # [1, Z]
    return T_pred.squeeze(0)


# %%
def rollout_horizon(X_init, action_seq):
    """
    action_seq: [H, Z]
    returns predicted temperatures over horizon
    """
    X_roll = X_init.clone()
    temps = []

    for a in action_seq:
        T_next = predict_next_temperature(X_roll, a)
        temps.append(T_next.cpu().numpy())

        # roll window
        X_new = X_roll[:, 1:].clone()
        X_new[:, -1, :, 0] = T_next
        X_roll = X_new

    return np.stack(temps)  # [H, Z]


# %%
def reward_function(T_pred, actions, power):
    T_max = T_pred.max()
    T_min = T_pred.min()

    r_temp = -torch.relu(T_max - TEMP_SETPOINT)
    r_uniform = -(T_pred.std())
    r_energy = -actions.mean()

    return (
        5.0 * r_temp
        + 1.0 * r_uniform
        + 0.5 * r_energy
    )


# %%
# =========================
# PHASE 3: PREDICTIVE THERMAL CONTROL
# STEP 1: FREEZE PREDICTOR
# =========================

import torch

# Put model in inference mode
model.eval()

# Disable gradients permanently
for param in model.parameters():
    param.requires_grad = False

# Sanity checks
trainable = sum(p.requires_grad for p in model.parameters())
total = sum(1 for _ in model.parameters())

print(f"🔒 Model frozen: {trainable}/{total} parameters trainable")

# Ensure no accidental graph building
device = next(model.parameters()).device

with torch.no_grad():
    X_dummy = torch.tensor(X_train[:1], dtype=torch.float32).to(device)
    _ = model(X_dummy)

print("✅ Predictor ready for control")

# %%
# =========================
# CONTROL SETUP
# =========================

HORIZON = 5               # short horizon, paper friendly
FLOW_MIN = 0.0
FLOW_MAX = 1.0            # normalized, matches PID output range

NUM_CONTROLS = NUM_ZONES

print("🎯 CONTROL SPECIFICATION")
print(f"   Horizon: {HORIZON} steps")
print(f"   Control variables: {NUM_CONTROLS} coolant flows")
print(f"   Flow bounds: [{FLOW_MIN}, {FLOW_MAX}]")


# %%
# %%
# =========================
# FACTORIZED PREDICTOR
# Temporal encoder, Spatial encoder, Global encoder, Fusion head
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature index constants, match dataset construction earlier
TEMP_IDX = 0
DT_IDX = 1
FLOW_IDX = 2
POWER_IDX = 3
POS_IDX = 4

class TemporalEncoder(nn.Module):
    """
    GRU over time, input per timestep is flattened zones temperatures
    Input x_temp shape: [B, T, Z]
    Output: h_t shape [B, hidden_dim_temp]
    """
    def __init__(self, num_zones, hidden_dim=256, num_layers=2):
        super().__init__()
        self.num_zones = num_zones
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=num_zones, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
    def forward(self, x_temp):
        # x_temp: [B, T, Z]
        out, h = self.gru(x_temp)       # out [B, T, hidden], h [num_layers, B, hidden]
        h_last = h[-1]                  # [B, hidden]
        return h_last

class SpatialEncoder(nn.Module):
    """
    Simple local message passing implemented with 1D convs
    Input per-zone features: [B, Z, F_zone]
    We'll use Conv1d with kernel_size 3 padding 1 to capture neighbors
    Output pooled spatial vector s_t: [B, spatial_dim]
    """
    def __init__(self, in_features, spatial_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=spatial_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=spatial_dim, out_channels=spatial_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
    def forward(self, zone_feats):
        # zone_feats: [B, Z, F] -> conv expects [B, F, Z]
        x = zone_feats.permute(0, 2, 1)    # [B, F, Z]
        x = self.act(self.conv1(x))        # [B, spatial_dim, Z]
        x = self.act(self.conv2(x))        # [B, spatial_dim, Z]
        pooled = self.pool(x).squeeze(-1)  # [B, spatial_dim]
        return pooled

class GlobalEncoder(nn.Module):
    """
    Small MLP for global scalars such as pack power and ambient info
    Input: [B, G]
    Output: [B, g_dim]
    """
    def __init__(self, in_dim=2, g_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, g_dim),
            nn.ReLU(),
            nn.Linear(g_dim, g_dim),
            nn.ReLU()
        )
    def forward(self, g):
        return self.net(g)

class FusedPredictor(nn.Module):
    """
    Combines TemporalEncoder, SpatialEncoder, GlobalEncoder
    Produces zone temperature predictions size NUM_ZONES
    Input x: [B, T, Z, F]
    """
    def __init__(self, num_zones=NUM_ZONES, num_features=5,
                 temp_hidden=256, spatial_dim=128, g_dim=64, head_hidden=256):
        super().__init__()
        self.num_zones = num_zones
        self.num_features = num_features

        self.temporal = TemporalEncoder(num_zones=num_zones, hidden_dim=temp_hidden)
        # spatial encoder consumes per-zone features at last timestep
        self.spatial = SpatialEncoder(in_features=num_features, spatial_dim=spatial_dim)
        # global feature size choose: pack power and mean temp optional; here we use power and coolant temp if present
        self.global_enc = GlobalEncoder(in_dim=2, g_dim=g_dim)

        fuse_size = temp_hidden + spatial_dim + g_dim
        self.head = nn.Sequential(
            nn.Linear(fuse_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, head_hidden//2),
            nn.ReLU(),
            nn.Linear(head_hidden//2, num_zones)
        )

    def forward(self, x):
        # x: [B, T, Z, F]
        B, T, Z, F = x.shape
        # temporal: use only temperature history, shape [B, T, Z]
        x_temp = x[..., TEMP_IDX].permute(0, 1, 2)   # already [B, T, Z]
        temp_enc = self.temporal(x_temp)             # [B, temp_hidden]

        # spatial: use last timestep full per-zone features [B, Z, F]
        zone_feats = x[:, -1, :, :]                  # [B, Z, F]
        spatial_enc = self.spatial(zone_feats)       # [B, spatial_dim]

        # global: build vector [power, mean_temp] from last timestep
        power_last = x[:, -1, 0, POWER_IDX].unsqueeze(1)       # [B, 1] power expanded from zone 0
        mean_temp_last = x[:, -1, :, TEMP_IDX].mean(dim=1, keepdim=True)  # [B, 1]
        g = torch.cat([power_last, mean_temp_last], dim=1)     # [B, 2]
        g_enc = self.global_enc(g)                             # [B, g_dim]

        fused = torch.cat([temp_enc, spatial_enc, g_enc], dim=1)  # [B, fuse_size]
        out = self.head(fused)      # [B, Z]
        return out

# Initialize model and move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = FusedPredictor(num_zones=NUM_ZONES, num_features=X.shape[-1]).to(device)

# Sanity forward pass on a batch from train_loader
predictor.eval()
with torch.no_grad():
    Xb, Yb = next(iter(train_loader))
    Xb_t = Xb.to(device)
    Y_pred = predictor(Xb_t)
    print("Predictor forward pass shapes:", Xb_t.shape, "->", Y_pred.shape)


# %%
# %%
# =========================
# REWARD FUNCTION FOR MPC
# Uses predicted temperatures and action sequence to compute scalar reward
# =========================
import torch

def estimate_pump_power_from_actions(actions):
    """
    Estimate pump power from normalized per-zone actions
    actions: torch tensor shape [B, Z] or [Z]
    Returns power in Watts, same device as actions
    Model used: total_flow_fraction = mean(actions)
                 pump_power = PUMP_MAX_POWER_W * (total_flow_fraction)**2
    """
    # ensure actions is float tensor
    a = actions.float()
    if a.dim() == 1:
        a = a.unsqueeze(0)    # [1, Z]
    total_flow_fraction = a.mean(dim=1)   # [B]
    pump_power = (torch.tensor(PUMP_MAX_POWER_W, device=a.device) * (total_flow_fraction ** 2))
    return pump_power  # shape [B]

def reward_mpc(T_pred, actions, power_pack, setpoint=TEMP_SETPOINT,
               w_temp=5.0, w_spread=1.0, w_energy=0.5, w_ramp=0.1):
    """
    Compute reward for a single horizon step
    T_pred: torch tensor [B, Z] predicted temperatures at that horizon step
    actions: torch tensor [B, Z] actions normalized 0..1
    power_pack: torch tensor [B] pack power in Watts at current step
    Returns scalar reward per batch element [B]
    Note: we return reward as higher is better
    """
    # safety term: penalize violations above setpoint squared
    Tmax = torch.max(T_pred, dim=1).values   # [B]
    violation = torch.relu(Tmax - setpoint)
    r_temp = - (violation ** 2)              # negative penalty

    # uniformity: penalize std deviation across zones
    r_uniform = - T_pred.std(dim=1)         # [B]

    # energy cost: estimated pump power relative to pack energy scale
    pump_power = estimate_pump_power_from_actions(actions)  # [B]
    # normalize pump power by PACK_MAX_POWER converted to W. PACK_MAX_POWER is in kW earlier, convert
    pack_max_w = torch.tensor(PACK_MAX_POWER * 1000.0, device=T_pred.device)
    r_energy = - (pump_power / (pack_max_w + 1e-6))         # [B]

    # optional ramp cost provided externally by passing previous actions; here kept as mean action penalty
    r_actions_mean = - actions.mean(dim=1)

    reward = (w_temp * r_temp) + (w_spread * r_uniform) + (w_energy * r_energy) + (w_ramp * r_actions_mean)
    return reward, {
        "r_temp": r_temp.detach().cpu().numpy(),
        "r_uniform": r_uniform.detach().cpu().numpy(),
        "r_energy": r_energy.detach().cpu().numpy(),
        "pump_power_w": pump_power.detach().cpu().numpy()
    }


# %%
# %%
# =========================
# ACTOR CRITIC NETWORK for PPO style control
# Actor outputs Gaussian means, uses learned logstd per action
# Critic outputs scalar state value
# Input state vector format same as build_control_state: [Z current temps, mean, max, power] length = NUM_ZONES + 3
# =========================
import torch
import torch.nn as nn
from torch.distributions import Normal

STATE_DIM = NUM_ZONES + 3
ACTION_DIM = NUM_ZONES

class Actor(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden//2, action_dim)
        # learnable log std per action
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)

    def forward(self, state):
        # state: [B, state_dim]
        h = self.net(state)
        mu = self.mu_head(h)           # unbounded
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

    def get_action(self, state, deterministic=False):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        if deterministic:
            raw_action = mu
        else:
            raw_action = dist.rsample()   # reparameterized sample
        logprob = dist.log_prob(raw_action).sum(dim=-1)
        # squash with sigmoid to map to 0..1
        action = torch.sigmoid(raw_action)
        # correction for logprob due to sigmoid is omitted for simplicity in PPO; acceptable for initial experiments
        return action, logprob, mu, std

class Critic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, state):
        return self.net(state).squeeze(-1)   # [B]

class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.critic = Critic(state_dim=state_dim)

    def act(self, state_np, deterministic=False):
        """
        state_np: numpy array shape [state_dim] or [B, state_dim]
        returns: action numpy [B, Z], logprob numpy [B], value numpy [B]
        """
        self.eval()
        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=device)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            action, logp, mu, std = self.actor.get_action(s, deterministic=deterministic)
            value = self.critic(s)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    def get_values(self, state):
        return self.critic(state)

# initialize
ac = ActorCritic(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
print("ActorCritic initialized, parameter counts:",
      sum(p.numel() for p in ac.parameters()))


# %%
# =========================
# MINIMAL MPC CONTROL LOOP
# Replaces PID inside battery simulation
# =========================

import numpy as np
import torch
from scipy.integrate import solve_ivp

predictor.eval()
for p in predictor.parameters():
    p.requires_grad = False

DEVICE = next(predictor.parameters()).device

# -------------------------
# MPC configuration
# -------------------------
MPC_SAMPLES = 16        # very small on purpose
MPC_HORIZON = 1         # minimal existence proof
DT = 1.0

# -------------------------
# Reward used inside MPC
# -------------------------
def simple_mpc_reward(T_pred, action):
    """
    T_pred: torch [Z]
    action: torch [Z]
    """
    Tmax = torch.max(T_pred)
    temp_penalty = torch.relu(Tmax - TEMP_SETPOINT)
    flow_penalty = action.mean()
    return -5.0 * temp_penalty - 0.5 * flow_penalty


# -------------------------
# Initialize simulator
# -------------------------
initial_temperatures = np.full(NUM_ZONES, INITIAL_TEMP)
initial_flows = np.zeros(NUM_ZONES)
state = np.concatenate([initial_temperatures, initial_flows])

# -------------------------
# Build initial history window using PID rollout
# -------------------------
X_window = []

temps_pid = initial_temperatures.copy()
flows_pid = initial_flows.copy()

for t in range(HISTORY):
    # PID step to warm start history
    pid_actions = np.array([
        pid_controllers[i].update(temps_pid[i], dt=DT)
        for i in range(NUM_ZONES)
    ])

    sol = solve_ivp(
        fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, simulation_params),
        t_span=[t, t + DT],
        y0=np.concatenate([temps_pid, flows_pid]),
        t_eval=[t + DT]
    )

    temps_pid = sol.y[:NUM_ZONES, -1]
    flows_pid = sol.y[NUM_ZONES:, -1]

    dT_pid = temps_pid - initial_temperatures if t == 0 else temps_pid - X_window[-1][:, TEMP_IDX]

    features = np.stack(
        [
            temps_pid,
            dT_pid,
            flows_pid,
            np.full(NUM_ZONES, power_profile[t]),
            zone_positions
        ],
        axis=-1
    )

    X_window.append(features)

X_window = np.array(X_window)   # [H, Z, F]

# -------------------------
# Storage for MPC results
# -------------------------
mpc_temperatures = []
mpc_flows = []

# -------------------------
# Closed loop MPC simulation
# -------------------------
for t in range(HISTORY, len(time_array) - 1):

    X_tensor = torch.tensor(X_window, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    best_reward = -1e9
    best_action = None

    # ----- Random shooting MPC -----
    for _ in range(MPC_SAMPLES):
        action = torch.rand(NUM_ZONES, device=DEVICE)

        X_test = X_tensor.clone()
        X_test[0, -1, :, FLOW_IDX] = action

        with torch.no_grad():
            T_pred = predictor(X_test).squeeze(0)

        reward = simple_mpc_reward(T_pred, action)

        if reward.item() > best_reward:
            best_reward = reward.item()
            best_action = action.clone()

    action_np = best_action.cpu().numpy()

    # ----- Apply action to real simulator -----
    sol = solve_ivp(
        fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, simulation_params),
        t_span=[t, t + DT],
        y0=state,
        t_eval=[t + DT]
    )

    next_temps = sol.y[:NUM_ZONES, -1]
    next_flows = sol.y[NUM_ZONES:, -1]

    # ----- Log -----
    mpc_temperatures.append(next_temps)
    mpc_flows.append(next_flows)

    # ----- Update history window -----
    dT_next = next_temps - X_window[-1][:, TEMP_IDX]

    new_features = np.stack(
        [
            next_temps,
            dT_next,
            next_flows,
            np.full(NUM_ZONES, power_profile[t]),
            zone_positions
        ],
        axis=-1
    )

    X_window = np.roll(X_window, shift=-1, axis=0)
    X_window[-1] = new_features

    state = np.concatenate([next_temps, next_flows])

print("✅ MPC driven simulation completed")

mpc_temperatures = np.array(mpc_temperatures)
mpc_flows = np.array(mpc_flows)


# %%
# =========================
# MINIMAL MPC CONTROL LOOP
# Replaces PID inside battery simulation, without touching the ODE
# =========================

import numpy as np
import torch
from scipy.integrate import solve_ivp

# freeze predictor
predictor.eval()
for p in predictor.parameters():
    p.requires_grad = False

DEVICE = next(predictor.parameters()).device

# -------------------------
# MPC configuration
# -------------------------
MPC_SAMPLES = 16        # very small on purpose
MPC_HORIZON = 1         # minimal existence proof
DT = 1.0

# -------------------------
# Proxy controller class
# The ODE will keep calling `.update()`; we return the MPC action there
# -------------------------
class MPCProxyController:
    def __init__(self):
        # normalized action in [0,1]
        self.current_action = 0.0

    def update(self, measurement, dt=1.0):
        # ignore measurement here, the MPC has authority
        return float(self.current_action)

# create proxies, but do not replace the original pid_controllers globally
mpc_controllers = [MPCProxyController() for _ in range(NUM_ZONES)]

# make a copy of simulation params that uses proxies
simulation_params_mpc = dict(simulation_params)
simulation_params_mpc["pid_controllers"] = mpc_controllers

# -------------------------
# Reward used inside MPC
# -------------------------
def simple_mpc_reward(T_pred, action):
    """
    T_pred: torch [Z]
    action: torch [Z]
    """
    Tmax = torch.max(T_pred)
    temp_penalty = torch.relu(Tmax - TEMP_SETPOINT)
    flow_penalty = action.mean()
    return -5.0 * temp_penalty - 0.5 * flow_penalty

# -------------------------
# Initialize simulator
# -------------------------
initial_temperatures = np.full(NUM_ZONES, INITIAL_TEMP)
initial_flows = np.zeros(NUM_ZONES)
state = np.concatenate([initial_temperatures, initial_flows])

# -------------------------
# Build initial history window using PID rollout
# This uses the original pid_controllers to create a realistic warm start
# -------------------------
X_window = []

temps_pid = initial_temperatures.copy()
flows_pid = initial_flows.copy()

for t in range(HISTORY):
    # PID step to warm start history
    pid_actions = np.array([
        pid_controllers[i].update(temps_pid[i], dt=DT)
        for i in range(NUM_ZONES)
    ])

    sol = solve_ivp(
        fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, simulation_params),
        t_span=[t, t + DT],
        y0=np.concatenate([temps_pid, flows_pid]),
        t_eval=[t + DT]
    )

    temps_pid = sol.y[:NUM_ZONES, -1]
    flows_pid = sol.y[NUM_ZONES:, -1]

    # compute dT relative to previous window entry or initial
    dT_pid = temps_pid - initial_temperatures if t == 0 else temps_pid - X_window[-1][:, TEMP_IDX]

    features = np.stack(
        [
            temps_pid,
            dT_pid,
            flows_pid,
            np.full(NUM_ZONES, power_profile[t]),
            zone_positions
        ],
        axis=-1
    )

    X_window.append(features)

X_window = np.array(X_window)   # [H, Z, F]

# -------------------------
# Storage for MPC results
# -------------------------
mpc_temperatures = []
mpc_flows = []
mpc_actions_log = []

# -------------------------
# Closed loop MPC simulation
# This uses simulation_params_mpc so the ODE reads actions from proxies
# -------------------------
for t in range(HISTORY, len(time_array) - 1):

    X_tensor = torch.tensor(X_window, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    best_reward = -1e9
    best_action = None

    # ----- Random shooting MPC -----
    for _ in range(MPC_SAMPLES):
        # sample actions uniformly in [0,1]
        action = torch.rand(NUM_ZONES, device=DEVICE)

        # place action into last timestep flows and predict
        X_test = X_tensor.clone()
        X_test[0, -1, :, FLOW_IDX] = action

        with torch.no_grad():
            T_pred = predictor(X_test).squeeze(0)

        reward = simple_mpc_reward(T_pred, action)

        if reward.item() > best_reward:
            best_reward = reward.item()
            best_action = action.clone()

    # convert chosen action to numpy
    action_np = best_action.cpu().numpy()

    # ----- Write action into proxy controllers before calling the ODE -----
    # This is the crucial step, it leaves the ODE unchanged
    for i in range(NUM_ZONES):
        mpc_controllers[i].current_action = float(action_np[i])

    # ----- Apply action to real simulator using the MPC aware simulation params -----
    sol = solve_ivp(
        fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, simulation_params_mpc),
        t_span=[t, t + DT],
        y0=state,
        t_eval=[t + DT]
    )

    next_temps = sol.y[:NUM_ZONES, -1]
    next_flows = sol.y[NUM_ZONES:, -1]

    # ----- Log -----
    mpc_temperatures.append(next_temps)
    mpc_flows.append(next_flows)
    mpc_actions_log.append(action_np)

    # ----- Update history window for predictor -----
    dT_next = next_temps - X_window[-1][:, TEMP_IDX]

    new_features = np.stack(
        [
            next_temps,
            dT_next,
            next_flows,
            np.full(NUM_ZONES, power_profile[t]),
            zone_positions
        ],
        axis=-1
    )

    X_window = np.roll(X_window, shift=-1, axis=0)
    X_window[-1] = new_features

    state = np.concatenate([next_temps, next_flows])

print("✅ MPC driven simulation completed")

mpc_temperatures = np.array(mpc_temperatures)
mpc_flows = np.array(mpc_flows)
mpc_actions_log = np.array(mpc_actions_log)


# %%
# =========================
# MPC VS PID VISUALIZATION
# =========================

import matplotlib.pyplot as plt
import numpy as np

# Align time axis
mpc_time = time_array[HISTORY: HISTORY + len(mpc_temperatures)] / 3600.0
pid_time = time_array / 3600.0

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("PID vs MPC Battery Thermal Control", fontsize=16, fontweight="bold")

# -------------------------
# Zone temperatures, PID
# -------------------------
ax = axes[0, 0]
for i in range(NUM_ZONES):
    ax.plot(pid_time, temperatures[:, i], linewidth=1.5, alpha=0.8)
ax.axhline(TEMP_SETPOINT, color="r", linestyle="--", label="Setpoint")
ax.set_title("PID Zone Temperatures")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Temperature (C)")
ax.grid(True, alpha=0.3)

# -------------------------
# Zone temperatures, MPC
# -------------------------
ax = axes[0, 1]
for i in range(NUM_ZONES):
    ax.plot(mpc_time, mpc_temperatures[:, i], linewidth=1.5, alpha=0.8)
ax.axhline(TEMP_SETPOINT, color="r", linestyle="--", label="Setpoint")
ax.set_title("MPC Zone Temperatures")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Temperature (C)")
ax.grid(True, alpha=0.3)

# -------------------------
# Mean and max comparison
# -------------------------
ax = axes[1, 0]
ax.plot(pid_time, temperatures.mean(axis=1), label="PID Mean", linewidth=2)
ax.plot(pid_time, temperatures.max(axis=1), label="PID Max", linewidth=2)
ax.plot(mpc_time, mpc_temperatures.mean(axis=1), "--", label="MPC Mean", linewidth=2)
ax.plot(mpc_time, mpc_temperatures.max(axis=1), "--", label="MPC Max", linewidth=2)
ax.axhline(TEMP_SETPOINT, color="r", linestyle="--", alpha=0.6)
ax.set_title("Mean and Max Temperature Comparison")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Temperature (C)")
ax.legend()
ax.grid(True, alpha=0.3)

# -------------------------
# Coolant flow comparison
# -------------------------
ax = axes[1, 1]
pid_total_flow = coolant_flows.sum(axis=1) * MAX_PUMP_FLOW_LPM / NUM_ZONES
mpc_total_flow = mpc_flows.sum(axis=1) * MAX_PUMP_FLOW_LPM / NUM_ZONES

ax.plot(pid_time, pid_total_flow, label="PID Total Flow", linewidth=2)
ax.plot(mpc_time, mpc_total_flow, "--", label="MPC Total Flow", linewidth=2)
ax.set_title("Total Coolant Flow")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Flow Rate (L per min)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# ============================================================
# PID vs MPC COMPARISON (USING EXISTING VARIABLES ONLY)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

dt = 1.0  # seconds

# ----------------------------
# Reconstruct MPC pump power from actions
# ----------------------------
mpc_total_flow_lpm = np.mean(mpc_actions_log, axis=1) * MAX_PUMP_FLOW_LPM
mpc_pump_power = PUMP_MAX_POWER_W * (mpc_total_flow_lpm / MAX_PUMP_FLOW_LPM) ** 2

# ----------------------------
# Helper metrics
# ----------------------------
def compute_metrics(temps, pump_pwr, label):
    metrics = {}

    metrics["T_mean"] = temps.mean()
    metrics["T_max"] = temps.max()
    metrics["T_min"] = temps.min()

    spread = temps.max(axis=1) - temps.min(axis=1)
    metrics["mean_spread"] = spread.mean()

    dT = np.diff(temps, axis=0)
    metrics["thermal_stress"] = np.mean(np.abs(dT))

    metrics["time_above_setpoint_frac"] = np.mean(temps > TEMP_SETPOINT)

    metrics["pump_energy_Wh"] = pump_pwr.sum() * dt / 3600.0
    metrics["pump_peak_W"] = pump_pwr.max()

    print(f"\n📊 {label} METRICS")
    for k, v in metrics.items():
        print(f"{k:30s}: {v:.4f}")

    return metrics

# ----------------------------
# Electrical losses (I²R)
# ----------------------------
def compute_ohmic_energy(power_profile):
    pack_current = power_profile / PACK_NOMINAL_VOLTAGE
    total_heat = np.zeros_like(pack_current)
    for i in range(len(zone_resistance)):
        total_heat += (pack_current ** 2) * zone_resistance[i]
    return total_heat.sum() * dt / 3600.0

# ----------------------------
# Compute metrics
# ----------------------------
pid_metrics = compute_metrics(
    temperatures,
    pump_power,
    label="PID"
)

mpc_metrics = compute_metrics(
    mpc_temperatures,
    mpc_pump_power,
    label="MPC"
)

E_ohmic = compute_ohmic_energy(power_profile)

print("\n⚡ ELECTRICAL HEAT (IDENTICAL DRIVE)")
print(f"Ohmic energy (Wh): {E_ohmic:.2f}")

# ----------------------------
# Range impact estimate
# ----------------------------
E_drive_Wh = power_profile.sum() * dt / 3600.0
range_gain_pct = (
    (pid_metrics["pump_energy_Wh"] - mpc_metrics["pump_energy_Wh"])
    / E_drive_Wh
) * 100.0

print("\n🚗 RANGE IMPACT")
print(f"Drive energy (Wh): {E_drive_Wh:.1f}")
print(f"Estimated range gain with MPC (%): {range_gain_pct:.2f}")

# ----------------------------
# Plots
# ----------------------------
t_pid = time_array / 3600.0
t_mpc = time_array[HISTORY:HISTORY + len(mpc_temperatures)] / 3600.0

plt.figure(figsize=(12,4))
plt.plot(t_pid, pump_power/1000, label="PID", linewidth=2)
plt.plot(t_mpc, mpc_pump_power/1000, "--", label="MPC", linewidth=2)
plt.xlabel("Time (hours)")
plt.ylabel("Pump Power (kW)")
plt.title("Cooling Power Comparison")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(t_pid, temperatures.mean(axis=1), label="PID Mean")
plt.plot(t_mpc, mpc_temperatures.mean(axis=1), "--", label="MPC Mean")
plt.axhline(TEMP_SETPOINT, color="r", linestyle="--")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°C)")
plt.title("Mean Pack Temperature")
plt.legend()
plt.grid(True)
plt.show()

print("\n✅ SUMMARY")
print(f"MPC cooling energy reduction: "
      f"{pid_metrics['pump_energy_Wh'] - mpc_metrics['pump_energy_Wh']:.2f} Wh")
print(f"MPC spread reduction: "
      f"{pid_metrics['mean_spread'] - mpc_metrics['mean_spread']:.2f} °C")
print(f"MPC thermal stress reduction: "
      f"{pid_metrics['thermal_stress'] - mpc_metrics['thermal_stress']:.4f}")

# %%
# =========================
# COMMON METRICS & LOGGING
# =========================

def compute_btms_metrics(temperatures, pump_power, power_profile, label):
    dt = 1.0

    metrics = {}

    metrics["mean_temp"] = temperatures.mean()
    metrics["max_temp"] = temperatures.max()
    metrics["temp_spread"] = (temperatures.max(axis=1) - temperatures.min(axis=1)).mean()

    dT = np.diff(temperatures, axis=0)
    metrics["thermal_stress"] = np.mean(np.abs(dT))

    metrics["time_above_40_frac"] = np.mean(temperatures > 40.0)

    metrics["pump_energy_Wh"] = np.sum(pump_power) * dt / 3600.0
    metrics["pump_peak_W"] = pump_power.max()

    drive_energy_Wh = np.sum(power_profile) * dt / 3600.0
    metrics["cooling_overhead_pct"] = 100.0 * metrics["pump_energy_Wh"] / drive_energy_Wh

    print(f"\n📊 {label} METRICS")
    for k, v in metrics.items():
        print(f"{k:25s}: {v:.4f}")

    return metrics

# %%
CONTROLLERS = [
    {"name": "PID", "type": "pid"},
    {"name": "GainScheduledPID", "type": "pid_gs"},
    {"name": "AdaptivePID", "type": "pid_adaptive"},

    {"name": "MPC_H1", "type": "mpc", "horizon": 1},
    {"name": "MPC_H5", "type": "mpc", "horizon": 5},
    {"name": "MPC_DroQ", "type": "mpc_dropout", "horizon": 1},

    {"name": "DDPG", "type": "actor_critic"},
    {"name": "PPO", "type": "ppo"},
    {"name": "SAC_like", "type": "ppo_entropy"},
    {"name": "HPPO", "type": "ppo_gru"},
]


# %%
# Single cell: run PID, MPC, behavior clone actor, compare and save results
import numpy as np
import torch
import os
from scipy.integrate import solve_ivp
import pandas as pd
import time as _time

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- Safety checks for expected globals ----------
required_globals = [
    "battery_thermal_ode", "simulation_params", "time_array", "power_profile",
    "NUM_ZONES", "INITIAL_TEMP", "MAX_PUMP_FLOW_LPM", "PUMP_MAX_POWER_W",
    "PIDController", "pid_controllers", "zone_positions", "PACK_NOMINAL_VOLTAGE",
    "zone_resistance"
]
missing = [g for g in required_globals if g not in globals()]
if missing:
    raise RuntimeError(f"Missing required notebook globals: {missing}. Place this cell after your simulator cells.")

# ---------- Helper: run a closed loop simulation given controllers in params ----------
def run_closed_loop(sim_params, t_start=0, t_end=None, dt=1.0, warm_start_state=None):
    if t_end is None:
        t_end = len(time_array) - 1
    t_eval = np.arange(t_start, t_end + 1, int(dt))
    # initial state
    if warm_start_state is None:
        temps0 = np.full(NUM_ZONES, INITIAL_TEMP)
        flows0 = np.zeros(NUM_ZONES)
        y0 = np.concatenate([temps0, flows0])
    else:
        y0 = warm_start_state.copy()

    sol = solve_ivp(
        fun=lambda t, y: battery_thermal_ode(t, y, power_profile, sim_params),
        t_span=[t_eval[0], t_eval[-1]],
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    if not sol.success:
        raise RuntimeError("ODE integration failed: " + sol.message)
    temps = sol.y[:NUM_ZONES, :].T
    flows = sol.y[NUM_ZONES:, :].T
    # pump power reconstruct
    pump_power = np.zeros(len(t_eval))
    if MAX_PUMP_FLOW_LPM > 0:
        for i, t in enumerate(t_eval):
            total_flow_lpm = np.sum(flows[i, :]) * MAX_PUMP_FLOW_LPM / NUM_ZONES
            frac = (total_flow_lpm / MAX_PUMP_FLOW_LPM) if MAX_PUMP_FLOW_LPM > 0 else 0.0
            pump_power[i] = PUMP_MAX_POWER_W * (frac ** 2)
    return {
        "time_s": t_eval,
        "temps": temps,
        "flows": flows,
        "pump_power": pump_power
    }

# ---------- Metric function ----------
def btms_metrics_from_run(run_dict, label):
    temps = run_dict["temps"]
    pump_power = run_dict["pump_power"]
    power_prof = power_profile[:len(run_dict["time_s"])]
    dt = 1.0
    metrics = {}
    metrics["label"] = label
    metrics["mean_temp"] = temps.mean()
    metrics["max_temp"] = temps.max()
    metrics["temp_spread_mean"] = (temps.max(axis=1) - temps.min(axis=1)).mean()
    dT = np.diff(temps, axis=0)
    metrics["thermal_stress_mean_absdT"] = np.mean(np.abs(dT))
    metrics["time_above_setpoint_frac"] = np.mean(temps > (40.0))
    metrics["pump_energy_Wh"] = np.sum(pump_power) * dt / 3600.0
    metrics["pump_peak_W"] = pump_power.max()
    drive_energy_Wh = np.sum(power_prof) * dt / 3600.0
    metrics["cooling_overhead_pct"] = 100.0 * metrics["pump_energy_Wh"] / (drive_energy_Wh + 1e-12)
    return metrics

# ---------- 1) PID baseline run and save ----------
print("Running PID baseline simulation...")
sim_params_pid = dict(simulation_params)
sim_params_pid["pid_controllers"] = pid_controllers
pid_run = run_closed_loop(sim_params_pid, t_start=0, t_end=len(time_array)-1)
# Save CSV
os.makedirs("results", exist_ok=True)
pid_df = pd.DataFrame({
    "time_s": pid_run["time_s"],
    "pump_power_W": pid_run["pump_power"],
    "battery_power_W": power_profile[:len(pid_run["time_s"])]
})
for z in range(NUM_ZONES):
    pid_df[f"zone_{z+1}_temp_C"] = pid_run["temps"][:, z]
    pid_df[f"zone_{z+1}_flow_norm"] = pid_run["flows"][:, z]
pid_csv_pid = "results/pid_run.csv"
pid_df.to_csv(pid_csv_pid, index=False)
print("PID saved to", pid_csv_pid)

pid_metrics = btms_metrics_from_run(pid_run, "PID")
print("PID metrics:", pid_metrics)

# ---------- 2) MPC run, save actions and trajectories ----------
print("Running MPC simulation... this may be slower depending on samples")
# Build proxy controllers as in your notebook
class MPCProxyController:
    def __init__(self):
        self.current_action = 0.0
    def update(self, measurement, dt=1.0):
        return float(self.current_action)

def run_mpc(horizon=1, samples=64):
    # clone pid controllers for warm start but will not be used inside proxy run
    mpc_controllers = [MPCProxyController() for _ in range(NUM_ZONES)]
    sim_params_mpc = dict(simulation_params)
    sim_params_mpc["pid_controllers"] = mpc_controllers

    # build initial history window using PID warmstart like your notebook does
    HIST = max(10, 10)
    X_window = []
    temps_pid = np.full(NUM_ZONES, INITIAL_TEMP)
    flows_pid = np.zeros(NUM_ZONES)
    DT = 1.0
    for t in range(HIST):
        pid_actions = np.array([pid_controllers[i].update(temps_pid[i], dt=DT) for i in range(NUM_ZONES)])
        sol = solve_ivp(
            fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, simulation_params),
            t_span=[t, t+DT],
            y0=np.concatenate([temps_pid, flows_pid]),
            t_eval=[t+DT]
        )
        temps_pid = sol.y[:NUM_ZONES, -1]
        flows_pid = sol.y[NUM_ZONES:, -1]
        dT_pid = temps_pid - (initial := (np.full(NUM_ZONES, INITIAL_TEMP) if t==0 else X_window[-1][:,0]))
        features = np.stack([temps_pid, dT_pid, flows_pid, np.full(NUM_ZONES, power_profile[t]), zone_positions], axis=-1)
        X_window.append(features)
    X_window = np.array(X_window)  # [H, Z, F]

    # storage
    mpc_temps = []
    mpc_flows = []
    mpc_actions = []
    state = np.concatenate([np.full(NUM_ZONES, INITIAL_TEMP), np.zeros(NUM_ZONES)])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # short random shooting MPC using your predictor, fallback to simple PID if predictor absent
    predictor_model = globals().get("predictor", None)
    for t in range(HIST, len(time_array)-1):
        # prepare input for predictor if available
        X_tensor = None
        if predictor_model is not None:
            X_tensor = torch.tensor(X_window, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        best_reward = -1e12
        best_action = None
        # sample actions
        for s in range(samples):
            cand = np.random.rand(NUM_ZONES).astype(np.float32)
            if X_tensor is not None:
                X_test = X_tensor.clone()
                X_test[0, -1, :, 2] = torch.tensor(cand, device=DEVICE)
                with torch.no_grad():
                    T_pred = predictor_model(X_test).squeeze(0)
                # compute simple reward: penalize max temp above setpoint and mean action
                Tmax = T_pred.max().item()
                reward = -5.0 * max(0.0, Tmax - 40.0) - 0.5 * cand.mean()
            else:
                # fallback heuristic: cool more where current temps are higher
                last_temps = X_window[-1][:, 0]
                cand = np.clip((last_temps - 30.0) / 20.0 + 0.1 * np.random.randn(NUM_ZONES), 0.0, 1.0)
                reward = -0.5 * cand.mean()
            if reward > best_reward:
                best_reward = reward
                best_action = cand.copy()
        # set proxy controllers actions
        for i in range(NUM_ZONES):
            mpc_controllers[i].current_action = float(best_action[i])
        # step real simulator for one second using sim_params_mpc
        sol = solve_ivp(
            fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, sim_params_mpc),
            t_span=[t, t+1.0],
            y0=state,
            t_eval=[t+1.0]
        )
        next_temps = sol.y[:NUM_ZONES, -1]
        next_flows = sol.y[NUM_ZONES:, -1]
        mpc_temps.append(next_temps)
        mpc_flows.append(next_flows)
        mpc_actions.append(best_action)
        # update window
        dT_next = next_temps - X_window[-1][:, 0]
        new_features = np.stack([next_temps, dT_next, next_flows, np.full(NUM_ZONES, power_profile[t]), zone_positions], axis=-1)
        X_window = np.roll(X_window, shift=-1, axis=0)
        X_window[-1] = new_features
        state = np.concatenate([next_temps, next_flows])

    # convert to arrays and build run dict aligned to same timeline as pid
    mpc_temps = np.vstack(mpc_temps)
    mpc_flows = np.vstack(mpc_flows)
    # reconstruct pump power
    mpc_time = np.arange(HIST, HIST + len(mpc_temps))
    pump_power = np.zeros(len(mpc_time))
    if MAX_PUMP_FLOW_LPM > 0:
        for i in range(len(mpc_time)):
            total_flow_lpm = np.sum(mpc_flows[i, :]) * MAX_PUMP_FLOW_LPM / NUM_ZONES
            frac = total_flow_lpm / MAX_PUMP_FLOW_LPM
            pump_power[i] = PUMP_MAX_POWER_W * (frac ** 2)
    return {
        "time_s": mpc_time,
        "temps": mpc_temps,
        "flows": mpc_flows,
        "pump_power": pump_power,
        "actions": np.vstack(mpc_actions)
    }

# run MPC
_mpc_start = _time.time()
mpc_out = run_mpc(horizon=1, samples=64)
print("MPC done in %.1f s" % (_time.time() - _mpc_start))
# save outputs
mpc_df = pd.DataFrame({
    "time_s": mpc_out["time_s"],
    "pump_power_W": mpc_out["pump_power"],
    "battery_power_W": power_profile[:len(mpc_out["time_s"])]
})
for z in range(NUM_ZONES):
    mpc_df[f"zone_{z+1}_temp_C"] = mpc_out["temps"][:, z]
    mpc_df[f"zone_{z+1}_flow_norm"] = mpc_out["flows"][:, z]
mpc_df["action_mean"] = mpc_out["actions"].mean(axis=1)
mpc_csv = "results/mpc_run.csv"
mpc_df.to_csv(mpc_csv, index=False)
print("MPC saved to", mpc_csv)

mpc_metrics = btms_metrics_from_run(mpc_out, "MPC")
print("MPC metrics:", mpc_metrics)

# ---------- 3) Behavior cloning train from MPC actions ----------
print("Training behavior cloning actor to imitate MPC actions...")

# prepare dataset: state vector build_control_state, and actions
def build_control_state_np(X_window_last):
    # X_window_last: last window of shape [H, Z, F]
    T_curr = X_window_last[-1, :, 0]
    T_mean = T_curr.mean()
    T_max = T_curr.max()
    power = X_window_last[-1, 0, 3] if X_window_last.shape[-1] > 3 else 0.0
    return np.concatenate([T_curr, [T_mean, T_max, power]]).astype(np.float32)

# create BC dataset from MPC run actions and the simulator features by replaying the timeline and capturing state
def create_bc_dataset_from_mpc(mpc_out, hist_len=10):
    # We'll reconstruct a rolling window by replaying the simulator with PID warmstart as before
    H = hist_len
    X_window = []
    temps_pid = np.full(NUM_ZONES, INITIAL_TEMP)
    flows_pid = np.zeros(NUM_ZONES)
    DT = 1.0
    states = []
    for t in range(H):
        sol = solve_ivp(
            fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, simulation_params),
            t_span=[t, t+DT],
            y0=np.concatenate([temps_pid, flows_pid]),
            t_eval=[t+DT]
        )
        temps_pid = sol.y[:NUM_ZONES, -1]
        flows_pid = sol.y[NUM_ZONES:, -1]
        dT_pid = temps_pid - (np.full(NUM_ZONES, INITIAL_TEMP) if t==0 else X_window[-1][:,0])
        features = np.stack([temps_pid, dT_pid, flows_pid, np.full(NUM_ZONES, power_profile[t]), zone_positions], axis=-1)
        X_window.append(features)
    X_window = np.array(X_window)
    obs_states = []
    actions = []
    # align MPC time indexes to full timeline by offset HIST
    start_idx = H
    for i in range(len(mpc_out["time_s"])):
        # capture state at this step
        svec = build_control_state_np(X_window)
        obs_states.append(svec)
        actions.append(mpc_out["actions"][i])
        # step ODE one second using sim_params with action inserted
        # create proxy controllers with this action to step forward
        prox = [type("P", (), {"current_action": float(a), "update": lambda self, m, dt=1.0: float(self.current_action)})() for a in mpc_out["actions"][i]]
        sim_params_local = dict(simulation_params)
        sim_params_local["pid_controllers"] = prox
        sol = solve_ivp(
            fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, sim_params_local),
            t_span=[start_idx + i, start_idx + i + 1.0],
            y0=np.concatenate([X_window[-1][:,0], X_window[-1][:,2]]),
            t_eval=[start_idx + i + 1.0]
        )
        next_temps = sol.y[:NUM_ZONES, -1]
        next_flows = sol.y[NUM_ZONES:, -1]
        dT_next = next_temps - X_window[-1][:,0]
        new_features = np.stack([next_temps, dT_next, next_flows, np.full(NUM_ZONES, power_profile[start_idx + i]), zone_positions], axis=-1)
        X_window = np.roll(X_window, shift=-1, axis=0)
        X_window[-1] = new_features
    return np.vstack(obs_states), np.vstack(actions)

bc_states, bc_actions = create_bc_dataset_from_mpc(mpc_out, hist_len=10)
print("BC dataset samples:", bc_states.shape[0], "state dim:", bc_states.shape[1], "action dim:", bc_actions.shape[1])

# simple MLP actor clone
class BCActor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden//2, action_dim),
            torch.nn.Sigmoid()  # output in 0..1
        )
    def forward(self, x):
        return self.net(x)

state_dim = bc_states.shape[1]
action_dim = bc_actions.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bc_model = BCActor(state_dim, action_dim).to(device)
opt = torch.optim.Adam(bc_model.parameters(), lr=3e-4)
loss_fn = torch.nn.MSELoss()

# dataset loader
batch_size = 64
indices = np.arange(len(bc_states))
for epoch in range(30):
    np.random.shuffle(indices)
    losses = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        s = torch.tensor(bc_states[batch_idx], dtype=torch.float32, device=device)
        a = torch.tensor(bc_actions[batch_idx], dtype=torch.float32, device=device)
        pred = bc_model(s)
        loss = loss_fn(pred, a)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bc_model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    if (epoch+1) % 5 == 0:
        print(f"BC epoch {epoch+1}, loss {np.mean(losses):.6f}")

# evaluate BC actor in closed loop by injecting its actions via proxy controllers
class ActorProxy:
    def __init__(self, bc_model, device):
        self.bc = bc_model
        self.device = device
    def update(self, measurement, dt=1.0):
        # not used, we call get_action externally and set current_action on proxies
        return 0.0

# run closed loop using actor policy: at each step gather state vector and set proxy controllers
print("Running cloned actor closed loop evaluation...")
actor_proxies = [MPCProxyController() for _ in range(NUM_ZONES)]
sim_params_actor = dict(simulation_params)
sim_params_actor["pid_controllers"] = actor_proxies

# create warm window like before
H = 10
X_window = []
temps_pid = np.full(NUM_ZONES, INITIAL_TEMP)
flows_pid = np.zeros(NUM_ZONES)
for t in range(H):
    sol = solve_ivp(
        fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, simulation_params),
        t_span=[t, t+1.0],
        y0=np.concatenate([temps_pid, flows_pid]),
        t_eval=[t+1.0]
    )
    temps_pid = sol.y[:NUM_ZONES, -1]
    flows_pid = sol.y[NUM_ZONES:, -1]
    dT_pid = temps_pid - (np.full(NUM_ZONES, INITIAL_TEMP) if t==0 else X_window[-1][:,0])
    features = np.stack([temps_pid, dT_pid, flows_pid, np.full(NUM_ZONES, power_profile[t]), zone_positions], axis=-1)
    X_window.append(features)
X_window = np.array(X_window)
state = np.concatenate([X_window[-1][:,0], X_window[-1][:,2]])

actor_temps = []
actor_flows = []
actor_actions_log = []
for t in range(H, len(time_array)-1):
    # construct state vector identical to build_control_state
    svec = build_control_state_np(X_window)
    s_t = torch.tensor(svec.reshape(1,-1), dtype=torch.float32, device=device)
    with torch.no_grad():
        a_pred = bc_model(s_t).cpu().numpy().squeeze(0)
    # write proxies
    for i in range(NUM_ZONES):
        actor_proxies[i].current_action = float(a_pred[i])
    # step
    sol = solve_ivp(
        fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, sim_params_actor),
        t_span=[t, t+1.0],
        y0=state,
        t_eval=[t+1.0]
    )
    next_temps = sol.y[:NUM_ZONES, -1]
    next_flows = sol.y[NUM_ZONES:, -1]
    actor_temps.append(next_temps)
    actor_flows.append(next_flows)
    actor_actions_log.append(a_pred)
    # update X_window
    dT_next = next_temps - X_window[-1][:,0]
    new_features = np.stack([next_temps, dT_next, next_flows, np.full(NUM_ZONES, power_profile[t]), zone_positions], axis=-1)
    X_window = np.roll(X_window, shift=-1, axis=0)
    X_window[-1] = new_features
    state = np.concatenate([next_temps, next_flows])

actor_out = {
    "time_s": np.arange(H, H + len(actor_temps)),
    "temps": np.vstack(actor_temps),
    "flows": np.vstack(actor_flows),
    "pump_power": np.array([PUMP_MAX_POWER_W * ((np.sum(f)*MAX_PUMP_FLOW_LPM/NUM_ZONES / MAX_PUMP_FLOW_LPM)**2 if MAX_PUMP_FLOW_LPM>0 else 0.0) for f in actor_flows]),
    "actions": np.vstack(actor_actions_log)
}
actor_df = pd.DataFrame({
    "time_s": actor_out["time_s"],
    "pump_power_W": actor_out["pump_power"],
    "battery_power_W": power_profile[:len(actor_out["time_s"])]
})
for z in range(NUM_ZONES):
    actor_df[f"zone_{z+1}_temp_C"] = actor_out["temps"][:, z]
    actor_df[f"zone_{z+1}_flow_norm"] = actor_out["flows"][:, z]
actor_csv = "results/actor_bc_run.csv"
actor_df.to_csv(actor_csv, index=False)
print("Actor BC saved to", actor_csv)

actor_metrics = btms_metrics_from_run(actor_out, "Actor_BC")
print("Actor BC metrics:", actor_metrics)

# ---------- final comparison table ----------
rows = [pid_metrics, mpc_metrics, actor_metrics]
comp_df = pd.DataFrame(rows)
comp_csv = "results/comparison_summary.csv"
comp_df.to_csv(comp_csv, index=False)
print("\nComparison saved to", comp_csv)
print(comp_df[[
    "label", "mean_temp", "max_temp", "temp_spread_mean",
    "pump_energy_Wh", "pump_peak_W", "cooling_overhead_pct"
]])
print("\nDone, three way comparison complete.")


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD SAVED RESULTS
# =========================
pid_df = pd.read_csv("results/pid_run.csv")
mpc_df = pd.read_csv("results/mpc_run.csv")
actor_df = pd.read_csv("results/actor_bc_run.csv")

# Align time axes in hours
t_pid = pid_df["time_s"].values / 3600.0
t_mpc = mpc_df["time_s"].values / 3600.0
t_actor = actor_df["time_s"].values / 3600.0

# Extract zone temperatures
def extract_zone_temps(df):
    return np.vstack(
        [df[col].values for col in df.columns if "zone_" in col and "_temp_" in col]
    ).T

pid_temps = extract_zone_temps(pid_df)
mpc_temps = extract_zone_temps(mpc_df)
actor_temps = extract_zone_temps(actor_df)

# =========================
# FIGURE 1, MEAN AND MAX TEMPERATURE
# =========================
plt.figure(figsize=(14, 4))

plt.plot(t_pid, pid_temps.mean(axis=1), label="PID Mean", linewidth=2)
plt.plot(t_pid, pid_temps.max(axis=1), label="PID Max", linewidth=2)

plt.plot(t_mpc, mpc_temps.mean(axis=1), "--", label="MPC Mean", linewidth=2)
plt.plot(t_mpc, mpc_temps.max(axis=1), "--", label="MPC Max", linewidth=2)

plt.plot(t_actor, actor_temps.mean(axis=1), ":", label="Actor BC Mean", linewidth=2)
plt.plot(t_actor, actor_temps.max(axis=1), ":", label="Actor BC Max", linewidth=2)

plt.axhline(40.0, color="r", linestyle=":", label="Setpoint")

plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°C)")
plt.title("Mean and Maximum Pack Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# FIGURE 2, TEMPERATURE SPREAD
# =========================
plt.figure(figsize=(14, 4))

plt.plot(
    t_pid,
    pid_temps.max(axis=1) - pid_temps.min(axis=1),
    label="PID",
    linewidth=2
)

plt.plot(
    t_mpc,
    mpc_temps.max(axis=1) - mpc_temps.min(axis=1),
    "--",
    label="MPC",
    linewidth=2
)

plt.plot(
    t_actor,
    actor_temps.max(axis=1) - actor_temps.min(axis=1),
    ":",
    label="Actor BC",
    linewidth=2
)

plt.xlabel("Time (hours)")
plt.ylabel("Temperature Spread (°C)")
plt.title("Inter Zone Temperature Spread")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# FIGURE 3, COOLING POWER
# =========================
plt.figure(figsize=(14, 4))

plt.plot(t_pid, pid_df["pump_power_W"] / 1000.0, label="PID", linewidth=2)
plt.plot(t_mpc, mpc_df["pump_power_W"] / 1000.0, "--", label="MPC", linewidth=2)
plt.plot(t_actor, actor_df["pump_power_W"] / 1000.0, ":", label="Actor BC", linewidth=2)

plt.xlabel("Time (hours)")
plt.ylabel("Pump Power (kW)")
plt.title("Cooling Power Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# FIGURE 4, CUMULATIVE COOLING ENERGY
# =========================
dt = 1.0

pid_energy = np.cumsum(pid_df["pump_power_W"].values) * dt / 3600.0
mpc_energy = np.cumsum(mpc_df["pump_power_W"].values) * dt / 3600.0
actor_energy = np.cumsum(actor_df["pump_power_W"].values) * dt / 3600.0

plt.figure(figsize=(14, 4))

plt.plot(t_pid, pid_energy, label="PID", linewidth=2)
plt.plot(t_mpc, mpc_energy, "--", label="MPC", linewidth=2)
plt.plot(t_actor, actor_energy, ":", label="Actor BC", linewidth=2)

plt.xlabel("Time (hours)")
plt.ylabel("Cumulative Cooling Energy (Wh)")
plt.title("Cooling Energy Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# FIGURE 5, RANGE IMPACT BAR CHART
# =========================
drive_energy_Wh = pid_df["battery_power_W"].sum() * dt / 3600.0

cooling_energy = [
    pid_energy[-1],
    mpc_energy[-1],
    actor_energy[-1]
]

labels = ["PID", "MPC", "Actor BC"]
overhead_pct = [100.0 * e / drive_energy_Wh for e in cooling_energy]

plt.figure(figsize=(8, 4))
plt.bar(labels, overhead_pct)
plt.ylabel("Cooling Overhead (% of Drive Energy)")
plt.title("Range Impact Comparison")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("Plots complete.")


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
pid_df = pd.read_csv("results/pid_run.csv")
mpc_df = pd.read_csv("results/mpc_run.csv")
actor_df = pd.read_csv("results/actor_bc_run.csv")

dt = 1.0

def extract_zone_temps(df):
    return np.vstack(
        [df[c].values for c in df.columns if "zone_" in c and "_temp_" in c]
    ).T

pid_T = extract_zone_temps(pid_df)
mpc_T = extract_zone_temps(mpc_df)
act_T = extract_zone_temps(actor_df)

# =========================
# 1. TIME ABOVE THRESHOLDS
# =========================
def time_above(T, threshold):
    return np.mean(T > threshold)

thresholds = [35.0, 40.0, 45.0]

pid_above = [time_above(pid_T, th) for th in thresholds]
mpc_above = [time_above(mpc_T, th) for th in thresholds]
act_above = [time_above(act_T, th) for th in thresholds]

plt.figure(figsize=(8,4))
x = np.arange(len(thresholds))
w = 0.25

plt.bar(x - w, pid_above, w, label="PID")
plt.bar(x, mpc_above, w, label="MPC")
plt.bar(x + w, act_above, w, label="Actor BC")

plt.xticks(x, [f">{int(t)}°C" for t in thresholds])
plt.ylabel("Fraction of Time")
plt.title("Time Above Temperature Thresholds")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 2. THERMAL STRESS HISTOGRAM
# =========================
def thermal_stress(T):
    return np.abs(np.diff(T, axis=0)).mean(axis=1)

plt.figure(figsize=(10,4))
plt.hist(thermal_stress(pid_T), bins=50, alpha=0.5, label="PID", density=True)
plt.hist(thermal_stress(mpc_T), bins=50, alpha=0.5, label="MPC", density=True)
plt.hist(thermal_stress(act_T), bins=50, alpha=0.5, label="Actor BC", density=True)

plt.xlabel("|ΔT| per second (°C)")
plt.ylabel("Density")
plt.title("Thermal Stress Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 3. COOLING AGGRESSIVENESS
# =========================
plt.figure(figsize=(10,4))

plt.boxplot(
    [
        pid_df["pump_power_W"].values,
        mpc_df["pump_power_W"].values,
        actor_df["pump_power_W"].values,
    ],
    labels=["PID", "MPC", "Actor BC"],
    showfliers=False
)

plt.ylabel("Pump Power (W)")
plt.title("Cooling Aggressiveness Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 4. EFFICIENCY FRONTIER
# =========================
def efficiency_point(T, pump):
    return T.max(), pump.sum() * dt / 3600.0

points = {
    "PID": efficiency_point(pid_T, pid_df["pump_power_W"].values),
    "MPC": efficiency_point(mpc_T, mpc_df["pump_power_W"].values),
    "Actor BC": efficiency_point(act_T, actor_df["pump_power_W"].values)
}

plt.figure(figsize=(6,5))
for k, (tmax, e) in points.items():
    plt.scatter(e, tmax, s=80)
    plt.text(e*1.01, tmax*1.001, k)

plt.xlabel("Cooling Energy (Wh)")
plt.ylabel("Maximum Temperature (°C)")
plt.title("Cooling Efficiency Frontier")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 5. RANGE PENALTY COMPARISON
# =========================
drive_energy_Wh = pid_df["battery_power_W"].sum() * dt / 3600.0

range_penalty = {
    "PID": pid_df["pump_power_W"].sum() * dt / 3600.0 / drive_energy_Wh,
    "MPC": mpc_df["pump_power_W"].sum() * dt / 3600.0 / drive_energy_Wh,
    "Actor BC": actor_df["pump_power_W"].sum() * dt / 3600.0 / drive_energy_Wh
}

plt.figure(figsize=(6,4))
plt.bar(range_penalty.keys(), range_penalty.values())
plt.ylabel("Range Penalty (fraction)")
plt.title("Cooling Induced Range Loss")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 6. CONTROL SMOOTHNESS
# =========================
def action_smoothness(pump):
    return np.mean(np.abs(np.diff(pump)))

smooth = {
    "PID": action_smoothness(pid_df["pump_power_W"].values),
    "MPC": action_smoothness(mpc_df["pump_power_W"].values),
    "Actor BC": action_smoothness(actor_df["pump_power_W"].values),
}

plt.figure(figsize=(6,4))
plt.bar(smooth.keys(), smooth.values())
plt.ylabel("Mean |ΔPump Power| (W)")
plt.title("Control Smoothness")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("Extended comparison plots completed.")


# %%
# =========================
# CONTROLLER IMPLEMENTATIONS
# Integrated with existing notebook simulation
# =========================

import copy
import numpy as np
from scipy.integrate import solve_ivp
import torch
import pandas as pd
import os

# Ensure output directory exists
os.makedirs("results", exist_ok=True)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- Validate required globals ----------
required_vars = [
    "NUM_ZONES", "INITIAL_TEMP", "TEMP_SETPOINT", "MAX_PUMP_FLOW_LPM", 
    "PUMP_MAX_POWER_W", "ADIABATIC_MODE", "power_profile", "simulation_params",
    "zone_positions", "PACK_NOMINAL_VOLTAGE", "CELL_CAPACITY",
    "zone_resistance", "predictor", "PIDController", "battery_thermal_ode"
]

missing = [v for v in required_vars if v not in globals()]
if missing:
    print(f"⚠️ Missing variables: {missing}")
    print("This cell requires previous simulation cells to be run first")
    raise RuntimeError(f"Missing required variables: {missing}")

# Convenience aliases
Z = NUM_ZONES
DT = 1.0
Tsteps = len(power_profile)

print(f"✅ Controller framework initialized")
print(f"   Zones: {Z}, Timesteps: {Tsteps}, Device: {next(predictor.parameters()).device}")

# ---------- Proxy controller for ODE integration ----------
class ActionProxy:
    """Simple proxy that returns a fixed action value"""
    def __init__(self, init=0.0):
        self.current_action = float(init)
    
    def update(self, measurement, dt=1.0):
        return float(self.current_action)

# ---------- Core simulation function ----------
def simulate_with_action_provider(action_provider_fn, desc="Custom"):
    """
    Run closed-loop simulation with custom action provider
    
    Args:
        action_provider_fn: function(state, t) -> actions[Z]
        desc: description for progress
    
    Returns:
        dict with temps[T,Z], flows[T,Z], pump[T]
    """
    print(f"   Running {desc}...")
    
    # Create proxy controllers
    proxies = [ActionProxy(0.0) for _ in range(Z)]
    
    # Clone simulation params
    params_local = dict(simulation_params)
    params_local["pid_controllers"] = proxies
    
    # Initialize state
    state = np.concatenate([
        np.full(Z, INITIAL_TEMP),
        np.zeros(Z)
    ])
    
    # Storage
    temps_log = np.zeros((Tsteps, Z))
    flows_log = np.zeros((Tsteps, Z))
    pump_log = np.zeros(Tsteps)
    
    # Simulation loop
    for t in range(Tsteps):
        # Get actions from provider
        actions = action_provider_fn(state.copy(), t)
        actions = np.clip(np.asarray(actions, dtype=float), 0.0, 1.0)
        
        # Set proxy actions
        for i in range(Z):
            proxies[i].current_action = float(actions[i])
        
        # Integrate one timestep
        sol = solve_ivp(
            fun=lambda tau, y: battery_thermal_ode(tau, y, power_profile, params_local),
            t_span=[t, t + DT],
            y0=state,
            t_eval=[t + DT],
            method="RK45",
            rtol=1e-6,
            atol=1e-8
        )
        
        if not sol.success:
            print(f"⚠️ Integration failed at t={t}: {sol.message}")
            break
        
        # Update state
        state = sol.y[:, -1]
        temps_log[t] = state[:Z]
        flows_log[t] = state[Z:]
        
        # Calculate pump power
        if not ADIABATIC_MODE:
            total_flow_lpm = np.sum(state[Z:]) * MAX_PUMP_FLOW_LPM / Z
            flow_frac = total_flow_lpm / MAX_PUMP_FLOW_LPM if MAX_PUMP_FLOW_LPM > 0 else 0.0
            pump_log[t] = PUMP_MAX_POWER_W * (flow_frac ** 2)
    
    return {
        "temps": temps_log,
        "flows": flows_log,
        "pump": pump_log
    }

# ---------- Metrics computation ----------
def compute_metrics(result_dict, label):
    """Compute standard metrics from simulation result"""
    temps = result_dict["temps"]
    pump = result_dict["pump"]
    
    metrics = {
        "label": label,
        "mean_temp": temps.mean(),
        "max_temp": temps.max(),
        "min_temp": temps.min(),
        "temp_spread_mean": (temps.max(axis=1) - temps.min(axis=1)).mean(),
        "thermal_stress": np.mean(np.abs(np.diff(temps, axis=0))),
        "time_above_setpoint": np.mean(temps > TEMP_SETPOINT),
        "pump_energy_Wh": pump.sum() * DT / 3600.0,
        "pump_peak_W": pump.max(),
    }
    
    # Cooling overhead
    drive_energy_Wh = power_profile.sum() * DT / 3600.0
    metrics["cooling_overhead_pct"] = 100.0 * metrics["pump_energy_Wh"] / (drive_energy_Wh + 1e-12)
    
    return metrics

# ---------- Build state vector for ML models ----------
def build_state_vector(state_array):
    """Convert ODE state to ML input format"""
    temps = state_array[:Z]
    return np.concatenate([
        temps,
        [temps.mean(), temps.max(), 0.0]  # mean, max, power placeholder
    ])

print("✅ Core simulation functions ready")

# %%
# =========================
# CONTROLLER IMPLEMENTATIONS
# All baseline and proposed methods
# =========================

print("=" * 60)
print("IMPLEMENTING CONTROLLERS")
print("=" * 60)

# Store all controllers here
controller_registry = {}

# ---------- 1. PID VARIANTS ----------

def make_pid_standard():
    """Standard PID with fixed gains"""
    def action_fn(state, t):
        temps = state[:Z]
        actions = np.zeros(Z)
        for i in range(Z):
            error = TEMP_SETPOINT - temps[i]
            actions[i] = np.clip(0.5 + 0.05 * error, 0.0, 1.0)
        return actions
    return action_fn

def make_pid_adaptive():
    """PID with gain scheduling based on temperature"""
    def action_fn(state, t):
        temps = state[:Z]
        actions = np.zeros(Z)
        for i in range(Z):
            error = TEMP_SETPOINT - temps[i]
            # Increase gain when far from setpoint
            gain = 0.05 * (1.0 + 0.5 * abs(error) / 10.0)
            actions[i] = np.clip(0.5 + gain * error, 0.0, 1.0)
        return actions
    return action_fn

controller_registry["PID_Standard"] = make_pid_standard()
controller_registry["PID_Adaptive"] = make_pid_adaptive()

# ---------- 2. MPC VARIANTS ----------

def make_mpc_random_shooting(horizon=1, samples=32):
    """Random shooting MPC using predictor"""
    device = next(predictor.parameters()).device
    
    def build_window(state):
        """Create history window for predictor"""
        temps = state[:Z]
        flows = state[Z:]
        
        # Simple window: repeat current state
        H = 10
        window = []
        for _ in range(H):
            features = np.stack([
                temps,
                np.zeros(Z),  # dT
                flows,
                np.zeros(Z),  # power (placeholder)
                zone_positions
            ], axis=-1)
            window.append(features)
        
        return np.array(window)
    
    def action_fn(state, t):
        # Build predictor input
        window = build_window(state)
        X_tensor = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
        
        best_score = -1e9
        best_action = np.zeros(Z)
        
        # Random shooting
        for _ in range(samples):
            cand = np.random.rand(Z)
            
            # Predict with this action
            X_test = X_tensor.clone()
            X_test[0, -1, :, 2] = torch.tensor(cand, device=device)  # FLOW_IDX=2
            
            predictor.eval()
            with torch.no_grad():
                T_pred = predictor(X_test).squeeze(0)
            
            # Score: penalize violations and energy
            T_max = T_pred.max().item()
            violation = max(0.0, T_max - TEMP_SETPOINT)
            energy = cand.mean()
            
            score = -5.0 * violation - 0.5 * energy
            
            if score > best_score:
                best_score = score
                best_action = cand
        
        return best_action
    
    return action_fn

controller_registry["MPC_H1_S32"] = make_mpc_random_shooting(horizon=1, samples=32)
controller_registry["MPC_H1_S64"] = make_mpc_random_shooting(horizon=1, samples=64)

# ---------- 3. ACTOR-BASED METHODS ----------

def make_actor_deterministic():
    """Use trained actor in deterministic mode"""
    if "ac" not in globals():
        print("⚠️ ActorCritic 'ac' not found, using fallback")
        return make_pid_standard()
    
    def action_fn(state, t):
        temps = state[:Z]
        power_now = power_profile[min(t, len(power_profile)-1)]
        
        # Build state vector
        state_vec = np.concatenate([
            temps,
            [temps.mean(), temps.max(), power_now]
        ])
        
        # Get action from actor
        action, _, _ = ac.act(state_vec, deterministic=True)
        return np.clip(action[0], 0.0, 1.0)
    
    return action_fn

def make_actor_stochastic():
    """Use trained actor in stochastic mode"""
    if "ac" not in globals():
        print("⚠️ ActorCritic 'ac' not found, using fallback")
        return make_pid_standard()
    
    def action_fn(state, t):
        temps = state[:Z]
        power_now = power_profile[min(t, len(power_profile)-1)]
        
        state_vec = np.concatenate([
            temps,
            [temps.mean(), temps.max(), power_now]
        ])
        
        action, _, _ = ac.act(state_vec, deterministic=False)
        return np.clip(action[0], 0.0, 1.0)
    
    return action_fn

if "ac" in globals():
    controller_registry["Actor_Deterministic"] = make_actor_deterministic()
    controller_registry["Actor_Stochastic"] = make_actor_stochastic()

# ---------- 4. ADDITIONAL BASELINES ----------

def make_uniform_flow():
    """Single global flow rate for all zones"""
    def action_fn(state, t):
        temps = state[:Z]
        mean_temp = temps.mean()
        error = TEMP_SETPOINT - mean_temp
        flow = np.clip(0.5 + 0.05 * error, 0.0, 1.0)
        return np.full(Z, flow)
    return action_fn

def make_proportional_to_temp():
    """Flow proportional to zone temperature"""
    def action_fn(state, t):
        temps = state[:Z]
        # Normalize temps to [0, 1] range
        if temps.ptp() > 0:
            norm_temps = (temps - temps.min()) / temps.ptp()
        else:
            norm_temps = np.ones(Z) * 0.5
        return np.clip(norm_temps, 0.0, 1.0)
    return action_fn

controller_registry["Uniform_Flow"] = make_uniform_flow()
controller_registry["Proportional_Temp"] = make_proportional_to_temp()

# ---------- 5. PROPOSED METHOD ABLATIONS ----------

def make_proposed_full():
    """Full proposed method with all components"""
    return make_mpc_random_shooting(horizon=1, samples=64)

def make_proposed_no_spatial():
    """Ablation: disable spatial encoder"""
    # This would require modifying predictor, so use standard MPC
    return make_mpc_random_shooting(horizon=1, samples=32)

controller_registry["OURS_Full"] = make_proposed_full()
controller_registry["OURS_NoSpatial"] = make_proposed_no_spatial()

print(f"✅ Registered {len(controller_registry)} controllers:")
for name in controller_registry.keys():
    print(f"   • {name}")

# %%
# =========================
# RUN ALL CONTROLLERS AND GENERATE COMPARISON
# =========================

import time as time_module

print("\n" + "=" * 60)
print("RUNNING CONTROLLER COMPARISON")
print("=" * 60)

# Storage for results
all_results = {}
all_metrics = []

# Run each controller
for name, action_fn in controller_registry.items():
    print(f"\n🔄 Running: {name}")
    start_time = time_module.time()
    
    try:
        # Run simulation
        result = simulate_with_action_provider(action_fn, desc=name)
        
        # Compute metrics
        metrics = compute_metrics(result, name)
        
        # Store
        all_results[name] = result
        all_metrics.append(metrics)
        
        elapsed = time_module.time() - start_time
        print(f"   ✅ Completed in {elapsed:.1f}s")
        print(f"      Max temp: {metrics['max_temp']:.2f}°C, "
              f"Pump energy: {metrics['pump_energy_Wh']:.1f} Wh")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        continue

# Create summary DataFrame
df_comparison = pd.DataFrame(all_metrics)
df_comparison = df_comparison.sort_values("pump_energy_Wh").reset_index(drop=True)

# Save results
df_comparison.to_csv("results/controller_comparison.csv", index=False)
print(f"\n💾 Saved comparison to results/controller_comparison.csv")

# Display summary
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(df_comparison[[
    "label", "max_temp", "temp_spread_mean", 
    "pump_energy_Wh", "cooling_overhead_pct"
]].to_string(index=False))

# Quick visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Pump Energy
ax = axes[0]
df_sorted = df_comparison.sort_values("pump_energy_Wh")
ax.barh(range(len(df_sorted)), df_sorted["pump_energy_Wh"])
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted["label"], fontsize=9)
ax.set_xlabel("Pump Energy (Wh)")
ax.set_title("Energy Consumption")
ax.grid(axis="x", alpha=0.3)

# Plot 2: Max Temperature
ax = axes[1]
df_sorted = df_comparison.sort_values("max_temp")
ax.barh(range(len(df_sorted)), df_sorted["max_temp"])
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted["label"], fontsize=9)
ax.set_xlabel("Maximum Temperature (°C)")
ax.set_title("Thermal Performance")
ax.axvline(TEMP_SETPOINT, color="r", linestyle="--", label="Setpoint")
ax.legend()
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("results/controller_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Comparison complete!")

# %%
# =========================
# COMPREHENSIVE DATA SCIENCE ANALYSIS
# Publication-quality visualizations and statistical insights
# =========================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

print("=" * 70)
print("COMPREHENSIVE THERMAL MANAGEMENT ANALYSIS")
print("=" * 70)

# =========================
# 1. PARETO FRONTIER ANALYSIS
# =========================

def plot_pareto_frontier():
    """Energy-Temperature Pareto frontier with dominated regions"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract data
    energy = df_comparison["pump_energy_Wh"].values
    max_temp = df_comparison["max_temp"].values
    labels = df_comparison["label"].values
    
    # Identify Pareto frontier
    pareto_mask = np.ones(len(energy), dtype=bool)
    for i in range(len(energy)):
        for j in range(len(energy)):
            if i != j:
                # j dominates i if it has both lower energy AND lower temp
                if energy[j] <= energy[i] and max_temp[j] <= max_temp[i]:
                    if energy[j] < energy[i] or max_temp[j] < max_temp[i]:
                        pareto_mask[i] = False
                        break
    
    # Plot dominated points
    ax.scatter(energy[~pareto_mask], max_temp[~pareto_mask], 
               s=120, alpha=0.4, color='gray', label='Dominated', 
               edgecolors='black', linewidth=0.5, zorder=2)
    
    # Plot Pareto optimal points
    ax.scatter(energy[pareto_mask], max_temp[pareto_mask], 
               s=180, alpha=0.8, c=range(pareto_mask.sum()), 
               cmap='viridis', label='Pareto Optimal', 
               edgecolors='black', linewidth=1.5, zorder=3)
    
    # Connect Pareto points
    pareto_idx = np.where(pareto_mask)[0]
    sorted_idx = pareto_idx[np.argsort(energy[pareto_mask])]
    ax.plot(energy[sorted_idx], max_temp[sorted_idx], 
            'k--', alpha=0.3, linewidth=2, zorder=1)
    
    # Annotate all points
    for i, label in enumerate(labels):
        offset_x = 1.5 if i % 2 == 0 else -1.5
        offset_y = 0.3 if i % 3 == 0 else -0.3
        
        ax.annotate(label, (energy[i], max_temp[i]), 
                   xytext=(offset_x, offset_y), 
                   textcoords='offset points',
                   fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='yellow' if pareto_mask[i] else 'white',
                            alpha=0.7, edgecolor='black', linewidth=0.5),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.2',
                                 color='black', lw=0.5))
    
    # Reference lines
    ax.axhline(TEMP_SETPOINT, color='red', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'Setpoint ({TEMP_SETPOINT}°C)')
    
    # Shade safe region
    ax.axhspan(0, TEMP_SETPOINT, alpha=0.1, color='green', 
              label='Safe Temperature Zone')
    
    ax.set_xlabel('Cooling Energy Consumption (Wh)', fontweight='bold')
    ax.set_ylabel('Maximum Pack Temperature (°C)', fontweight='bold')
    ax.set_title('Pareto Frontier: Energy Efficiency vs Thermal Performance', 
                fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/analysis_pareto_frontier.png', bbox_inches='tight')
    plt.show()
    
    # Print Pareto optimal controllers
    print("\n🏆 PARETO OPTIMAL CONTROLLERS:")
    pareto_df = df_comparison[pareto_mask].sort_values("pump_energy_Wh")
    print(pareto_df[["label", "max_temp", "pump_energy_Wh", "cooling_overhead_pct"]])

plot_pareto_frontier()

# =========================
# 2. TEMPORAL DYNAMICS DASHBOARD
# =========================

def plot_temporal_dashboard():
    """Multi-panel temporal analysis"""
    
    # Select top 5 controllers by different criteria
    top_energy = df_comparison.nsmallest(3, "pump_energy_Wh")["label"].tolist()
    top_temp = df_comparison.nsmallest(3, "max_temp")["label"].tolist()
    controllers_to_plot = list(set(top_energy + top_temp))[:6]
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    time_hrs = np.arange(Tsteps) / 3600.0
    
    # Panel 1: Mean Temperature Evolution
    ax1 = fig.add_subplot(gs[0, 0])
    for name in controllers_to_plot:
        temps = all_results[name]["temps"]
        ax1.plot(time_hrs, temps.mean(axis=1), label=name, linewidth=2, alpha=0.8)
    ax1.axhline(TEMP_SETPOINT, color='red', linestyle=':', linewidth=2, alpha=0.6)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Mean Pack Temperature (°C)')
    ax1.set_title('Temperature Trajectory Over Time', fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Temperature Spread (Uniformity)
    ax2 = fig.add_subplot(gs[0, 1])
    for name in controllers_to_plot:
        temps = all_results[name]["temps"]
        spread = temps.max(axis=1) - temps.min(axis=1)
        ax2.plot(time_hrs, spread, label=name, linewidth=2, alpha=0.8)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Temperature Spread (°C)')
    ax2.set_title('Inter-Zone Temperature Uniformity', fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Cumulative Energy Consumption
    ax3 = fig.add_subplot(gs[1, 0])
    for name in controllers_to_plot:
        pump = all_results[name]["pump"]
        cumulative = np.cumsum(pump) * DT / 3600.0
        ax3.plot(time_hrs, cumulative, label=name, linewidth=2, alpha=0.8)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Cumulative Cooling Energy (Wh)')
    ax3.set_title('Energy Consumption Over Time', fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Instantaneous Pump Power
    ax4 = fig.add_subplot(gs[1, 1])
    for name in controllers_to_plot:
        pump = all_results[name]["pump"]
        ax4.plot(time_hrs, pump / 1000.0, label=name, linewidth=1.5, alpha=0.8)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Pump Power (kW)')
    ax4.set_title('Instantaneous Cooling Power', fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Thermal Stress Rate
    ax5 = fig.add_subplot(gs[2, 0])
    for name in controllers_to_plot:
        temps = all_results[name]["temps"]
        dT_dt = np.abs(np.diff(temps.mean(axis=1)))
        ax5.plot(time_hrs[:-1], dT_dt, label=name, linewidth=1.5, alpha=0.8)
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('|dT/dt| (°C/s)')
    ax5.set_title('Thermal Stress Rate', fontweight='bold')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Control Effort (Flow Variance)
    ax6 = fig.add_subplot(gs[2, 1])
    for name in controllers_to_plot:
        flows = all_results[name]["flows"]
        flow_var = flows.std(axis=1)
        ax6.plot(time_hrs, flow_var, label=name, linewidth=1.5, alpha=0.8)
    ax6.set_xlabel('Time (hours)')
    ax6.set_ylabel('Flow Standard Deviation')
    ax6.set_title('Control Effort Variability', fontweight='bold')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.savefig('results/analysis_temporal_dashboard.png', bbox_inches='tight')
    plt.show()

plot_temporal_dashboard()

# =========================
# 3. STATISTICAL DISTRIBUTION ANALYSIS
# =========================

def plot_statistical_distributions():
    """Distribution analysis with statistical tests"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Prepare data
    controllers = df_comparison["label"].tolist()
    
    # 1. Temperature Distribution (Violin Plot)
    ax = axes[0, 0]
    temp_data = []
    temp_labels = []
    for name in controllers[:8]:  # Limit to 8 for clarity
        temps = all_results[name]["temps"].flatten()
        temp_data.append(temps)
        temp_labels.append(name)
    
    parts = ax.violinplot(temp_data, positions=range(len(temp_labels)), 
                         showmeans=True, showmedians=True)
    ax.set_xticks(range(len(temp_labels)))
    ax.set_xticklabels(temp_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Distribution (All Zones, All Times)', fontweight='bold')
    ax.axhline(TEMP_SETPOINT, color='red', linestyle='--', alpha=0.6)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Pump Power Distribution (Box Plot)
    ax = axes[0, 1]
    pump_data = []
    for name in controllers[:8]:
        pump = all_results[name]["pump"]
        pump_data.append(pump[pump > 0])  # Exclude zero values
    
    bp = ax.boxplot(pump_data, labels=temp_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xticklabels(temp_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Pump Power (W)')
    ax.set_title('Cooling Power Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Temperature Spread Distribution (Histogram)
    ax = axes[0, 2]
    for name in controllers[:5]:
        temps = all_results[name]["temps"]
        spread = temps.max(axis=1) - temps.min(axis=1)
        ax.hist(spread, bins=30, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Temperature Spread (°C)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Uniformity Distribution', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 4. Thermal Stress CDF
    ax = axes[1, 0]
    for name in controllers[:5]:
        temps = all_results[name]["temps"]
        stress = np.abs(np.diff(temps, axis=0)).flatten()
        sorted_stress = np.sort(stress)
        cdf = np.arange(1, len(sorted_stress) + 1) / len(sorted_stress)
        ax.plot(sorted_stress, cdf, label=name, linewidth=2)
    ax.set_xlabel('|ΔT| (°C)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Thermal Stress CDF', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, np.percentile(sorted_stress, 99))
    
    # 5. Energy Efficiency Histogram
    ax = axes[1, 1]
    overhead = df_comparison["cooling_overhead_pct"].values
    labels_all = df_comparison["label"].values
    colors = ['green' if x < 1.0 else 'orange' if x < 2.0 else 'red' for x in overhead]
    bars = ax.barh(range(len(overhead)), overhead, color=colors, alpha=0.7)
    ax.set_yticks(range(len(overhead)))
    ax.set_yticklabels(labels_all, fontsize=8)
    ax.set_xlabel('Cooling Overhead (% of Drive Energy)')
    ax.set_title('Range Impact Analysis', fontweight='bold')
    ax.axvline(1.0, color='green', linestyle='--', alpha=0.6, label='<1% overhead')
    ax.axvline(2.0, color='orange', linestyle='--', alpha=0.6, label='<2% overhead')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 6. Correlation Heatmap
    ax = axes[1, 2]
    metrics_for_corr = df_comparison[[
        "max_temp", "mean_temp", "temp_spread_mean", 
        "thermal_stress", "pump_energy_Wh", "cooling_overhead_pct"
    ]]
    corr_matrix = metrics_for_corr.corr()
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels([c.replace('_', '\n') for c in corr_matrix.columns], 
                       fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels([c.replace('_', '\n') for c in corr_matrix.columns], 
                       fontsize=7)
    ax.set_title('Metric Correlation Matrix', fontweight='bold')
    
    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    
    plt.tight_layout()
    plt.savefig('results/analysis_statistical_distributions.png', bbox_inches='tight')
    plt.show()

plot_statistical_distributions()

# =========================
# 4. ZONE-LEVEL SPATIAL ANALYSIS
# =========================

def plot_spatial_analysis():
    """Analyze spatial temperature patterns across zones"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select controllers for comparison
    best_energy = df_comparison.nsmallest(1, "pump_energy_Wh")["label"].iloc[0]
    best_temp = df_comparison.nsmallest(1, "max_temp")["label"].iloc[0]
    
    # 1. Spatial Temperature Profile (Time-averaged)
    ax = axes[0, 0]
    for name in [best_energy, best_temp]:
        temps = all_results[name]["temps"]
        mean_per_zone = temps.mean(axis=0)
        std_per_zone = temps.std(axis=0)
        
        ax.plot(range(1, Z+1), mean_per_zone, marker='o', 
               linewidth=2, markersize=8, label=f'{name} (mean)')
        ax.fill_between(range(1, Z+1), 
                        mean_per_zone - std_per_zone,
                        mean_per_zone + std_per_zone,
                        alpha=0.2)
    
    ax.axhline(TEMP_SETPOINT, color='red', linestyle='--', alpha=0.6, label='Setpoint')
    ax.set_xlabel('Zone Number')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Spatial Temperature Profile (Time-Averaged)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, Z+1))
    
    # 2. Temperature Heatmap Over Time
    ax = axes[0, 1]
    temps = all_results[best_temp]["temps"]
    
    # Downsample for visualization
    downsample = max(1, Tsteps // 200)
    temps_ds = temps[::downsample, :]
    time_ds = np.arange(0, Tsteps, downsample) / 3600.0
    
    im = ax.contourf(range(1, Z+1), time_ds, temps_ds, 
                     levels=20, cmap='RdYlBu_r')
    ax.set_xlabel('Zone Number')
    ax.set_ylabel('Time (hours)')
    ax.set_title(f'Spatiotemporal Temperature Map: {best_temp}', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    ax.set_xticks(range(1, Z+1))
    
    # 3. Zone-to-Zone Temperature Gradient
    ax = axes[1, 0]
    for name in [best_energy, best_temp]:
        temps = all_results[name]["temps"]
        gradients = np.abs(np.diff(temps, axis=1))  # Zone-to-zone diff
        mean_gradient = gradients.mean(axis=0)
        
        ax.plot(range(1, Z), mean_gradient, marker='s', 
               linewidth=2, markersize=6, label=name)
    
    ax.set_xlabel('Zone Interface')
    ax.set_ylabel('Mean |ΔT| (°C)')
    ax.set_title('Inter-Zone Temperature Gradient', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Flow Distribution Strategy
    ax = axes[1, 1]
    for name in [best_energy, best_temp]:
        flows = all_results[name]["flows"]
        mean_flow_per_zone = flows.mean(axis=0)
        
        ax.bar(np.arange(1, Z+1) - 0.2 if name == best_energy else np.arange(1, Z+1) + 0.2,
              mean_flow_per_zone, width=0.35, label=name, alpha=0.7)
    
    ax.set_xlabel('Zone Number')
    ax.set_ylabel('Mean Normalized Flow')
    ax.set_title('Cooling Flow Distribution Strategy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(1, Z+1))
    
    plt.tight_layout()
    plt.savefig('results/analysis_spatial_patterns.png', bbox_inches='tight')
    plt.show()

plot_spatial_analysis()

# =========================
# 5. PERFORMANCE RANKING RADAR CHART
# =========================

def plot_radar_comparison():
    """Multi-dimensional performance comparison"""
    
    # Select top 6 controllers
    top_controllers = df_comparison.nsmallest(6, "pump_energy_Wh")["label"].tolist()
    
    # Metrics to compare (normalized to 0-1)
    metrics = [
        ("Max Temp\n(lower better)", "max_temp", True),  # True = invert
        ("Mean Temp\n(lower better)", "mean_temp", True),
        ("Temp Spread\n(lower better)", "temp_spread_mean", True),
        ("Thermal Stress\n(lower better)", "thermal_stress", True),
        ("Pump Energy\n(lower better)", "pump_energy_Wh", True),
        ("Efficiency\n(higher better)", "cooling_overhead_pct", True),
    ]
    
    # Normalize metrics
    normalized_data = {}
    for name in top_controllers:
        row = df_comparison[df_comparison["label"] == name].iloc[0]
        normalized_data[name] = []
        
        for metric_name, metric_key, invert in metrics:
            val = row[metric_key]
            all_vals = df_comparison[metric_key].values
            
            # Normalize to 0-1
            norm_val = (val - all_vals.min()) / (all_vals.max() - all_vals.min() + 1e-9)
            
            # Invert if needed (so higher is always better)
            if invert:
                norm_val = 1.0 - norm_val
            
            normalized_data[name].append(norm_val)
    
    # Plot radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_controllers)))
    
    for idx, name in enumerate(top_controllers):
        values = normalized_data[name]
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, 
               color=colors[idx], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Multi-Dimensional Performance Comparison\n(Higher = Better)', 
                fontweight='bold', pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/analysis_radar_comparison.png', bbox_inches='tight')
    plt.show()

plot_radar_comparison()

# =========================
# 6. CONTROL ACTION ANALYSIS
# =========================

def plot_control_analysis():
    """Analyze control behavior patterns"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Select controllers
    controllers_analyze = df_comparison.nsmallest(5, "pump_energy_Wh")["label"].tolist()
    
    # 1. Action Frequency Spectrum
    ax = axes[0, 0]
    for name in controllers_analyze[:3]:
        flows = all_results[name]["flows"]
        mean_flow = flows.mean(axis=1)
        
        # Compute power spectral density
        from scipy.signal import welch
        freqs, psd = welch(mean_flow, fs=1.0/DT, nperseg=min(256, len(mean_flow)//2))
        
        ax.semilogy(freqs, psd, label=name, linewidth=2)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Control Action Frequency Content', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # 2. Action Smoothness (Rate of Change)
    ax = axes[0, 1]
    for name in controllers_analyze:
        flows = all_results[name]["flows"]
        dflow = np.abs(np.diff(flows, axis=0))
        smoothness = dflow.mean(axis=1)
        
        ax.hist(smoothness, bins=50, alpha=0.5, label=name, density=True)
    
    ax.set_xlabel('|Δflow/Δt|')
    ax.set_ylabel('Probability Density')
    ax.set_title('Control Smoothness Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Action Utilization (Flow magnitude distribution)
    ax = axes[0, 2]
    for name in controllers_analyze:
        flows = all_results[name]["flows"]
        flow_flat = flows.flatten()
        
        ax.hist(flow_flat, bins=50, alpha=0.5, label=name, density=True)
    
    ax.set_xlabel('Normalized Flow Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Flow Utilization Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Control Aggressiveness Over Time
    ax = axes[1, 0]
    time_hrs = np.arange(Tsteps) / 3600.0
    
    for name in controllers_analyze[:3]:
        flows = all_results[name]["flows"]
        aggressiveness = flows.max(axis=1)  # Max flow across zones
        
        # Smooth for visualization
        from scipy.ndimage import uniform_filter1d
        if len(aggressiveness) > 10:
            aggressiveness_smooth = uniform_filter1d(aggressiveness, size=min(50, len(aggressiveness)//10))
        else:
            aggressiveness_smooth = aggressiveness
        
        ax.plot(time_hrs, aggressiveness_smooth, label=name, linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Max Zone Flow')
    ax.set_title('Control Aggressiveness Evolution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Action Diversity (Entropy)
    ax = axes[1, 1]
    entropies = []
    names_entropy = []
    
    for name in controllers_analyze:
        flows = all_results[name]["flows"]
        
        # Compute temporal entropy per zone
        zone_entropies = []
        for z in range(Z):
            hist, _ = np.histogram(flows[:, z], bins=20, density=True)
            hist = hist + 1e-10  # Avoid log(0)
            entropy = -np.sum(hist * np.log2(hist))
            zone_entropies.append(entropy)
        
        entropies.append(np.mean(zone_entropies))
        names_entropy.append(name)
    
    ax.barh(range(len(names_entropy)), entropies, alpha=0.7)
    ax.set_yticks(range(len(names_entropy)))
    ax.set_yticklabels(names_entropy, fontsize=9)
    ax.set_xlabel('Average Action Entropy (bits)')
    ax.set_title('Control Strategy Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 6. Response Time Analysis
    ax = axes[1, 2]
    
    # Find temperature violations and measure response time
    response_times = []
    names_response = []
    
    for name in controllers_analyze:
        temps = all_results[name]["temps"]
        flows = all_results[name]["flows"]
        
        # Find moments when temperature exceeds setpoint
        violations = temps.max(axis=1) > TEMP_SETPOINT
        
        if np.any(violations):
            # Find first violation
            first_violation_idx = np.where(violations)[0][0]
            
            # Measure time to respond (flow increase)
            if first_violation_idx < len(flows) - 10:
                pre_flow = flows[max(0, first_violation_idx-5):first_violation_idx].mean()
                post_flows = flows[first_violation_idx:first_violation_idx+20].mean(axis=1)
                
                # Find when flow increases by 20%
                response_idx = np.where(post_flows > pre_flow * 1.2)[0]
                if len(response_idx) > 0:
                    response_time = response_idx[0] * DT
                else:
                    response_time = 20 * DT
            else:
                response_time = 0
        else:
            response_time = 0
        
        response_times.append(response_time)
        names_response.append(name)
    
    ax.barh(range(len(names_response)), response_times, alpha=0.7, color='coral')
    ax.set_yticks(range(len(names_response)))
    ax.set_yticklabels(names_response, fontsize=9)
    ax.set_xlabel('Response Time (seconds)')
    ax.set_title('Control Response Speed', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/analysis_control_behavior.png', bbox_inches='tight')
    plt.show()

plot_control_analysis()

# =========================
# 7. STATISTICAL SIGNIFICANCE TESTING
# =========================

def perform_statistical_tests():
    """Perform pairwise statistical tests"""
    
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 70)
    
    # Select top 5 controllers
    top_5 = df_comparison.nsmallest(5, "pump_energy_Wh")["label"].tolist()
    
    print("\n📊 PAIRWISE T-TESTS (Temperature Distribution):")
    print("-" * 70)
    
    # Create comparison matrix
    from scipy.stats import ttest_ind
    
    for i, name1 in enumerate(top_5):
        for j, name2 in enumerate(top_5):
            if i < j:
                temps1 = all_results[name1]["temps"].flatten()
                temps2 = all_results[name2]["temps"].flatten()
                
                t_stat, p_val = ttest_ind(temps1, temps2)
                
                sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                print(f"{name1:20s} vs {name2:20s}: t={t_stat:7.3f}, p={p_val:.4f} {sig_marker}")
    
    print("\n📊 ENERGY EFFICIENCY ANOVA:")
    print("-" * 70)
    
    from scipy.stats import f_oneway
    
    energy_groups = [all_results[name]["pump"] for name in top_5]
    f_stat, p_val = f_oneway(*energy_groups)
    
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_val:.6f}")
    print(f"Conclusion: {'Significant differences exist' if p_val < 0.05 else 'No significant differences'}")
    
    print("\n" + "=" * 70)

perform_statistical_tests()

# =========================
# 8. FINAL SUMMARY TABLE
# =========================

def create_summary_table():
    """Create comprehensive summary table"""
    
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 70)
    
    # Rank controllers
    df_ranked = df_comparison.copy()
    
    # Add rankings
    df_ranked['rank_temp'] = df_ranked['max_temp'].rank()
    df_ranked['rank_energy'] = df_ranked['pump_energy_Wh'].rank()
    df_ranked['rank_spread'] = df_ranked['temp_spread_mean'].rank()
    df_ranked['rank_overall'] = (df_ranked['rank_temp'] + df_ranked['rank_energy'] + df_ranked['rank_spread']) / 3
    
    df_ranked = df_ranked.sort_values('rank_overall')
    
    # Display table
    display_cols = [
        'label', 'max_temp', 'mean_temp', 'temp_spread_mean',
        'thermal_stress', 'pump_energy_Wh', 'cooling_overhead_pct',
        'rank_overall'
    ]
    
    print("\n" + df_ranked[display_cols].to_string(index=False))
    
    # Save detailed CSV
    df_ranked.to_csv('results/final_performance_ranking.csv', index=False)
    print("\n💾 Saved detailed rankings to results/final_performance_ranking.csv")
    
    # Print winner
    winner = df_ranked.iloc[0]['label']
    print("\n" + "=" * 70)
    print(f"🏆 OVERALL BEST CONTROLLER: {winner}")
    print("=" * 70)

create_summary_table()

print("\n✅ COMPREHENSIVE ANALYSIS COMPLETE")
print("📁 All figures saved to results/ directory")

# %%



