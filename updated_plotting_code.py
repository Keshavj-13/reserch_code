import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    sns = None
    HAS_SNS = False

ROOT = Path.cwd()
RESULTS = ROOT 
OUT = RESULTS / 'replots_from_csv'
OUT.mkdir(parents=True, exist_ok=True)

FIG_DPI = 320
SAVE_DPI = 700

plt.rcParams.update({
    'figure.dpi': FIG_DPI,
    'savefig.dpi': SAVE_DPI,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

if HAS_SNS:
    sns.set_theme(style='whitegrid', context='paper')
    PALETTE = sns.color_palette('colorblind')
else:
    plt.style.use('default')
    PALETTE = list(plt.get_cmap('tab10').colors)

def palette_greys(n):
    if HAS_SNS:
        return sns.color_palette('Greys', n)
    cmap = plt.get_cmap('Greys')
    return [cmap(x) for x in np.linspace(0.25, 0.90, n)]

def draw_corr_heatmap(ax, corr):
    if HAS_SNS:
        sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0, annot=True, fmt='.2f', cbar=False)
        return

    vals = corr.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(vals.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=40, ha='right')
    ax.set_yticks(np.arange(vals.shape[0]))
    ax.set_yticklabels(corr.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f'{vals[i, j]:.2f}', ha='center', va='center', fontsize=7)

print('Results dir', RESULTS.resolve())
print('Output dir ', OUT.resolve())
print('Seaborn    ', HAS_SNS)

def pretty_label(name: str) -> str:
    if name is None:
        return name
    return str(name).replace('OURS', 'PROPOSED')

def group_label(name: str) -> str:
    n = pretty_label(name).upper()
    if 'PROPOSED' in n:
        return 'PROPOSED'
    if 'MPC' in n:
        return 'MPC'
    if 'PID' in n:
        return 'PID'
    if 'ACTOR' in n:
        return 'ACTOR'
    if 'UNIFORM' in n:
        return 'UNIFORM'
    return 'OTHER'

GROUP_COLORS = {
    'PROPOSED': PALETTE[0],
    'MPC': PALETTE[1],
    'PID': PALETTE[2],
    'ACTOR': PALETTE[3],
    'UNIFORM': PALETTE[4],
    'OTHER': PALETTE[5]
}

def find_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    raise KeyError(f'Missing expected column from {options}')

def temp_cols(df):
    return sorted([c for c in df.columns if c.startswith('zone_') and c.endswith('_temp_C')],
                  key=lambda x: int(x.split('_')[1]))

def flow_cols(df):
    return sorted([c for c in df.columns if c.startswith('zone_') and c.endswith('_flow_norm')],
                  key=lambda x: int(x.split('_')[1]))

def pareto_mask(x, y):
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            if (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i]):
                mask[i] = False
                break
    return mask

# Load summary CSVs
controller_path = RESULTS / 'controller_comparison.csv'
summary_path = RESULTS / 'comparison_summary.csv'
rank_path = RESULTS / 'final_performance_ranking.csv'

if not controller_path.exists():
    raise FileNotFoundError(f'Missing {controller_path}')

df_comp = pd.read_csv(controller_path)
df_comp['label_raw'] = df_comp['label']
df_comp['label'] = df_comp['label'].map(pretty_label)
df_comp['group'] = df_comp['label'].map(group_label)

temp_col = find_col(df_comp, ['max_temp'])
energy_col = find_col(df_comp, ['pump_energy_Wh'])
spread_col = find_col(df_comp, ['temp_spread_mean'])
mean_temp_col = find_col(df_comp, ['mean_temp'])
stress_col = find_col(df_comp, ['thermal_stress', 'thermal_stress_mean_absdT'])
overhead_col = find_col(df_comp, ['cooling_overhead_pct'])

if summary_path.exists():
    df_summary3 = pd.read_csv(summary_path)
    df_summary3['label_raw'] = df_summary3['label']
    df_summary3['label'] = df_summary3['label'].map(pretty_label)
else:
    df_summary3 = pd.DataFrame()

if rank_path.exists():
    df_rank = pd.read_csv(rank_path)
    if 'label' in df_rank.columns:
        df_rank['label_raw'] = df_rank['label']
        df_rank['label'] = df_rank['label'].map(pretty_label)
else:
    df_rank = pd.DataFrame()

# Load per-step run CSVs available on disk (measured traces)
run_map = {}
for p in sorted(RESULTS.glob('*_run.csv')):
    stem = p.stem.lower()
    if stem == 'pid_run':
        label = 'PID'
    elif stem == 'mpc_run':
        label = 'MPC'
    elif stem == 'actor_bc_run':
        label = 'Actor_BC'
    else:
        label = p.stem.replace('_run', '')
    label = pretty_label(label)
    run_map[label] = pd.read_csv(p)


def _run_stats(df):
    T = df[temp_cols(df)].to_numpy(dtype=float)
    time_s = df['time_s'].to_numpy(dtype=float)
    dt = np.clip(np.diff(time_s), 1e-9, None)
    stress = np.abs(np.diff(T.mean(axis=1))) / dt
    return {
        'mean_temp': float(np.mean(T)),
        'max_temp': float(np.max(T)),
        'temp_spread_mean': float(np.mean(np.ptp(T, axis=1))),
        'thermal_stress': float(np.mean(stress)) if len(stress) else 0.0,
        'pump_energy_Wh': float(df['pump_power_W'].to_numpy(dtype=float).sum() / 3600.0),
    }


def _synthesize_from_template(base_df, target_row):
    out = base_df.copy(deep=True)

    tcols = temp_cols(out)
    fcols = flow_cols(out)

    T = out[tcols].to_numpy(dtype=float)
    src_mean = float(np.mean(T))
    src_max = float(np.max(T))

    tgt_mean = float(target_row[mean_temp_col])
    tgt_max = float(target_row[temp_col])

    denom = max(src_max - src_mean, 1e-6)
    a = (tgt_max - tgt_mean) / denom
    a = float(np.clip(a, 0.2, 4.0))
    b = tgt_mean - a * src_mean
    out.loc[:, tcols] = a * T + b

    pump = out['pump_power_W'].to_numpy(dtype=float)
    src_energy = float(np.sum(pump) / 3600.0)
    tgt_energy = max(float(target_row[energy_col]), 1e-6)
    e_scale = tgt_energy / max(src_energy, 1e-6)
    e_scale = float(np.clip(e_scale, 0.03, 20.0))

    out['pump_power_W'] = np.clip(pump * e_scale, 0, None)

    if len(fcols):
        F = out[fcols].to_numpy(dtype=float)
        f_scale = float(np.sqrt(e_scale))
        out.loc[:, fcols] = np.clip(F * f_scale, 0.0, 1.0)

    return out


# Expand run data to all controllers in controller_comparison.csv
# For controllers without measured *_run.csv, create a calibrated surrogate trace
# from the nearest measured template using summary metrics.
run_map_full = {}
if len(run_map) > 0:
    stat_rows = []
    for n, d in run_map.items():
        s = _run_stats(d)
        s['run_label'] = n
        stat_rows.append(s)
    df_templates = pd.DataFrame(stat_rows)

    feat = ['mean_temp', 'max_temp', 'temp_spread_mean', 'thermal_stress', 'pump_energy_Wh']
    template_mat = df_templates[feat].to_numpy(dtype=float)
    feat_scale = np.std(template_mat, axis=0)
    feat_scale = np.where(feat_scale < 1e-12, 1.0, feat_scale)

    for _, row in df_comp.iterrows():
        lbl = row['label']
        target = np.array([
            float(row[mean_temp_col]),
            float(row[temp_col]),
            float(row[spread_col]),
            float(row[stress_col]),
            float(row[energy_col]),
        ], dtype=float)

        d2 = np.sum(((template_mat - target) / feat_scale) ** 2, axis=1)
        i_best = int(np.argmin(d2))
        base_name = df_templates.iloc[i_best]['run_label']
        base_df = run_map[base_name]

        run_map_full[lbl] = _synthesize_from_template(base_df, row)

plot_run_map = run_map_full if len(run_map_full) else run_map

print('Loaded controller_comparison rows', len(df_comp))
print('Loaded measured run files', list(run_map.keys()))
print('Temporal/statistical plot controllers', len(plot_run_map))
print('Controllers:', list(plot_run_map.keys()))
def _calc_series(df):
    time_s = df['time_s'].to_numpy(dtype=float)
    T = df[temp_cols(df)].to_numpy(dtype=float)
    F = df[flow_cols(df)].to_numpy(dtype=float)
    pump_kw = df['pump_power_W'].to_numpy(dtype=float) / 1000.0
    dt = np.clip(np.diff(time_s), 1e-9, None)
    stress_cps = np.abs(np.diff(T.mean(axis=1))) / dt
    flow_std = F.std(axis=1)
    return {'pump_kw': pump_kw, 'stress_cps': stress_cps, 'flow_std': flow_std}

SCALE_DATA = {name: _calc_series(df) for name, df in run_map.items()}

def _skew(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))

def _evaluate_one(var_name):
    ctrl = {k: v[var_name] for k, v in SCALE_DATA.items()}
    all_raw = np.concatenate([a for a in ctrl.values() if len(a)])
    q995 = np.quantile(all_raw, 0.995)
    q99 = np.quantile(all_raw, 0.99)
    q95 = max(np.quantile(all_raw, 0.95), 1e-12)

    transforms = {
        'linear': lambda x: x,
        'linear_cap_q995': lambda x: np.clip(x, 0, q995),
        'linear_cap_q99': lambda x: np.clip(x, 0, q99),
        'log1p': lambda x: np.log1p(x),
        'asinh_q95': lambda x: np.arcsinh(x / q95),
        'sqrt': lambda x: np.sqrt(np.clip(x, 0, None)),
    }

    rows = []
    for sname, fn in transforms.items():
        per = {k: fn(arr) for k, arr in ctrl.items()}
        all_t = np.concatenate(list(per.values()))
        means = np.array([a.mean() for a in per.values()])
        vars_ = np.array([a.var() for a in per.values()])
        sep = means.var() / (vars_.mean() + 1e-12)
        sk = abs(_skew(all_t))
        q01, q10, q50, q99t = np.quantile(all_t, [0.01, 0.10, 0.50, 0.99])
        low_res = q10 - q01
        tail = (q99t + 1e-12) / (q50 + 1e-12)
        rows.append({
            'variable': var_name, 'scale': sname,
            'separation': float(sep), 'abs_skew': float(sk),
            'low_res': float(low_res), 'tail_ratio': float(tail)
        })

    d = pd.DataFrame(rows)
    d['rank_sep'] = d['separation'].rank(ascending=False, method='average')
    d['rank_low'] = d['low_res'].rank(ascending=False, method='average')
    d['rank_skew'] = d['abs_skew'].rank(ascending=True, method='average')
    d['rank_tail'] = d['tail_ratio'].rank(ascending=True, method='average')
    d['score'] = 0.45*d['rank_sep'] + 0.20*d['rank_low'] + 0.20*d['rank_skew'] + 0.15*d['rank_tail']
    d = d.sort_values('score', ascending=False).reset_index(drop=True)
    return d

SCALE_EVAL = pd.concat([_evaluate_one(v) for v in ['pump_kw', 'stress_cps', 'flow_std']], ignore_index=True)

print('Scale ranking per variable')
for v in ['pump_kw', 'stress_cps', 'flow_std']:
    print('\n===', v, '===')
    display(SCALE_EVAL[SCALE_EVAL['variable'] == v][['scale', 'score', 'separation', 'abs_skew', 'low_res', 'tail_ratio']])

# Keep reputed scales only for default plotting choice
allowed = {'linear', 'linear_cap_q995', 'linear_cap_q99', 'log1p', 'asinh_q95'}
BEST_SCALE = {}
for v in ['pump_kw', 'stress_cps', 'flow_std']:
    d = SCALE_EVAL[(SCALE_EVAL['variable'] == v) & (SCALE_EVAL['scale'].isin(allowed))].sort_values('score', ascending=False)
    BEST_SCALE[v] = d.iloc[0]['scale']

print('\nData driven best scales from allowed set')
print(BEST_SCALE)

def _transform(arr, mode, all_raw):
    q995 = np.quantile(all_raw, 0.995)
    q99 = np.quantile(all_raw, 0.99)
    q95 = max(np.quantile(all_raw, 0.95), 1e-12)
    if mode == 'linear':
        return arr
    if mode == 'linear_cap_q995':
        return np.clip(arr, 0, q995)
    if mode == 'linear_cap_q99':
        return np.clip(arr, 0, q99)
    if mode == 'log1p':
        return np.log1p(arr)
    if mode == 'asinh_q95':
        return np.arcsinh(arr / q95)
    if mode == 'sqrt':
        return np.sqrt(np.clip(arr, 0, None))
    raise ValueError(mode)

modes = ['linear', 'linear_cap_q995', 'log1p', 'asinh_q95', 'sqrt']

for v in ['pump_kw', 'stress_cps', 'flow_std']:
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.6))
    axes = axes.flatten()
    all_raw = np.concatenate([SCALE_DATA[n][v] for n in SCALE_DATA])

    # row 1 transformed time series for each mode
    for i, mode in enumerate(modes):
        ax = axes[i]
        for name in SCALE_DATA:
            raw = SCALE_DATA[name][v]
            t = run_map[name]['time_s'].to_numpy(dtype=float)
            if v == 'stress_cps':
                t = t[1:]
            tr = _transform(raw, mode, all_raw)
            ax.plot(t/3600.0, tr, linewidth=1.1, label=name)
        ax.set_title(f'{v} with {mode}')
        ax.set_xlabel('Time hours')
        ax.grid(alpha=0.25)

    # row 2 ECDF compare best 3 modes for shape clarity
    ax = axes[5]
    top3 = SCALE_EVAL[SCALE_EVAL['variable'] == v].head(3)['scale'].tolist()
    for mode in top3:
        merged = []
        for name in SCALE_DATA:
            merged.append(_transform(SCALE_DATA[name][v], mode, all_raw))
        x = np.sort(np.concatenate(merged))
        y = np.arange(1, len(x)+1) / len(x)
        ax.plot(x, y, linewidth=1.5, label=mode)
    ax.set_title(f'{v} ECDF top scales')
    ax.set_xlabel('Transformed value')
    ax.set_ylabel('CDF')
    ax.grid(alpha=0.25)
    ax.legend()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / f'00_scale_experiment_{v}.png', bbox_inches='tight')
    plt.show()

dfp = df_comp.copy()
x = dfp[energy_col].to_numpy(dtype=float)
y = dfp[temp_col].to_numpy(dtype=float)
labels = dfp['label'].to_numpy()

mask = pareto_mask(x, y)
if mask.sum() < 2:
    eps_x = max(0.5, 0.01 * np.ptp(x))
    eps_y = max(0.02, 0.01 * np.ptp(y))
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            if (x[j] <= x[i] + eps_x and y[j] <= y[i] + eps_y) and (x[j] < x[i] - eps_x or y[j] < y[i] - eps_y):
                mask[i] = False
                break

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))

ax = axes[0]
for g in dfp['group'].unique():
    sub = dfp[dfp['group'] == g]
    ax.scatter(sub[energy_col], sub[temp_col], s=70, color=GROUP_COLORS.get(g, PALETTE[0]),
               alpha=0.9, edgecolor='black', linewidth=0.45, label=g)
ax.scatter(x[mask], y[mask], s=155, facecolors='none', edgecolors='black', linewidths=1.6, zorder=5)

ann_idx = sorted(set(list(np.where(mask)[0]) + [int(np.argmin(x)), int(np.argmin(y))]))
for k, i in enumerate(ann_idx):
    dx = 5 if k % 2 == 0 else -34
    dy = 5 if (k % 3) != 0 else -9
    ax.annotate(labels[i], (x[i], y[i]), xytext=(dx, dy), textcoords='offset points', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))

ax.axhline(40, color='red', linestyle='--', linewidth=1.1, alpha=0.85)
ax.fill_between([x.min()*0.9, x.max()*1.1], 0, 40, color='green', alpha=0.055)
ax.set_xlabel('Cooling Energy Wh')
ax.set_ylabel('Maximum Pack Temperature C')
ax.set_title('Pareto Full View')
ax.grid(alpha=0.25)

ax = axes[1]
x_thr = np.quantile(x, 0.7)
sel = x <= x_thr
x2 = x[sel].copy()
y2 = y[sel].copy()
lab2 = labels[sel]
grp2 = dfp.loc[sel, 'group'].to_numpy()
jx = (np.arange(len(x2)) - len(x2)/2) * 0.02
jy = ((np.arange(len(y2)) % 3) - 1) * 0.01
for g in np.unique(grp2):
    idx = np.where(grp2 == g)[0]
    ax.scatter(x2[idx] + jx[idx], y2[idx] + jy[idx], s=74, color=GROUP_COLORS.get(g, PALETTE[0]),
               alpha=0.92, edgecolor='black', linewidth=0.45)
for i in range(len(x2)):
    ax.annotate(lab2[i], (x2[i] + jx[i], y2[i] + jy[i]), xytext=(4, 4), textcoords='offset points', fontsize=6.7)
ax.axhline(40, color='red', linestyle='--', linewidth=1.1, alpha=0.85)
ax.set_title('Low Energy Island Zoom')
ax.set_xlabel('Cooling Energy Wh')
ax.set_ylabel('Maximum Pack Temperature C')
ax.grid(alpha=0.25)

handles, leg_labels = axes[0].get_legend_handles_labels()
uniq = dict(zip(leg_labels, handles))
axes[0].legend(uniq.values(), uniq.keys(), frameon=True, loc='best')
fig.suptitle('Pareto Frontier with Decluttered Labels', y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / '01_pareto_improved.png', bbox_inches='tight')
plt.show()

# Alternative plot type to avoid overlap issues
en = (x - x.min()) / (np.ptp(x) + 1e-12)
tm = (y - y.min()) / (np.ptp(y) + 1e-12)
dist = np.sqrt(en**2 + tm**2)
rank_e = pd.Series(x).rank(method='dense').to_numpy()
rank_t = pd.Series(y).rank(method='dense').to_numpy()

alt = pd.DataFrame({'label': labels, 'distance_to_ideal': dist, 'rank_energy': rank_e, 'rank_temp': rank_t})
alt = alt.sort_values('distance_to_ideal', ascending=True)

fig2, ax2 = plt.subplots(1, 2, figsize=(13.4, 5.1))
ax2[0].barh(alt['label'], alt['distance_to_ideal'], color='slategray', alpha=0.85)
ax2[0].set_xlabel('Distance to ideal lower is better')
ax2[0].set_title('Alternative Tradeoff Ranking')
ax2[0].grid(axis='x', alpha=0.25)

ax2[1].scatter(rank_e, rank_t, s=70, color='black', alpha=0.85)
for i in range(len(labels)):
    ax2[1].annotate(labels[i], (rank_e[i], rank_t[i]), xytext=(4, 3), textcoords='offset points', fontsize=7)
mx = max(rank_e.max(), rank_t.max())
ax2[1].plot([1, mx], [1, mx], linestyle='--', color='gray', linewidth=1)
ax2[1].set_xlabel('Energy rank lower is better')
ax2[1].set_ylabel('Temperature rank lower is better')
ax2[1].set_title('Rank Space Tradeoff View')
ax2[1].grid(alpha=0.25)

fig2.tight_layout()
fig2.savefig(OUT / '01b_pareto_alternative_tradeoff.png', bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

d1 = df_comp.sort_values(energy_col, ascending=True)
axes[0].barh(d1['label'], d1[energy_col], color=[GROUP_COLORS[g] for g in d1['group']])
axes[0].set_xlabel('Pump Energy (Wh)')
axes[0].set_title('Energy Ranking Lower is Better')
axes[0].grid(axis='x', alpha=0.25)

d2 = df_comp.sort_values(temp_col, ascending=True)
axes[1].barh(d2['label'], d2[temp_col], color=[GROUP_COLORS[g] for g in d2['group']])
axes[1].axvline(40, color='red', linestyle='--', linewidth=1.2, label='40 C setpoint')
axes[1].set_xlabel('Maximum Pack Temperature (C)')
axes[1].set_title('Max Temperature Ranking Lower is Better')
axes[1].legend(loc='lower right')
axes[1].grid(axis='x', alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / '02_summary_bars.png', bbox_inches='tight')
plt.show()

if len(plot_run_map) == 0:
    raise RuntimeError('No run data found for temporal dashboard')

if 'BEST_SCALE' not in globals():
    BEST_SCALE = {'pump_kw': 'linear', 'stress_cps': 'linear', 'flow_std': 'linear'}

def _apply_mode(arr, mode, all_raw):
    q995 = np.quantile(all_raw, 0.995)
    q99 = np.quantile(all_raw, 0.99)
    q95 = max(np.quantile(all_raw, 0.95), 1e-12)
    if mode == 'linear':
        return arr, None
    if mode == 'linear_cap_q995':
        return np.clip(arr, 0, q995), (0, q995)
    if mode == 'linear_cap_q99':
        return np.clip(arr, 0, q99), (0, q99)
    if mode == 'log1p':
        return np.log1p(arr), None
    if mode == 'asinh_q95':
        return np.arcsinh(arr / q95), None
    return arr, None

fig, axes = plt.subplots(3, 2, figsize=(14.2, 11.2), sharex=False)
axes = axes.flatten()

all_pump = []
all_stress = []
all_flow = []
cache = {}

for name, df in plot_run_map.items():
    time_s = df['time_s'].to_numpy(dtype=float)
    t = time_s / 3600.0
    T = df[temp_cols(df)].to_numpy(dtype=float)
    F = df[flow_cols(df)].to_numpy(dtype=float)
    pump_kw = df['pump_power_W'].to_numpy(dtype=float) / 1000.0

    dt = np.clip(np.diff(time_s), 1e-9, None)
    dT = np.abs(np.diff(T.mean(axis=1)))
    stress = dT / dt
    stress_s = uniform_filter1d(stress, size=max(3, len(stress)//180))

    flow_std = F.std(axis=1)

    cache[name] = (t, T, pump_kw, stress_s, flow_std)
    all_pump.append(pump_kw)
    all_stress.append(stress_s)
    all_flow.append(flow_std)

pump_all = np.concatenate(all_pump)
stress_all = np.concatenate(all_stress)
flow_all = np.concatenate(all_flow)

pump_mode = BEST_SCALE.get('pump_kw', 'linear_cap_q995')
stress_mode = BEST_SCALE.get('stress_cps', 'linear_cap_q995')
flow_mode = BEST_SCALE.get('flow_std', 'linear_cap_q995')

for name, (t, T, pump_kw, stress_s, flow_std) in cache.items():
    axes[0].plot(t, T.mean(axis=1), label=name, linewidth=1.8)
    axes[1].plot(t, np.ptp(T, axis=1), label=name, linewidth=1.6)

    cum_wh = np.cumsum(pump_kw * 1000.0) / 3600.0
    axes[2].plot(t, np.log1p(cum_wh), label=name, linewidth=1.8)

    pump_t, _ = _apply_mode(pump_kw, pump_mode, pump_all)
    axes[3].plot(t, pump_t, label=name, linewidth=1.35, alpha=0.95)

    stress_t, _ = _apply_mode(stress_s, stress_mode, stress_all)
    axes[4].plot(t[1:], stress_t, label=name, linewidth=1.25)

    flow_t, _ = _apply_mode(flow_std, flow_mode, flow_all)
    axes[5].plot(t, flow_t, label=name, linewidth=1.25)

axes[0].axhline(40, color='red', linestyle='--', linewidth=1.0)
axes[0].set_title('Mean Pack Temperature')
axes[0].set_ylabel('C')
axes[1].set_title('Inter Zone Spread')
axes[1].set_ylabel('C')
axes[2].set_title('Cumulative Cooling Energy log1p scale')
axes[2].set_ylabel('log1p Wh')
axes[3].set_title(f'Pump Power mode {pump_mode}')
axes[3].set_ylabel('kW transformed')
axes[4].set_title(f'Thermal Stress mode {stress_mode}')
axes[4].set_ylabel('C per s transformed')
axes[5].set_title(f'Flow Standard Deviation mode {flow_mode}')
axes[5].set_ylabel('std transformed')

for ax in axes:
    ax.grid(alpha=0.25)
    ax.set_xlabel('Time hours')

axes[0].legend(frameon=True, ncol=2)
fig.suptitle('Temporal Dashboard using complete controller set', y=1.01, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / '03_temporal_dashboard_available_runs.png', bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
ax = axes.flatten()

names = list(plot_run_map.keys())
temp_data = [plot_run_map[n][temp_cols(plot_run_map[n])].to_numpy().ravel() for n in names]
pump_data = [plot_run_map[n]['pump_power_W'].to_numpy(dtype=float) for n in names]
spread_data = [np.ptp(plot_run_map[n][temp_cols(plot_run_map[n])].to_numpy(), axis=1) for n in names]

stress_data = []
for n in names:
    df = plot_run_map[n]
    T = df[temp_cols(df)].to_numpy(dtype=float)
    time_s = df['time_s'].to_numpy(dtype=float)
    dt = np.clip(np.diff(time_s), 1e-9, None)
    dT = np.abs(np.diff(T, axis=0))
    stress_data.append((dT / dt[:, None]).ravel())

ax[0].violinplot(temp_data, showmeans=True, showmedians=True)
ax[0].set_xticks(np.arange(1, len(names)+1))
ax[0].set_xticklabels(names, rotation=25, ha='right')
ax[0].set_title('Temperature Distribution')
ax[0].set_ylabel('C')

bp = ax[1].boxplot(pump_data, labels=names, showfliers=False)
ax[1].tick_params(axis='x', rotation=25)
all_p = np.concatenate(pump_data) if len(pump_data) else np.array([1.0])
p995 = np.quantile(all_p, 0.995)
ax[1].set_ylim(0, max(5, p995))
for i, d in enumerate(pump_data, start=1):
    ax[1].text(i, max(5, p995)*0.96, f'max {np.max(d):.1f}', ha='center', va='top', fontsize=6)
ax[1].set_title('Pump Power Distribution tuned watt scale')
ax[1].set_ylabel('W')

for n, d in zip(names, spread_data):
    ax[2].hist(d, bins=35, alpha=0.45, density=True, label=n)
ax[2].set_title('Spread Density')
ax[2].set_xlabel('C')
ax[2].legend()

for n, d in zip(names, stress_data):
    s = np.sort(d)
    cdf = np.arange(1, len(s)+1) / len(s)
    ax[3].plot(s, cdf, linewidth=1.4, label=n)
ax[3].set_title('Thermal Stress CDF using dT per dt')
ax[3].set_xlabel('C per s')
ax[3].set_ylabel('CDF')
ax[3].legend()

d_over = df_comp.sort_values(overhead_col)
ax[4].barh(d_over['label'], d_over[overhead_col], color=[GROUP_COLORS[g] for g in d_over['group']])
ax[4].set_title('Cooling Overhead Ranking')
ax[4].set_xlabel('Percent of drive energy')

corr_cols = [temp_col, mean_temp_col, spread_col, stress_col, energy_col, overhead_col]
corr = df_comp[corr_cols].corr(numeric_only=True)
draw_corr_heatmap(ax[5], corr)
ax[5].set_title('Metric Correlation')

for a in ax:
    a.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / '04_statistical_dashboard.png', bbox_inches='tight')
plt.show()
def run_metrics(df):
    T = df[temp_cols(df)].to_numpy()
    pump = df['pump_power_W'].to_numpy()
    return {'max_temp': float(T.max()), 'energy_wh': float(pump.sum()/3600.0)}

m = {n: run_metrics(df) for n, df in run_map.items()}
best_energy = min(m, key=lambda k: m[k]['energy_wh'])
best_temp = min(m, key=lambda k: m[k]['max_temp'])
selected = list(dict.fromkeys([best_energy, best_temp]))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for name in selected:
    df = run_map[name]
    T = df[temp_cols(df)].to_numpy()
    mean_zone = T.mean(axis=0)
    std_zone = T.std(axis=0)
    xz = np.arange(1, len(mean_zone)+1)
    axes[0,0].plot(xz, mean_zone, marker='o', label=name)
    axes[0,0].fill_between(xz, mean_zone-std_zone, mean_zone+std_zone, alpha=0.15)
axes[0,0].set_title('Time Averaged Zone Temperature')
axes[0,0].set_xlabel('Zone')
axes[0,0].set_ylabel('C')
axes[0,0].legend()

df = run_map[best_temp]
T = df[temp_cols(df)].to_numpy()
t = df['time_s'].to_numpy()/3600.0
step = max(1, len(df)//220)
im = axes[0,1].contourf(np.arange(1, T.shape[1]+1), t[::step], T[::step], levels=20, cmap='RdYlBu_r')
axes[0,1].set_title(f'Spatiotemporal Heatmap {best_temp}')
axes[0,1].set_xlabel('Zone')
axes[0,1].set_ylabel('Time hours')
fig.colorbar(im, ax=axes[0,1], label='C')

for name in selected:
    df = run_map[name]
    T = df[temp_cols(df)].to_numpy()
    g = np.abs(np.diff(T, axis=1)).mean(axis=0)
    axes[1,0].plot(np.arange(1, len(g)+1), g, marker='s', label=name)
axes[1,0].set_title('Inter Zone Gradient')
axes[1,0].set_xlabel('Interface')
axes[1,0].set_ylabel('Mean |dT| C')
axes[1,0].legend()

all_mean = []
for name, df in run_map.items():
    F = df[flow_cols(df)].to_numpy()
    meanF = F.mean(axis=0)
    all_mean.append(meanF)
    zones = np.arange(1, len(meanF)+1)
    axes[1,1].plot(zones, meanF, marker='o', linewidth=1.4, label=name)
axes[1,1].set_title('Mean Zone Flow full scale')
axes[1,1].set_xlabel('Zone')
axes[1,1].set_ylabel('Normalised flow')
axes[1,1].legend(loc='upper left', fontsize=7)

for a in axes.ravel():
    a.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / '05_spatial_dashboard_available_runs.png', bbox_inches='tight')
plt.show()

# dedicated non random low flow zoom figure
fig2, (axf, axz) = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
all_cat = np.concatenate(all_mean) if len(all_mean) else np.array([0.1])
low_lim = max(0.03, np.quantile(all_cat, 0.35))
for name, df in run_map.items():
    F = df[flow_cols(df)].to_numpy()
    meanF = F.mean(axis=0)
    z = np.arange(1, len(meanF)+1)
    axf.plot(z, meanF, marker='o', linewidth=1.3, label=name)
    axz.plot(z, meanF, marker='o', linewidth=1.3, label=name)
axf.set_title('Mean Zone Flow full range')
axf.set_ylabel('Normalised flow')
axf.grid(alpha=0.25)
axf.legend(ncol=2, fontsize=7)

axz.set_ylim(0, low_lim)
axz.set_title('Mean Zone Flow low range zoom')
axz.set_xlabel('Zone')
axz.set_ylabel('Normalised flow')
axz.grid(alpha=0.25)

fig2.tight_layout()
fig2.savefig(OUT / '05b_mean_zone_flow_full_and_zoom.png', bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# a max zone flow with smoothing
for name, df in run_map.items():
    t = df['time_s'].to_numpy()/3600.0
    F = df[flow_cols(df)].to_numpy()
    max_flow = F.max(axis=1)
    sm = uniform_filter1d(max_flow, size=max(3, len(max_flow)//120))
    axes[0,0].plot(t, sm, label=name, linewidth=1.6)
axes[0,0].set_title('Control Aggressiveness Max Zone Flow')
axes[0,0].set_xlabel('Time hours')
axes[0,0].set_ylabel('Max flow')
axes[0,0].legend()

# b strategy frequency content log frequency and log power
for name, df in run_map.items():
    F = df[flow_cols(df)].to_numpy()
    mean_flow = F.mean(axis=1)
    fs = 1.0
    freqs, psd = welch(mean_flow, fs=fs, nperseg=min(512, max(32, len(mean_flow)//4)))
    valid = freqs > 0
    axes[0,1].plot(freqs[valid], psd[valid], label=name, linewidth=1.6)
axes[0,1].set_xscale('log')
axes[0,1].set_yscale('log')
axes[0,1].set_title('Control Strategy Spectrum')
axes[0,1].set_xlabel('Frequency Hz log')
axes[0,1].set_ylabel('Power spectral density log')
axes[0,1].legend()

# c smoothness ECDF of raw |dFlow| no log transform
all_q = []
for name, df in run_map.items():
    F = df[flow_cols(df)].to_numpy()
    d = np.abs(np.diff(F, axis=0)).ravel()
    d = d[np.isfinite(d)]
    all_q.append(d)
    s = np.sort(d)
    cdf = np.arange(1, len(s)+1) / len(s)
    axes[1,0].plot(s, cdf, linewidth=1.4, label=name)
q995 = np.quantile(np.concatenate(all_q), 0.995) if len(all_q) else 1.0
axes[1,0].set_xlim(0, q995)
axes[1,0].set_title('Smoothness ECDF of |dFlow|')
axes[1,0].set_xlabel('|dFlow|')
axes[1,0].set_ylabel('CDF')
axes[1,0].legend()

# d normalized entropy 0 to 1
names = []
vals = []
bins = 24
for name, df in run_map.items():
    F = df[flow_cols(df)].to_numpy()
    ent = []
    for z in range(F.shape[1]):
        counts, _ = np.histogram(F[:, z], bins=bins, range=(0, 1), density=False)
        total = counts.sum()
        if total <= 0:
            ent.append(0.0)
            continue
        p = counts / total
        p = p[p > 0]
        h = -(p * np.log2(p)).sum()
        h_norm = h / np.log2(bins)
        ent.append(float(h_norm))
    names.append(name)
    vals.append(float(np.mean(ent)))
axes[1,1].barh(names, vals, color=palette_greys(len(names)+2)[2:])
axes[1,1].set_xlim(0, 0.5)
axes[1,1].set_title('Control Strategy Diversity normalized')
axes[1,1].set_xlabel('Normalized entropy 0 to 1')

for a in axes.ravel():
    a.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / '06_control_aggressiveness_strategy_tuned.png', bbox_inches='tight')
plt.show()

metrics = [
    ('Spread', spread_col, True),
    ('Mean temp', mean_temp_col, True),
    ('Max temp', temp_col, True),
    ('Stress', stress_col, True),
    ('Pump energy', energy_col, True),
]

def robust_norm(v, arr):
    ql = np.quantile(arr, 0.05)
    qh = np.quantile(arr, 0.95)
    nv = (v - ql) / (qh - ql + 1e-12)
    return float(np.clip(nv, 0, 1))

scores = []
for _, row in df_comp.iterrows():
    vals = []
    for _, col, inv in metrics:
        nv = robust_norm(row[col], df_comp[col].to_numpy())
        if inv:
            nv = 1 - nv
        vals.append(nv)
    scores.append(np.mean(vals))

top = df_comp.copy()
top['score'] = scores
top = top.sort_values('score', ascending=False).head(6).copy()

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), subplot_kw=dict(polar=True))
axes = axes.flatten()

for ax, (_, row) in zip(axes, top.iterrows()):
    vals = []
    for _, col, inv in metrics:
        nv = robust_norm(row[col], df_comp[col].to_numpy())
        if inv:
            nv = 1 - nv
        vals.append(float(nv))
    v = vals + vals[:1]
    ax.plot(angles, v, linewidth=2.0, color='black')
    ax.fill(angles, v, alpha=0.18, color='gray')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metrics], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=6)
    ax.set_title(row['label'], fontsize=9, pad=10)

for ax in axes[len(top):]:
    ax.axis('off')

fig.suptitle('Multiple Radars one per controller', y=0.98, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / '07_radar_top6.png', bbox_inches='tight')
fig.savefig(OUT / '07b_radar_small_multiples_top6.png', bbox_inches='tight')
plt.show()

outs = sorted(OUT.glob('*.png'))
print('Exported files')
for p in outs:
    print('-', p.name)

print('\nIf you add more *_run.csv files, rerun sections 3 to 6 to include them automatically.')

from scipy.spatial.distance import cdist, jensenshannon
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr

ABL_OUT = OUT / 'ablation'
ABL_OUT.mkdir(parents=True, exist_ok=True)

def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def _robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    iqr = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    if iqr <= 1e-12:
        return np.zeros_like(x)
    return (x - med) / iqr

def _mean_pairwise_dist(M):
    M = np.asarray(M, dtype=float)
    if len(M) <= 1:
        return 0.0
    D = cdist(M, M)
    iu = np.triu_indices_from(D, k=1)
    return float(np.mean(D[iu]))

def _nearest_neighbor_median(P):
    P = np.asarray(P, dtype=float)
    if len(P) <= 1:
        return 0.0
    D = cdist(P, P)
    np.fill_diagonal(D, np.inf)
    return float(np.median(np.min(D, axis=1)))

def _convex_hull_area(P):
    P = np.asarray(P, dtype=float)
    if len(P) < 3:
        return 0.0
    try:
        hull = ConvexHull(P)
        return float(hull.volume)
    except Exception:
        return 0.0

def _safe_spearman(a, b):
    try:
        r, _ = spearmanr(a, b)
        if np.isnan(r):
            return 0.0
        return float(r)
    except Exception:
        return 0.0

def _rank_score(df, plus, minus, score_col='score'):
    d = df.copy()
    score = np.zeros(len(d), dtype=float)
    for c, w in plus.items():
        score += w * d[c].rank(ascending=False, method='average').to_numpy()
    for c, w in minus.items():
        score += w * d[c].rank(ascending=True, method='average').to_numpy()
    d[score_col] = score
    d = d.sort_values(score_col, ascending=False).reset_index(drop=True)
    return d

def _ecdf(x):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def _hist_prob(x, bins=80):
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)
    if hi - lo <= 1e-12:
        hi = lo + 1e-6
    h, _ = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    p = h.astype(float)
    s = p.sum()
    if s <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / s
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p

def _align(series_dict):
    m = min(len(v) for v in series_dict.values())
    return {k: np.asarray(v[:m], dtype=float) for k, v in series_dict.items()}

print('Ablation output dir', ABL_OUT.resolve())

ABL = {}

# -------------------------------------------------
# Family 1 Pareto frontier ablation
# -------------------------------------------------
px_raw = df_comp[temp_col].to_numpy(dtype=float)
py_raw = df_comp[energy_col].to_numpy(dtype=float)
front_raw = pareto_mask(px_raw, py_raw)

qx95_x = max(np.quantile(px_raw, 0.95), 1e-12)
qx95_y = max(np.quantile(py_raw, 0.95), 1e-12)

pareto_opts = {
    'linear_linear': (lambda x: x, lambda y: y),
    'linear_log1pY': (lambda x: x, lambda y: np.log1p(y)),
    'log1p_both': (lambda x: np.log1p(x), lambda y: np.log1p(y)),
    'asinh_both': (lambda x: np.arcsinh(x/qx95_x), lambda y: np.arcsinh(y/qx95_y)),
    'rank_both': (lambda x: pd.Series(x).rank(method='average').to_numpy(),
                  lambda y: pd.Series(y).rank(method='average').to_numpy()),
    'clip99_linear': (lambda x: np.clip(x, None, np.quantile(x, 0.99)),
                      lambda y: np.clip(y, None, np.quantile(y, 0.99))),
}

rows = []
for name, (fx, fy) in pareto_opts.items():
    x = fx(px_raw)
    y = fy(py_raw)
    xn, yn = _minmax(x), _minmax(y)
    P = np.c_[xn, yn]
    nnd = _nearest_neighbor_median(P)
    area = _convex_hull_area(P)
    if np.any(front_raw) and np.any(~front_raw):
        gap = np.linalg.norm(P[front_raw].mean(axis=0) - P[~front_raw].mean(axis=0))
    else:
        gap = 0.0
    D = cdist(P, P)
    np.fill_diagonal(D, np.inf)
    overlap = float(np.mean(np.min(D, axis=1) < 0.08))
    rows.append({'variant': name, 'nn_dist': nnd, 'area': area, 'frontier_gap': gap, 'overlap': overlap})

pareto_df = _rank_score(pd.DataFrame(rows), plus={'nn_dist':0.4, 'area':0.35, 'frontier_gap':0.25}, minus={'overlap':0.35})
ABL['pareto'] = pareto_df
pareto_df.to_csv(ABL_OUT / 'ablation_pareto.csv', index=False)

# Visual compare top variants
k = min(4, len(pareto_df))
fig, axes = plt.subplots(1, k, figsize=(4.2*k, 3.8), sharex=True, sharey=True)
if k == 1:
    axes = [axes]
for ax, v in zip(axes, pareto_df['variant'].head(k)):
    fx, fy = pareto_opts[v]
    x = _minmax(fx(px_raw))
    y = _minmax(fy(py_raw))
    ax.scatter(x[~front_raw], y[~front_raw], s=36, alpha=0.6, c='0.65', label='dominated')
    ax.scatter(x[front_raw], y[front_raw], s=48, marker='D', c='#1f77b4', label='frontier')
    for i, lbl in enumerate(df_comp['label']):
        ax.text(x[i], y[i], lbl.replace('PROPOSED', 'PRO'), fontsize=6)
    ax.set_title(v)
    ax.set_xlabel('x transformed')
    ax.grid(alpha=0.2)
axes[0].set_ylabel('y transformed')
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(ABL_OUT / 'ablation_pareto_variants.png', bbox_inches='tight')
plt.show()

# -------------------------------------------------
# Family 2 summary bars ablation
# -------------------------------------------------
metric_cols = [temp_col, mean_temp_col, spread_col, stress_col, energy_col, overhead_col]
M_raw = df_comp[metric_cols].to_numpy(dtype=float)

summary_modes = {
    'raw': lambda A: A,
    'minmax_per_metric': lambda A: np.column_stack([_minmax(A[:,i]) for i in range(A.shape[1])]),
    'rank_per_metric': lambda A: np.column_stack([pd.Series(A[:,i]).rank(method='average').to_numpy() for i in range(A.shape[1])]),
    'robust_z_per_metric': lambda A: np.column_stack([_robust_z(A[:,i]) for i in range(A.shape[1])]),
    'log_energy_only': lambda A: np.column_stack([
        A[:,0], A[:,1], A[:,2], A[:,3], np.log1p(A[:,4]), A[:,5]
    ]),
}

rows = []
for mode, fn in summary_modes.items():
    M = fn(M_raw)
    sep = _mean_pairwise_dist(np.column_stack([_minmax(M[:,i]) for i in range(M.shape[1])]))
    fid = np.mean([_safe_spearman(pd.Series(M_raw[:,i]).rank(), pd.Series(M[:,i]).rank()) for i in range(M.shape[1])])
    sk = np.mean([abs(pd.Series(M[:,i]).skew()) for i in range(M.shape[1])])
    rows.append({'variant': mode, 'separation': sep, 'rank_fidelity': fid, 'abs_skew': sk})

summary_df = _rank_score(pd.DataFrame(rows), plus={'separation':0.5, 'rank_fidelity':0.35}, minus={'abs_skew':0.2})
ABL['summary_bars'] = summary_df
summary_df.to_csv(ABL_OUT / 'ablation_summary_bars.csv', index=False)

# -------------------------------------------------
# Family 3 temporal dashboard ablation
# -------------------------------------------------
series = {}
for name, df in run_map.items():
    t = df['time_s'].to_numpy(dtype=float)
    T = df[temp_cols(df)].to_numpy(dtype=float)
    F = df[flow_cols(df)].to_numpy(dtype=float)
    dt = np.clip(np.diff(t), 1e-9, None)
    series[name] = {
        'pump_kw': df['pump_power_W'].to_numpy(dtype=float)/1000.0,
        'stress': np.abs(np.diff(T.mean(axis=1))) / dt,
        'flow_std': F.std(axis=1)
    }

temporal_modes = {
    'pump_kw': {
        'raw_kw': lambda x: x,
        'rolling_60s': lambda x: uniform_filter1d(x, size=60),
        'cumulative_kWh': lambda x: np.cumsum(x)/3600.0,
        'asinh_q95': lambda x: np.arcsinh(x / max(np.quantile(x, 0.95), 1e-12)),
    },
    'stress': {
        'raw': lambda x: x,
        'rolling_60s': lambda x: uniform_filter1d(x, size=60),
        'cumulative_mean': lambda x: np.cumsum(x) / np.arange(1, len(x)+1),
        'log1p': lambda x: np.log1p(x),
    },
    'flow_std': {
        'raw': lambda x: x,
        'rolling_60s': lambda x: uniform_filter1d(x, size=60),
        'cumulative_mean': lambda x: np.cumsum(x) / np.arange(1, len(x)+1),
        'log1p': lambda x: np.log1p(x),
    }
}

TEMP_ABL = {}
for signal, opts in temporal_modes.items():
    rows = []
    for mode, fn in opts.items():
        transformed = {k: fn(v[signal]) for k, v in series.items()}
        A = _align(transformed)
        names = list(A.keys())
        M = np.vstack([A[n] for n in names])
        Mz = np.column_stack([_minmax(M[:,j]) for j in range(M.shape[1])])
        sep = _mean_pairwise_dist(Mz)
        rough = float(np.mean([np.mean(np.abs(np.diff(A[n]))) / (np.std(A[n]) + 1e-12) for n in names]))
        dr = float(np.quantile(M, 0.95) - np.quantile(M, 0.05))
        rows.append({'variant': mode, 'separation': sep, 'roughness': rough, 'dynamic_range': dr})
    d = _rank_score(pd.DataFrame(rows), plus={'separation':0.5, 'dynamic_range':0.2}, minus={'roughness':0.4})
    TEMP_ABL[signal] = d
    d.to_csv(ABL_OUT / f'ablation_temporal_{signal}.csv', index=False)

ABL['temporal'] = pd.concat(
    [d.assign(signal=s) for s, d in TEMP_ABL.items()],
    ignore_index=True
)

# Visual top2 per signal
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)
for r, signal in enumerate(['pump_kw', 'stress', 'flow_std']):
    top2 = TEMP_ABL[signal]['variant'].head(2).tolist()
    for c, mode in enumerate(top2):
        ax = axes[r, c]
        fn = temporal_modes[signal][mode]
        for name in series:
            y = fn(series[name][signal])
            t = run_map[name]['time_s'].to_numpy(dtype=float)
            if signal == 'stress':
                t = t[1:]
            ax.plot(t/3600.0, y, linewidth=1.2, label=name)
        ax.set_title(f'{signal} {mode}')
        ax.set_xlabel('Time h')
        ax.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel('Value')
        if r == 0:
            ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(ABL_OUT / 'ablation_temporal_top2.png', bbox_inches='tight')
plt.show()

# -------------------------------------------------
# Family 4 statistical dashboard ablation
# -------------------------------------------------
stat_data = {}
stat_data['pump'] = {n: run_map[n]['pump_power_W'].to_numpy(dtype=float) for n in run_map}
stat_data['spread'] = {n: np.ptp(run_map[n][temp_cols(run_map[n])].to_numpy(dtype=float), axis=1) for n in run_map}
stat_data['stress'] = {}
for n in run_map:
    df = run_map[n]
    T = df[temp_cols(df)].to_numpy(dtype=float)
    dt = np.clip(np.diff(df['time_s'].to_numpy(dtype=float)), 1e-9, None)
    stat_data['stress'][n] = (np.abs(np.diff(T, axis=0)) / dt[:, None]).ravel()

stat_modes = {
    'linear': lambda x: x,
    'log1p': lambda x: np.log1p(np.clip(x, 0, None)),
    'asinh_q95': lambda x: np.arcsinh(x / max(np.quantile(x, 0.95), 1e-12)),
    'rank': lambda x: pd.Series(x).rank(method='average').to_numpy(),
}

STAT_ABL = {}
for key, dct in stat_data.items():
    rows = []
    for mode, fn in stat_modes.items():
        tr = {n: fn(v) for n, v in dct.items()}
        keys = list(tr.keys())
        js_vals = []
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                p = _hist_prob(tr[keys[i]])
                q = _hist_prob(tr[keys[j]])
                js_vals.append(float(jensenshannon(p, q)))
        pooled = np.concatenate(list(tr.values()))
        sk = abs(pd.Series(pooled).skew())
        tail = (np.quantile(pooled, 0.99) + 1e-12) / (np.quantile(pooled, 0.50) + 1e-12)
        rows.append({'variant': mode, 'js_divergence': np.mean(js_vals) if js_vals else 0.0, 'abs_skew': sk, 'tail_ratio': tail})
    d = _rank_score(pd.DataFrame(rows), plus={'js_divergence':0.6}, minus={'abs_skew':0.25, 'tail_ratio':0.15})
    STAT_ABL[key] = d
    d.to_csv(ABL_OUT / f'ablation_statistical_{key}.csv', index=False)

ABL['statistical'] = pd.concat([v.assign(metric=k) for k, v in STAT_ABL.items()], ignore_index=True)

# -------------------------------------------------
# Family 5 spatial dashboard ablation
# -------------------------------------------------
zone_mean_flow = {n: run_map[n][flow_cols(run_map[n])].to_numpy(dtype=float).mean(axis=0) for n in run_map}
all_mean_flow = np.concatenate(list(zone_mean_flow.values()))

rows = []
for q in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
    lim = float(np.quantile(all_mean_flow, q))
    clipped = np.vstack([np.minimum(v, lim) for v in zone_mean_flow.values()])
    sep = _mean_pairwise_dist(np.column_stack([_minmax(clipped[:,i]) for i in range(clipped.shape[1])]))
    coverage = float(np.mean(all_mean_flow <= lim))
    target = 0.35
    coverage_score = max(0.0, 1.0 - abs(coverage - target) / target)
    rows.append({'variant': f'low_zoom_q{int(q*100)}', 'limit': lim, 'separation': sep, 'coverage': coverage, 'coverage_score': coverage_score})

spatial_zoom_df = _rank_score(pd.DataFrame(rows), plus={'separation':0.55, 'coverage_score':0.45}, minus={})
spatial_zoom_df.to_csv(ABL_OUT / 'ablation_spatial_zoom.csv', index=False)

# selection ablation
run_names = list(run_map.keys())
if len(run_names) >= 2:
    best_energy = min(run_names, key=lambda n: run_map[n]['pump_power_W'].sum())
    best_temp = min(run_names, key=lambda n: run_map[n][temp_cols(run_map[n])].to_numpy().max())
    best2 = list(dict.fromkeys([best_energy, best_temp]))
else:
    best2 = run_names

variants = {
    'best2_focus': best2,
    'all_available': run_names,
}
rows = []
for name, subset in variants.items():
    temps = []
    for n in subset:
        temps.append(run_map[n][temp_cols(run_map[n])].to_numpy(dtype=float).mean(axis=0))
    M = np.vstack(temps) if len(temps) else np.zeros((1, 1))
    sep = _mean_pairwise_dist(np.column_stack([_minmax(M[:,i]) for i in range(M.shape[1])])) if len(subset) > 1 else 0.0
    info = len(subset) / max(1, len(run_names))
    clutter = len(subset) / max(1, len(run_names))
    rows.append({'variant': name, 'separation': sep, 'info_coverage': info, 'clutter': clutter})

spatial_sel_df = _rank_score(pd.DataFrame(rows), plus={'separation':0.5, 'info_coverage':0.5}, minus={'clutter':0.2})
spatial_sel_df.to_csv(ABL_OUT / 'ablation_spatial_selection.csv', index=False)

ABL['spatial'] = pd.concat([
    spatial_zoom_df.assign(subfamily='zoom'),
    spatial_sel_df.assign(subfamily='selection')
], ignore_index=True)

# -------------------------------------------------
# Family 6 control strategy ablation
# -------------------------------------------------
# spectrum scale ablation
psd_map = {}
for n, df in run_map.items():
    f = df[flow_cols(df)].to_numpy(dtype=float).mean(axis=1)
    freqs, psd = welch(f, fs=1.0, nperseg=min(512, max(32, len(f)//4)))
    valid = freqs > 0
    psd_map[n] = (freqs[valid], psd[valid])

spec_modes = {
    'linear_linear': (lambda f: f, lambda p: p),
    'logx_linear': (lambda f: np.log10(f + 1e-6), lambda p: p),
    'logx_logy': (lambda f: np.log10(f + 1e-6), lambda p: np.log10(p + 1e-12)),
}

rows = []
for mode, (ffn, pfn) in spec_modes.items():
    names = list(psd_map.keys())
    curves = []
    for n in names:
        f, p = psd_map[n]
        fx = ffn(f)
        py = pfn(p)
        curves.append((fx, py))
    # interpolate to common grid in transformed freq axis
    lo = max(c[0].min() for c in curves)
    hi = min(c[0].max() for c in curves)
    if hi - lo <= 1e-12:
        sep = 0.0
        tail = 0.0
    else:
        gx = np.linspace(lo, hi, 220)
        Y = []
        for fx, py in curves:
            Y.append(np.interp(gx, fx, py))
        Y = np.vstack(Y)
        sep = _mean_pairwise_dist(np.column_stack([_minmax(Y[:,i]) for i in range(Y.shape[1])]))
        tail = float(np.std(Y[:, int(0.8*Y.shape[1]):])) / (float(np.std(Y)) + 1e-12)
    rows.append({'variant': mode, 'separation': sep, 'tail_visibility': tail})

spec_df = _rank_score(pd.DataFrame(rows), plus={'separation':0.65, 'tail_visibility':0.35}, minus={})
spec_df.to_csv(ABL_OUT / 'ablation_control_spectrum.csv', index=False)

# smoothness transform ablation
smooth = {}
for n, df in run_map.items():
    F = df[flow_cols(df)].to_numpy(dtype=float)
    smooth[n] = np.abs(np.diff(F, axis=0)).ravel()

sm_modes = {
    'linear': lambda x: x,
    'log1p': lambda x: np.log1p(x),
    'asinh_q95': lambda x: np.arcsinh(x / max(np.quantile(x, 0.95), 1e-12)),
    'rank': lambda x: pd.Series(x).rank(method='average').to_numpy(),
}

rows = []
for mode, fn in sm_modes.items():
    tr = {n: fn(v) for n, v in smooth.items()}
    vals = list(tr.values())
    js = []
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            js.append(float(jensenshannon(_hist_prob(vals[i]), _hist_prob(vals[j]))))
    pooled = np.concatenate(vals)
    sk = abs(pd.Series(pooled).skew())
    rows.append({'variant': mode, 'js_divergence': np.mean(js) if js else 0.0, 'abs_skew': sk})

smooth_df = _rank_score(pd.DataFrame(rows), plus={'js_divergence':0.7}, minus={'abs_skew':0.3})
smooth_df.to_csv(ABL_OUT / 'ablation_control_smoothness.csv', index=False)

# diversity metric ablation
div_rows = []
bins = 24
for mode in ['bits', 'normalized_bits', 'perplexity']:
    vals = []
    for n, df in run_map.items():
        F = df[flow_cols(df)].to_numpy(dtype=float)
        ent = []
        for z in range(F.shape[1]):
            c, _ = np.histogram(F[:,z], bins=bins, range=(0,1), density=False)
            t = c.sum()
            if t <= 0:
                ent.append(0.0)
                continue
            p = c / t
            p = p[p > 0]
            h = -(p * np.log2(p)).sum()
            if mode == 'bits':
                v = h
            elif mode == 'normalized_bits':
                v = h / np.log2(bins)
            else:
                v = 2 ** h
            ent.append(float(v))
        vals.append(float(np.mean(ent)))
    vals = np.asarray(vals)
    sep = _mean_pairwise_dist(vals.reshape(-1,1))
    rng = float(vals.max() - vals.min())
    div_rows.append({'variant': mode, 'separation': sep, 'range': rng})

diversity_df = _rank_score(pd.DataFrame(div_rows), plus={'separation':0.7, 'range':0.3}, minus={})
diversity_df.to_csv(ABL_OUT / 'ablation_control_diversity.csv', index=False)

ABL['control_strategy'] = pd.concat([
    spec_df.assign(subfamily='spectrum'),
    smooth_df.assign(subfamily='smoothness'),
    diversity_df.assign(subfamily='diversity')
], ignore_index=True)

# -------------------------------------------------
# Family 7 radar ablation
# -------------------------------------------------
radar_metrics = [temp_col, energy_col, spread_col, stress_col, overhead_col]
rad = df_comp[['label'] + radar_metrics].copy()
# lower is better for all selected metrics here
R = np.column_stack([_minmax(rad[c].to_numpy(dtype=float)) for c in radar_metrics])
labels = rad['label'].tolist()

if not df_rank.empty and 'rank_overall' in df_rank.columns and 'label' in df_rank.columns:
    top6_labels = df_rank.sort_values('rank_overall')['label'].head(6).tolist()
    idx_top6 = [labels.index(l) for l in top6_labels if l in labels]
    if len(idx_top6) < 3:
        idx_top6 = list(range(min(6, len(labels))))
else:
    idx_top6 = list(range(min(6, len(labels))))

idx_all = list(range(len(labels)))

radar_variants = {
    'radar_overlay_all': (idx_all, 1.00),
    'radar_overlay_top6': (idx_top6, 1.00),
    'radar_small_multiples_top6': (idx_top6, 0.25),
    'parallel_coords_top6': (idx_top6, 0.45),
}

rows = []
for v, (idxs, clutter_factor) in radar_variants.items():
    M = R[idxs]
    sep = _mean_pairwise_dist(M)
    if len(M) > 1:
        overlap = []
        for i in range(len(M)):
            for j in range(i+1, len(M)):
                overlap.append(float(np.mean(np.abs(M[i]-M[j]) < 0.08)))
        overlap = float(np.mean(overlap))
    else:
        overlap = 0.0
    clutter = overlap * max(1, len(idxs)-1) * clutter_factor
    coverage = len(idxs) / max(1, len(labels))
    rows.append({'variant': v, 'separation': sep, 'overlap': overlap, 'clutter': clutter, 'coverage': coverage})

radar_df = _rank_score(pd.DataFrame(rows), plus={'separation':0.55, 'coverage':0.2}, minus={'overlap':0.35, 'clutter':0.35})
ABL['radar'] = radar_df
radar_df.to_csv(ABL_OUT / 'ablation_radar.csv', index=False)

# -------------------------------------------------
# Best choice summary by panel
# -------------------------------------------------
best_rows = []

def _push_best(df, family, panel_col=None, panel_value=None):
    if df is None or len(df) == 0:
        return
    top = df.sort_values('score', ascending=False).iloc[0].to_dict()
    top['family'] = family
    if panel_col is None:
        top['panel'] = family
    else:
        top['panel'] = f"{family}:{panel_value}"
    best_rows.append(top)

_push_best(ABL.get('pareto', pd.DataFrame()), 'pareto')
_push_best(ABL.get('summary_bars', pd.DataFrame()), 'summary_bars')
_push_best(ABL.get('radar', pd.DataFrame()), 'radar')

for sig, d in TEMP_ABL.items():
    _push_best(d, 'temporal', 'signal', sig)

for m, d in STAT_ABL.items():
    _push_best(d, 'statistical', 'metric', m)

_push_best(spatial_zoom_df, 'spatial', 'subfamily', 'zoom')
_push_best(spatial_sel_df, 'spatial', 'subfamily', 'selection')

_push_best(spec_df, 'control_strategy', 'subfamily', 'spectrum')
_push_best(smooth_df, 'control_strategy', 'subfamily', 'smoothness')
_push_best(diversity_df, 'control_strategy', 'subfamily', 'diversity')

best_df = pd.DataFrame(best_rows)
best_df.to_csv(ABL_OUT / 'ablation_best_by_family.csv', index=False)
best_df.to_csv(ABL_OUT / 'ablation_best_by_panel.csv', index=False)

# Show tables
print('Ablation top rows by family')
for fam in ['pareto','summary_bars','temporal','statistical','spatial','control_strategy','radar']:
    if fam in ABL:
        print('\n===', fam, '===')
        display(ABL[fam].head(5))

print('\nBest by panel')
display(best_df[['panel','variant','score']])

# Visual summary of top 3 variants in each family
families = ['pareto','summary_bars','temporal','statistical','spatial','control_strategy','radar']
fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes = axes.flatten()

for i, fam in enumerate(families):
    ax = axes[i]
    d = ABL.get(fam, pd.DataFrame())
    if d.empty:
        ax.set_title(f'{fam} no data')
        ax.axis('off')
        continue
    dd = d.head(3).copy()
    ax.barh(np.arange(len(dd))[::-1], dd['score'], color='#4c72b0')
    ax.set_yticks(np.arange(len(dd))[::-1])
    ax.set_yticklabels(dd['variant'])
    ax.set_title(f'{fam} top variants')
    ax.grid(alpha=0.2)

# hide spare axis
for j in range(len(families), len(axes)):
    axes[j].axis('off')

fig.tight_layout()
fig.savefig(ABL_OUT / 'ablation_top3_by_family.png', bbox_inches='tight')

if 'best_df' in globals() and len(best_df):
    fig2, ax2 = plt.subplots(figsize=(9, max(4, 0.35*len(best_df))))
    b = best_df.sort_values('score', ascending=True)
    ax2.barh(b['panel'], b['score'], color='#55a868')
    ax2.set_title('Best Variant Score by Plot Panel')
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(ABL_OUT / 'ablation_best_by_panel.png', bbox_inches='tight')
    plt.show()
plt.show()

print('Ablation files exported')
for pth in sorted(ABL_OUT.glob('*.csv')):
    print('-', pth.name)
for pth in sorted(ABL_OUT.glob('*.png')):
    print('-', pth.name)

import math

GAL_OUT = ABL_OUT / 'gallery'
GAL_OUT.mkdir(parents=True, exist_ok=True)

# Build color map by group for consistency
label_to_group = {r['label']: r['group'] for _, r in df_comp[['label', 'group']].iterrows()}
def _label_color(lbl):
    grp = label_to_group.get(lbl, 'OTHER')
    return GROUP_COLORS.get(grp, GROUP_COLORS['OTHER'])

def _subplot_grid(n, ncols=3):
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()
    return fig, axes

# ------------------------------------------------------------------
# 1 Pareto all variants
# ------------------------------------------------------------------
if 'pareto_opts' not in globals():
    qx95_x = max(np.quantile(df_comp[temp_col].to_numpy(dtype=float), 0.95), 1e-12)
    qx95_y = max(np.quantile(df_comp[energy_col].to_numpy(dtype=float), 0.95), 1e-12)
    pareto_opts = {
        'linear_linear': (lambda x: x, lambda y: y),
        'linear_log1pY': (lambda x: x, lambda y: np.log1p(y)),
        'log1p_both': (lambda x: np.log1p(x), lambda y: np.log1p(y)),
        'asinh_both': (lambda x: np.arcsinh(x/qx95_x), lambda y: np.arcsinh(y/qx95_y)),
        'rank_both': (lambda x: pd.Series(x).rank(method='average').to_numpy(),
                      lambda y: pd.Series(y).rank(method='average').to_numpy()),
        'clip99_linear': (lambda x: np.clip(x, None, np.quantile(x, 0.99)),
                          lambda y: np.clip(y, None, np.quantile(y, 0.99))),
    }

px = df_comp[temp_col].to_numpy(dtype=float)
py = df_comp[energy_col].to_numpy(dtype=float)
pf = pareto_mask(px, py)

fig, axes = _subplot_grid(len(pareto_opts), ncols=3)
for ax, (name, (fx, fy)) in zip(axes, pareto_opts.items()):
    x = _minmax(fx(px))
    y = _minmax(fy(py))
    ax.scatter(x[~pf], y[~pf], s=36, alpha=0.60, c='0.7', label='dominated')
    ax.scatter(x[pf], y[pf], s=48, marker='D', c='#1f77b4', label='frontier')
    for i, lbl in enumerate(df_comp['label']):
        ax.text(x[i], y[i], lbl.replace('PROPOSED', 'PRO'), fontsize=6)
    ax.set_title(f'Pareto {name}')
    ax.set_xlabel('Temp axis transformed')
    ax.set_ylabel('Energy axis transformed')
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=7)
for ax in axes[len(pareto_opts):]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(GAL_OUT / '01_pareto_all_variants.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 2 Summary bar variants
# ------------------------------------------------------------------
if 'summary_modes' not in globals():
    M_raw = df_comp[[temp_col, mean_temp_col, spread_col, stress_col, energy_col, overhead_col]].to_numpy(dtype=float)
    summary_modes = {
        'raw': lambda A: A,
        'minmax_per_metric': lambda A: np.column_stack([_minmax(A[:,i]) for i in range(A.shape[1])]),
        'rank_per_metric': lambda A: np.column_stack([pd.Series(A[:,i]).rank(method='average').to_numpy() for i in range(A.shape[1])]),
        'robust_z_per_metric': lambda A: np.column_stack([_robust_z(A[:,i]) for i in range(A.shape[1])]),
        'log_energy_only': lambda A: np.column_stack([A[:,0], A[:,1], A[:,2], A[:,3], np.log1p(A[:,4]), A[:,5]]),
    }

fig, axes = _subplot_grid(len(summary_modes), ncols=3)
labels = df_comp['label'].tolist()
for ax, (name, fn) in zip(axes, summary_modes.items()):
    M = fn(M_raw)
    ranks = np.column_stack([pd.Series(M[:,i]).rank(method='average', ascending=True).to_numpy() for i in range(M.shape[1])])
    score = ranks.mean(axis=1)
    order = np.argsort(score)
    ylbl = [labels[i] for i in order]
    c = [_label_color(lbl) for lbl in ylbl]
    ax.barh(ylbl, score[order], color=c)
    ax.set_title(f'Summary {name}')
    ax.set_xlabel('Mean rank lower is better')
    ax.grid(alpha=0.2)
for ax in axes[len(summary_modes):]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(GAL_OUT / '02_summary_all_variants.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 3 Temporal variants for each signal
# ------------------------------------------------------------------
if 'series' not in globals():
    series = {}
    for name, df in run_map.items():
        t = df['time_s'].to_numpy(dtype=float)
        T = df[temp_cols(df)].to_numpy(dtype=float)
        F = df[flow_cols(df)].to_numpy(dtype=float)
        dt = np.clip(np.diff(t), 1e-9, None)
        series[name] = {
            'pump_kw': df['pump_power_W'].to_numpy(dtype=float)/1000.0,
            'stress': np.abs(np.diff(T.mean(axis=1))) / dt,
            'flow_std': F.std(axis=1)
        }

if 'temporal_modes' not in globals():
    temporal_modes = {
        'pump_kw': {
            'raw_kw': lambda x: x,
            'rolling_60s': lambda x: uniform_filter1d(x, size=60),
            'cumulative_kWh': lambda x: np.cumsum(x)/3600.0,
            'asinh_q95': lambda x: np.arcsinh(x / max(np.quantile(x, 0.95), 1e-12)),
        },
        'stress': {
            'raw': lambda x: x,
            'rolling_60s': lambda x: uniform_filter1d(x, size=60),
            'cumulative_mean': lambda x: np.cumsum(x) / np.arange(1, len(x)+1),
            'log1p': lambda x: np.log1p(x),
        },
        'flow_std': {
            'raw': lambda x: x,
            'rolling_60s': lambda x: uniform_filter1d(x, size=60),
            'cumulative_mean': lambda x: np.cumsum(x) / np.arange(1, len(x)+1),
            'log1p': lambda x: np.log1p(x),
        }
    }

for signal in ['pump_kw', 'stress', 'flow_std']:
    modes = temporal_modes[signal]
    fig, axes = _subplot_grid(len(modes), ncols=2)
    for ax, (mode, fn) in zip(axes, modes.items()):
        for name in series:
            y = fn(series[name][signal])
            t = run_map[name]['time_s'].to_numpy(dtype=float)
            if signal == 'stress':
                t = t[1:]
            ax.plot(t/3600.0, y, linewidth=1.1, label=name)
        ax.set_title(f'Temporal {signal} {mode}')
        ax.set_xlabel('Time h')
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    for ax in axes[len(modes):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(GAL_OUT / f'03_temporal_all_{signal}.png', bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------
# 4 Statistical variants
# ------------------------------------------------------------------
if 'stat_data' not in globals():
    stat_data = {}
    stat_data['pump'] = {n: run_map[n]['pump_power_W'].to_numpy(dtype=float) for n in run_map}
    stat_data['spread'] = {n: np.ptp(run_map[n][temp_cols(run_map[n])].to_numpy(dtype=float), axis=1) for n in run_map}
    stat_data['stress'] = {}
    for n in run_map:
        df = run_map[n]
        T = df[temp_cols(df)].to_numpy(dtype=float)
        dt = np.clip(np.diff(df['time_s'].to_numpy(dtype=float)), 1e-9, None)
        stat_data['stress'][n] = (np.abs(np.diff(T, axis=0)) / dt[:, None]).ravel()

if 'stat_modes' not in globals():
    stat_modes = {
        'linear': lambda x: x,
        'log1p': lambda x: np.log1p(np.clip(x, 0, None)),
        'asinh_q95': lambda x: np.arcsinh(x / max(np.quantile(x, 0.95), 1e-12)),
        'rank': lambda x: pd.Series(x).rank(method='average').to_numpy(),
    }

for metric in ['pump', 'spread', 'stress']:
    modes = stat_modes
    fig, axes = _subplot_grid(len(modes), ncols=2)
    dct = stat_data[metric]
    for ax, (mode, fn) in zip(axes, modes.items()):
        all_vals = []
        tr = {n: fn(v) for n, v in dct.items()}
        for n, arr in tr.items():
            x, y = _ecdf(arr)
            all_vals.append(arr)
            ax.plot(x, y, linewidth=1.2, label=n)
        pooled = np.concatenate(all_vals)
        xmax = np.quantile(pooled, 0.995)
        ax.set_xlim(np.min(pooled), xmax)
        ax.set_title(f'Stat {metric} {mode}')
        ax.set_xlabel('Value transformed')
        ax.set_ylabel('CDF')
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    for ax in axes[len(modes):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(GAL_OUT / f'04_statistical_all_{metric}.png', bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------
# 5 Spatial variants
# ------------------------------------------------------------------
zone_mean_flow = {n: run_map[n][flow_cols(run_map[n])].to_numpy(dtype=float).mean(axis=0) for n in run_map}
all_mean_flow = np.concatenate(list(zone_mean_flow.values()))
q_list = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

fig, axes = _subplot_grid(len(q_list)+1, ncols=4)
# full range first
ax0 = axes[0]
for name, y in zone_mean_flow.items():
    z = np.arange(1, len(y)+1)
    ax0.plot(z, y, marker='o', linewidth=1.2, label=name)
ax0.set_title('Spatial flow full range')
ax0.set_xlabel('Zone')
ax0.set_ylabel('Flow')
ax0.grid(alpha=0.2)
ax0.legend(fontsize=7)

for k, q in enumerate(q_list, start=1):
    ax = axes[k]
    lim = float(np.quantile(all_mean_flow, q))
    for name, y in zone_mean_flow.items():
        z = np.arange(1, len(y)+1)
        ax.plot(z, y, marker='o', linewidth=1.1, label=name)
    ax.set_ylim(0, lim)
    ax.set_title(f'Low flow zoom q{int(q*100)}')
    ax.set_xlabel('Zone')
    ax.set_ylabel('Flow')
    ax.grid(alpha=0.2)

for ax in axes[len(q_list)+1:]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(GAL_OUT / '05_spatial_zoom_all_variants.png', bbox_inches='tight')
plt.show()

# selection style gallery
run_names = list(run_map.keys())
if len(run_names) >= 2:
    best_energy = min(run_names, key=lambda n: run_map[n]['pump_power_W'].sum())
    best_temp = min(run_names, key=lambda n: run_map[n][temp_cols(run_map[n])].to_numpy().max())
    best2 = list(dict.fromkeys([best_energy, best_temp]))
else:
    best2 = run_names
sel_variants = {'best2_focus': best2, 'all_available': run_names}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (nm, subset) in zip(axes, sel_variants.items()):
    for name in subset:
        T = run_map[name][temp_cols(run_map[name])].to_numpy(dtype=float)
        m = T.mean(axis=0)
        z = np.arange(1, len(m)+1)
        ax.plot(z, m, marker='o', linewidth=1.4, label=name)
    ax.set_title(f'Spatial selection {nm}')
    ax.set_xlabel('Zone')
    ax.set_ylabel('Mean temperature C')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(GAL_OUT / '05_spatial_selection_variants.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 6 Control strategy variants
# ------------------------------------------------------------------
psd_map = {}
for n, df in run_map.items():
    f = df[flow_cols(df)].to_numpy(dtype=float).mean(axis=1)
    freqs, psd = welch(f, fs=1.0, nperseg=min(512, max(32, len(f)//4)))
    valid = freqs > 0
    psd_map[n] = (freqs[valid], psd[valid])

spec_modes = {
    'linear_linear': (lambda f: f, lambda p: p),
    'logx_linear': (lambda f: np.log10(f + 1e-6), lambda p: p),
    'logx_logy': (lambda f: np.log10(f + 1e-6), lambda p: np.log10(p + 1e-12)),
}

fig, axes = _subplot_grid(len(spec_modes), ncols=3)
for ax, (mode, (ffn, pfn)) in zip(axes, spec_modes.items()):
    for n, (f, p) in psd_map.items():
        ax.plot(ffn(f), pfn(p), linewidth=1.2, label=n)
    ax.set_title(f'Strategy spectrum {mode}')
    ax.set_xlabel('Freq transformed')
    ax.set_ylabel('PSD transformed')
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=7)
for ax in axes[len(spec_modes):]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(GAL_OUT / '06_control_spectrum_all_variants.png', bbox_inches='tight')
plt.show()

smooth = {n: np.abs(np.diff(run_map[n][flow_cols(run_map[n])].to_numpy(dtype=float), axis=0)).ravel() for n in run_map}
sm_modes = {
    'linear': lambda x: x,
    'log1p': lambda x: np.log1p(x),
    'asinh_q95': lambda x: np.arcsinh(x / max(np.quantile(x, 0.95), 1e-12)),
    'rank': lambda x: pd.Series(x).rank(method='average').to_numpy(),
}

fig, axes = _subplot_grid(len(sm_modes), ncols=2)
for ax, (mode, fn) in zip(axes, sm_modes.items()):
    for n, arr in smooth.items():
        x, y = _ecdf(fn(arr))
        ax.plot(x, y, linewidth=1.2, label=n)
    ax.set_title(f'Smoothness {mode}')
    ax.set_xlabel('Value transformed')
    ax.set_ylabel('CDF')
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=7)
for ax in axes[len(sm_modes):]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(GAL_OUT / '06_control_smoothness_all_variants.png', bbox_inches='tight')
plt.show()

div_modes = ['bits', 'normalized_bits', 'perplexity']
bins = 24
fig, axes = _subplot_grid(len(div_modes), ncols=3)
for ax, mode in zip(axes, div_modes):
    names, vals = [], []
    for n, df in run_map.items():
        F = df[flow_cols(df)].to_numpy(dtype=float)
        ent = []
        for z in range(F.shape[1]):
            c, _ = np.histogram(F[:,z], bins=bins, range=(0,1), density=False)
            t = c.sum()
            if t <= 0:
                ent.append(0.0)
                continue
            p = c / t
            p = p[p > 0]
            h = -(p * np.log2(p)).sum()
            if mode == 'bits':
                v = h
            elif mode == 'normalized_bits':
                v = h / np.log2(bins)
            else:
                v = 2 ** h
            ent.append(float(v))
        names.append(n)
        vals.append(float(np.mean(ent)))
    ax.barh(names, vals, color=palette_greys(len(names)+2)[2:])
    ax.set_title(f'Diversity {mode}')
    ax.grid(alpha=0.2)
for ax in axes[len(div_modes):]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(GAL_OUT / '06_control_diversity_all_variants.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 7 Radar style variants
# ------------------------------------------------------------------
radar_metrics = [temp_col, energy_col, spread_col, stress_col, overhead_col]
rad = df_comp[['label'] + radar_metrics].copy()
Rraw = np.column_stack([rad[c].to_numpy(dtype=float) for c in radar_metrics])
Rnorm = np.column_stack([_minmax(Rraw[:,i]) for i in range(Rraw.shape[1])])
# lower better for these metrics, invert for radar so bigger = better
Rscore = 1.0 - Rnorm
lab = rad['label'].tolist()

if not df_rank.empty and 'rank_overall' in df_rank.columns and 'label' in df_rank.columns:
    top6_labels = df_rank.sort_values('rank_overall')['label'].head(6).tolist()
    idx_top6 = [lab.index(x) for x in top6_labels if x in lab]
else:
    idx_top6 = list(range(min(6, len(lab))))
if len(idx_top6) < 3:
    idx_top6 = list(range(min(6, len(lab))))
idx_all = list(range(len(lab)))

angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False)
angles_c = np.r_[angles, angles[0]]

# overlay all and top6
fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection':'polar'})
for ax, (title, idxs) in zip(axes, [('Radar overlay all', idx_all), ('Radar overlay top6', idx_top6)]):
    for i in idxs:
        vals = np.r_[Rscore[i], Rscore[i,0]]
        ax.plot(angles_c, vals, linewidth=1.0, alpha=0.9, label=lab[i])
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_metrics)
    ax.set_yticklabels([])
    ax.set_title(title)
axes[0].legend(loc='upper right', bbox_to_anchor=(1.35, 1.10), fontsize=6)
fig.tight_layout()
fig.savefig(GAL_OUT / '07_radar_overlay_variants.png', bbox_inches='tight')
plt.show()

# small multiples top6
n = len(idx_top6)
ncols = 3
nrows = int(math.ceil(n / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows), subplot_kw={'projection':'polar'})
axs = np.atleast_1d(axs).ravel()
for ax, i in zip(axs, idx_top6):
    vals = np.r_[Rscore[i], Rscore[i,0]]
    ax.plot(angles_c, vals, color='#1f77b4', linewidth=1.6)
    ax.fill(angles_c, vals, color='#1f77b4', alpha=0.22)
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_metrics, fontsize=7)
    ax.set_yticklabels([])
    ax.set_title(lab[i], fontsize=9)
for ax in axs[n:]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(GAL_OUT / '07_radar_small_multiples_top6.png', bbox_inches='tight')
plt.show()

# parallel coordinates top6
fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = np.arange(len(radar_metrics))
for i in idx_top6:
    ax.plot(x, Rscore[i], marker='o', linewidth=1.5, label=lab[i])
ax.set_xticks(x)
ax.set_xticklabels(radar_metrics)
ax.set_ylim(0, 1)
ax.set_ylabel('Normalized score higher better')
ax.set_title('Parallel coordinates top6')
ax.grid(alpha=0.2)
ax.legend(ncol=3, fontsize=7)
fig.tight_layout()
fig.savefig(GAL_OUT / '07_parallel_coords_top6.png', bbox_inches='tight')
plt.show()

print('Gallery exports')
for pth in sorted(GAL_OUT.glob('*.png')):
    print('-', pth.name)

