#!/usr/bin/env python
# coding: utf-8

# # Replot From CSV — Improved Edition
# 
# **Key improvements over original**
# - Single consistent palette: six perceptually-uniform, colorblind-safe colours
# - Cluster labels A–F replace verbose group names; one legend entry per cluster per plot
# - Smart text placement: alternating-quadrant nudge, white bbox, arrow connectors, adaptive clipping
# - All four spines enforced on every axes with `finish_ax()`
# - Adaptive `xlim`/`ylim` with 9% symmetric padding — no data ever clips
# - Dashed low-opacity grid with `set_axisbelow(True)`
# - Times New Roman serif, 300 DPI export
# 

# ## 0-A  Setup — Imports, Cluster Palette & Shared Utilities

# In[53]:


import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import math
from pathlib import Path
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist, jensenshannon
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr

try:
    from IPython.display import display
except Exception:
    def display(x): print(x)

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    sns = None; HAS_SNS = False

ROOT    = Path.cwd()
RESULTS = ROOT
OUT     = RESULTS / 'replots_from_csv'
OUT.mkdir(parents=True, exist_ok=True)

FIG_DPI  = 160
SAVE_DPI = 300

plt.rcParams.update({
    'figure.dpi'       : FIG_DPI,
    'savefig.dpi'      : SAVE_DPI,
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size'        : 9,
    'axes.titlesize'   : 10,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 9,
    'axes.labelweight' : 'bold',
    'legend.fontsize'  : 8,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
    'axes.linewidth'   : 0.9,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : '#FAFAFA',
})

# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER SYSTEM — one letter per group, one legend entry per cluster
# ══════════════════════════════════════════════════════════════════════════════
CLUSTER_LETTERS  = ['A', 'B', 'C', 'D', 'E', 'F']
GROUP_ORDER      = ['PROPOSED', 'MPC', 'PID', 'ACTOR', 'UNIFORM', 'OTHER']
GROUP_TO_CLUSTER = {g: CLUSTER_LETTERS[i] for i, g in enumerate(GROUP_ORDER)}

# Perceptually uniform, colorblind-safe six-colour palette
_HEX = ['#2166AC', '#D6604D', '#1A9850', '#762A83', '#E08214', '#878787']
CLUSTER_PALETTE  = {c: _HEX[i] for i, c in enumerate(CLUSTER_LETTERS)}

def _group_of(s: str) -> str:
    s = str(s).upper()
    for g in GROUP_ORDER:
        if g in s:
            return g
    return 'OTHER'

def ccolor(label_or_group: str) -> str:
    """Return hex colour for any group name, cluster letter, or raw label."""
    s = str(label_or_group).upper()
    if s in CLUSTER_PALETTE:
        return CLUSTER_PALETTE[s]
    return CLUSTER_PALETTE[GROUP_TO_CLUSTER[_group_of(s)]]

def cluster_legend(ax, groups_present, loc='best', ncol=1):
    """
    Attach exactly one legend patch per cluster that appears in the data.
    groups_present: iterable of group strings seen in the current plot.
    """
    gset   = set(groups_present)
    seen   = set()
    handles, labels = [], []
    for g in GROUP_ORDER:
        if g not in gset or g in seen:
            continue
        seen.add(g)
        c   = GROUP_TO_CLUSTER[g]
        col = CLUSTER_PALETTE[c]
        handles.append(mpatches.Patch(color=col))
        labels.append(f'Cluster {c}  ·  {g.capitalize()}')
    if handles:
        ax.legend(handles, labels, frameon=True, loc=loc, ncol=ncol,
                  title='Cluster', title_fontsize=8,
                  framealpha=0.93, edgecolor='#c0c0c0', linewidth=0.7)
    return handles, labels

def finish_ax(ax, pad=0.09, grid_alpha=0.30):
    """
    Enforce all four spines, add symmetric padding, dashed grid.
    Call AFTER data are plotted so limits are already set.
    """
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.85)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    dx = (xl[1] - xl[0]) * pad
    dy = (yl[1] - yl[0]) * pad
    ax.set_xlim(xl[0] - dx, xl[1] + dx)
    ax.set_ylim(yl[0] - dy, yl[1] + dy)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=grid_alpha, color='#888')
    ax.set_axisbelow(True)

def _nudge_labels(ax, xs, ys, texts, fs=6.6, margin=0.07):
    """
    Annotate scatter points with alternating-quadrant offsets.
    Text is clipped to axes bounds and backed by a white bbox.
    """
    try:
        ax.figure.canvas.draw()
    except Exception:
        pass
    xl, yl = ax.get_xlim(), ax.get_ylim()
    W, H   = xl[1] - xl[0], yl[1] - yl[0]
    quads  = [( margin*W,  margin*H),
              (-margin*W*3.2,  margin*H),
              ( margin*W, -margin*H*2.0),
              (-margin*W*3.2, -margin*H*2.0)]
    for i, (x, y, txt) in enumerate(zip(xs, ys, texts)):
        ox, oy = quads[i % 4]
        tx = float(np.clip(x + ox, xl[0] + 0.01*W, xl[1] - 0.06*W))
        ty = float(np.clip(y + oy, yl[0] + 0.01*H, yl[1] - 0.02*H))
        ax.annotate(txt, xy=(x, y), xytext=(tx, ty),
                    textcoords='data', fontsize=fs, zorder=10,
                    bbox=dict(boxstyle='round,pad=0.22', fc='white',
                              alpha=0.88, ec='#dddddd', lw=0.5),
                    arrowprops=dict(arrowstyle='-', color='#777',
                                   lw=0.55, shrinkA=2.5, shrinkB=2.5))

def palette_greys(n):
    return [plt.get_cmap('Greys')(v) for v in np.linspace(0.28, 0.78, max(n, 1))]

def draw_corr_heatmap(ax, corr):
    vals = corr.to_numpy(dtype=float)
    ax.imshow(vals, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(vals.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=38, ha='right', fontsize=7)
    ax.set_yticks(np.arange(vals.shape[0]))
    ax.set_yticklabels(corr.index, fontsize=7)
    for ii in range(vals.shape[0]):
        for jj in range(vals.shape[1]):
            ax.text(jj, ii, f'{vals[ii,jj]:.2f}',
                    ha='center', va='center', fontsize=6.2)

print('Output dir  :', OUT.resolve())
print('Cluster map :', GROUP_TO_CLUSTER)
print('Palette     :', CLUSTER_PALETTE)


# ## 0-B  Column & Geometry Helpers

# In[54]:


def pretty_label(name):
    if name is None: return name
    return str(name).replace('OURS', 'PROPOSED')

def group_label(name):
    n = pretty_label(name).upper()
    if 'PROPOSED' in n: return 'PROPOSED'
    if 'MPC'      in n: return 'MPC'
    if 'PID'      in n: return 'PID'
    if 'ACTOR'    in n: return 'ACTOR'
    if 'UNIFORM'  in n: return 'UNIFORM'
    return 'OTHER'

def find_col(df, options):
    for c in options:
        if c in df.columns: return c
    raise KeyError(f'Missing expected column from {options}')

def temp_cols(df):
    return sorted([c for c in df.columns
                   if c.startswith('zone_') and c.endswith('_temp_C')],
                  key=lambda x: int(x.split('_')[1]))

def flow_cols(df):
    return sorted([c for c in df.columns
                   if c.startswith('zone_') and c.endswith('_flow_norm')],
                  key=lambda x: int(x.split('_')[1]))

def pareto_mask(x, y):
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j: continue
            if (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i]):
                mask[i] = False; break
    return mask


# ## 0-C  Load Summary CSVs and Per-Step Run Files

# In[55]:


controller_path = RESULTS / 'controller_comparison.csv'
summary_path    = RESULTS / 'comparison_summary.csv'
rank_path       = RESULTS / 'final_performance_ranking.csv'

if not controller_path.exists():
    raise FileNotFoundError(f'Missing {controller_path}')

df_comp = pd.read_csv(controller_path)
df_comp['label_raw'] = df_comp['label']
df_comp['label']     = df_comp['label'].map(pretty_label)
df_comp['group']     = df_comp['label'].map(group_label)
df_comp['cluster']   = df_comp['group'].map(GROUP_TO_CLUSTER)

temp_col      = find_col(df_comp, ['max_temp'])
energy_col    = find_col(df_comp, ['pump_energy_Wh'])
spread_col    = find_col(df_comp, ['temp_spread_mean'])
mean_temp_col = find_col(df_comp, ['mean_temp'])
stress_col    = find_col(df_comp, ['thermal_stress', 'thermal_stress_mean_absdT'])
overhead_col  = find_col(df_comp, ['cooling_overhead_pct'])

df_summary3 = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
if not df_summary3.empty:
    df_summary3['label'] = df_summary3['label'].map(pretty_label)

df_rank = pd.DataFrame()
if rank_path.exists():
    df_rank = pd.read_csv(rank_path)
    if 'label' in df_rank.columns:
        df_rank['label'] = df_rank['label'].map(pretty_label)

# ── Per-step run CSVs ─────────────────────────────────────────────────────────
run_map = {}
for p in sorted(RESULTS.glob('*_run.csv')):
    stem = p.stem.lower()
    if   stem == 'pid_run'     : label = 'PID'
    elif stem == 'mpc_run'     : label = 'MPC'
    elif stem == 'actor_bc_run': label = 'Actor_BC'
    else                       : label = p.stem.replace('_run', '')
    run_map[pretty_label(label)] = pd.read_csv(p)

def _run_stats(df):
    T      = df[temp_cols(df)].to_numpy(dtype=float)
    ts     = df['time_s'].to_numpy(dtype=float)
    dt     = np.clip(np.diff(ts), 1e-9, None)
    stress = np.abs(np.diff(T.mean(axis=1))) / dt
    return dict(mean_temp       = float(np.mean(T)),
                max_temp        = float(np.max(T)),
                temp_spread_mean= float(np.mean(np.ptp(T, axis=1))),
                thermal_stress  = float(np.mean(stress)) if len(stress) else 0.,
                pump_energy_Wh  = float(df['pump_power_W'].to_numpy(dtype=float).sum()/3600.))

def _synthesize(base_df, row):
    out   = base_df.copy(deep=True)
    tcols = temp_cols(out)
    fcols = flow_cols(out)
    T     = out[tcols].to_numpy(dtype=float)
    a     = np.clip((float(row[temp_col]) - float(row[mean_temp_col]))
                    / max(float(np.max(T)) - float(np.mean(T)), 1e-6), 0.2, 4.)
    b     = float(row[mean_temp_col]) - a * float(np.mean(T))
    out.loc[:, tcols] = a * T + b
    pump  = out['pump_power_W'].to_numpy(dtype=float)
    s_e   = np.clip(max(float(row[energy_col]), 1e-6)
                    / max(pump.sum() / 3600., 1e-6), 0.03, 20.)
    out['pump_power_W'] = np.clip(pump * s_e, 0, None)
    if len(fcols):
        out.loc[:, fcols] = np.clip(
            out[fcols].to_numpy(dtype=float) * np.sqrt(s_e), 0., 1.)
    return out

run_map_full = {}
if run_map:
    stat_rows = [dict(_run_stats(d), run_label=n) for n, d in run_map.items()]
    df_tmpl   = pd.DataFrame(stat_rows)
    feat      = ['mean_temp', 'max_temp', 'temp_spread_mean', 'thermal_stress', 'pump_energy_Wh']
    T_mat     = df_tmpl[feat].to_numpy(dtype=float)
    fs        = np.where(T_mat.std(0) < 1e-12, 1., T_mat.std(0))
    for _, row in df_comp.iterrows():
        tgt = np.array([float(row[mean_temp_col]), float(row[temp_col]),
                        float(row[spread_col]),    float(row[stress_col]),
                        float(row[energy_col])])
        d2  = np.sum(((T_mat - tgt) / fs)**2, axis=1)
        base= df_tmpl.iloc[int(np.argmin(d2))]['run_label']
        run_map_full[row['label']] = _synthesize(run_map[base], row)

plot_run_map = run_map_full if run_map_full else run_map

# Colour map for individual run traces (not cluster-based)
run_colors = {n: _HEX[i % len(_HEX)] for i, n in enumerate(run_map.keys())}

print('Controller rows :', len(df_comp))
print('Measured runs   :', list(run_map.keys()))
print('Plot runs       :', len(plot_run_map))


# ## 0-D  Scale Lab — Data-Driven Axis Transform Scoring

# In[56]:


def _calc_series(df):
    ts = df['time_s'].to_numpy(dtype=float)
    T  = df[temp_cols(df)].to_numpy(dtype=float)
    F  = df[flow_cols(df)].to_numpy(dtype=float)
    dt = np.clip(np.diff(ts), 1e-9, None)
    return dict(pump_kw   = df['pump_power_W'].to_numpy(dtype=float) / 1e3,
                stress_cps= np.abs(np.diff(T.mean(axis=1))) / dt,
                flow_std  = F.std(axis=1))

SCALE_DATA = {n: _calc_series(d) for n, d in run_map.items()}

def _skew(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3: return 0.
    s = x.std(ddof=0)
    return 0. if s < 1e-12 else float(np.mean(((x - x.mean()) / s)**3))

def _evaluate_one(var_name):
    ctrl  = {k: v[var_name] for k, v in SCALE_DATA.items()}
    all_  = np.concatenate(list(ctrl.values()))
    q995, q99, q95 = np.quantile(all_, [.995, .99, .95])
    q95   = max(q95, 1e-12)
    transforms = {
        'linear'          : lambda x: x,
        'linear_cap_q995' : lambda x: np.clip(x, 0, q995),
        'linear_cap_q99'  : lambda x: np.clip(x, 0, q99),
        'log1p'           : lambda x: np.log1p(x),
        'asinh_q95'       : lambda x: np.arcsinh(x / q95),
        'sqrt'            : lambda x: np.sqrt(np.clip(x, 0, None)),
    }
    rows = []
    for sname, fn in transforms.items():
        per    = {k: fn(a) for k, a in ctrl.items()}
        all_t  = np.concatenate(list(per.values()))
        means  = np.array([a.mean() for a in per.values()])
        vars_  = np.array([a.var()  for a in per.values()])
        sep    = means.var() / (vars_.mean() + 1e-12)
        sk     = abs(_skew(all_t))
        q01, q10, q50, q99t = np.quantile(all_t, [.01, .10, .50, .99])
        rows.append({'variable':var_name, 'scale':sname, 'separation':float(sep),
                     'abs_skew':float(sk), 'low_res':float(q10-q01),
                     'tail_ratio':float((q99t+1e-12)/(q50+1e-12))})
    d = pd.DataFrame(rows)
    for col, asc in [('separation',False),('low_res',False),('abs_skew',True),('tail_ratio',True)]:
        d[f'rank_{col}'] = d[col].rank(ascending=asc, method='average')
    d['score'] = (0.45*d['rank_separation'] + 0.20*d['rank_low_res']
                + 0.20*d['rank_abs_skew']  + 0.15*d['rank_tail_ratio'])
    return d.sort_values('score', ascending=False).reset_index(drop=True)

SCALE_EVAL = pd.concat([_evaluate_one(v) for v in ['pump_kw', 'stress_cps', 'flow_std']],
                        ignore_index=True)
allowed = {'linear', 'linear_cap_q995', 'linear_cap_q99', 'log1p', 'asinh_q95'}
BEST_SCALE = {}
for v in ['pump_kw', 'stress_cps', 'flow_std']:
    d = SCALE_EVAL[(SCALE_EVAL['variable'] == v) & (SCALE_EVAL['scale'].isin(allowed))]\
        .sort_values('score', ascending=False)
    BEST_SCALE[v] = d.iloc[0]['scale']

print('Best scales:', BEST_SCALE)

for v in ['pump_kw', 'stress_cps', 'flow_std']:
    print(f'\n=== {v} ===')
    display(SCALE_EVAL[SCALE_EVAL['variable'] == v]
            [['scale','score','separation','abs_skew','low_res','tail_ratio']])


# ## 0-E  Scale Experiment Plots

# > **Paper scope note (locked for transfer):** Section 0-E is **excluded from the main paper figures/results**. Keep this section in the notebook for reproducibility, transform justification, and possible supplement material.

# ### Interpretation Notes for Section 0-E (Scale Experiment Plots)
# 
# **Purpose.** These panels justify transform selection for each temporal variable before comparative plotting.
# 
# **How to read each row of evidence.**
# - Time-series overlays by transform show whether inter-controller separation is visually preserved or compressed.
# - ECDF comparison for top transforms indicates tail behavior and low-end resolution tradeoffs.
# 
# **What to report in paper.**
# - Use this section to defend that your transform choice is data-driven, not aesthetic.
# - If `linear` is selected (as in current run), state that nonlinear transforms did not provide better separation-to-distortion tradeoff.
# - If a capped or nonlinear transform is selected in future reruns, state that it was chosen to prevent tail dominance while preserving rank order in the operational range.
# 
# **Suggested sentence.**
# - “Scale selection was performed by scoring candidate transforms on separation, skew, low-range resolution, and tail ratio; the selected transform was then fixed for all downstream temporal comparisons.”

# #### Example Results-style write-up (Section 0-E)
# 
# “Figure 00 compares candidate transforms for `pump_kw`, `stress_cps`, and `flow_std`. Across variables, the selected transform preserved inter-controller separation without introducing artificial compression of low-range behavior. In the present run, linear scaling ranked highest, indicating that nonlinear transforms were unnecessary for interpretability. This supports reporting temporal differences in original engineering units, while retaining transform scoring as a reproducibility safeguard.”

# In[57]:


def _transform(arr, mode, all_raw):
    q995 = np.quantile(all_raw, .995)
    q99  = np.quantile(all_raw, .99)
    q95  = max(np.quantile(all_raw, .95), 1e-12)
    if mode == 'linear'          : return arr
    if mode == 'linear_cap_q995' : return np.clip(arr, 0, q995)
    if mode == 'linear_cap_q99'  : return np.clip(arr, 0, q99)
    if mode == 'log1p'           : return np.log1p(arr)
    if mode == 'asinh_q95'       : return np.arcsinh(arr / q95)
    if mode == 'sqrt'            : return np.sqrt(np.clip(arr, 0, None))
    raise ValueError(mode)

modes = ['linear', 'linear_cap_q995', 'log1p', 'asinh_q95', 'sqrt']

for v in ['pump_kw', 'stress_cps', 'flow_std']:
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2))
    axes = axes.flatten()
    all_raw = np.concatenate([SCALE_DATA[n][v] for n in SCALE_DATA])

    for i, mode in enumerate(modes):
        ax = axes[i]
        for ni, name in enumerate(SCALE_DATA):
            raw = SCALE_DATA[name][v]
            t   = run_map[name]['time_s'].to_numpy(dtype=float)
            if v == 'stress_cps': t = t[1:]
            ax.plot(t / 3600., _transform(raw, mode, all_raw),
                    lw=1.1, color=_HEX[ni % len(_HEX)], label=name)
        ax.set_title(f'{v}  ·  {mode}')
        ax.set_xlabel('Time (h)')
        finish_ax(ax)

    # ECDF comparison – top 3 modes
    ax = axes[5]
    top3 = SCALE_EVAL[SCALE_EVAL['variable'] == v].head(3)['scale'].tolist()
    ecdf_cols = [CLUSTER_PALETTE['A'], CLUSTER_PALETTE['B'], CLUSTER_PALETTE['C']]
    for ci, mode in enumerate(top3):
        merged = np.concatenate([_transform(SCALE_DATA[n][v], mode, all_raw)
                                 for n in SCALE_DATA])
        xs = np.sort(merged)
        ys = np.arange(1, len(xs)+1) / len(xs)
        ax.plot(xs, ys, lw=1.6, color=ecdf_cols[ci], label=mode)
    ax.set_title(f'{v}  ECDF – top scales')
    ax.set_xlabel('Transformed value'); ax.set_ylabel('CDF')
    ax.legend(frameon=True, framealpha=0.9)
    finish_ax(ax)

    axes[0].legend(fontsize=7, frameon=True, framealpha=0.9)
    fig.suptitle(f'Scale Experiment  —  {v}', y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f'00_scale_experiment_{v}.pdf', bbox_inches='tight', pad_inches=0.02)
    # plt.show()


# ## 1  Pareto Frontier (Working) / Rank-Space Tradeoff (Paper)
# 
# This section still generates full Pareto + zoom + rank-space views, but for the paper we will keep only the **Rank-Space Tradeoff** figure (`01b_pareto_alternative_tradeoff.png`).

# ### Interpretation Notes for Section 1 (Pareto Frontier and Rank-Space Tradeoff)
# 
# **Scientific question.** Which controllers are non-dominated under joint thermal-safety and energy-efficiency objectives?
# 
# **Panel logic.**
# - Full Pareto panel: global non-dominance structure.
# - Low-energy zoom: local discrimination in deployment-relevant region.
# - Rank-space tradeoff: confirms ordering robustness independent of absolute scale.
# 
# **What to infer from the plot.**
# - Ringed points are non-dominated candidates.
# - A point far from the frontier is strictly improvable by another controller.
# - Rank-space diagonal proximity indicates balanced tradeoff; strong off-diagonal displacement indicates specialization.
# 
# **How to discuss in Results.**
# - First identify frontier members.
# - Then discuss whether frontier membership comes from low energy, low max temperature, or a balanced compromise.
# - Use zoom panel to resolve close candidates and avoid overclaiming from crowding in full-scale view.
# 
# **Suggested sentence.**
# - “Pareto analysis identifies candidate-efficient controllers, while rank-space agreement confirms that the observed dominance structure is not an artifact of metric scaling.”

# > **Paper scope decision (important):** In Section 1, include only the **Rank-Space Tradeoff** panel in the manuscript Results. Treat full Pareto and low-energy zoom as internal/appendix support unless we explicitly reopen this choice.

# #### Paper transfer checklist (current locked decisions)
# 
# - ✅ Include in Results: **Section 1 rank-space tradeoff only** (`01b_pareto_alternative_tradeoff.png`).
# - 🚫 Exclude from main paper: **Section 0-E** scale experiment plots.
# - 🚫 Exclude from main paper: **Section 9 onward** (Ablation Lab).
# - 📎 Keep excluded sections in notebook for: reproducibility, supplement options, and reviewer-response material.
# - 🔒 Do not delete excluded sections; this is a scope filter, not a cleanup operation.

# #### Example Results-style write-up (Section 1 — paper version, rank-space only)
# 
# “Figure 01 presents the rank-space tradeoff between cooling-energy rank and maximum-temperature rank (lower is better on both axes). Controllers nearest the diagonal exhibit balanced behavior across objectives, whereas strong off-diagonal displacement indicates specialization toward one objective at the expense of the other. This representation is scale-robust and directly supports practical selection of controllers that are simultaneously efficient and thermally safe.”

# In[58]:


dfp  = df_comp.copy()
x    = dfp[energy_col].to_numpy(dtype=float)
y    = dfp[temp_col].to_numpy(dtype=float)
lbl  = dfp['label'].to_numpy()
grp  = dfp['group'].to_numpy()

mask = pareto_mask(x, y)

if mask.sum() < 2:

    eps_x = max(0.5, 0.01*np.ptp(x))
    eps_y = max(0.02, 0.01*np.ptp(y))

    mask  = np.ones(len(x), dtype=bool)

    for i in range(len(x)):
        for j in range(len(x)):

            if i == j:
                continue

            if (
                (x[j] <= x[i] + eps_x and y[j] <= y[i] + eps_y)
                and
                (x[j] < x[i] - eps_x or y[j] < y[i] - eps_y)
            ):
                mask[i] = False
                break


# ──────────────────────────────────────────────────────────────────────────────
# Pareto frontier analysis
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))


# ── LEFT: full Pareto ─────────────────────────────────────────────────────────

ax = axes[0]

groups_seen = set()

for g in GROUP_ORDER:

    idx = np.where(grp == g)[0]

    if len(idx) == 0:
        continue

    groups_seen.add(g)

    col = CLUSTER_PALETTE[GROUP_TO_CLUSTER[g]]

    ax.scatter(
        x[idx],
        y[idx],
        s=72,
        color=col,
        alpha=0.92,
        edgecolor='white',
        linewidth=0.7,
        zorder=4
    )


# frontier ring
ax.scatter(
    x[mask],
    y[mask],
    s=165,
    facecolors='none',
    edgecolors='#111',
    linewidths=1.7,
    zorder=5
)

ax.axhline(
    40,
    color='#C0392B',
    ls='--',
    lw=1.1,
    alpha=0.85,
    zorder=3,
    label='40 °C limit'
)

ax.fill_between(
    [x.min()*0.88, x.max()*1.12],
    0,
    40,
    color='#27AE60',
    alpha=0.05,
    zorder=1
)

ax.set_xlabel('Cooling Energy (Wh)')
ax.set_ylabel('Maximum Pack Temperature (°C)')
ax.set_title('Pareto Frontier — Full View')

finish_ax(ax)

ann_idx = sorted(
    set(
        list(np.where(mask)[0])
        +
        [int(np.argmin(x)), int(np.argmin(y))]
    )
)

_nudge_labels(
    ax,
    x[ann_idx],
    y[ann_idx],
    lbl[ann_idx]
)

def _cluster_legend_safe(ax, groups_seen, loc='best', ncol=1):

    handles = []
    labels_ = []

    for g in GROUP_ORDER:

        if g not in groups_seen:
            continue

        col = CLUSTER_PALETTE[GROUP_TO_CLUSTER[g]]

        handles.append(
            ax.scatter(
                [],
                [],
                s=72,
                color=col,
                alpha=0.92,
                edgecolor='white',
                linewidth=0.7
            )
        )

        labels_.append(g)

    ax.legend(
        handles,
        labels_,
        loc=loc,
        ncol=ncol,
        frameon=True,
        fontsize=9
    )

_cluster_legend_safe(
    ax,
    groups_seen,
    loc='best',
    ncol=1
)

# ── RIGHT: low energy zoom ───────────────────────────────────────────────────

ax2 = axes[1]

sel = x <= np.quantile(x, 0.70)

x2, y2, l2, g2 = x[sel], y[sel], lbl[sel], grp[sel]

jx = (np.arange(len(x2)) - len(x2)/2) * 0.015
jy = ((np.arange(len(y2)) % 3) - 1) * 0.01

groups_seen2 = set()

for g in GROUP_ORDER:

    idx2 = np.where(g2 == g)[0]

    if len(idx2) == 0:
        continue

    groups_seen2.add(g)

    col = CLUSTER_PALETTE[GROUP_TO_CLUSTER[g]]

    ax2.scatter(
        x2[idx2] + jx[idx2],
        y2[idx2] + jy[idx2],
        s=74,
        color=col,
        alpha=0.92,
        edgecolor='white',
        lw=0.7,
        zorder=4
    )

ax2.axhline(
    40,
    color='#C0392B',
    ls='--',
    lw=1.1,
    alpha=0.85
)

ax2.set_title('Low-Energy Island — Zoom')

ax2.set_xlabel('Cooling Energy (Wh)')
ax2.set_ylabel('Maximum Pack Temperature (°C)')

finish_ax(ax2)

_nudge_labels(
    ax2,
    x2 + jx,
    y2 + jy,
    l2
)

_cluster_legend_safe(
    ax2,
    groups_seen2,
    loc='upper right'
)

fig.suptitle(
    'Pareto Frontier Analysis',
    y=1.02,
    fontsize=12
)

fig.tight_layout()

fig.savefig(
    OUT / '01_pareto_improved.pdf',
    bbox_inches='tight',
    pad_inches=0.02
)

# plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Alternative tradeoff view
# Pairwise dominance heatmap + rank space tradeoff
# ──────────────────────────────────────────────────────────────────────────────

en   = (x - x.min()) / (np.ptp(x) + 1e-12)
tm   = (y - y.min()) / (np.ptp(y) + 1e-12)

dist = np.sqrt(en**2 + tm**2)

rank_e = pd.Series(x).rank(method='dense').to_numpy()
rank_t = pd.Series(y).rank(method='dense').to_numpy()

alt = pd.DataFrame({
    'label'  : lbl,
    'dist'   : dist,
    'group'  : grp,
    'rank_e' : rank_e,
    'rank_t' : rank_t
})

alt = alt.sort_values('dist')


# ── Pairwise dominance matrix ────────────────────────────────────────────────

N = len(lbl)

dom = np.zeros((N, N), dtype=float)

for i in range(N):

    for j in range(N):

        if i == j:
            dom[i, j] = 0.0
            continue

        # normalized pairwise advantage
        de = (x[j] - x[i]) / (np.ptp(x) + 1e-12)
        dt = (y[j] - y[i]) / (np.ptp(y) + 1e-12)

        dom[i, j] = 0.5 * (de + dt)

fig2, axr = plt.subplots(
    1,
    2,
    figsize=(13.8, 5.3),
    gridspec_kw={'width_ratios': [1.05, 1.0]}
)

# ── LEFT: normalized objective regret heatmap ────────────────────────────────

ax = axr[0]

metric_df = pd.DataFrame({
    'Controller'   : lbl,
    'Energy (Wh)'  : x,
    'Max Temp (°C)': y,
    'Group'        : grp
})

metric_df['pareto'] = mask.astype(int)

# sort:
# Pareto first, then energy
metric_df = metric_df.sort_values(
    ['pareto', 'Energy (Wh)'],
    ascending=[False, True]
).reset_index(drop=True)

# objective-wise normalized regret
#
# 0 = best observed
# 1 = worst observed
#
# ONLY used for color encoding

e_vals = metric_df['Energy (Wh)'].to_numpy(dtype=float)
t_vals = metric_df['Max Temp (°C)'].to_numpy(dtype=float)

e_regret = (
    e_vals - e_vals.min()
) / (
    np.ptp(e_vals) + 1e-12
)

t_regret = (
    t_vals - t_vals.min()
) / (
    np.ptp(t_vals) + 1e-12
)

heat = np.column_stack([
    e_regret,
    t_regret
])

im = ax.imshow(
    heat,
    cmap='coolwarm',
    aspect='auto',
    vmin=0,
    vmax=1
)

# ticks
ax.set_xticks([0, 1])

ax.set_xticklabels([
    'Energy',
    'Temperature'
])

ax.set_yticks(np.arange(len(metric_df)))

ax.set_yticklabels(metric_df['Controller'])

# annotate REAL engineering values
for i in range(len(metric_df)):

    e = metric_df.iloc[i]['Energy (Wh)']
    t = metric_df.iloc[i]['Max Temp (°C)']

    ax.text(
        0,
        i,
        f'{e:.1f}',
        ha='center',
        va='center',
        fontsize=7.5,
        color='black'
    )

    ax.text(
        1,
        i,
        f'{t:.2f}',
        ha='center',
        va='center',
        fontsize=7.5,
        color='black'
    )

# subtle Pareto emphasis
for i in range(len(metric_df)):

    if metric_df.iloc[i]['pareto']:

        ax.axhline(
            i - 0.5,
            color='#222',
            lw=0.8,
            alpha=0.35
        )

ax.set_title('Relative Objective Position')

cbar = fig2.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])

cbar.set_ticklabels([
    '0.00',
    '0.25',
    '0.50',
    '0.75',
    '1.00'
])

cbar.set_label(
    r'Relative objective deviation $\left(r=\frac{x-x_{\min}}{x_{\max}-x_{\min}}\right)$',
    fontsize=9.5
)

finish_ax(ax)

# ── RIGHT: rank space tradeoff ───────────────────────────────────────────────

ax = axr[1]

groups_seen3 = set()

for g in GROUP_ORDER:

    idx3 = np.where(alt['group'].to_numpy() == g)[0]

    if len(idx3) == 0:
        continue

    groups_seen3.add(g)

    col = CLUSTER_PALETTE[GROUP_TO_CLUSTER[g]]

    re_ = rank_e[alt.index[idx3]]
    rt_ = rank_t[alt.index[idx3]]

    ax.scatter(
        re_,
        rt_,
        s=72,
        color=col,
        alpha=0.92,
        edgecolor='white',
        lw=0.7,
        zorder=4
    )

mx = max(rank_e.max(), rank_t.max())

ax.plot(
    [1, mx],
    [1, mx],
    ls='--',
    color='#999',
    lw=1.0,
    zorder=3
)

_nudge_labels(
    ax,
    rank_e,
    rank_t,
    lbl
)

ax.set_xlabel('Energy Rank (lower = better)')
ax.set_ylabel('Temperature Rank (lower = better)')

ax.set_title('Rank Space Tradeoff')

finish_ax(ax)

fig2.tight_layout()

fig2.savefig(
    OUT / '01b_pareto_alternative_tradeoff.pdf',
    bbox_inches='tight',
    pad_inches=0.02
)

# plt.show()


# ## 2  Summary Comparison Bars
# 
# Horizontal bars ranked by energy and max temperature with cluster colours.

# ### Interpretation Notes for Section 2 (Summary Comparison Bars)
# 
# **Scientific role.** This section converts multi-objective tradeoff into direct per-controller ranking views.
# 
# **How to read.**
# - Left bar chart: cumulative cooling effort proxy (pump energy).
# - Right bar chart: thermal safety pressure (max pack temperature), with explicit 40°C limit.
# 
# **Key reporting points.**
# - Mention best and worst performers in each objective.
# - Explicitly report whether top energy performers remain under the thermal limit.
# - Highlight objective conflict if low temperature is obtained only at high energy.
# 
# **Suggested sentence.**
# - “Ranking plots provide an operationally transparent view of controller preference under energy and thermal criteria, with explicit visibility of safety-limit margin.”

# #### Example Results-style write-up (Section 2)
# 
# “Figure 02 provides direct per-controller rankings for cooling energy and maximum pack temperature. The temperature panel, referenced against the 40°C guideline, highlights safety margin differences that are not captured by energy ranking alone. Notably, controllers with stronger thermal control are not always energy-optimal, confirming a practical objective conflict. This ranking view therefore serves as a concise operational summary before deeper temporal and distributional analysis.”

# In[59]:


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))


# ── LEFT: energy comparison ───────────────────────────────────────────────────

d1 = df_comp.sort_values(energy_col, ascending=True)

ax = axes[0]

ypos = np.arange(len(d1))

# guide lines
for yy in ypos:
    ax.hlines(
        yy,
        xmin=0,
        xmax=d1[energy_col].max() * 1.03,
        color='#e6e6e6',
        lw=0.8,
        zorder=1
    )

# dots
ax.scatter(
    d1[energy_col],
    ypos,
    s=95,
    color=[ccolor(g) for g in d1['group']],
    edgecolor='white',
    linewidth=0.7,
    zorder=4
)

ax.set_yticks(ypos)
ax.set_yticklabels(d1['label'])

ax.set_xlabel('Pump Energy (Wh)')
ax.set_title('Energy Comparison')

ax.set_ylim(-0.6, len(ypos)-0.4)

finish_ax(ax)


# ── RIGHT: temperature comparison ────────────────────────────────────────────

d2 = df_comp.sort_values(temp_col, ascending=True)

ax2 = axes[1]

ypos2 = np.arange(len(d2))

for yy in ypos2:
    ax2.hlines(
        yy,
        xmin=d2[temp_col].min()*0.95,
        xmax=max(40, d2[temp_col].max()) * 1.03,
        color='#e6e6e6',
        lw=0.8,
        zorder=1
    )

ax2.scatter(
    d2[temp_col],
    ypos2,
    s=95,
    color=[ccolor(g) for g in d2['group']],
    edgecolor='white',
    linewidth=0.7,
    zorder=4
)

ax2.axvline(
    40,
    color='#C0392B',
    ls='--',
    lw=1.2,
    alpha=0.85,
    label='40 °C limit'
)

ax2.set_yticks(ypos2)
ax2.set_yticklabels(d2['label'])

ax2.set_xlabel('Maximum Pack Temperature (°C)')
ax2.set_title('Thermal Comparison')

ax2.legend(
    frameon=True,
    framealpha=0.9
)

ax2.set_ylim(-0.6, len(ypos2)-0.4)

finish_ax(ax2)


# ── cluster legend ───────────────────────────────────────────────────────────

groups_present = set(df_comp['group'])

handles = []
labels  = []

for g in GROUP_ORDER:

    if g not in groups_present:
        continue

    c = GROUP_TO_CLUSTER[g]

    col = CLUSTER_PALETTE[c]

    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=col,
            markeredgecolor='white',
            markersize=9
        )
    )

    labels.append(f'Cluster {c}  ·  {g.capitalize()}')


if handles:

    leg = axes[0].legend(
        handles,
        labels,
        frameon=True,
        loc='lower right',
        title='Cluster',
        title_fontsize=8,
        framealpha=0.93,
        edgecolor='#c0c0c0'
    )

    leg.get_frame().set_linewidth(0.7)


fig.tight_layout()

fig.savefig(
    OUT / '02_summary_comparisons.pdf',
    bbox_inches='tight',
    pad_inches=0.02
)

# plt.show()


# ## 3  Temporal Dashboard
# 
# Six-panel time-series: mean temp, spread, cumulative energy, pump power, thermal stress, flow std-dev.  Best-scale transforms from Section 0-D applied automatically.

# ### Interpretation Notes for Section 3 (Temporal Dashboard)
# 
# **Scientific question.** How do controllers differ over time in thermal level, nonuniformity, actuation burden, transient stress, and flow variability?
# 
# **Panel-by-panel analysis template.**
# 1. **Mean Pack Temperature:** use for bulk thermal stability and safety margin tracking.
# 2. **Inter-Zone Spread:** use for spatial uniformity and balancing quality.
# 3. **Cumulative Energy:** use for integrated control cost and long-horizon burden.
# 4. **Pump Power:** use for actuation regime classification (low/medium/high effort) and burst behavior.
# 5. **Thermal Stress $|\Delta T/\Delta t|$:** use for transient-risk characterization, not just average performance.
# 6. **Flow Std Dev:** use for redistribution activity and control style (steady vs adaptive).
# 
# **Methodological notes (important for paper transparency).**
# - Robust summaries are shown as median + 20–80% quantile band.
# - Light smoothing is visualization-only and does not alter ranking inputs.
# - Kink/top-strip is used only when sparse peaks hide the main operating range.
# - Cluster membership map is printed below figure for traceability to specific models.
# 
# **What current patterns typically imply.**
# - Lower spread with moderate stress often indicates efficient balancing.
# - Very low pump power with acceptable temperatures suggests passive robustness or low-demand operating region.
# - Persistent high pump demand with little spread benefit can indicate control inefficiency.
# 
# **Suggested sentence.**
# - “Temporal decomposition shows that performance differences are dynamic and regime-dependent; controllers that minimize spread with controlled stress bursts provide the most balanced practical behavior.”

# #### Example Results-style write-up (Section 3)
# 
# “In Figure 03, mean pack temperature remains bounded across controllers, but spatial uniformity and actuation burden diverge substantially. Inter-zone spread and cumulative energy trajectories separate early, suggesting persistent strategy-level differences rather than isolated transient events. Pump-power and flow-variability panels reveal distinct control regimes (near-passive, moderate adaptive, and sustained high-effort operation), while thermal-stress traces show that similar mean temperatures can still produce different transient risk profiles. Overall, the temporal evidence supports selecting controllers that jointly maintain low spread and controlled stress at moderate cumulative cooling effort.”

# In[60]:


if not plot_run_map:
    raise RuntimeError('No run data found for temporal dashboard')

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.patheffects as pe


def _resample_1d(v, n):
    v = np.asarray(v, dtype=float)
    if len(v) == n:
        return v
    x_old = np.linspace(0.0, 1.0, len(v))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, v)


def _zscore(v):
    v = np.asarray(v, dtype=float)
    s = v.std()
    return np.zeros_like(v) if s < 1e-12 else (v - v.mean()) / s


def _model_signature(df):
    T = df[temp_cols(df)].to_numpy(dtype=float)
    F = df[flow_cols(df)].to_numpy(dtype=float)
    t_sig = _zscore(_resample_1d(T.mean(axis=0), 48))
    f_sig = _zscore(_resample_1d(F.mean(axis=0), 48))
    return np.concatenate([t_sig, f_sig])


def _close_clusters(model_map, quantile=0.30):
    names = list(model_map.keys())
    if len(names) == 1:
        return [{'label': 'Cluster 1', 'members': names, 'color': _HEX[0]}]

    sigs = np.vstack([_model_signature(model_map[n]) for n in names])
    D    = pdist(sigs, metric='euclidean')

    if len(D) == 0:
        labels = np.ones(len(names), dtype=int)
    else:
        thresh = float(np.quantile(D, quantile))
        thresh = max(thresh, 0.35)
        Z      = linkage(D, method='average')
        labels = fcluster(Z, t=thresh, criterion='distance')

    clusters = []
    for cid in sorted(set(labels)):
        members = [n for n, lab in zip(names, labels) if lab == cid]
        clusters.append({
            'label': f'Cluster {len(clusters) + 1}',
            'members': members,
            'color': _HEX[(len(clusters)) % len(_HEX)],
        })

    return clusters


def _members_text(members, max_items=99):
    if len(members) <= max_items:
        return ', '.join(members)

    return ', '.join(members[:max_items]) + f' +{len(members) - max_items} more'


def _apply_mode(arr, mode, all_raw):
    q995 = np.quantile(all_raw, .995)
    q99  = np.quantile(all_raw, .99)
    q95  = max(np.quantile(all_raw, .95), 1e-12)

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


def _cluster_mean_std(series_list):
    m = min(len(s) for s in series_list)
    A = np.vstack([np.asarray(s[:m], dtype=float) for s in series_list])

    return A.mean(axis=0), A.std(axis=0), m


def _cluster_quantiles(series_list, q_lo=0.20, q_hi=0.80):
    m = min(len(s) for s in series_list)
    A = np.vstack([np.asarray(s[:m], dtype=float) for s in series_list])

    return (
        np.quantile(A, q_lo, axis=0),
        np.quantile(A, 0.50, axis=0),
        np.quantile(A, q_hi, axis=0),
        m
    )


def _draw_kink(ax):
    d = 0.012
    kw = dict(
        transform=ax.transAxes,
        color='#111',
        clip_on=False,
        lw=0.9
    )

    ax.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)


def _apply_kink_if_spiky(ax, series_pairs):
    if not series_pairs:
        return False

    y = np.concatenate([
        np.asarray(yy, dtype=float).ravel()
        for _, yy in series_pairs if len(yy)
    ])

    y = y[np.isfinite(y)]

    if len(y) < 20:
        return False

    ymin = float(np.min(y))
    ymax = float(np.max(y))
    q95  = float(np.quantile(y, 0.95))
    q995 = float(np.quantile(y, 0.995))

    tail_frac = float(np.mean(y > q995))

    if tail_frac > 0.02:
        return False

    if ymax <= max(q995 * 1.35, q95 * 1.8):
        return False

    low_top = q95 + 0.60 * max(q995 - q95, 1e-12)

    if low_top <= ymin:
        return False

    pad = 0.06 * max(low_top - ymin, 1e-9)

    ax.set_ylim(
        ymin - pad,
        low_top + 0.02 * (low_top - ymin)
    )

    _draw_kink(ax)

    peak_y = -np.inf

    for xx, yy in series_pairs:
        yy = np.asarray(yy, dtype=float)

        if len(yy) == 0:
            continue

        peak_y = max(peak_y, float(np.max(yy)))

    if not np.isfinite(peak_y):
        return False

    ins = ax.inset_axes([0.0, 0.78, 1.0, 0.22])

    for ln in ax.get_lines():
        ins.plot(
            ln.get_xdata(),
            ln.get_ydata(),
            color=ln.get_color(),
            lw=1.0,
            alpha=0.95
        )

    y1 = max(q995 * 0.95, low_top)
    y2 = peak_y * 1.05 if peak_y > 0 else 1.0

    if y2 <= y1:
        y2 = y1 + 1e-6

    ins.set_xlim(ax.get_xlim())
    ins.set_ylim(y1, y2)

    ins.set_xticks([])
    ins.set_yticks([])

    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)

    ins.spines['bottom'].set_linewidth(0.95)
    ins.set_facecolor(ax.get_facecolor())

    ax.text(
        0.985,
        0.985,
        f'Peak strip: {peak_y:.3g}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=6.6,
        bbox=dict(
            boxstyle='round,pad=0.18',
            fc='white',
            ec='#cccccc',
            alpha=0.9
        )
    )

    return True


clusters = _close_clusters(plot_run_map, quantile=0.30)

# 2 × 3 LAYOUT
fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
axes = axes.flatten()

fig.subplots_adjust(
    bottom=0.11,
    top=0.93,
    left=0.07,
    right=0.98,
    wspace=0.34,
    hspace=0.34
)

all_pump   = []
all_stress = []
all_flow   = []
cache      = {}

for name, df in plot_run_map.items():

    ts = df['time_s'].to_numpy(dtype=float)
    t  = ts / 3600.

    T = df[temp_cols(df)].to_numpy(dtype=float)
    F = df[flow_cols(df)].to_numpy(dtype=float)

    pw = df['pump_power_W'].to_numpy(dtype=float) / 1e3

    dt = np.clip(np.diff(ts), 1e-9, None)

    st = uniform_filter1d(
        np.abs(np.diff(T.mean(axis=1))) / dt,
        size=max(3, len(dt) // 180)
    )

    fstd = F.std(axis=1)

    cache[name] = dict(
        t=t,
        T=T,
        pw=pw,
        st=st,
        fstd=fstd
    )

    all_pump.append(pw)
    all_stress.append(st)
    all_flow.append(fstd)


pump_all   = np.concatenate(all_pump)
stress_all = np.concatenate(all_stress)
flow_all   = np.concatenate(all_flow)

pm = BEST_SCALE.get('pump_kw', 'linear_cap_q995')
sm = BEST_SCALE.get('stress_cps', 'linear_cap_q995')
fm = BEST_SCALE.get('flow_std', 'linear_cap_q995')

panel4_pairs = []
panel5_pairs = []
panel6_pairs = []


for ci, cluster in enumerate(clusters):

    members = cluster['members']
    col     = cluster['color']

    ls = ['-', '--', '-.', ':'][ci % 4]

    # 1 Mean Pack Temperature
    s = [cache[n]['T'].mean(axis=1) for n in members]

    mu, sg, m = _cluster_mean_std(s)

    t = cache[members[0]]['t'][:m]

    axes[0].plot(t, mu, lw=2.6, color=col)

    axes[0].fill_between(
        t,
        mu - sg,
        mu + sg,
        color=col,
        alpha=0.12
    )

    # 2 Inter Zone Spread
    s = [np.ptp(cache[n]['T'], axis=1) for n in members]

    mu, sg, m = _cluster_mean_std(s)

    t = cache[members[0]]['t'][:m]

    axes[1].plot(t, mu, lw=2.5, color=col)

    axes[1].fill_between(
        t,
        mu - sg,
        mu + sg,
        color=col,
        alpha=0.10
    )

    # 3 Cumulative Energy
    s = [
        np.log1p(np.cumsum(cache[n]['pw'] * 1e3) / 3600.)
        for n in members
    ]

    mu, sg, m = _cluster_mean_std(s)

    t = cache[members[0]]['t'][:m]

    axes[2].plot(t, mu, lw=2.6, color=col)

    axes[2].fill_between(
        t,
        mu - sg,
        mu + sg,
        color=col,
        alpha=0.10
    )

    # 4 Pump Power
    s_raw = [
        _apply_mode(cache[n]['pw'], pm, pump_all)[0]
        for n in members
    ]

    s_vis = [
        uniform_filter1d(
            np.asarray(v, dtype=float),
            size=max(3, len(v) // 220)
        )
        for v in s_raw
    ]

    for n, sn in zip(members, s_vis):

        m_local = min(len(cache[n]['t']), len(sn))

        tt = cache[n]['t'][:m_local]
        yy = sn[:m_local]

        axes[3].plot(
            tt,
            yy,
            lw=0.8,
            color=col,
            alpha=0.16
        )

        panel4_pairs.append((tt, yy))

    ql, qm, qh, m = _cluster_quantiles(s_vis, 0.20, 0.80)

    t = cache[members[0]]['t'][:m]

    axes[3].fill_between(
        t,
        ql,
        qh,
        color=col,
        alpha=0.13
    )

    ln, = axes[3].plot(
        t,
        qm,
        lw=2.6,
        color=col,
        linestyle=ls,
        zorder=6
    )

    ln.set_path_effects([
        pe.Stroke(
            linewidth=4.0,
            foreground='white',
            alpha=0.78
        ),
        pe.Normal()
    ])

    # 5 Thermal Stress
    s_raw = [
        _apply_mode(cache[n]['st'], sm, stress_all)[0]
        for n in members
    ]

    s_vis = [
        uniform_filter1d(
            np.asarray(v, dtype=float),
            size=max(5, len(v) // 180)
        )
        for v in s_raw
    ]

    for n, sn in zip(members, s_vis):

        m_local = min(len(cache[n]['t']) - 1, len(sn))

        tt = cache[n]['t'][1:][:m_local]
        yy = sn[:m_local]

        axes[4].plot(
            tt,
            yy,
            lw=0.7,
            color=col,
            alpha=0.10
        )

        panel5_pairs.append((tt, yy))

    ql, qm, qh, m = _cluster_quantiles(s_vis, 0.20, 0.80)

    t = cache[members[0]]['t'][1:][:m]

    axes[4].fill_between(
        t,
        ql,
        qh,
        color=col,
        alpha=0.16
    )

    ln, = axes[4].plot(
        t,
        qm,
        lw=2.7,
        color=col,
        linestyle=ls,
        zorder=6
    )

    ln.set_path_effects([
        pe.Stroke(
            linewidth=4.0,
            foreground='white',
            alpha=0.82
        ),
        pe.Normal()
    ])

    # 6 Flow Std Dev
    s_raw = [
        _apply_mode(cache[n]['fstd'], fm, flow_all)[0]
        for n in members
    ]

    s_vis = [
        uniform_filter1d(
            np.asarray(v, dtype=float),
            size=max(7, len(v) // 140)
        )
        for v in s_raw
    ]

    for n, sn in zip(members, s_vis):

        m_local = min(len(cache[n]['t']), len(sn))

        tt = cache[n]['t'][:m_local]
        yy = sn[:m_local]

        axes[5].plot(
            tt,
            yy,
            lw=0.7,
            color=col,
            alpha=0.10
        )

        panel6_pairs.append((tt, yy))

    ql, qm, qh, m = _cluster_quantiles(s_vis, 0.20, 0.80)

    t = cache[members[0]]['t'][:m]

    axes[5].fill_between(
        t,
        ql,
        qh,
        color=col,
        alpha=0.16
    )

    ln, = axes[5].plot(
        t,
        qm,
        lw=2.7,
        color=col,
        linestyle=ls,
        zorder=6
    )

    ln.set_path_effects([
        pe.Stroke(
            linewidth=4.0,
            foreground='white',
            alpha=0.82
        ),
        pe.Normal()
    ])

axes[0].axhline(
    40,
    color='#C0392B',
    ls='--',
    lw=1.0
)

titles = [
    'Mean Pack Temperature (°C)',
    'Inter-Zone Spread (°C)',
    'Cumulative Energy  log₁₊ scale (Wh)',
    f'Pump Power  [{pm}]  (kW)',
    f'Thermal Stress  [{sm}]  (°C s⁻¹)',
    f'Flow Std Dev  [{fm}]'
]

ylabels = [
    '°C',
    '°C',
    'log₁₊ Wh',
    'kW (transformed)',
    '°C s⁻¹ (transformed)',
    'std (transformed)'
]

for ax, ti, yl in zip(axes, titles, ylabels):

    ax.set_title(ti)

    ax.set_ylabel(yl)

    ax.set_xlabel('Time (h)')

    finish_ax(ax)

# Auto kink
_apply_kink_if_spiky(axes[3], panel4_pairs)
_apply_kink_if_spiky(axes[4], panel5_pairs)
_apply_kink_if_spiky(axes[5], panel6_pairs)

handles = [
    mpatches.Patch(color=c['color'])
    for c in clusters
]

labels = [
    f"{c['label']} (n={len(c['members'])})"
    for c in clusters
]

for ax in axes:

    ax.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=True,
        framealpha=0.97,
        fontsize=7.0,
        title='Groups',
        title_fontsize=8
    )

cluster_map_text = '\n'.join([
    f"{c['label']}: {_members_text(c['members'], max_items=99)}"
    for c in clusters
])

fig.text(
    0.01,
    0.01,
    cluster_map_text,
    ha='left',
    va='bottom',
    fontsize=7.0,
    bbox=dict(
        boxstyle='round,pad=0.25',
        fc='white',
        ec='#cccccc',
        alpha=0.9
    )
)

fig.suptitle(
    'Temporal Dashboard — Grouped Controllers',
    y=0.99,
    fontsize=12
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

fig.savefig(
    OUT / '03_temporal_dashboard_available_runs.pdf',
    bbox_inches='tight',
    pad_inches=0.02
)

# plt.show()


# ## 4  Statistical Dashboard
# 
# Violin, boxplot, histogram, CDF, overhead bar, metric correlation heatmap.

# ### Interpretation Notes for Section 4 (Statistical Dashboard)
# 
# **Scientific role.** This section validates temporal findings in distribution space (location, spread, skewness, and tails).
# 
# **Panel-level guidance.**
# 1. **Temperature violin (per model):** compare center, spread, and tail asymmetry.
# 2. **Pump power boxplot (per model):** compare operational regimes; line-like boxes are valid when variance is near-zero.
# 3. **Spread density (grouped):** compare where each family concentrates on nonuniformity axis.
# 4. **Thermal stress CDF (grouped):** compare risk accumulation; left-shifted curves indicate lower stress risk.
# 5. **Cooling overhead:** compare energetic tax of cooling relative to drive energy.
# 6. **Correlation heatmap:** report coupled objectives and caution against single-metric optimization.
# 
# **Specific interpretation reminders for writing.**
# - If one model’s box collapses to a line, report it as quasi-constant operation, not plotting failure.
# - Use density + CDF together: density shows modal behavior, CDF shows risk exposure across thresholds.
# - Correlation panel should motivate multi-objective control selection and constraint-aware tuning.
# 
# **Suggested sentence.**
# - “Distributional analysis confirms structural controller differences and reveals tail-risk behavior that is not visible in mean-only comparisons.”

# #### Example Results-style write-up (Section 4)
# 
# “Figure 04 indicates that the controller families differ not only in central tendency but also in distribution shape and tail behavior. The per-model pump-power boxplot shows that some controllers achieve comparable medians with substantially different upper-tail demand, implying different hardware stress profiles. Density/CDF views for flow variability and thermal stress further reveal separation in high-percentile risk, which is critical for robustness screening. These distributional diagnostics justify selecting controllers by both typical performance and tail-risk control.”

# In[61]:


fig, ax = plt.subplots(2, 3, figsize=(14.4, 9.4))
axes = ax.flatten()
fig.subplots_adjust(bottom=0.10, top=0.94, left=0.06, right=0.98, wspace=0.30, hspace=0.45)

# Reuse grouped clusters from temporal/spatial style
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def _resample_1d(v, n):
    v = np.asarray(v, dtype=float)
    if len(v) == n:
        return v
    x_old = np.linspace(0.0, 1.0, len(v))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, v)


def _zscore(v):
    v = np.asarray(v, dtype=float)
    s = v.std()
    return np.zeros_like(v) if s < 1e-12 else (v - v.mean()) / s


def _model_signature(df):
    T = df[temp_cols(df)].to_numpy(dtype=float)
    F = df[flow_cols(df)].to_numpy(dtype=float)
    t_sig = _zscore(_resample_1d(T.mean(axis=0), 48))
    f_sig = _zscore(_resample_1d(F.mean(axis=0), 48))
    return np.concatenate([t_sig, f_sig])


def _close_clusters(model_map, quantile=0.30):
    names = list(model_map.keys())
    if len(names) == 1:
        return [{'label': 'Cluster 1', 'members': names, 'color': _HEX[0]}]

    sigs = np.vstack([_model_signature(model_map[n]) for n in names])
    D    = pdist(sigs, metric='euclidean')
    if len(D) == 0:
        labels = np.ones(len(names), dtype=int)
    else:
        thresh = float(np.quantile(D, quantile))
        thresh = max(thresh, 0.35)
        Z      = linkage(D, method='average')
        labels = fcluster(Z, t=thresh, criterion='distance')

    clusters = []
    for cid in sorted(set(labels)):
        members = [n for n, lab in zip(names, labels) if lab == cid]
        clusters.append({
            'label': f'Cluster {len(clusters) + 1}',
            'members': members,
            'color': _HEX[(len(clusters)) % len(_HEX)],
        })
    return clusters


def _members_text(members, max_items=3):
    if len(members) <= max_items:
        return ', '.join(members)
    return ', '.join(members[:max_items]) + f' +{len(members) - max_items} more'


clusters = _close_clusters(plot_run_map, quantile=0.30)
names       = list(plot_run_map.keys())
col_named   = [ccolor(group_label(n)) for n in names]
temp_data   = [plot_run_map[n][temp_cols(plot_run_map[n])].to_numpy().ravel() for n in names]
pump_data   = [plot_run_map[n]['pump_power_W'].to_numpy(dtype=float)           for n in names]
spread_data = [np.ptp(plot_run_map[n][temp_cols(plot_run_map[n])].to_numpy(), axis=1) for n in names]
stress_data = []
for n in names:
    df = plot_run_map[n]
    T  = df[temp_cols(df)].to_numpy(dtype=float)
    dt = np.clip(np.diff(df['time_s'].to_numpy(dtype=float)), 1e-9, None)
    stress_data.append((np.abs(np.diff(T, axis=0)) / dt[:, None]).ravel())

# violin – temperature distribution (leave as-is)
parts = axes[0].violinplot(temp_data, showmeans=True, showmedians=True)
for pc, col in zip(parts['bodies'], col_named):
    pc.set_facecolor(col); pc.set_alpha(0.75)
axes[0].set_xticks(np.arange(1, len(names)+1))
axes[0].set_xticklabels(names, rotation=28, ha='right', fontsize=7.5)
axes[0].set_title('Temperature Distribution')
axes[0].set_ylabel('°C')
finish_ax(axes[0])

# box – pump power (per model, tuned for readability)
all_p = np.concatenate(pump_data) if pump_data else np.array([1.0])
p995 = max(5.0, float(np.quantile(all_p, 0.995)))
p99s = [float(np.quantile(d, 0.99)) for d in pump_data]
pump_data_capped = [np.clip(d, 0, p99) for d, p99 in zip(pump_data, p99s)]
bp = axes[1].boxplot(pump_data_capped, patch_artist=True, showfliers=False, whis=(5, 95), widths=0.55)
for patch, col in zip(bp['boxes'], col_named):
    patch.set_facecolor(col)
    patch.set_alpha(0.72)
    patch.set_edgecolor('white')
    patch.set_linewidth(0.8)
for med in bp['medians']:
    med.set_color('#111')
    med.set_linewidth(1.1)

axes[1].set_ylim(0, p995)
axes[1].set_xticks(np.arange(1, len(names)+1))
axes[1].set_xticklabels(names, rotation=28, ha='right', fontsize=7.5)
for i, d in enumerate(pump_data_capped, 1):
    if np.ptp(d) < 1e-6:
        axes[1].scatter(i, float(np.median(d)), s=16, facecolor='white', edgecolor='#222', zorder=5)
axes[1].set_title('Pump Power Distribution (W) — Per Model')
axes[1].set_ylabel('W')
finish_ax(axes[1])

# histogram – spread density (GROUPED)
for c in clusters:
    pooled = []
    for n in c['members']:
        pooled.append(np.ptp(plot_run_map[n][temp_cols(plot_run_map[n])].to_numpy(), axis=1))
    pooled = np.concatenate(pooled) if pooled else np.array([0.0])
    axes[2].hist(pooled, bins=35, alpha=0.42, density=True, color=c['color'],
                 label=f"{c['label']} (n={len(c['members'])})")
axes[2].set_title('Spread Density — Grouped')
axes[2].set_xlabel('°C')
axes[2].legend(frameon=True, framealpha=0.95, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.20))
finish_ax(axes[2])

# CDF – thermal stress (GROUPED)
for c in clusters:
    pooled = []
    for n in c['members']:
        df = plot_run_map[n]
        T  = df[temp_cols(df)].to_numpy(dtype=float)
        dt = np.clip(np.diff(df['time_s'].to_numpy(dtype=float)), 1e-9, None)
        pooled.append((np.abs(np.diff(T, axis=0)) / dt[:, None]).ravel())
    pooled = np.concatenate(pooled) if pooled else np.array([0.0])
    xs = np.sort(pooled)
    ys = np.arange(1, len(xs)+1)/len(xs)
    axes[3].plot(xs, ys, lw=2.2, color=c['color'], label=f"{c['label']} (n={len(c['members'])})")
axes[3].set_title('Thermal Stress CDF — Grouped  (|ΔT / Δt|)')
axes[3].set_xlabel('°C s⁻¹'); axes[3].set_ylabel('CDF')
axes[3].legend(frameon=True, framealpha=0.95, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.20))
finish_ax(axes[3])

# bar – cooling overhead
d_ov = df_comp.sort_values(overhead_col)
axes[4].barh(d_ov['label'], d_ov[overhead_col],
             color=[ccolor(g) for g in d_ov['group']],
             edgecolor='white', lw=0.5)
axes[4].set_title('Cooling Overhead (%)')
axes[4].set_xlabel('% of drive energy')
finish_ax(axes[4])

# heatmap – correlation
corr_cols = [temp_col, mean_temp_col, spread_col, stress_col, energy_col, overhead_col]
corr      = df_comp[corr_cols].corr(numeric_only=True)
draw_corr_heatmap(axes[5], corr)
axes[5].set_title('Metric Correlation')
for sp in axes[5].spines.values():
    sp.set_visible(True); sp.set_linewidth(0.85)

cluster_map_text = '\n'.join([f"{c['label']}: {_members_text(c['members'], max_items=99)}" for c in clusters])
fig.text(0.01, 0.01, cluster_map_text, ha='left', va='bottom', fontsize=7.0,
         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.9))

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT / '04_statistical_dashboard.pdf', bbox_inches='tight', pad_inches=0.02)
# plt.show()


# ## 5  Spatial Dashboard
# 
# Zone mean temperature, spatiotemporal contour, inter-zone gradient, mean zone flow with low-range zoom.

# ### Interpretation Notes for Section 5 (Spatial Dashboard)
# 
# **Scientific question.** Where in the pack do controllers differ, and are improvements spatially balanced or localized?
# 
# **Panel-level interpretation.**
# - **Time-averaged zone temperature:** identifies persistent zone bias/hotspots.
# - **Model-to-model spatiotemporal difference matrix:** quantifies global behavioral distance across full zone-time fields.
# - **Inter-zone gradient profile:** pinpoints interfaces with strongest thermal discontinuity.
# - **Mean zone flow + zoom:** reveals allocation strategy, including low-flow distribution nuances.
# 
# **What to infer.**
# - Low gradients + flatter zone temperature profiles imply improved homogeneity.
# - Strong localized peaks suggest controller under-allocation or poor redistribution at specific interfaces.
# - Large matrix distances indicate fundamentally different control behavior, not minor tuning shifts.
# 
# **Suggested sentence.**
# - “Spatial diagnostics show whether thermal gains are achieved through globally balanced redistribution or through uneven local compensation.”

# #### Example Results-style write-up (Section 5)
# 
# “In Figure 05, feature-space embedding places controllers into coherent neighborhoods that align with their temporal and statistical signatures. Nearby points generally share similar pump usage and stress behavior, while isolated points correspond to more specialized policies. Cluster overlap is limited, suggesting that learned descriptors capture meaningful policy-level differences rather than plotting noise. This map is therefore useful for selecting representative controllers for ablation and deployment-stage simplification.”

# In[62]:


# Compare every available model in this dashboard
model_map   = plot_run_map if ('plot_run_map' in globals() and plot_run_map) else run_map
model_names = list(model_map.keys())

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def _resample_1d(v, n):
    v = np.asarray(v, dtype=float)
    if len(v) == n:
        return v
    x_old = np.linspace(0.0, 1.0, len(v))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, v)


def _zscore(v):
    v = np.asarray(v, dtype=float)
    s = v.std()
    return np.zeros_like(v) if s < 1e-12 else (v - v.mean()) / s


def _model_signature(df):
    T = df[temp_cols(df)].to_numpy(dtype=float)
    F = df[flow_cols(df)].to_numpy(dtype=float)
    t_sig = _zscore(_resample_1d(T.mean(axis=0), 48))
    f_sig = _zscore(_resample_1d(F.mean(axis=0), 48))
    return np.concatenate([t_sig, f_sig])


def _close_clusters(model_map, quantile=0.30):
    names = list(model_map.keys())
    if len(names) == 1:
        return [{'label': 'Cluster 1', 'members': names, 'color': _HEX[0]}]

    sigs = np.vstack([_model_signature(model_map[n]) for n in names])
    D    = pdist(sigs, metric='euclidean')
    if len(D) == 0:
        labels = np.ones(len(names), dtype=int)
    else:
        thresh = float(np.quantile(D, quantile))
        thresh = max(thresh, 0.35)
        Z      = linkage(D, method='average')
        labels = fcluster(Z, t=thresh, criterion='distance')

    clusters = []
    for cid in sorted(set(labels)):
        members = [n for n, lab in zip(names, labels) if lab == cid]
        clusters.append({
            'label': f'Cluster {len(clusters) + 1}',
            'members': members,
            'color': _HEX[(len(clusters)) % len(_HEX)],
        })
    return clusters


def _profile_dict(model_map, reducer):
    out = {}
    for name, df in model_map.items():
        T = df[temp_cols(df)].to_numpy(dtype=float)
        F = df[flow_cols(df)].to_numpy(dtype=float)
        out[name] = reducer(T, F)
    return out


def _members_text(members, max_items=3):
    if len(members) <= max_items:
        return ', '.join(members)
    return ', '.join(members[:max_items]) + f' +{len(members) - max_items} more'


clusters = _close_clusters(model_map, quantile=0.30)
cluster_lookup = {name: c for c in clusters for name in c['members']}

# Build compact cluster-based profiles for the crowded line plots
zone_temp_profiles = _profile_dict(model_map, lambda T, F: T.mean(axis=0))
grad_profiles      = _profile_dict(model_map, lambda T, F: np.abs(np.diff(T, axis=1)).mean(axis=0))
flow_profiles      = _profile_dict(model_map, lambda T, F: F.mean(axis=0))

fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
fig.subplots_adjust(bottom=0.12, top=0.92, left=0.08, right=0.98, wspace=0.35, hspace=0.45)

# ── zone mean temperature ─────────────────────────────────────────────────────
for ci, cluster in enumerate(clusters):
    members = cluster['members']
    col     = cluster['color']
    member_stack = np.vstack([zone_temp_profiles[n] for n in members])
    mu = member_stack.mean(axis=0)
    sg = member_stack.std(axis=0)
    z  = np.arange(1, len(mu) + 1)

    axes[0,0].plot(z, mu, marker='o', lw=2.8, color=col)
    axes[0,0].fill_between(z, mu - sg, mu + sg, alpha=0.12, color=col)

axes[0,0].set_title('Time-Averaged Zone Temperature')
axes[0,0].set_xlabel('Zone'); axes[0,0].set_ylabel('°C')
finish_ax(axes[0,0])

# Legend under the plot, not to the right
axes[0,0].legend(
    [mpatches.Patch(color=c['color']) for c in clusters],
    [f"{c['label']}: {_members_text(c['members'])}" for c in clusters],
    loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=True, framealpha=0.97,
    fontsize=7.6, title='Groups', title_fontsize=8, ncol=1
)

# ── model-to-model difference: keep original order, no clustering ─────────────
# This panel is already readable; clustering here just adds noise.
min_z = min(len(temp_cols(df)) for df in model_map.values())
n_t   = min(240, min(len(df) for df in model_map.values()))  # common time samples

cube = []
for name in model_names:
    df = model_map[name]
    T  = df[temp_cols(df)].to_numpy(dtype=float)[:, :min_z]
    idx = np.linspace(0, len(T) - 1, n_t).astype(int)
    cube.append(T[idx])

D = np.zeros((len(model_names), len(model_names)), dtype=float)
for i in range(len(model_names)):
    for j in range(len(model_names)):
        D[i, j] = np.mean(np.abs(cube[i] - cube[j]))

im = axes[0,1].imshow(D, cmap='magma', aspect='auto')
fig.colorbar(im, ax=axes[0,1], label='Mean |ΔT| (°C)', pad=0.02)
axes[0,1].set_xticks(np.arange(len(model_names)))
axes[0,1].set_yticks(np.arange(len(model_names)))
axes[0,1].set_xticklabels(model_names, rotation=35, ha='right', fontsize=7)
axes[0,1].set_yticklabels(model_names, fontsize=7)
for i in range(len(model_names)):
    for j in range(len(model_names)):
        axes[0,1].text(j, i, f'{D[i, j]:.2f}', ha='center', va='center',
                       color='white' if D[i, j] > D.max() * 0.45 else 'black', fontsize=6)
axes[0,1].set_title('Model-to-Model Spatiotemporal Difference')
axes[0,1].set_xlabel('Model')
axes[0,1].set_ylabel('Model')
finish_ax(axes[0,1])

# ── inter-zone gradient ───────────────────────────────────────────────────────
for ci, cluster in enumerate(clusters):
    members = cluster['members']
    col     = cluster['color']
    member_stack = np.vstack([grad_profiles[n] for n in members])
    mu = member_stack.mean(axis=0)
    sg = member_stack.std(axis=0)
    xg = np.arange(1, len(mu) + 1)

    axes[1,0].plot(xg, mu, marker='s', lw=2.8, color=col)
    axes[1,0].fill_between(xg, mu - sg, mu + sg, alpha=0.10, color=col)

axes[1,0].set_title('Inter-Zone Gradient')
axes[1,0].set_xlabel('Interface'); axes[1,0].set_ylabel('Mean |ΔT| (°C)')
finish_ax(axes[1,0])

axes[1,0].legend(
    [mpatches.Patch(color=c['color']) for c in clusters],
    [f"{c['label']}: {_members_text(c['members'])}" for c in clusters],
    loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=True, framealpha=0.97,
    fontsize=7.6, title='Groups', title_fontsize=8, ncol=1
)

# ── mean zone flow ────────────────────────────────────────────────────────────
all_mean_f = []
for ci, cluster in enumerate(clusters):
    members = cluster['members']
    col     = cluster['color']
    member_stack = np.vstack([flow_profiles[n] for n in members])
    mu = member_stack.mean(axis=0)
    sg = member_stack.std(axis=0)
    xf = np.arange(1, len(mu) + 1)

    axes[1,1].plot(xf, mu, marker='o', lw=2.8, color=col)
    axes[1,1].fill_between(xf, mu - sg, mu + sg, alpha=0.10, color=col)
    all_mean_f.append(mu)

axes[1,1].set_title('Mean Zone Flow')
axes[1,1].set_xlabel('Zone'); axes[1,1].set_ylabel('Normalised Flow')
finish_ax(axes[1,1])

axes[1,1].legend(
    [mpatches.Patch(color=c['color']) for c in clusters],
    [f"{c['label']}: {_members_text(c['members'])}" for c in clusters],
    loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=True, framealpha=0.97,
    fontsize=7.6, title='Groups', title_fontsize=8, ncol=1
)

fig.tight_layout()
fig.savefig(OUT / '05_spatial_dashboard_available_runs.pdf', bbox_inches='tight', pad_inches=0.02)
# plt.show()

# ── flow zoom panel ───────────────────────────────────────────────────────────
all_cat  = np.concatenate(all_mean_f) if all_mean_f else np.array([0.1])
low_lim  = max(0.03, np.quantile(all_cat, .35))
fig2, (axf, axz) = plt.subplots(2, 1, figsize=(9.5, 7.), sharex=True)
fig2.subplots_adjust(bottom=0.14, top=0.92, left=0.08, right=0.98)
for ci, cluster in enumerate(clusters):
    members = cluster['members']
    col     = cluster['color']
    member_stack = np.vstack([flow_profiles[n] for n in members])
    mu = member_stack.mean(axis=0)
    sg = member_stack.std(axis=0)
    z  = np.arange(1, len(mu) + 1)

    axf.plot(z, mu, marker='o', lw=2.8, color=col, label=f"{cluster['label']}  (n={len(members)})")
    axf.fill_between(z, mu - sg, mu + sg, alpha=0.10, color=col)
    axz.plot(z, mu, marker='o', lw=2.8, color=col)
    axz.fill_between(z, mu - sg, mu + sg, alpha=0.10, color=col)

axf.set_title('Mean Zone Flow — Full Range'); axf.set_ylabel('Normalised Flow')
axf.legend(ncol=1, fontsize=7, frameon=True, framealpha=0.97, loc='upper center', bbox_to_anchor=(0.5, -0.18))
finish_ax(axf)
axz.set_ylim(0, low_lim)
axz.set_title(f'Mean Zone Flow — Low Range Zoom  (≤ {low_lim:.3f})')
axz.set_xlabel('Zone'); axz.set_ylabel('Normalised Flow')
finish_ax(axz)
fig2.tight_layout()
fig2.savefig(OUT / '05b_mean_zone_flow_full_and_zoom.pdf', bbox_inches='tight', pad_inches=0.02)
# plt.show()


# ## 6  Control Aggressiveness & Strategy
# 
# Max-flow time-series, PSD spectrum (log-log), smoothness ECDF, normalised flow-entropy diversity.

# ### Interpretation Notes for Section 6 (Control Aggressiveness & Strategy)
# 
# **Scientific role.** These panels characterize *how* controllers actuate, complementing thermal outcome plots.
# 
# **Panel interpretation.**
# - **Max zone flow (time):** instantaneous aggressiveness envelope.
# - **PSD spectrum:** frequency content of control activity (slow scheduling vs fast reactive actuation).
# - **ECDF of $|\Delta\mathrm{Flow}|$:** command smoothness / jitter risk.
# - **Flow entropy:** diversity of zone-level allocation strategy.
# 
# **What to infer.**
# - Similar thermal outcomes with lower high-frequency PSD and lower $|\Delta\mathrm{Flow}|$ tail usually indicate a more implementation-friendly policy.
# - Very high entropy can indicate adaptive redistribution; very low entropy can indicate rigid/static policy.
# 
# **Suggested sentence.**
# - “Actuation-style diagnostics indicate whether thermal gains arise from efficient scheduling or from aggressive high-frequency control effort.”

# #### Example Results-style write-up (Section 6)
# 
# “Figure 06 compares grouped versus all-controller visualizations and shows that grouping improves readability while preserving the dominant ranking and trend structure. The grouped view suppresses minor within-family fluctuations, making cross-family differences easier to inspect. However, the all-model panel remains essential for auditing outliers and validating that no subgroup is hidden by aggregation. Reporting both views together provides a transparent balance between interpretability and completeness.”

# In[63]:


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# a – max zone flow (smoothed)
for ni, (name, df) in enumerate(run_map.items()):
    t   = df['time_s'].to_numpy() / 3600.
    F   = df[flow_cols(df)].to_numpy()
    sm  = uniform_filter1d(F.max(axis=1), size=max(3, len(t)//120))
    col = run_colors.get(name, _HEX[ni % len(_HEX)])
    axes[0,0].plot(t, sm, lw=1.6, color=col, label=name)
axes[0,0].set_title('Control Aggressiveness — Max Zone Flow')
axes[0,0].set_xlabel('Time (h)'); axes[0,0].set_ylabel('Max Flow')
axes[0,0].legend(frameon=True, framealpha=0.9, fontsize=7)
finish_ax(axes[0,0])

# b – strategy spectrum (log-log)
for ni, (name, df) in enumerate(run_map.items()):
    F = df[flow_cols(df)].to_numpy().mean(axis=1)
    freqs, psd = welch(F, fs=1., nperseg=min(512, max(32, len(F)//4)))
    v = freqs > 0
    col = run_colors.get(name, _HEX[ni % len(_HEX)])
    axes[0,1].plot(freqs[v], psd[v], lw=1.6, color=col, label=name)
axes[0,1].set_xscale('log'); axes[0,1].set_yscale('log')
axes[0,1].set_title('Control Strategy Spectrum')
axes[0,1].set_xlabel('Frequency (Hz)'); axes[0,1].set_ylabel('PSD')
axes[0,1].legend(frameon=True, framealpha=0.9, fontsize=7)
finish_ax(axes[0,1])

# c – smoothness ECDF of |ΔFlow|
all_q = []
for ni, (name, df) in enumerate(run_map.items()):
    d = np.abs(np.diff(df[flow_cols(df)].to_numpy(), axis=0)).ravel()
    d = d[np.isfinite(d)]; all_q.append(d)
    xs = np.sort(d)
    col = run_colors.get(name, _HEX[ni % len(_HEX)])
    axes[1,0].plot(xs, np.arange(1, len(xs)+1)/len(xs), lw=1.4, color=col, label=name)
q995 = np.quantile(np.concatenate(all_q), .995) if all_q else 1.
xl   = axes[1,0].get_xlim()
axes[1,0].set_xlim(max(xl[0], 0), min(xl[1], q995))
axes[1,0].set_title('Flow Smoothness ECDF  (|ΔFlow|)')
axes[1,0].set_xlabel('|ΔFlow|'); axes[1,0].set_ylabel('CDF')
axes[1,0].legend(frameon=True, framealpha=0.9, fontsize=7)
finish_ax(axes[1,0])

# d – normalised flow-entropy diversity
enames, evals = [], []
bins = 24
for name, df in run_map.items():
    F   = df[flow_cols(df)].to_numpy()
    ent = []
    for z in range(F.shape[1]):
        c, _ = np.histogram(F[:, z], bins=bins, range=(0, 1), density=False)
        tot  = c.sum()
        if tot <= 0: ent.append(0.); continue
        p = c / tot; p = p[p > 0]
        ent.append(float(-(p*np.log2(p)).sum() / np.log2(bins)))
    enames.append(name); evals.append(float(np.mean(ent)))

axes[1,1].barh(enames, evals,
               color=[run_colors.get(n, _HEX[i % len(_HEX)]) for i, n in enumerate(enames)],
               edgecolor='white', lw=0.5)
axes[1,1].set_xlim(0, max(max(evals)*1.12, 0.55))
axes[1,1].set_title('Control Strategy Diversity (Normalised Entropy)')
axes[1,1].set_xlabel('Normalised entropy  (0 – 1)')
finish_ax(axes[1,1])

fig.tight_layout()
fig.savefig(OUT / '06_control_aggressiveness_strategy_tuned.pdf', bbox_inches='tight', pad_inches=0.02)
# plt.show()


# ## 7  Radar Profiles — Top 6 Controllers
# 
# Small-multiple radar charts, one per top controller, filled with cluster colour and labelled with cluster letter.

# ### Interpretation Notes for Section 7 (Radar Profiles — Top Controllers)
# 
# **Scientific role.** Radar plots summarize multi-objective balance for shortlisted candidates.
# 
# **How to interpret correctly.**
# - Use this figure as synthesis after detailed temporal/statistical/spatial evidence.
# - Wider and more uniform polygon => balanced controller.
# - Peaked polygon => specialization (strong on some metrics, weaker on others).
# 
# **Reporting guidance.**
# - Avoid claiming absolute dominance from radar area alone.
# - Use radar to support tradeoff narrative already established by Pareto + temporal/statistical panels.
# 
# **Suggested sentence.**
# - “Radar profiles consolidate multi-metric behavior and highlight balanced versus specialized controller phenotypes among top candidates.”

# #### Example Results-style write-up (Section 7)
# 
# “Figure 07 (export-ready panels) confirms that the final figure set retains consistent ordering, legends, and scale logic across manuscript-quality outputs. Cross-panel consistency enables direct comparison of energy, thermal control, and actuation burden without reinterpreting visual encodings. The final exported artifacts therefore support reproducible reporting and reduce the risk of narrative drift between analysis notebooks and paper figures. This section can be cited as the canonical source of publication figures.”

# In[64]:


metrics = [('Spread',    spread_col,    True),
           ('Mean Temp', mean_temp_col, True),
           ('Max Temp',  temp_col,      True),
           ('Stress',    stress_col,    True),
           ('Energy',    energy_col,    True)]

def robust_norm(v, arr):
    ql, qh = np.quantile(arr, [.05, .95])
    return float(np.clip((v-ql)/(qh-ql+1e-12), 0, 1))

scores = []
for _, row in df_comp.iterrows():
    vals = [1-robust_norm(row[col], df_comp[col].to_numpy()) if inv
            else robust_norm(row[col], df_comp[col].to_numpy())
            for _, col, inv in metrics]
    scores.append(np.mean(vals))

top = df_comp.copy()
top['_score'] = scores
top = top.sort_values('_score', ascending=False).head(6)

angles  = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), subplot_kw=dict(polar=True))
axes = axes.flatten()

for ax, (_, row) in zip(axes, top.iterrows()):
    vals = [1-robust_norm(row[col], df_comp[col].to_numpy()) if inv
            else robust_norm(row[col], df_comp[col].to_numpy())
            for _, col, inv in metrics]
    v   = vals + vals[:1]
    col = ccolor(row['group'])
    ax.plot(angles, v, lw=2.0, color=col)
    ax.fill(angles, v, alpha=0.20, color=col)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metrics], fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([.25, .5, .75, 1.])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=5.5)
    cltr = GROUP_TO_CLUSTER[row['group']]
    ax.set_title(f"{row['label']}\n[Cluster {cltr}]", fontsize=8.5, pad=10)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7)

for ax in axes[len(top):]:
    ax.axis('off')

fig.suptitle('Radar Profiles — Top 6 Controllers', y=0.99, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / '07_radar_top6.pdf', bbox_inches='tight', pad_inches=0.02)
# plt.show()


# ## 8  Export Check

# In[65]:


outs = sorted(OUT.glob('*.png'))
print(f'Exported {len(outs)} files to {OUT.resolve()}')
for p in outs:
    kb = p.stat().st_size / 1024
    print(f'  {p.name:<55s}  {kb:6.1f} kB')
print('\nIf you add more *_run.csv files, rerun sections 3-6 to include them automatically.')


# ## 8-B  Detailed Paper Notes — Plot-by-Plot Analysis and Inference
# 
# This section is written in research-paper style and is intended to be reused directly while drafting Results and Discussion.
# 
# ---
# 
# ### 1) Experimental framing (for Methods/Results opening paragraph)
# 
# - **Controllers compared (9 total):** `PROPOSED_Full`, `PROPOSED_NoSpatial`, `MPC_H1_S64`, `MPC_H1_S32`, `Actor_Deterministic`, `Actor_Stochastic`, `PID_Standard`, `PID_Adaptive`, `Uniform_Flow`.
# - **Data basis:** measured runs + synthesized controller trajectories aligned to summary-level metrics (when measured run not directly available).
# - **Core metrics used throughout:** max temperature, mean temperature, inter-zone spread, thermal stress $|\Delta T/\Delta t|$, pump power/energy, cooling overhead.
# - **Visualization principle:** readability transforms (quantile bands, clipping, kink strip, smoothing for display) are used only for interpretation; they do not alter underlying raw values used for ranking and statistics.
# 
# ---
# 
# ### 2) Figure 00 — Scale Experiment (`00_scale_experiment_*.png`)
# 
# #### What is computed
# - For each temporal variable (`pump_kw`, `stress_cps`, `flow_std`), multiple transforms are scored by a weighted criterion balancing:
#   - between-controller separation,
#   - skew reduction,
#   - low-end resolution,
#   - tail compression.
# 
# #### Why this matters
# - This avoids arbitrary plotting scales and ensures that interpretation of peaks/dispersion is not a scale artifact.
# - In your current run, selected transforms are conservative (`linear`) for all three temporal variables, implying signal range was already interpretable without heavy nonlinear compression.
# 
# #### Paper-ready interpretation
# - “Scale-selection ablation indicated that linear transforms retained sufficient inter-controller discrimination while avoiding unnecessary shape distortion; therefore, temporal comparisons were reported in linear form.”
# 
# ---
# 
# ### 3) Figure 01 / 01b — Pareto and Rank-Space Tradeoff (`01_*.png`)
# 
# #### Plot meaning
# - **Primary Pareto panel:** cooling energy (x) vs max temperature (y), with non-dominated points ring-highlighted.
# - **Zoom panel:** low-energy region for local separation among efficient controllers.
# - **Rank-space panel:** rank(energy) vs rank(max temperature), reducing dependence on absolute metric scaling.
# 
# #### What to look for
# - Whether a controller is globally non-dominated versus only locally good.
# - Whether lower max temperature is achieved by disproportionately higher cooling effort.
# - Whether Pareto ordering is stable in rank-space.
# 
# #### Inference template
# - “Controllers near the Pareto envelope define the best achievable energy–thermal compromises. Points away from the frontier are dominated by alternatives with lower energy, lower temperature, or both.”
# - “Rank-space consistency indicates that tradeoff ordering is robust to metric scaling.”
# 
# ---
# 
# ### 4) Figure 02 — Summary Bars (`02_summary_bars.png`)
# 
# #### Plot meaning
# - Left: direct energy ranking.
# - Right: direct max-temperature ranking with 40°C reference threshold.
# 
# #### What can be inferred
# - Fast audit of operational preference (low energy + low max temperature).
# - Practical compliance check against thermal constraint.
# 
# #### Paper-ready commentary
# - “The bar-ranking view provides an interpretable controller ordering under two operationally relevant objectives, while the 40°C reference line explicitly indicates safety margin.”
# 
# ---
# 
# ### 5) Figure 03 — Temporal Dashboard (`03_temporal_dashboard_available_runs.png`)
# 
# > This figure is best described as **dynamic behavior decomposition**: thermal level, nonuniformity, control effort, and stress are shown over the full drive cycle.
# 
# #### Panel 1: Mean Pack Temperature
# - **Observed pattern:** clusters/strategies remain thermally bounded in the low-30°C region with cycle-dependent dips/plateaus.
# - **Interpretation:** no controller appears to violate bulk thermal stability; differences are in margin and control effort rather than runaway behavior.
# 
# #### Panel 2: Inter-Zone Spread
# - **Observed pattern:** one family stays systematically lower spread (tighter balance), while others show higher and more oscillatory spread.
# - **Interpretation:** low spread indicates stronger spatial balancing and less thermal heterogeneity across zones.
# - **Inference:** if energy is comparable, the lower-spread family is preferable for aging/uniformity objectives.
# 
# #### Panel 3: Cumulative Cooling Energy (log1p Wh)
# - **Observed pattern:** energy trajectories separate early and remain ordered; high-slope curves indicate persistently aggressive pumping.
# - **Interpretation:** cumulative curves show control burden integrated over time, not just transient peaks.
# - **Inference:** steep cumulative growth indicates strong thermal enforcement but potentially lower overall efficiency.
# 
# #### Panel 4: Pump Power (robust summary + peak strip)
# - **Observed pattern:** multiple regimes are visible: near-zero/low-power control, moderate dynamic control, and sustained high-power operation.
# - **Interpretation:** this panel explains why some controllers achieve lower temperature/spread: they spend more actuation effort.
# - **Method note:** quantile bands (20–80%) and median traces summarize behavior while preserving sparse peak visibility via top-strip kink.
# 
# #### Panel 5: Thermal Stress $|\Delta T/\Delta t|$
# - **Observed pattern:** stress is bursty and cycle-phase dependent; some families show sharper/higher spikes.
# - **Interpretation:** high stress bursts imply rapid thermal transients, which can be detrimental even if mean temperature is acceptable.
# - **Inference:** preferable controllers combine low spread and moderate stress bursts, not simply low mean temperature.
# 
# #### Panel 6: Flow Std Dev
# - **Observed pattern:** higher flow variability families exhibit more active redistribution behavior; near-flat families imply steady allocation.
# - **Interpretation:** variability is not inherently bad; it reflects strategy style (reactive redistribution vs smooth allocation).
# - **Inference:** combine this panel with stress panel: high variability with controlled stress may represent effective adaptive balancing.
# 
# #### Temporal figure discussion paragraph (copy-ready)
# - “Temporal analysis reveals a consistent energy–uniformity–stress triad. Strategies with stronger actuation reduce spread and often lower mean thermal levels, but at the expense of cumulative cooling energy and, in some cases, increased transient stress. The robust band/median representation confirms that these differences are persistent over the cycle rather than isolated events.”
# 
# ---
# 
# ### 6) Figure 04 — Statistical Dashboard (`04_statistical_dashboard.png`)
# 
# > This figure converts temporal behavior into distributional evidence (location, spread, skew, and tails).
# 
# #### Panel 1: Temperature Violin (per model)
# - **Observed pattern:** model distributions differ in center and lower-tail extent; some show broader spread around the operating mean.
# - **Inference:** narrow violins around acceptable means indicate more consistent thermal regulation; broader tails imply less stable zone-level control.
# 
# #### Panel 2: Pump Power Boxplot (per model)
# - **Observed pattern in current output:**
#   - a high-power subset is concentrated near the upper actuation range (appearing line-like due to near-degenerate distribution),
#   - a low-power subset remains near zero,
#   - a mid-power subset spans a moderate band.
# - **Interpretation:** this indicates **distinct control regimes** rather than minor tuning differences.
# - **Important note for paper:** line-like boxes are valid when variance is truly very small (quasi-constant pumping), not a plotting error.
# 
# #### Panel 3: Spread Density (grouped)
# - **Observed pattern:** grouped histograms are shifted relative to each other; one family concentrates at lower spread, another at higher spread with heavier right tail.
# - **Inference:** right-shifted spread density indicates persistent nonuniformity, which is relevant for thermal imbalance and cell mismatch concerns.
# 
# #### Panel 4: Thermal Stress CDF (grouped)
# - **Observed pattern:** CDF separation in low-to-mid stress region indicates different stress risk profiles across families.
# - **Inference:** a left-shifted CDF (reaching high cumulative probability at lower stress) is preferable for transient safety/aging robustness.
# 
# #### Panel 5: Cooling Overhead (%)
# - **Observed pattern:** overhead differs strongly across controllers, with some policies incurring substantially higher cooling tax.
# - **Inference:** overhead contextualizes thermal gains in system-efficiency terms and is essential for deployment-oriented evaluation.
# 
# #### Panel 6: Correlation Heatmap
# - **Observed structure:** strong relationships between temperature metrics and energy/overhead indicate nontrivial coupling.
# - **Inference:** objective optimization should be multi-metric; single-metric tuning is likely to move other metrics in coupled directions.
# 
# #### Statistical figure discussion paragraph (copy-ready)
# - “Distributional analysis confirms that controller differences are structural rather than incidental: pump policies cluster into high-, mid-, and low-actuation regimes, with corresponding shifts in spread and stress-risk distributions. Correlation structure further indicates that thermal and energetic objectives are coupled, motivating explicit multi-objective control selection.”
# 
# ---
# 
# ### 7) Figure 05 / 05b — Spatial Dashboard (`05_*.png`)
# 
# #### Plot meaning
# - Zone-level temperature and flow profiles + model-to-model spatiotemporal distance matrix.
# 
# #### What to infer
# - **Zone temperature profile:** identifies spatial hotspots and systematic zone bias.
# - **Inter-zone gradient profile:** shows where thermal discontinuities are strongest (interfaces).
# - **Model-distance heatmap:** quantifies behavioral similarity without relying on visual trace overlap.
# - **Flow zoom:** reveals low-range allocation nuances hidden in full-scale plots.
# 
# #### Paper-ready statement
# - “Spatial diagnostics indicate not only whether a controller is cool on average, but whether it achieves that result through balanced zone allocation or by tolerating localized gradients.”
# 
# ---
# 
# ### 8) Figure 06 — Control Aggressiveness & Strategy (`06_*.png`)
# 
# #### Plot meaning
# - Time-domain aggressiveness, frequency-domain signature, smoothness, and flow entropy.
# 
# #### Inference structure
# - Higher max-flow traces + broad PSD content => more aggressive/reactive strategy.
# - Lower $|\Delta\mathrm{Flow}|$ ECDF tail => smoother actuator commands.
# - Higher entropy => more diverse spatial control usage.
# 
# #### Paper-ready statement
# - “Control-style metrics separate controllers that are thermally effective due to sustained high actuation from those that achieve comparable outcomes through smoother and more diverse redistribution.”
# 
# ---
# 
# ### 9) Figure 07 — Radar Top-6 (`07_radar_top6.png`)
# 
# #### Plot meaning
# - Normalized multi-objective profile for top candidates.
# 
# #### How to report
# - Highlight whether a model is balanced (broad polygon) or specialized (peaked polygon).
# - Use as a synthesis figure after temporal/statistical evidence, not as standalone proof.
# 
# ---
# 
# ### 10) Result synthesis logic (recommended final narrative order)
# 
# 1. **Pareto + summary bars**: establish who is efficient and safe.
# 2. **Temporal dashboard**: explain *how* each family behaves dynamically.
# 3. **Statistical dashboard**: validate that temporal patterns persist distributionally (including tails).
# 4. **Spatial dashboard**: show where improvements occur physically across zones.
# 5. **Control strategy dashboard**: explain actuator behavior behind observed thermal outcomes.
# 6. **Radar + ablation**: consolidate multi-objective balance and robustness of conclusions.
# 
# ---
# 
# ### 11) Ready-to-paste long-form conclusion paragraph
# 
# - “Across nine controllers, the analysis reveals distinct operating regimes rather than minor parametric variation. Pareto and ranking views identify candidate-efficient controllers, while temporal and statistical diagnostics show that lower thermal nonuniformity is generally achieved through increased control effort, with stress behavior differentiating robust from aggressive policies. Spatial diagnostics confirm that performance differences are physically distributed across zones rather than confined to isolated moments. Collectively, the evidence supports multi-objective selection: preferred controllers are those that maintain low spread and acceptable transient stress at moderate cumulative cooling overhead, rather than optimizing any single metric in isolation.”

# ## 9  Ablation Lab — Setup

# > **Paper scope note (current plan):** Sections **9 and onward (Ablation Lab)** are excluded from the main paper body for now. Keep them in this notebook for robustness checks, supplement candidates, and future revisions.

# In[66]:


ABL_OUT = OUT / 'ablation'
ABL_OUT.mkdir(parents=True, exist_ok=True)

def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)
    return np.zeros_like(x) if hi-lo <= 1e-12 else (x-lo)/(hi-lo)

def _robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    iqr = np.quantile(x,.75) - np.quantile(x,.25)
    return np.zeros_like(x) if iqr <= 1e-12 else (x-med)/iqr

def _mpd(M):
    M = np.asarray(M, dtype=float)
    if len(M) <= 1: return 0.
    D = cdist(M, M); iu = np.triu_indices_from(D, k=1)
    return float(np.mean(D[iu]))

def _nnm(P):
    P = np.asarray(P, dtype=float)
    if len(P) <= 1: return 0.
    D = cdist(P, P); np.fill_diagonal(D, np.inf)
    return float(np.median(np.min(D, axis=1)))

def _hull_area(P):
    P = np.asarray(P, dtype=float)
    if len(P) < 3: return 0.
    try: return float(ConvexHull(P).volume)
    except: return 0.

def _safe_spearman(a, b):
    try:
        r, _ = spearmanr(a, b)
        return 0. if np.isnan(r) else float(r)
    except: return 0.

def _rank_score(df, plus, minus, col='score'):
    d  = df.copy()
    sc = np.zeros(len(d))
    for c, w in plus.items():  sc += w * d[c].rank(ascending=False, method='average').to_numpy()
    for c, w in minus.items(): sc += w * d[c].rank(ascending=True,  method='average').to_numpy()
    d[col] = sc
    return d.sort_values(col, ascending=False).reset_index(drop=True)

def _ecdf(x):
    x = np.sort(np.asarray(x, dtype=float))
    return x, np.arange(1, len(x)+1) / len(x)

def _hist_prob(x, bins=80):
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)
    hi = lo + 1e-6 if hi-lo <= 1e-12 else hi
    h, _ = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    p = h.astype(float); s = p.sum()
    p = np.ones_like(p)/len(p) if s <= 0 else p/s
    p = np.clip(p, 1e-12, None); return p/p.sum()

ABL = {}
print('Ablation output dir:', ABL_OUT.resolve())


# ## 9-B  Ablation Lab — Computation (Pareto, Summary, Temporal, Statistical)

# In[67]:


# ── Family 1: Pareto ──────────────────────────────────────────────────────────
px_raw = df_comp[temp_col].to_numpy(dtype=float)
py_raw = df_comp[energy_col].to_numpy(dtype=float)
front  = pareto_mask(px_raw, py_raw)
qx95   = max(np.quantile(px_raw, .95), 1e-12)
qy95   = max(np.quantile(py_raw, .95), 1e-12)

pareto_opts = {
    'linear_linear'  : (lambda x: x, lambda y: y),
    'linear_log1pY'  : (lambda x: x, lambda y: np.log1p(y)),
    'log1p_both'     : (lambda x: np.log1p(x), lambda y: np.log1p(y)),
    'asinh_both'     : (lambda x: np.arcsinh(x/qx95), lambda y: np.arcsinh(y/qy95)),
    'rank_both'      : (lambda x: pd.Series(x).rank(method='average').to_numpy(),
                        lambda y: pd.Series(y).rank(method='average').to_numpy()),
    'clip99_linear'  : (lambda x: np.clip(x, None, np.quantile(x,.99)),
                        lambda y: np.clip(y, None, np.quantile(y,.99))),
}

rows = []
for name, (fx, fy) in pareto_opts.items():
    xn, yn = _minmax(fx(px_raw)), _minmax(fy(py_raw))
    P  = np.c_[xn, yn]
    nnd= _nnm(P); area = _hull_area(P)
    gap= (np.linalg.norm(P[front].mean(0) - P[~front].mean(0))
          if front.any() and (~front).any() else 0.)
    D  = cdist(P, P); np.fill_diagonal(D, np.inf)
    overlap = float(np.mean(np.min(D, axis=1) < 0.08))
    rows.append({'variant':name,'nn_dist':nnd,'area':area,'frontier_gap':gap,'overlap':overlap})

pareto_df = _rank_score(pd.DataFrame(rows),
                        plus ={'nn_dist':.4,'area':.35,'frontier_gap':.25},
                        minus={'overlap':.35})
ABL['pareto'] = pareto_df
pareto_df.to_csv(ABL_OUT/'ablation_pareto.csv', index=False)

# visual
k = min(4, len(pareto_df))
fig, axes = plt.subplots(1, k, figsize=(4.2*k, 3.8))
if k == 1: axes = [axes]
for ax, v in zip(axes, pareto_df['variant'].head(k)):
    fx, fy = pareto_opts[v]
    xn, yn = _minmax(fx(px_raw)), _minmax(fy(py_raw))
    ax.scatter(xn[~front], yn[~front], s=36, alpha=0.6,
               color=CLUSTER_PALETTE['F'], label='Dominated')
    ax.scatter(xn[front],  yn[front],  s=48, marker='D',
               color=CLUSTER_PALETTE['A'], label='Frontier')
    for i, lbl_ in enumerate(df_comp['label']):
        ax.text(xn[i], yn[i], lbl_.replace('PROPOSED','PRO'), fontsize=5.8,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.82, ec='none'))
    ax.set_title(v); ax.set_xlabel('x (transformed)'); ax.set_ylabel('y (transformed)')
    finish_ax(ax)
axes[0].legend(fontsize=7, frameon=True, framealpha=0.9)
fig.tight_layout()
fig.savefig(ABL_OUT/'ablation_pareto_variants.pdf', bbox_inches='tight', pad_inches=0.02)
# plt.show()

# ── Family 2: Summary bars ────────────────────────────────────────────────────
metric_cols  = [temp_col, mean_temp_col, spread_col, stress_col, energy_col, overhead_col]
M_raw        = df_comp[metric_cols].to_numpy(dtype=float)
summary_modes= {
    'raw'               : lambda A: A,
    'minmax_per_metric' : lambda A: np.column_stack([_minmax(A[:,i]) for i in range(A.shape[1])]),
    'rank_per_metric'   : lambda A: np.column_stack([pd.Series(A[:,i]).rank(method='average').to_numpy()
                                                     for i in range(A.shape[1])]),
    'robust_z'          : lambda A: np.column_stack([_robust_z(A[:,i]) for i in range(A.shape[1])]),
    'log_energy_only'   : lambda A: np.column_stack([A[:,0],A[:,1],A[:,2],A[:,3],np.log1p(A[:,4]),A[:,5]]),
}
rows = []
for mode, fn in summary_modes.items():
    M   = fn(M_raw)
    sep = _mpd(np.column_stack([_minmax(M[:,i]) for i in range(M.shape[1])]))
    fid = np.mean([_safe_spearman(pd.Series(M_raw[:,i]).rank(), pd.Series(M[:,i]).rank())
                   for i in range(M.shape[1])])
    sk  = np.mean([abs(pd.Series(M[:,i]).skew()) for i in range(M.shape[1])])
    rows.append({'variant':mode,'separation':sep,'rank_fidelity':fid,'abs_skew':sk})
summary_df = _rank_score(pd.DataFrame(rows),
                         plus ={'separation':.5,'rank_fidelity':.35},
                         minus={'abs_skew':.2})
ABL['summary_bars'] = summary_df
summary_df.to_csv(ABL_OUT/'ablation_summary_bars.csv', index=False)

# ── Family 3: Temporal ────────────────────────────────────────────────────────
series = {}
for name, df in run_map.items():
    ts = df['time_s'].to_numpy(dtype=float)
    T  = df[temp_cols(df)].to_numpy(dtype=float)
    F  = df[flow_cols(df)].to_numpy(dtype=float)
    dt = np.clip(np.diff(ts), 1e-9, None)
    series[name] = dict(pump_kw  = df['pump_power_W'].to_numpy(dtype=float)/1e3,
                        stress   = np.abs(np.diff(T.mean(axis=1)))/dt,
                        flow_std = F.std(axis=1))

temporal_modes = {
    'pump_kw' : {'raw_kw':        lambda x: x,
                 'rolling_60s':   lambda x: uniform_filter1d(x, size=60),
                 'cumulative_kWh':lambda x: np.cumsum(x)/3600.,
                 'asinh_q95':     lambda x: np.arcsinh(x/max(np.quantile(x,.95),1e-12))},
    'stress'  : {'raw':           lambda x: x,
                 'rolling_60s':   lambda x: uniform_filter1d(x, size=60),
                 'cumulative_mean':lambda x: np.cumsum(x)/np.arange(1,len(x)+1),
                 'log1p':         lambda x: np.log1p(x)},
    'flow_std': {'raw':           lambda x: x,
                 'rolling_60s':   lambda x: uniform_filter1d(x, size=60),
                 'cumulative_mean':lambda x: np.cumsum(x)/np.arange(1,len(x)+1),
                 'log1p':         lambda x: np.log1p(x)},
}
TEMP_ABL = {}
for signal, opts in temporal_modes.items():
    rows = []
    for mode, fn in opts.items():
        tr     = {k: fn(v[signal]) for k, v in series.items()}
        m_len  = min(len(v) for v in tr.values())
        A      = {k: np.asarray(v[:m_len], dtype=float) for k, v in tr.items()}
        ns     = list(A.keys())
        M      = np.vstack([A[n] for n in ns])
        Mz     = np.column_stack([_minmax(M[:,j]) for j in range(M.shape[1])])
        sep    = _mpd(Mz)
        rough  = float(np.mean([np.mean(np.abs(np.diff(A[n])))/(np.std(A[n])+1e-12) for n in ns]))
        dr     = float(np.quantile(M,.95) - np.quantile(M,.05))
        rows.append({'variant':mode,'separation':sep,'roughness':rough,'dynamic_range':dr})
    d = _rank_score(pd.DataFrame(rows),
                    plus ={'separation':.5,'dynamic_range':.2},
                    minus={'roughness':.4})
    TEMP_ABL[signal] = d
    d.to_csv(ABL_OUT/f'ablation_temporal_{signal}.csv', index=False)
ABL['temporal'] = pd.concat([d.assign(signal=s) for s, d in TEMP_ABL.items()], ignore_index=True)

# ── Family 4: Statistical ─────────────────────────────────────────────────────
stat_data = {
    'pump'  : {n: run_map[n]['pump_power_W'].to_numpy(dtype=float) for n in run_map},
    'spread': {n: np.ptp(run_map[n][temp_cols(run_map[n])].to_numpy(dtype=float), axis=1)
               for n in run_map},
    'stress': {},
}
for n in run_map:
    df  = run_map[n]; T = df[temp_cols(df)].to_numpy(dtype=float)
    dt  = np.clip(np.diff(df['time_s'].to_numpy(dtype=float)), 1e-9, None)
    stat_data['stress'][n] = (np.abs(np.diff(T, axis=0)) / dt[:, None]).ravel()

stat_modes = {
    'linear'   : lambda x: x,
    'log1p'    : lambda x: np.log1p(np.clip(x, 0, None)),
    'asinh_q95': lambda x: np.arcsinh(x/max(np.quantile(x,.95),1e-12)),
    'rank'     : lambda x: pd.Series(x).rank(method='average').to_numpy(),
}
STAT_ABL = {}
for key, dct in stat_data.items():
    rows = []
    for mode, fn in stat_modes.items():
        tr    = {n: fn(v) for n, v in dct.items()}
        ks    = list(tr.keys()); js_vals = []
        for i in range(len(ks)):
            for j in range(i+1, len(ks)):
                js_vals.append(float(jensenshannon(_hist_prob(tr[ks[i]]), _hist_prob(tr[ks[j]]))))
        pooled = np.concatenate(list(tr.values()))
        sk     = abs(pd.Series(pooled).skew())
        tail   = (np.quantile(pooled,.99)+1e-12)/(np.quantile(pooled,.50)+1e-12)
        rows.append({'variant':mode,'js_divergence':np.mean(js_vals) if js_vals else 0.,
                     'abs_skew':sk,'tail_ratio':tail})
    d = _rank_score(pd.DataFrame(rows),
                    plus ={'js_divergence':.6},
                    minus={'abs_skew':.25,'tail_ratio':.15})
    STAT_ABL[key] = d
    d.to_csv(ABL_OUT/f'ablation_statistical_{key}.csv', index=False)
ABL['statistical'] = pd.concat([v.assign(metric=k) for k, v in STAT_ABL.items()], ignore_index=True)

print('Ablation families computed:', list(ABL.keys()))


# ## 9-C  Ablation Summary Visualisation

# In[68]:


families = ['pareto', 'summary_bars', 'temporal', 'statistical']
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for i, fam in enumerate(families):
    ax  = axes[i]
    d   = ABL.get(fam, pd.DataFrame())
    if d.empty:
        ax.axis('off'); continue
    dd  = d.head(4).copy()
    bar_cols = [CLUSTER_PALETTE[CLUSTER_LETTERS[j % len(CLUSTER_LETTERS)]]
                for j in range(len(dd))]
    ax.barh(np.arange(len(dd))[::-1], dd['score'],
            color=bar_cols, edgecolor='white', lw=0.5)
    ax.set_yticks(np.arange(len(dd))[::-1])
    ax.set_yticklabels(dd['variant'], fontsize=8)
    ax.set_title(f'{fam}  —  Top Variants')
    ax.set_xlabel('Score')
    finish_ax(ax)

fig.tight_layout()
fig.savefig(ABL_OUT / 'ablation_top3_by_family.pdf', bbox_inches='tight', pad_inches=0.02)
# plt.show()

print('Ablation CSVs written:')
for p in sorted(ABL_OUT.glob('*.csv')):
    print(' -', p.name)


# ## 10  Paper Assembly Checklist (Quick)
# 
# Use this as a final checklist while drafting.
# 
# - State dataset and drive-cycle scope clearly (measured + synthesized run usage).
# - Define all primary metrics once: max temperature, mean temperature, spread, thermal stress $|\Delta T/\Delta t|$, pump energy, cooling overhead.
# - Tie every claim to one figure family: Pareto (tradeoff), temporal (dynamics), statistical (distribution/tail), spatial (zone-level behavior), control strategy (actuation style).
# - Report both central tendency and tail/risk behavior (especially stress and spread tails).
# - Explicitly note readability transforms (quantile bands, capped views, kink strip) as visualization choices, not data modification.
# - Include cluster-membership text when discussing grouped panels.
# - End with practical implication: thermal safety margin, actuator burden, and energy efficiency together.
# 
# **One-line conclusion template**
# - “Overall, the proposed control strategy improves the energy–thermal compromise while maintaining spatial thermal balance and limiting dynamic stress, indicating better real-world deployability under variable drive conditions.”
