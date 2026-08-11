"""
plot_results.py — Reproducible analysis and plotting script for:
"GSOA: Gravitational Slingshot Optimization Algorithm with p-best Guidance
and Adaptive Restart"

Generates all figures and tables reported in the paper from the raw
30-run result data in data/cec2022/ and data/mechanical_design/.

Usage:
    python3 analysis/plot_results.py

Outputs are written to analysis/outputs/. Requires: numpy, scipy, matplotlib.

File naming conventions:
  CEC2022:          data/cec2022/d{10,20}/{ALGO}_{funcID}_{D}.txt
  Mechanical:       data/mechanical_design/{PROB}/{ALGO}_{PROB}.txt
  NL-SHADE-RSP-MID: NL-SHADE-RSP_MID{fid}_{D}_pop_{50|100}.txt (D=10: pop=50, D=20: pop=100)

Each result file is a whitespace-delimited matrix of shape (16, 30):
  rows 0-15: best-so-far error / objective at 16 log-spaced FES checkpoints
  columns 0-29: independent runs (seed = 42 + run_index)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json
from scipy.stats import friedmanchisquare, wilcoxon
from pathlib import Path

OUT_DIR = Path('analysis/outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGOS_CEC = ['GSOA', 'EA4eig', 'S-LSHADE-DP', 'NL-SHADE-LBC', 'NL-SHADE-RSP-MID', 'GSA', 'DE', 'PSO', 'GA']
ALGOS_MECH = ['GSOA', 'EA4eig', 'S-LSHADE-DP', 'NL-SHADE-LBC', 'NL-SHADE-RSP-MID', 'GSA', 'DE', 'PSO', 'GA']

COLORS = {
    'GSOA':'#e74c3c', 'EA4eig':'#1abc9c', 'S-LSHADE-DP':'#27ae60',
    'NL-SHADE-LBC':'#8e44ad', 'NL-SHADE-RSP-MID':'#f39c12',
    'GSA':'#95a5a6', 'DE':'#7f8c8d', 'PSO':'#e67e22', 'GA':'#2c3e50',
}
STYLES = {
    'GSOA':'-', 'EA4eig':'--', 'S-LSHADE-DP':'-.', 'NL-SHADE-LBC':':',
    'NL-SHADE-RSP-MID':(0,(3,1,1,1)), 'GSA':'-', 'DE':'-.', 'PSO':'--', 'GA':':',
}
MECH_NAMES = {
    'B1':'Pressure Vessel Design', 'B2':'Tension/Compression Spring Design',
    'B3':'Welded Beam Design', 'C1':'Multiple Disc Clutch Brake Design',
    'C2':'Robot Gripper Design', 'C4':'Hydrodynamic Thrust Bearing Design',
}
MECH_DIMS = {'B1':4, 'B2':3, 'B3':4, 'C1':5, 'C2':7, 'C4':4}

def get_cec_fname(alg, fid, D):
    if alg == 'NL-SHADE-RSP-MID':
        pop = 50 if D == 10 else 100
        return f'NL-SHADE-RSP_MID{fid}_{D}_pop_{pop}.txt'
    return f'{alg}_{fid}_{D}.txt'

def load_cec(alg, fid, D):
    path = f'data/cec2022/d{D}/{get_cec_fname(alg, fid, D)}'
    return np.loadtxt(path)

def load_mech(alg, prob):
    return np.loadtxt(f'data/mechanical_design/{prob}/{alg}_{prob}.txt')

# ==============================================================
# 1. CEC2022 CONVERGENCE PLOTS (Figs 1-2)
# ==============================================================
def plot_cec_convergence(D, maxfes):
    checkpoints = np.array([round(D**(i/5.0-3.0)*maxfes) for i in range(16)])
    fig, axes = plt.subplots(4, 3, figsize=(21, 25))
    axes = axes.flatten()
    for fid in range(1, 13):
        ax = axes[fid-1]
        for a in ALGOS_CEC:
            mat = load_cec(a, fid, D)
            curve = np.maximum(mat[:16, :].mean(axis=1), 1e-10)
            lw = 2.6 if a == 'GSOA' else 1.5
            ax.semilogy(checkpoints, curve, color=COLORS[a], linestyle=STYLES[a],
                        lw=lw, marker='o', markersize=3, label=a,
                        zorder=10 if a == 'GSOA' else 5)
            if a == 'GSOA':
                std_c = mat[:16, :].std(axis=1)
                ax.fill_between(checkpoints,
                                np.maximum(curve - std_c, 1e-10),
                                np.maximum(curve + std_c, 1e-10),
                                color=COLORS[a], alpha=0.18, zorder=1,
                                label='GSOA ±1 std')
        ax.set_title(f'F{fid}', fontsize=13, fontweight='bold')
        ax.set_xlabel('FES', fontsize=10)
        ax.set_ylabel('Mean Error (log)', fontsize=10)
        ax.grid(True, which='both', ls='--', alpha=0.3)
        if fid == 1:
            ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'CEC2022 D={D} — Convergence (mean of 30 runs, shaded = GSOA ±1 std)\n'
                 f'GSOA vs EA4eig, S-LSHADE-DP, NL-SHADE-LBC, NL-SHADE-RSP-MID, GSA, DE, PSO, GA',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / f'GSOA_CEC2022_D{D}_convergence.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ==============================================================
# 2. CEC2022 STATISTICS (Tables 3-4)
# ==============================================================
def compute_cec_stats(D):
    stats = {}
    for a in ALGOS_CEC:
        stats[a] = {}
        for fid in range(1, 13):
            finals = load_cec(a, fid, D)[15, :]
            stats[a][fid] = {'min':finals.min(), 'max':finals.max(),
                             'mean':finals.mean(), 'std':finals.std()}
    return stats

def plot_stats_table(stats, D, ALGOS):
    col_labels = ['F', 'Algorithm', 'Min', 'Max', 'Mean', 'Std Dev']
    rows, colors = [], []
    for fid in range(1, 13):
        best = min(stats[a][fid]['mean'] for a in ALGOS)
        for i, a in enumerate(ALGOS):
            v = stats[a][fid]
            mark = ' *' if v['mean'] == best else ''
            rows.append([f'F{fid}' if i == 0 else '', a,
                         f"{v['min']:.3e}", f"{v['max']:.3e}",
                         f"{v['mean']:.3e}{mark}", f"{v['std']:.3e}"])
            colors.append('#d5f5e3' if v['mean'] == best else '#ffffff')
    fig, ax = plt.subplots(figsize=(14, max(10, len(rows)*0.185+1.5)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center', bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for j in range(len(col_labels)):
        tbl[(0,j)].set_facecolor('#1a252f')
        tbl[(0,j)].set_text_props(color='white', fontweight='bold', fontsize=9)
    for i, c in enumerate(colors, start=1):
        for j in range(len(col_labels)): tbl[(i,j)].set_facecolor(c)
    fig.suptitle(f'CEC2022 D={D} — Min / Max / Mean / Std Dev (30 runs)  |  * = best mean per function',
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()
    out = OUT_DIR / f'GSOA_CEC2022_stats_table_D{D}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ==============================================================
# 3. FRIEDMAN + WLT (Tables 5-6)
# ==============================================================
def compute_friedman_wlt(stats, ALGOS, proposed='GSOA', n_problems=12):
    fid_list = list(range(1, n_problems + 1)) if isinstance(list(stats[ALGOS[0]].keys())[0], int) else list(stats[ALGOS[0]].keys())
    fr = {a: [] for a in ALGOS}
    for fid in fid_list:
        means = [stats[a][fid]['mean'] for a in ALGOS]
        rks = np.argsort(np.argsort(means)) + 1
        for i, a in enumerate(ALGOS): fr[a].append(int(rks[i]))
    rm = np.array([fr[a] for a in ALGOS]).T
    stat_f, pval_f = friedmanchisquare(*[rm[:, i] for i in range(len(ALGOS))])
    mean_ranks = {a: float(np.mean(fr[a])) for a in ALGOS}
    prop_means = np.array([stats[proposed][fid]['mean'] for fid in fid_list])
    wlt = {}
    for a in ALGOS:
        if a == proposed: continue
        opp = np.array([stats[a][fid]['mean'] for fid in fid_list])
        w = int(np.sum(prop_means < opp)); l = int(np.sum(prop_means > opp)); t = int(np.sum(prop_means == opp))
        diff = prop_means - opp
        try:
            st, p = wilcoxon(diff, alternative='two-sided', zero_method='wilcox')
            st = float(st)
        except Exception:
            st, p = None, 1.0
        wlt[a] = {'W': w, 'L': l, 'T': t, 'stat': st, 'p': float(p)}
    return mean_ranks, sorted(ALGOS, key=lambda a: mean_ranks[a]), float(stat_f), float(pval_f), wlt

def plot_friedman(mean_ranks, sorted_algs, chi2, pval, prefix, title_suffix=''):
    fig, ax = plt.subplots(figsize=(11, 6))
    mrs = [mean_ranks[a] for a in sorted_algs]
    bars = ax.bar(sorted_algs, mrs, color=[COLORS[a] for a in sorted_algs], edgecolor='black', lw=0.8)
    for bar, v in zip(bars, mrs):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.08, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Friedman Rank', fontsize=11)
    ax.set_title(f'{title_suffix}\nChi2={chi2:.2f}, p={pval:.2e}  |  Lower = Better', fontsize=12)
    ax.set_ylim([0, len(sorted_algs)+0.8]); ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', ls='--', alpha=0.4)
    plt.tight_layout()
    out = OUT_DIR / f'{prefix}_friedman.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

def plot_wlt(wlt, proposed, prefix, n_note=''):
    sig = lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    verdict = lambda w,l,p: 'No sig. difference' if p>=0.05 else (f'{proposed} sig. better' if w>l else f'{proposed} sig. worse')
    rows, cell_colors = [], []
    for a, r in wlt.items():
        s = sig(r['p']); v = verdict(r['W'], r['L'], r['p'])
        st = f"{r['stat']:.2f}" if r['stat'] is not None else 'N/A'
        rows.append([a, str(r['W']), str(r['L']), str(r['T']), st, f"{r['p']:.4f}", s, v])
        c = '#eafaf1' if 'better' in v else ('#f9ebea' if 'worse' in v else '#fef9e7')
        cell_colors.append([c]*8)
    col_labels = ['Opponent','W','L','T','W-stat','p-value','Sig.','Verdict']
    fig, ax = plt.subplots(figsize=(13, max(3, len(rows)*0.55+1)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center', bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for j in range(len(col_labels)):
        tbl[(0,j)].set_facecolor('#1a252f')
        tbl[(0,j)].set_text_props(color='white', fontweight='bold', fontsize=9)
    for i, cr in enumerate(cell_colors, start=1):
        for j, c in enumerate(cr): tbl[(i,j)].set_facecolor(c)
        tbl[(i,0)].set_text_props(fontweight='bold')
    note = f'\n({n_note})' if n_note else ''
    fig.suptitle(f'Win/Loss/Tie + Wilcoxon Signed-Rank — {proposed} vs each  |  alpha=0.05{note}',
                 fontsize=11, fontweight='bold', y=1.1)
    plt.tight_layout()
    out = OUT_DIR / f'{prefix}_wlt.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ==============================================================
# 4. MECHANICAL DESIGN CONVERGENCE (Figs 3-8)
# ==============================================================
def plot_mech_convergence(prob):
    D = MECH_DIMS[prob]
    checkpoints = np.array([round(D**(i/5.0-3.0)*200000) for i in range(16)])
    fig, ax = plt.subplots(figsize=(10, 7))
    for a in ALGOS_MECH:
        mat = load_mech(a, prob)
        curve = np.maximum(mat.mean(axis=1), 1e-17)
        lw = 2.6 if a == 'GSOA' else 1.5
        ax.semilogy(checkpoints, curve, color=COLORS[a], linestyle=STYLES[a],
                    lw=lw, marker='o', markersize=4, label=a,
                    zorder=10 if a == 'GSOA' else 5)
        if a == 'GSOA':
            std_c = mat.std(axis=1)
            ax.fill_between(checkpoints,
                            np.maximum(curve-std_c, 1e-17),
                            np.maximum(curve+std_c, 1e-17),
                            color=COLORS[a], alpha=0.18, zorder=1, label='GSOA ±1 std')
    ax.set_xlabel('FES', fontsize=12)
    ax.set_ylabel('Mean Best Penalized Objective (log)', fontsize=12)
    ax.set_title(f'{MECH_NAMES[prob]}\nGSOA vs EA4eig, S-LSHADE-DP, NL-SHADE-LBC, NL-SHADE-RSP-MID, GSA, DE, PSO, GA\n'
                 f'(mean of 30 runs, shaded = GSOA ±1 std)', fontsize=11)
    ax.legend(fontsize=9.5, loc='upper right')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f'{prob}_30run_convergence.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ==============================================================
# MAIN
# ==============================================================
print("=== CEC2022 ===")
for D, maxfes in [(10, 200000), (20, 1000000)]:
    plot_cec_convergence(D, maxfes)
    stats = compute_cec_stats(D)
    plot_stats_table(stats, D, ALGOS_CEC)
    mr, sa, chi2, pval, wlt = compute_friedman_wlt(stats, ALGOS_CEC, 'GSOA', 12)
    plot_friedman(mr, sa, chi2, pval, f'GSOA_CEC2022_D{D}',
                  f'CEC2022 D={D} — Friedman Mean Rank (30 runs, 12 functions, 9 algorithms)')
    plot_wlt(wlt, 'GSOA', f'GSOA_CEC2022_D{D}')

print("=== Mechanical Design ===")
mech_stats = {}
for prob in ['B1','B2','B3','C1','C2','C4']:
    plot_mech_convergence(prob)
    mech_stats[prob] = {}
    for a in ALGOS_MECH:
        finals = load_mech(a, prob)[15, :]
        mech_stats[prob][a] = {'min':finals.min(),'max':finals.max(),'mean':finals.mean(),'std':finals.std()}

# mech stats table
col_labels = ['Problem','Algorithm','Min','Max','Mean','Std Dev']
rows, colors = [], []
for prob in ['B1','B2','B3','C1','C2','C4']:
    best = min(mech_stats[prob][a]['mean'] for a in ALGOS_MECH)
    for i, a in enumerate(ALGOS_MECH):
        v = mech_stats[prob][a]
        mark = ' *' if v['mean'] == best else ''
        rows.append([MECH_NAMES[prob].split()[0] if i==0 else '', a,
                     f"{v['min']:.3e}", f"{v['max']:.3e}",
                     f"{v['mean']:.3e}{mark}", f"{v['std']:.3e}"])
        colors.append('#d5f5e3' if v['mean']==best else '#ffffff')
fig, ax = plt.subplots(figsize=(14, max(10, len(rows)*0.185+1.5)))
ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center', bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for j in range(len(col_labels)):
    tbl[(0,j)].set_facecolor('#1a252f')
    tbl[(0,j)].set_text_props(color='white',fontweight='bold',fontsize=9)
for i,c in enumerate(colors, start=1):
    for j in range(len(col_labels)): tbl[(i,j)].set_facecolor(c)
fig.suptitle('Mechanical Design — Min / Max / Mean / Std Dev (30 runs)  |  * = best mean per problem',
             fontsize=13, fontweight='bold', y=1.0)
plt.tight_layout()
out = OUT_DIR / 'mech_stats_table.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# mech Friedman + WLT
mech_flat = {a:{i+1: mech_stats[prob][a] for i, prob in enumerate(['B1','B2','B3','C1','C2','C4'])} for a in ALGOS_MECH}
mr, sa, chi2, pval, wlt = compute_friedman_wlt(mech_flat, ALGOS_MECH, 'GSOA', 6)
plot_friedman(mr, sa, chi2, pval, 'mech',
              'Mechanical Design — Friedman Mean Rank (30 runs, 6 problems, 9 algorithms)')
plot_wlt(wlt, 'GSOA', 'mech', 'n=6 problems limits max possible significance to p=0.03125')

print("\n=== All outputs saved to analysis/outputs/ ===")
