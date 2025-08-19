import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg") # headless mode
import matplotlib.pyplot as plt
import seaborn as sns


# =======================================================================================
#          SETUP & LOAD
# =======================================================================================

# -------- Get session_id from first arg --------
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <session_id>")
    sys.exit(1)

session_id = sys.argv[1]

# -------- Paths --------
project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"
sesh_path = data_root / f"session_{session_id}"

# load output of stimulus_spike_counts.py
pivot = pd.read_csv(sesh_path / f"s{session_id}_pivot_counts_rates_by_region.csv")


# =======================================================================================
#          REPORTS & PLOTS
# =======================================================================================

# collapse movie variants for plottings
pivot_plot = pivot.copy()
pivot_plot["stimulus_collapsed"] = pivot_plot["stimulus_name"].replace({
    "natural_movie_one": "natural_movies",
    "natural_movie_three": "natural_movies",
})

# -------- Plot outdir --------
plot_outdir = sesh_path / f"session_{session_id}_plots"
plot_outdir.mkdir(parents=True, exist_ok=True)

# -------- Choose regions (top 5 by unique units) --------
units_per_region = (
    pivot.groupby("ecephys_structure_acronym")["unit_id"]
        .nunique()
        .sort_values(ascending=False)
)
regions = units_per_region.head(5).index.tolist()
if not regions:
    print("[plots] No regions available to plot.")
else:
    print("[plots] Regions selected:", regions)

# -------- Stimulus set & colors --------
stim_order = ["drifting_gratings", "natural_movies"]
present_stims = [s for s in stim_order if s in set(pivot_plot["stimulus_collapsed"].unique())]
if not present_stims:
    print("[plots] No known stimuli present among", stim_order)

stim_color = {
    "drifting_gratings": "tab:orange",
    "natural_movies": "tab:green",
}

# ======== A) Delta histograms by stimulus per region ========
USE_DENSITY = False
SHOW_MEDIAN = True

if regions and present_stims:
    nrows = len(present_stims)
    ncols = len(regions)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4*ncols, 2.9*nrows), squeeze=False, sharex=True, sharey=False)

    XMIN, XMAX = -75, 75
    NBINS = 40
    common_bins = np.linspace(XMIN, XMAX, NBINS+1)
    for row, stim_name in enumerate(present_stims):
        # pooled deltas -> common bins within the row
        pooled = []
        per_region_vals = {}
        for reg in regions:
            vals = pivot_plot.loc[
                (pivot_plot["stimulus_collapsed"] == stim_name) &
                (pivot_plot["ecephys_structure_acronym"] == reg),
                "delta_rate_hz"
            ].dropna().to_numpy()
            per_region_vals[reg] = vals
            if vals.size:
                pooled.append(vals)
        if pooled:
            pooled_all = np.concatenate(pooled)
            vmin, vmax = float(np.nanmin(pooled_all)), float(np.nanmax(pooled_all))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, max(1.0, vmax if np.isfinite(vmax) else 1.0)
        else:
            vmin, vmax = 0.0, 1.0
        bins = np.linspace(vmin, vmax, 41)

        for col, reg in enumerate(regions):
            ax = axes[row, col]
            vals = per_region_vals.get(reg, np.array([]))
            if vals.size:
                ax.hist(
                    vals, bins=common_bins, density=USE_DENSITY, alpha=0.9,
                    color=stim_color.get(stim_name, None),
                    label=f"n={len(vals)} trials"
                )
                if SHOW_MEDIAN and vals.size:
                    med = float(np.median(vals))
                    ax.axvline(med, color="k", linewidth=1, alpha=0.5)
                ax.set_xlim(XMIN, XMAX)
            ax.axvline(0, linestyle="--", linewidth=1, color="gray")
            n_units = pivot_plot.loc[pivot_plot["ecephys_structure_acronym"] == reg, "unit_id"].nunique()
            if row == 0:
                ax.set_title(f"{reg} (units={n_units})", fontsize=11)
            # y-labels show stimulus name at row start
            if col == 0:
                ylabel = "count" if not USE_DENSITY else "density"
                ax.set_ylabel(f"{stim_name}\n\n{ylabel}", fontsize=10)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("delta rate (Hz)", fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        f"Delta firing rate - drifting_gratings vs natural_movies (rows) x region (cols)\n"
        f"session {session_id}",
        y=1.05, fontsize=14
    )
    plt.tight_layout()
    out_path_grid = plot_outdir / f"s{session_id}_delta_rate_by_region.png"
    plt.savefig(out_path_grid, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved 2x5 delta grid histograms: {out_path_grid}")

# ======== B) Grouped bar - median delta per region x stimulus ========
per_unit_collapsed = (
    pivot_plot.groupby(["unit_id","ecephys_structure_acronym","stimulus_collapsed"], as_index=False)
            .agg(delta_rate_hz_mean=("delta_rate_hz","mean"),
                 n_trials=("delta_rate_hz","size"))
)
summ_unit = (
    per_unit_collapsed.groupby(["ecephys_structure_acronym", "stimulus_collapsed"], as_index = False)
        .agg(
            n_units=('delta_rate_hz_mean', 'size'),
            median_delta=('delta_rate_hz_mean', 'median'),
            mean_delta=('delta_rate_hz_mean', 'mean'),
            frac_pos=('delta_rate_hz_mean', lambda x: (x > 0).mean())
        )
)

if regions and present_stims:
    plot_summ = summ_unit[summ_unit["ecephys_structure_acronym"].isin(regions)].copy()
    plot_summ["stimulus_collapsed"] = pd.Categorical(
        plot_summ["stimulus_collapsed"], categories=present_stims, ordered=True)
    plot_summ = plot_summ.sort_values(["ecephys_structure_acronym","stimulus_collapsed"])

    reg_list = [r for r in regions if r in plot_summ["ecephys_structure_acronym"].unique()]
    x = np.arange(len(reg_list))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, 1.6*len(reg_list)), 4.6))
    for j, s in enumerate(present_stims):
        sub = plot_summ[plot_summ["stimulus_collapsed"] == s]
        # align to reg_list order; fill missing with 0 and mark n_units for labels
        med_map = {k: v for k, v in zip(sub["ecephys_structure_acronym"], sub["median_delta"])}
        n_map   = {k: v for k, v in zip(sub["ecephys_structure_acronym"], sub["n_units"])}
        y = np.array([med_map.get(r, np.nan) for r in reg_list], dtype=float)
        n = np.array([n_map.get(r, 0) for r in reg_list], dtype=int)
        pos = x + (j - 0.5) * width*0.9
        ax.bar(pos, y, width=width, label=s, color=stim_color.get(s, None))
        # annotate n_units above bars
        for xp, yp, nn in zip(pos, y, n):
            if np.isfinite(yp):
                ax.text(xp, yp + (0.03 if yp >= 0 else -0.03),
                        f"n={nn}", ha="center",
                        va="bottom" if yp>=0 else "top", fontsize=8)
    ax.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(reg_list, rotation=30, ha="right")
    ax.set_ylabel("median delta rate (Hz)\n(unit-weighted)")
    ax.set_title(f"Region x Stimulus preference — session {session_id}")
    ax.legend(frameon=False, ncol=len(present_stims))
    plt.tight_layout()
    out_path2 = plot_outdir / f"s{session_id}_median_delta_by_region_by_stim.png"
    plt.savefig(out_path2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved grouped bar chart: {out_path2}")

# ======== C) Baseling vs Evoked grids - one figure per collapsed stimulus ========
if regions and present_stims:
    rows=1
    cols=len(regions)
    for stim_name in present_stims:
        fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 2.8), squeeze=False)
        for i, r in enumerate(regions):
            ax = axes[0,i]
            sub = pivot_plot[(pivot_plot["ecephys_structure_acronym"] == r) &
                             (pivot_plot["stimulus_collapsed"] == stim_name)]
            b = sub["rate_hz_baseline"].dropna().to_numpy()
            w = sub["rate_hz_evoked"].dropna().to_numpy()
            if len(b) == 0 and len(w) == 0:
                ax.set_title(f"{r} (no data)")
                ax.axis("off")
                continue
            # common bins so shapes are comparable
            both = np.concatenate([b, w]) if b.size and w.size else (b if b.size else w)
            vmin, vmax = float(np.nanmin(both)), float(np.nanmax(both))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, max(1.0, vmax if np.isfinite(vmax) else 1.0)
            bins = np.linspace(vmin, vmax, 41)
        
            ax.hist(b, bins=bins, alpha=0.5, label=f"baseline (n={len(b)})")
            ax.hist(w, bins=bins, alpha=0.5, label=f"evoked (n={len(w)})")
            ax.set_title(f"{r}")
            ax.set_xlabel("rate (Hz)")
            ax.set_ylabel("count")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Baseline vs Evoked firing rate by region - {stim_name} - session {session_id}", y=1.05)
        plt.tight_layout()
        out_path3 = plot_outdir / f"s{session_id}_baseline_vs_evoked_by_region_{stim_name}.png"
        plt.savefig(out_path3, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[plots] Saved baseline-vs-evoked grid: {out_path3}")

# ======== D) Boxplot delta_rate_hz split by region and stimulus ========
df = pivot_plot[pivot_plot["stimulus_collapsed"].isin(present_stims)]

plt.figure(figsize=(14, 6))
sns.boxplot(
    data=df,
    x="ecephys_structure_acronym",   # regions on x-axis
    y="delta_rate_hz",               # delta rate
    hue="stimulus_collapsed",        # split by stimulus
    showfliers=False                 # hide extreme outliers for readability
)

plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.title("Delta firing rates by Region and Stimulus")
plt.ylabel("Delta firing rate (Hz)")
plt.xlabel("Region")
plt.legend(title="Stimulus")
plt.tight_layout()
out_path4 = plot_outdir / f"s{session_id}_boxplots_delta_rate.png"
plt.savefig(out_path4, dpi=300)
plt.close()

# =================================================================

# -------- TEXT SUMMARIES --------
# --- stimulus-aware trial-weighted summary (top 5 per stimulus) ---
region_summary = (
    pivot_plot.groupby([ "ecephys_structure_acronym", "stimulus_collapsed"])
            .agg(mean_delta=("delta_rate_hz", "mean"),
                 median_delta=("delta_rate_hz", "median"),
                 n=("delta_rate_hz", "size"))
            .reset_index()
            .sort_values(by=["stimulus_collapsed", "mean_delta"], ascending=[True,False])
)
top5_per_stim = region_summary.groupby("stimulus_collapsed", group_keys=False).head(5)
print("\nTop regions per stimuli by mean delta rate (Hz):")
print(top5_per_stim)

region_stim_unitweighted = (
    per_unit_collapsed.groupby(["ecephys_structure_acronym","stimulus_collapsed"])
            .agg(n_units=("delta_rate_hz_mean","size"),
                 mean_delta=("delta_rate_hz_mean","mean"),
                 median_delta=("delta_rate_hz_mean","median"),
                 frac_pos=("delta_rate_hz_mean", lambda x: (x > 0).mean()))
            .reset_index()
            .sort_values(by=["stimulus_collapsed","mean_delta"], ascending=[True,False])
)
print("\nUnit-weighted Region x Stimulus summary:")
print(region_stim_unitweighted.head(20).to_string(index=False))

# --- write summaries to outdir ---
summary_dir = (plot_outdir / f"summaries")
summary_dir.mkdir(parents=True, exist_ok=True)

region_summary.to_csv(summary_dir / f"s{session_id}_region_summary.csv", index=False)
top5_per_stim.to_csv(summary_dir / f"s{session_id}_top5_per_stim.csv", index=False)
region_stim_unitweighted.to_csv(summary_dir / f"s{session_id}_unweighted_summary.csv", index=False)

print("done")