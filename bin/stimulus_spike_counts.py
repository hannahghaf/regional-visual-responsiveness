import sys
import numpy as np
import pandas as pd
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg") # headless mode
import matplotlib.pyplot as plt
import re

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


# -------- Load files --------
spikes = pd.read_csv(
    sesh_path / "spike_times.csv",
    usecols=["unit", "t"],
    dtype={"unit": "int32", "t": "float64"})
units = pd.read_csv(
    data_root / "units.csv",
    usecols=["id", "ecephys_channel_id"],
    dtype={"id": "int32", "ecephys_channel_id": "int32"})
channels = pd.read_csv(
    data_root / "channels.csv",
    usecols=["id", "ecephys_probe_id", "ecephys_structure_acronym"],
    dtype={"id": "int32", "ecephys_probe_id": "int32"})
stim = pd.read_csv(
    sesh_path / "stimulus_presentations.csv",
    usecols=["start_time","stop_time","stimulus_name"],
    low_memory=False)
stim = stim.sort_values("start_time").reset_index(drop=True)

#series = channels["ecephys_structure_acronym"].unique()


# -------- Keep only cortex visual-proessing areas --------
VIS_AREAS = {
    "VISrl", "VIS", "VISp", "VISpm", "VISam", "VISal",
    "VISl", "VISmma", "VISmmp", "VISli", "POL"
}

# filter channels to desired regions
channels = channels[channels["ecephys_structure_acronym"].isin(VIS_AREAS)].copy()
# keep only units that sit on those channels
keep_chan_ids = set(channels["id"].tolist())
units = units[units["ecephys_channel_id"].isin(keep_chan_ids)].copy()
# keep only spikes from those units (save time)
keep_unit_ids = set(units["id"].tolist())
spikes = spikes[spikes["unit"].isin(keep_unit_ids)].copy()

print(f"Filtering VIS_AREAS: Kept {len(channels)} channels, {len(units)} units, {len(spikes)} spikes in visual areas.")


# -------- Stimulus counts check --------
print("\nCounts per stimulus before window building:")
print(stim["stimulus_name"].value_counts(dropna=False).rename_axis("stimulus").to_frame("n"))


# -------- Merge spike times w unit and channel metadata --------
channels_with_rate = (
    spikes.merge(units, left_on="unit", right_on="id", how="left")
        .drop(columns=["id"])
        .rename(columns={"unit": "unit_id", "t": "spike_timestamp"})
        [["ecephys_channel_id", "unit_id", "spike_timestamp"]]
)

channel_unit_t_struct = (
    channels_with_rate.merge(channels, left_on="ecephys_channel_id", right_on="id", how="left")
                    .drop(columns=["id"])
                    .rename(columns={"ecephys_channel_id": "channel_id",
                                     "ecephys_probe_id": "probe_id"})
                                     [["unit_id", "spike_timestamp", "ecephys_structure_acronym"]]
                                     .sort_values("spike_timestamp", ascending=True, ignore_index=True)
)


# -------- Stimulus sanity check --------
needed = {"start_time", "stop_time", "stimulus_name"}
missing = needed - set(stim.columns)
if missing:
    raise ValueError(f"stimulus_presentations.csv missing required columns: {missing}")


# =======================================================================================
#          BUILD WINDOWS
# =======================================================================================

# -------- Window policy implementation (0.5s for both) --------
B_SEC = 0.50    # baseline length for both drifting_gratings and natural_movies
W_SEC = 0.50    # evoked length for both drifting_gratings and natural_movies
EPS   = 1e-9
MOVIE_NAMES = ["natural_movie_one", "natural_movie_three"]
CLIP_SEC    = 30.0      # movie repeat length (one trial)
GAP_THRESHOLD = 0.10    # gap > this (s) = new movie block
MERGE_GAP   = 0.50      # merge spontaneous frames into blocks
MAX_SPONT_DIST = 120.0  # max seconds away for spontaneous baseline (movies)


# -------- Build SPONTANEOUS blocks (for movie baselines) --------
spont = stim[stim["stimulus_name"] == "spontaneous"].copy()
spont = spont.sort_values("start_time").reset_index(drop=True)

# merge consecutive spontaneous rows if short gaps between them
sp_rows = []
cur_s = cur_e = None
for s, e in zip(spont["start_time"], spont["stop_time"]):
    s = float(s); e = float(e)
    if cur_s is None:
        cur_s, cur_e = s, e
    else:
        if s - cur_e <= MERGE_GAP:
            cur_e = max(cur_e, e)
        else:
            sp_rows.append((cur_s, cur_e))
            cur_s, cur_e = s, e
if cur_s is not None:
    sp_rows.append((cur_s, cur_e))

spont_blocks = pd.DataFrame(sp_rows, columns=["s_start","s_end"])
spont_blocks["s_dur"] = spont_blocks["s_end"] - spont_blocks["s_start"]
spont_blocks = spont_blocks.sort_values("s_start").reset_index(drop=True)

SSTART = spont_blocks["s_start"].to_numpy(float)
SEND   = spont_blocks["s_end"].to_numpy(float)
SDUR   = spont_blocks["s_dur"].to_numpy(float)

def baseline_from_spont_before(t0: float, L: float, max_dist: float):
    """
    - Prefer the spontaneous block that ends <= t0 and is closest to t0
    - Return the last L seconds inside that block: [s_end - L, s_end)
    - If none before within max_dist, try nearest block after t0 (first L seconds)
    - If none, return (None, None)
    """
    idx = np.searchsorted(SEND, t0, side="right") - 1
    # search backward (before t0)
    j = idx
    while j >= 0:
        dist = t0 - SEND[j]
        if dist > max_dist:
            break
        if SDUR[j] >= L - EPS:
            b_end = SEND[j]
            b_start = max(SSTART[j], b_end - L)
            if b_end - b_start >= L - EPS:
                return (b_start, b_end)
        j -= 1
    # plan B: search forward (after t0)
    k = np.searchsorted(SSTART, t0, side="left")
    if k < len(SSTART):
        dist = SSTART[k] - t0
        if dist <= max_dist and SDUR[k] >= L - EPS:
            b_start = SSTART[k]
            b_end   = b_start + L
            return (b_start, b_end)
    return (None, None)


# -------- Build MOVIE trials (NM1, NM3) into 30s repeats --------
movie_rows = []
for name in MOVIE_NAMES:
    mv = stim[stim["stimulus_name"] == name].copy()
    if mv.empty:
        print(f"[movie-trials] No rows for {name}")
        continue
    mv = mv.sort_values("start_time").reset_index(drop=True)
    mv["prev_stop"] = mv["stop_time"].shift(1)
    mv["gap"] = mv["start_time"] - mv["prev_stop"]
    mv["new_block"] = mv["gap"].isna() | (mv["gap"] > GAP_THRESHOLD)
    mv["block_index"] = mv["new_block"].cumsum() # 1-based within this movie name

    blocks = (mv.groupby("block_index")
                .agg(block_start=("start_time","min"),
                     block_end=("stop_time","max"),
                     n_frames=("start_time","size"))
                .reset_index())

    for _, b in blocks.iterrows():
        block_idx   = int(b["block_index"])
        block_start = float(b["block_start"])
        block_end   = float(b["block_end"])
        block_len   = block_end - block_start
        n_repeats = max(1, int(np.floor((block_len + EPS) / CLIP_SEC)))

        short = "one" if name.endswith("_one") else "three"

        for r in range(1, n_repeats + 1):
            t0 = block_start + (r - 1) * CLIP_SEC
            t1 = min(t0 + CLIP_SEC, block_end)
            trial_id = f"{short}_{block_idx}_{r}"
            # evoked = first 0.5s of repeat
            ev_start = t0
            ev_end   = t0 + W_SEC
            # baseline = last 0.5s of nearest prev spontaneous block
            bl_start, bl_end = baseline_from_spont_before(t0, B_SEC, MAX_SPONT_DIST)
            movie_rows.append({
                "trial_key": trial_id,
                "stimulus_name": name,
                "phase": "baseline",
                "start": bl_start,
                "end": bl_end,
                "block_index": block_idx,
                "repeat_index": r
            })
            movie_rows.append({
                "trial_key": trial_id,
                "stimulus_name": name,
                "phase": "evoked",
                "start": ev_start,
                "end": ev_end,
                "block_index": block_idx,
                "repeat_index": r
            })

movie_wins = pd.DataFrame(movie_rows)
# drop trials where baseline couldn't be found
if not movie_wins.empty:
    bad = movie_wins["start"].isna() | movie_wins["end"].isna()
    keep_trials = set(movie_wins.loc[~bad, "trial_key"].unique())
    movie_wins = movie_wins[movie_wins["trial_key"].isin(keep_trials)].copy()
    print(f"Dropped {bad.sum()} trials")

# --- De-overlap movie baselines by staggering inside each spontaneous block ---
# only baselines from movies
movie_baselines = movie_wins[movie_wins["phase"] == "baseline"].copy()
# group by spont block end time
adjusted_baselines = []
for _, group in movie_baselines.groupby("end"):
    group = group.sort_values("start")  # order by trial time
    for i, (_, row) in enumerate(group.iterrows()):
        new_end = row["end"] - i * B_SEC
        new_start = new_end - B_SEC
        row["start"] = new_start
        row["end"] = new_end
        adjusted_baselines.append(row)
# replace the original movie baselines
movie_wins.loc[movie_wins["phase"] == "baseline"] = pd.DataFrame(adjusted_baselines)


# -------- Build DRIFTING_GRATINGS onset windows (0.5/0.5) --------
dg = stim[stim["stimulus_name"] == "drifting_gratings"].copy()
dg = dg.sort_values("start_time").reset_index(drop=True)

# helper: last stop of ANY stimulus before a time
ALL_STOPS = stim["stop_time"].to_numpy(float)
def last_stop_before_any(t):
    idx = np.searchsorted(ALL_STOPS, float(t), side="left") - 1
    return ALL_STOPS[idx] if idx >= 0 else -np.inf

dg_rows = []
for _, r in dg.iterrows():
    t0 = float(r["start_time"])
    tstop = float(r["stop_time"])
    # evoked window
    ev_start = t0
    ev_end   = min(t0 + W_SEC, tstop)
    # baseline window: inside pre-onset gray (clip to previous stop)
    bl_end   = t0
    prev_end = last_stop_before_any(t0)
    bl_start = max(bl_end - B_SEC, prev_end)
    if (ev_end - ev_start) >= W_SEC - EPS and (bl_end - bl_start) >= B_SEC - EPS:
        trial_id = f"dg_{int(_)+1}"
        dg_rows.append({"trial_key": trial_id, "stimulus_name": "drifting_gratings",
                        "phase": "baseline", "start": bl_start, "end": bl_end})
        dg_rows.append({"trial_key": trial_id, "stimulus_name": "drifting_gratings",
                        "phase": "evoked", "start": ev_start, "end": ev_end})

dg_wins = pd.DataFrame(dg_rows)


# -------- Combine windows (movies + DG) --------
windows = pd.concat([dg_wins, movie_wins], ignore_index=True) if not dg_wins.empty else movie_wins.copy()
if windows.empty:
    raise RuntimeError("No valid windows built. Check stimuli and baseline rules.")

windows = windows.copy()
windows["duration"] = windows["end"] - windows["start"]
windows = windows[windows["duration"] > 0].reset_index(drop=True)

# give a numeric trial index for pivot compatibility, but keep the string key
windows["trial"] = pd.factorize(windows["trial_key"])[0].astype("int32") # assign numeric trial id


# -------- DEBUG & SANITY CHECKS delta_rate_hz --------
## do W exactly follow B for each trial?
wb = windows.pivot_table(index="trial", columns="phase", values=["start","end"], aggfunc="first")
wb.columns = [f"{a}_{b}" for a,b in wb.columns.to_flat_index()]
print("\nWindow sanity (first 10):")
print(wb.head(10))
## should be: end_baseline == start_evoked (or close)
mismatch = np.any(np.abs(wb["end_baseline"].to_numpy() - wb["start_evoked"].to_numpy()) > 1e-9)
print("Baseline end != Evoked start for any trial?:", mismatch)
## durations
print("\nDuration summary (sec):")
print(windows.groupby("phase")["duration"].describe())
## overlaps (no two windows should overlap)
w_sorted = windows.sort_values("start").reset_index(drop=True)
any_overlap = (w_sorted["start"].iloc[1:].to_numpy() < w_sorted["end"].iloc[:-1].to_numpy()).any()
print("Any overlaps among windows?:", any_overlap)


# =======================================================================================
#          MAP SPIKES
# =======================================================================================

# -------- Map spikes to windows with IntervalIndex --------
intervals = pd.IntervalIndex.from_arrays(
    windows["start"].to_numpy(),
    windows["end"].to_numpy(),
    closed="left")
idx = intervals.get_indexer(channel_unit_t_struct["spike_timestamp"].to_numpy())
valid = idx >= 0

map_df = pd.DataFrame({
    "interval_id": idx[valid].astype("int32"),
    "unit_id": channel_unit_t_struct.loc[valid, "unit_id"].astype("int32").to_numpy(),
})


# -------- Count spikes per (interval, unit) --------
counts = (
    map_df.value_counts(["interval_id", "unit_id"])
        .rename("count").reset_index()
)


# -------- Join window & unit metadata --------
win_meta = (
    windows.reset_index(names="interval_id")
        [["interval_id","trial","stimulus_name","phase","duration"]]
)
counts = counts.merge(win_meta, on="interval_id", how="left")

unit_struct = (
    channel_unit_t_struct[["unit_id","ecephys_structure_acronym"]]
    .drop_duplicates("unit_id")
)
counts = counts.merge(unit_struct, on="unit_id", how="left")

# -------- Compute rates --------
## spikes per second (count / duration)
counts["rate_hz"] = counts["count"] / counts["duration"].replace(0, np.nan)


# -------- Pivot baseline vs evoked side-by-side per (trial, unit) --------
pivot = (
    counts.pivot_table(
        index=["trial","stimulus_name","unit_id","ecephys_structure_acronym"],
        columns="phase",
        values=["count","rate_hz"],
        aggfunc="first"
    )
)
pivot.columns = [f"{a}_{b}" for a,b in pivot.columns.to_flat_index()]
pivot = pivot.reset_index()

# fill NaNs for counts where a unit only fired in one window
pivot["count_baseline"] = pivot["count_baseline"].fillna(0)
pivot["count_evoked"] = pivot["count_evoked"].fillna(0)

# deltas
pivot["delta_count"] = pivot["count_evoked"] - pivot["count_baseline"]
pivot["delta_rate_hz"] = pivot["rate_hz_evoked"] - pivot["rate_hz_baseline"]

# save analysis table
analysis_out = sesh_path / f"s{session_id}_pivot_counts_rates_by_region.csv"
pivot.to_csv(analysis_out, index=False) 
print(f"Saved analysis table to: {analysis_out}")


# =======================================================================================
#          REPORTS & PLOTS
# =======================================================================================

# helper
def _safe(s: str) -> str: #######################################UNUSED
    # make strings safe for filenames
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s))

print(pivot.columns)

# -------- Plot of distribution --------
plot_outdir = sesh_path / f"session_{session_id}_plots"
plot_outdir.mkdir(parents=True, exist_ok=True)

# pick "top" regions by how many unit-trials they contain
region_counts = pivot["ecephys_structure_acronym"].value_counts()
regions = region_counts.head(8).index.tolist()
if not regions:
    print("[plots] No regions available to plot.")
else:
    print("[plots] Regions selected:", regions)

stim_order = ["drifting_gratings", "natural_movie_one", "natural_movie_three"]
present_stims = [s for s in stim_order if s in pivot["stimulus_name"].unique()]
if not present_stims:
    print("[plots] No known stimuli present among", stim_order)

# color map for the three stimuli
stim_color = {
    "drifting_gratings": "tab:orange",
    "natural_movie_one": "tab:blue",
    "natural_movie_three": "tab:green"
}

# -- A) delta-rate histograms by stimulus per region --
cols = 4
rows = math.ceil(len(regions)/cols) if regions else 1
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), squeeze=False)

for i, r in enumerate(regions):
    ax = axes[i//cols][i%cols]

    # gather per-stimulus delta-arrays for this region
    deltas_by_stim = {}
    for s in present_stims:
        vals = pivot.loc[
            (pivot["ecephys_structure_acronym"] == r) &
            (pivot["stimulus_name"] == s),
            "delta_rate_hz"
        ].dropna().to_numpy()
        if vals.size:
            deltas_by_stim[s] = vals
    if not deltas_by_stim:
        ax.set_title(f"{r} (no data)")
        ax.axis("off")
        continue

    # shared bins across stimuli within this region
    all_vals = np.concatenate(list(deltas_by_stim.values()))
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, max(1.0, vmax if np.isfinite(vmax) else 1.0)
    bins = np.linspace(vmin, vmax, 41)

    # overlay
    for s, vals in deltas_by_stim.items():
        ax.hist(vals, bins=bins, alpha=0.5, label=f"{s} (n={len(vals)})",
                color=stim_color.get(s, None))
    ax.axvline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_title(f"{r} (n={len(vals)})")
    ax.set_xlabel("delta rate (Hz) [evoked - baseline]")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=8)
# hide empty panels
last_i = len(regions) - 1
for k in range(last_i + 1, rows*cols):
    axes[k//cols][k%cols].axis("off")

fig.suptitle(f"Delta firing rate by region (overlayed by stimulus) - session {session_id}", y=1.02)
plt.tight_layout()
out_path = plot_outdir / f"s{session_id}_delta_rate_by_region.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"[plots] Saved delta histograms by stimulus: {out_path}")

# -- A2) Grouped bar chart of median delta per region × stimulus (unit-weighted) --
# unit-weighted per-unit summary for fair region ranking
per_unit = (
    pivot.groupby(["unit_id","ecephys_structure_acronym","stimulus_name"], as_index=False)
         .agg(delta_rate_hz_mean=("delta_rate_hz","mean"),
              n_trials=("delta_rate_hz","size"))
)
summ_unit = (
    per_unit.groupby(["ecephys_structure_acronym", "stimulus_name"], as_index = False)
        .agg(
            n_units=('delta_rate_hz_mean', 'size'),
            median_delta=('delta_rate_hz_mean', 'median'),
            mean_delta=('delta_rate_hz_mean', 'mean'),
            frac_pos=('delta_rate_hz_mean', lambda x: (x > 0).mean())
        )
)

if regions and present_stims:
    plot_summ = summ_unit[summ_unit["ecephys_structure_acronym"].isin(regions)].copy()
    # keep consistent stimulus order
    plot_summ["stimulus_name"] = pd.Categorical(plot_summ["stimulus_name"],
                                                categories=present_stims, ordered=True)
    plot_summ = plot_summ.sort_values(["ecephys_structure_acronym","stimulus_name"])

    # bar positions
    reg_list = [r for r in regions if r in plot_summ["ecephys_structure_acronym"].unique()]
    x = np.arange(len(reg_list))
    width = 0.25 if len(present_stims) == 3 else 0.35

    fig, ax = plt.subplots(figsize=(max(8, 1.5*len(reg_list)), 4.5))
    for j, s in enumerate(present_stims):
        sub = plot_summ[plot_summ["stimulus_name"] == s]
        # align to reg_list order; fill missing with 0 and mark n_units for labels
        med_map = {k: v for k, v in zip(sub["ecephys_structure_acronym"], sub["median_delta"])}
        n_map   = {k: v for k, v in zip(sub["ecephys_structure_acronym"], sub["n_units"])}
        y = np.array([med_map.get(r, np.nan) for r in reg_list], dtype=float)
        n = np.array([n_map.get(r, 0) for r in reg_list], dtype=int)
        pos = x + (j - (len(present_stims)-1)/2.0) * width
        ax.bar(pos, y, width=width, label=f"{s}", color=stim_color.get(s, None))
        # annotate n_units above bars
        for xp, yp, nn in zip(pos, y, n):
            if np.isfinite(yp):
                ax.text(xp, yp + (0.02 if yp >= 0 else -0.02), f"n={nn}", ha="center", va="bottom" if yp>=0 else "top", fontsize=8)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(reg_list, rotation=30, ha="right")
    ax.set_ylabel("median delta rate (Hz)\n(unit-weighted)")
    ax.set_title(f"Region x Stimulus preference — session {session_id}")
    ax.legend(frameon=False, ncol=len(present_stims))
    plt.tight_layout()
    out_path2 = plot_outdir / f"s{session_id}_median_delta_by_region_by_stim.png"
    plt.savefig(out_path2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved grouped bar chart (median unit delta): {out_path2}")

# -- B) per-stimulus grids: baseline vs. evoked within regions --
if regions and present_stims:
    for stim_name in present_stims:
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), squeeze=False)
        for i, r in enumerate(regions):
            ax = axes[i//cols][i%cols]
            sub = pivot[(pivot["ecephys_structure_acronym"] == r) &
                        (pivot["stimulus_name"] == stim_name)]
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

        #hide empty panels
        last_i = len(regions) - 1
        for k in range(last_i + 1, rows*cols):
            axes[k//cols][k%cols].axis("off")

        fig.suptitle(f"Baseline vs Evoked firing rate by region - {stim_name} - session {session_id}", y=1.02)
        plt.tight_layout()
        out_path_b = plot_outdir / f"s{session_id}_baseline_vs_evoked_by_region_{stim_name}.png"
        plt.savefig(out_path_b, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[plots] Saved baseline-vs-evoked grid for {stim_name}: {out_path_b}")
# =================================================================

# -------- TEXT SUMMARIES --------
# --- stimulus-aware trial-weighted summary (top 5 per stimulus) ---
region_summary = (
    pivot.groupby([ "ecephys_structure_acronym", "stimulus_name"])
    .agg(mean_delta=("delta_rate_hz", "mean"),
         median_delta=("delta_rate_hz", "median"),
         n=("delta_rate_hz", "size"))
    .reset_index()
    .sort_values(by=["stimulus_name", "mean_delta"], ascending=[True,False])
)
top5_per_stim = region_summary.groupby("stimulus_name", group_keys=False).head(5)
print("\nTop regions per stimuli by mean delta rate (Hz):")
print(top5_per_stim)

region_stim_unitweighted = (
    per_unit.groupby(["ecephys_structure_acronym","stimulus_name"])
            .agg(n_units=("delta_rate_hz_mean","size"),
                 mean_delta=("delta_rate_hz_mean","mean"),
                 median_delta=("delta_rate_hz_mean","median"),
                 frac_pos=("delta_rate_hz_mean", lambda x: (x > 0).mean()))
            .reset_index()
            .sort_values(by=["stimulus_name","mean_delta"], ascending=[True,False])
)
print("\nUnit-weighted Region x Stimulus summary:")
print(region_stim_unitweighted.head(20).to_string(index=False))

print("done")