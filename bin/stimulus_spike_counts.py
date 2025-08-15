import sys
import numpy as np
import pandas as pd
from pathlib import Path

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

#series = channels["ecephys_structure_acronym"].unique()

# -------- Keep only cortex visual-proessing areas --------
VIS_AREAS = {
    "VISrl", "VIS", "VISp", "VISpm", "VISam", "VISal",
    "VISl", "VISmma", "VISmmp", "VISli", "POL"
}

# filder channels to desired regions
channels = channels[channels["ecephys_structure_acronym"].isin(VIS_AREAS)].copy()
# keep only units that sit on those channels
keep_chan_ids = set(channels["id"].tolist())
units = units[units["ecephys_channel_id"].isin(keep_chan_ids)].copy()
# keep only spikes from those units (save time)
keep_unit_ids = set(units["id"].tolist())
spikes = spikes[spikes["unit"].isin(keep_unit_ids)].copy()

print(f"Filtering VIS_AREAS: Kept {len(channels)} channels, {len(units)} units, {len(spikes)} spikes in visual areas.")

# -------- Filter stimuli for this project's analysis --------
## filter for only specific stimulus types (project looking at drifting_gratings vs. natural_movies)
''''
stim["stimulus_name"] = stim["stimulus_name"].replace({
    "natural_movie_one": "natural_movies",
    "natural_movie_three": "natural_movies"
})
stim = stim[stim["stimulus_name"].isin(["drifting_gratings", "natural_movies"])].reset_index(drop=True)
'''

#################### ####################
print("#################### ####################")
print("\nCounts per stimulus BEFORE window building:")
print(stim["stimulus_name"].value_counts(dropna=False).rename_axis("stimulus").to_frame("n"))
print("#################### ####################")
#################### ####################

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

# ------------ Build trial windows & counts --------
needed = {"start_time", "stop_time", "stimulus_name"}
missing = needed - set(stim.columns)
if missing:
    raise ValueError(f"stimulus_presentations.csv missing required columns: {missing}")

stim = stim.reset_index(drop=True)
stim["duration"] = stim["stop_time"] - stim["start_time"]

# -------- Define evoked (W) and baseline (B) windows --------
BASELINE_SEC = 0.25 # 500 ms baseline window
EVOKED_SEC = 0.25   # 250 ms post-onset for transient response
# evoked_sec picked smaller than shortest stimulus ######################################

w_start = stim["start_time"].to_numpy()             # evoked start
stim_end = stim["stop_time"].to_numpy()
w_end = np.minimum(w_start + EVOKED_SEC, stim_end)  # evoked end

# B = Baseline window
## activity just before stimulus starts ("control period" neuron at rest)
b_end = w_start
b_start_exact = b_end - BASELINE_SEC

min_spike_t = float(channel_unit_t_struct["spike_timestamp"].min())
prev_w_end = np.r_[-np.inf, w_end[:-1]] # previous trial's evoked end
# forces baseline start to be no earlier than prev stimulus end; prevent overlapping
b_start = np.maximum(b_start_exact, np.maximum(min_spike_t, prev_w_end))
'''
# ---- Drop truncated trials (where B or W < intended length) ----
# keep a copy before filtering
stim_raw = stim.copy()
#############################

b_dur = b_end - b_start
w_dur = w_end = w_start
eps = 1e-9  # float tolerance

keep = (b_dur >= BASELINE_SEC - eps) & (w_dur >= EVOKED_SEC - eps)

total = len(stim)
kept  = int(keep.sum())
dropped = total - kept
print(f"Trials total: {total} | kept: {kept} ({kept/total:.1%}) | dropped (truncated): {dropped} ({dropped/total:.1%})")

# Filter stim and arrays to *kept* trials
w_start = w_start[keep]; w_end = w_end[keep]
b_start = b_start[keep]; b_end = b_end[keep]
stim = stim.loc[keep].reset_index(drop=True)

#################### ####################
print("#################### ####################")
print("\nCounts per stimulus AFTER dropping truncated windows:")
print(stim["stimulus_name"].value_counts(dropna=False).rename_axis("stimulus").to_frame("n"))
#################### ####################
#################### ####################
# kept/dropped report by stimulus (using the original stim index alignment)

# ... later, when you have 'keep' boolean for the same rows in stim_raw ...
drop_report = (
    pd.DataFrame({"stimulus_name": stim_raw["stimulus_name"], "keep": keep})
      .groupby(["stimulus_name","keep"])
      .size()
      .unstack(fill_value=0)
      .rename(columns={False:"dropped", True:"kept"})
)
print("\nKept vs Dropped by stimulus:")
print(drop_report)
print("#################### ####################")
#################### ####################
'''
############################
'''
dropped_mask = ~keep
print(f"Kept {keep.sum()} / {len(keep)} trials "
      f"({keep.mean():.1%}); Dropped {dropped_mask.sum()} ({dropped_mask.mean():.1%}).")

# Where did drops come from? (by stimulus type)
stim_drop_report = (
    stim_raw.assign(keep=keep)
            .groupby(["stimulus_name","keep"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={False:"dropped", True:"kept"})
)
print("\nDrop report by stimulus:")
print(stim_drop_report)
'''

# see the adjusted baselines
#B = pd.DataFrame({"trial": stim.index, "stim": stim["stimulus_name"], "start": b_start, "end": b_end})
#W = pd.DataFrame({"trial": stim.index, "stim": stim["stimulus_name"], "start": w_start, "end": w_end})

# B/W windows: 2 rows per trial (baseline, evoked)
## e.g.  drifting_gratings   baseline    8.0     10.0
##       drifting_gratings   evoked      10.0    12.0
windows = pd.DataFrame({
    "trial": np.repeat(stim.index.values, 2),
    "stimulus_name": np.repeat(stim["stimulus_name"].values, 2),
    "phase": np.tile(["baseline", "evoked"], len(stim)),
    "start": np.concatenate([b_start,   w_start]),
    "end":   np.concatenate([b_end,     w_end]),
})
windows["duration"] = windows["end"] - windows["start"]
# drop any zero/negative-duration windows
windows = windows[windows["duration"] > 0].reset_index(drop=True)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# ---------- Stimulus timeline (condensed, with explicit NM1/NM3) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# exactly the three you want, plotted separately
PLOT_NAMES = ["drifting_gratings", "natural_movie_one", "natural_movie_three"]
GAP_THR = 1.0  # merge frames within 1s into a single block; bump if still "toothpicks"

st = stim[stim["stimulus_name"].isin(PLOT_NAMES)].copy()
st = st.sort_values(["stimulus_name", "start_time"]).reset_index(drop=True)

# Condense adjacent rows of the same stimulus if the gap is small
rows = []
for name, g in st.groupby("stimulus_name"):
    g = g.sort_values("start_time").reset_index(drop=True)
    cur_start = cur_end = None
    for _, r in g.iterrows():
        s, e = float(r["start_time"]), float(r["stop_time"])
        if cur_start is None:
            cur_start, cur_end = s, e
        else:
            gap = s - cur_end
            if gap <= GAP_THR:
                cur_end = max(cur_end, e)
            else:
                rows.append((name, cur_start, cur_end))
                cur_start, cur_end = s, e
    if cur_start is not None:
        rows.append((name, cur_start, cur_end))

condensed = pd.DataFrame(rows, columns=["stimulus_name", "start", "end"])
condensed["duration"] = condensed["end"] - condensed["start"]

# Figure out which of the requested names actually have segments
present_names = condensed["stimulus_name"].unique().tolist() if not condensed.empty else []
missing_names = [n for n in PLOT_NAMES if n not in present_names]
if missing_names:
    print(f"(timeline) No segments for: {missing_names}")

# Build lane positions dynamically to avoid KeyError
ordered_names = [n for n in PLOT_NAMES if n in present_names]
if not ordered_names:
    print("(timeline) Nothing to plot after condensing.")
else:
    y0, step, h = 10, 12, 8
    lane_y = {name: y0 + i * step for i, name in enumerate(ordered_names)}

    # simple color map; auto-pick if not specified
    base_colors = {
        "drifting_gratings": "tab:orange",
        "natural_movie_one": "tab:blue",
        "natural_movie_three": "tab:green",
    }

    fig, ax = plt.subplots(figsize=(12, 3.5))
    for name in ordered_names:
        segs = condensed[condensed["stimulus_name"] == name]
        spans = [(float(s), float(d)) for s, d in zip(segs["start"], segs["duration"])]
        if spans:
            ax.broken_barh(spans, (lane_y[name], h), facecolors=base_colors.get(name, None))

    ax.set_xlabel("Time (s)")
    ax.set_yticks([lane_y[n] + h/2 for n in ordered_names])
    ax.set_yticklabels(ordered_names)
    ax.set_title(f"Stimulus timeline (condensed) — session {session_id}")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6)

    plot_outdir = sesh_path / f"session_{session_id}_plots"
    plot_outdir.mkdir(parents=True, exist_ok=True)
    tl_path = plot_outdir / f"s{session_id}_stim_timeline.png"
    plt.tight_layout()
    plt.savefig(tl_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved condensed stimulus timeline to: {tl_path}")
print("# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$




# ===================================================================
# ---- DEBUG & SANITY CHECKS delta_rate_hz ----

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

## is evoked shorter than the actual stimulus anywhere?
min_stim_len = (stim["stop_time"] - stim["start_time"]).min()
print("Shortest stimulus duration (sec):", float(min_stim_len))
print("Your EVOKED_SEC:", float(EVOKED_SEC))
print('-----------------------------------------------------------------')
print("Duration summary (sec) after fix:")
print(windows.groupby("phase")["duration"].describe())
# ===================================================================

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

# ==========================================================================================
# -------- Plot of distribution --------
import math
import matplotlib
matplotlib.use("Agg") # headless mode
import matplotlib.pyplot as plt

print("Plotting distributions...")

plot_outdir = sesh_path / f"session_{session_id}_plots"
plot_outdir.mkdir(parents=True, exist_ok=True)

# pick regions to show #############################################################
region_counts = pivot["ecephys_structure_acronym"].value_counts()
regions = region_counts.head(8).index.tolist()  # tweak how many you want

# -------- A) delta-rate histograms per region --------
cols = 4
rows = math.ceil(len(regions)/cols) if regions else 1
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), squeeze=False)

for i, r in enumerate(regions):
    ax = axes[i//cols][i%cols]
    vals = pivot.loc[pivot["ecephys_structure_acronym"] == r, "delta_rate_hz"].dropna().to_numpy()
    if len(vals) == 0:
        ax.set_title(f"{r} (no data)")
        ax.axis("off")
        continue
    ax.hist(vals, bins=40)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title(f"{r} (n={len(vals)})")
    ax.set_xlabel("delta rate (Hz)")
    ax.set_ylabel("count")
# hide empty panels
for k in range(i+1, rows*cols):
    axes[k//cols][k%cols].axis("off")

fig.suptitle(f"delta firing rate (evoked - baseline) by region - session {session_id}", y=1.02)
plt.tight_layout()
out_path = plot_outdir / f"s{session_id}_delta_rate_by_region.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved plot: {out_path}")

# -------- B) baseline vs evoked per region (overlaid) --------
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), squeeze=False)

for i, r in enumerate(regions):
    ax = axes[i//cols][i%cols]
    sub = pivot[pivot["ecephys_structure_acronym"] == r]
    b = sub["rate_hz_baseline"].dropna().to_numpy()
    w = sub["rate_hz_evoked"].dropna().to_numpy()
    if len(b) == 0 and len(w) == 0:
        ax.set_title(f"{r} (no data)")
        ax.axis("off")
        continue
    # overlay histograms; matplotlib will auto-pick colors
    ax.hist(b, bins=40, alpha=0.5, label="baseline")
    ax.hist(w, bins=40, alpha=0.5, label="evoked")
    ax.set_title(f"{r} (n={len(sub)})")
    ax.set_xlabel("rate (Hz)")
    ax.set_ylabel("count")
    ax.legend(frameon=False)
# hide empty panels
for k in range(i+1, rows*cols):
    axes[k//cols][k%cols].axis("off")

fig.suptitle("Baseline vs Evoked firing rate by region - session {session_id}", y=1.02)
plt.tight_layout()
out_path2 = plot_outdir / f"s{session_id}_baseline_vs_evoked_by_region.png"
plt.savefig(out_path2, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved plot: {out_path2}")

print("Distribution plotting complete.")
# ==========================================================================================

# -------- small text summary --------
# -------- Stimulus-aware trial-weighted summary (top 5 per stimulus) --------
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
#------------------------
# -------- Unit-weighted summary (one vote per neuron per stimulus) --------
per_unit = (
    pivot.groupby(["unit_id","ecephys_structure_acronym","stimulus_name"], as_index=False)
         .agg(delta_rate_hz_mean=("delta_rate_hz","mean"),
              n_trials=("delta_rate_hz","size"))
)
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
print(region_stim_unitweighted.head(20))

print("done")