import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Get session_id from first arg ----
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <session_id>")
    sys.exit(1)

session_id = sys.argv[1]

# ---- Paths -----
project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"
sesh_path = data_root / f"session_{session_id}"

# ---- Load files ----
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

# filter for only specific stimulus types (project looking at drifting_gratings & natural_scenes)
stim = stim[stim["stimulus_name"].isin(["drifting_gratings", "natural_scenes"])].reset_index(drop=True)

# ---- Merge spike times w unit and channel metadata ----
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

# ---- Build trial windows & counts ----
needed = {"start_time", "stop_time", "stimulus_name"}
missing = needed - set(stim.columns)
if missing:
    raise ValueError(f"stimulus_presentations.csv missing required columns: {missing}")

stim = stim.reset_index(drop=True)
stim["duration"] = stim["stop_time"] - stim["start_time"]

# ---- Define evoked (W) and baseline (B) windows ----
w_start = stim["start_time"].to_numpy()     # evoked start
w_end = stim["stop_time"].to_numpy()        # evoked end

# B = Baseline window
## activity just before stimulus starts ("control period" neuron at rest)
min_spike_t = float(channel_unit_t_struct["spike_timestamp"].min())
b_start_clipped = np.maximum(
    (stim["start_time"] - stim["duration"]).to_numpy(), 
    min_spike_t
)
b_end = stim["start_time"].to_numpy()

# prevent baseline from overlapping the previous trial's ekoved
prev_w_end = np.r_[-np.inf, w_end[:-1]]  # previous trial's evoked end, aligned to current trial
b_start_nolap = np.maximum(b_start_clipped, prev_w_end)

####################################optional debug
######### pptional: see the adjusted baselines
B = pd.DataFrame({"trial": stim.index, "start": b_start_clipped, "end": b_end})
W = pd.DataFrame({"trial": stim.index, "start": w_start,         "end": w_end})
print(B.head(), W.head())
print('B and W ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

# B/W windows: 2 rows per trial (baseline, evoked)
## e.g.  drifting_gratings   baseline    8.0     10.0
##       drifting_gratings   evoked      10.0    12.0
n = len(stim)
windows = pd.DataFrame({
    "trial": np.repeat(stim.index.values, 2),
    "stimulus_name": np.repeat(stim["stimulus_name"].values, 2),
    "phase": np.tile(["baseline", "evoked"], n),
    "start": np.concatenate([b_start_nolap, w_start]),
    "end":   np.concatenate([b_end, w_end]),
})
windows["duration"] = windows["end"] - windows["start"]

######### limit to just drifting_gratings and natural_scenes
print(windows.head()) ##########################################
print("WINDOWS HEAD 1 ------------------------------------------------")

# drop any zero/negative-duration windows
windows = windows[windows["duration"] > 0].reset_index(drop=True)

print(windows.head())
print("WINDOWS HEAD drop zeros ------------------------------------------------")

# ---- Map spikes to windows with IntervalIndex ----
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

# ---- Count spikes per (interval, unit) ----
counts = (
    map_df.value_counts(["interval_id", "unit_id"])
        .rename("count").reset_index()
)

# ---- Join window & unit metadata ----
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

# ---- Computer rates ----
## spikes per second (count / duration)
counts["rate_hz"] = counts["count"] / counts["duration"].replace(0, np.nan)

# ---- Pivot baseline vs evoked side-by-side per trial x unit ---- ##################################
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
########################
pivot["count_baseline"] = pivot["count_baseline"].fillna(0)
pivot["count_evoked"] = pivot["count_evoked"].fillna(0)

# deltas
pivot["delta_count"] = pivot["count_evoked"] - pivot["count_baseline"]
pivot["delta_rate_hz"] = pivot["rate_hz_evoked"] - pivot["rate_hz_baseline"]

print("==================")
print(pivot.head())
print(pivot.columns)
print("pivot head --------------------------------------------")

# save analysis table
#analysis_out = sesh_path / f"s{session_id}_trial_unit_rates_by_region.csv"
#pivot.to_csv(analysis_out, index=False)
#print(f"Saved analysis table to: {analysis_out}")

# Example quick sanity checks
print(pivot.head(5))
print(
    pivot.groupby(["ecephys_structure_acronym","stimulus_name"])["delta_rate_hz"]
         .mean()
         .sort_values(ascending=False)
         .head(10)
)