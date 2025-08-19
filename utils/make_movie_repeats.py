import sys
import numpy as np
import pandas as pd
from pathlib import Path

session_id = sys.argv[1]

project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"
sesh_path = data_root / f"session_{session_id}"

# --- Parameters you can tune ---
MOVIE_NAMES   = ["natural_movie_one", "natural_movie_three"]
CLIP_SEC      = 30.0     # expected clip duration (s)
GAP_THRESHOLD = 0.10     # gap > this (s) = new movie block
BASELINE_SEC  = 0.50     # block-level baseline length (s)
EPS           = 1e-6

# Use the RAW stimulus table to avoid side-effects from earlier filters/renames
stim_df = pd.read_csv(sesh_path / "stimulus_presentations.csv", low_memory=False)

# Keep only columns we need and sort by time once
stim_all = stim_df[["stimulus_name","start_time","stop_time"]].sort_values("start_time").reset_index(drop=True)

# Helper: for a given time t, find last stop_time before t (to clip the baseline)
prev_stop_all = stim_all["stop_time"].to_numpy()

def last_stop_before(t):
    # binary search via numpy since stim_all is sorted
    idx = np.searchsorted(prev_stop_all, t, side="left") - 1
    return prev_stop_all[idx] if idx >= 0 else -np.inf

rows = []

for name in MOVIE_NAMES:
    mv = stim_all[stim_all["stimulus_name"] == name].copy()
    if mv.empty:
        print(f"[movie-trials] No rows for {name}")
        continue

    mv = mv.sort_values("start_time").reset_index(drop=True)
    # Mark movie blocks (contiguous frames; gap > threshold => new block)
    mv["prev_stop"] = mv["stop_time"].shift(1)
    mv["gap"] = mv["start_time"] - mv["prev_stop"]
    mv["new_block"] = mv["gap"].isna() | (mv["gap"] > GAP_THRESHOLD)
    mv["block_index"] = mv["new_block"].cumsum()  # 1-based within this movie name

    blocks = (mv.groupby("block_index")
                .agg(block_start=("start_time","min"),
                     block_end=("stop_time","max"),
                     n_frames=("start_time","size"))
                .reset_index())

    if blocks.empty:
        print(f"[movie-trials] No blocks detected for {name}")
        continue

    # Build block-level baseline once per block, then slice into 30 s repeats
    for _, b in blocks.iterrows():
        block_idx   = int(b["block_index"])
        block_start = float(b["block_start"])
        block_end   = float(b["block_end"])
        block_len   = block_end - block_start

        # ---- baseline for this block (shared by all repeats in this block) ----
        baseline_end   = block_start
        baseline_start = baseline_end - BASELINE_SEC

        # clip baseline to avoid overlap with the previous stimulus
        prev_stim_end  = last_stop_before(block_start - EPS)
        if np.isfinite(prev_stim_end):
            baseline_start = max(baseline_start, prev_stim_end)

        baseline_duration = max(0.0, baseline_end - baseline_start)

        # optional: warn if baseline shorter than requested
        if baseline_duration + 1e-9 < BASELINE_SEC:
            print(f"[movie-trials] WARNING: baseline shortened in {name} block {block_idx} "
                  f"({baseline_duration:.3f}s < {BASELINE_SEC:.3f}s).")

        # ---- subdivide the block into ~30 s repeats ----
        # how many full repeats fit? (allow small slack)
        n_repeats = int(np.floor((block_len + EPS) / CLIP_SEC))
        if n_repeats <= 0:
            # if the block is shorter than one clip, still emit one truncated repeat
            n_repeats = 1

        # short label for trial_id (one/three)
        short = "one" if name.endswith("_one") else "three"

        for r in range(1, n_repeats + 1):  # 1-based repeat index
            t0 = block_start + (r - 1) * CLIP_SEC
            t1 = min(t0 + CLIP_SEC, block_end)
            trial_id = f"{short}_{block_idx}_{r}"
            rows.append({
                "trial_id": trial_id,
                "stimulus_name": name,
                "block_index": block_idx,
                "repeat_index": r,
                "start_time": t0,
                "end_time": t1,
                "clip_duration": t1 - t0,
                "baseline_start": baseline_start,
                "baseline_end": baseline_end,
                "baseline_duration": baseline_duration,
                "block_start": block_start,
                "block_end": block_end,
                "block_duration": block_len,
                "n_frames_block": int(b["n_frames"]),
            })

movie_trials = pd.DataFrame(rows).sort_values(["stimulus_name","block_index","repeat_index"]).reset_index(drop=True)

print("\n[movie-trials] summary counts:")
print(movie_trials.groupby(["stimulus_name","block_index"]).size().rename("repeats_per_block"))
print("\n[movie-trials] head:")
print(movie_trials.tail(25))
print(len(movie_trials))
movie_trials.to_csv(sesh_path / "movie_trials.csv")


# =======================================================================================

def find_stimulus_gaps(stim_df):
    """
    stim_df: full stimulus_presentations.csv as DataFrame
    - must have 'start_time' and 'stop_time' in seconds
    
    returns a DataFrame of all gaps between any two stimuli,
    with the gap start/end/duration, sorted by gap duration
    """
    # sort by start time
    stim_df = stim_df.sort_values("start_time").reset_index(drop=True)
    
    # get all stop times and next start times
    stops = stim_df["stop_time"].values
    starts = stim_df["start_time"].values[1:]  # next stimulus start
    
    # gap start is prev stimulus stop, gap end is next stimulus start
    gap_starts = stops[:-1]
    gap_ends = starts
    gap_durations = gap_ends - gap_starts
    
    gaps_df = pd.DataFrame({
        "gap_start": gap_starts,
        "gap_end": gap_ends,
        "gap_duration": gap_durations
    })
    
    # only keep positive gaps (drop overlaps or zero)
    gaps_df = gaps_df[gaps_df["gap_duration"] > 0].reset_index(drop=True)
    
    return gaps_df.sort_values("gap_duration", ascending=False)


gaps_df = find_stimulus_gaps(stim_df)

print(f"Found {len(gaps_df)} gaps")
print(gaps_df.head(20))  # top 20 biggest gaps

print(gaps_df)

# ==============================================
# determine gaps between drifting_gratings frames

dg = stim_df[stim_df["stimulus_name"] == "drifting_gratings"].sort_values("start_time")

# calculate gap from previous drifting grating
dg["gap_from_prev_dg"] = dg["start_time"] - dg["stop_time"].shift(1)

# calculate gap from any previous stimulus
dg_sorted = stim_df.sort_values("start_time")
prev_stop_any = []
for start in dg["start_time"]:
    prev_stops = dg_sorted[dg_sorted["stop_time"] <= start]["stop_time"]
    prev_stop_any.append(prev_stops.max() if not prev_stops.empty else None)

dg["gap_from_prev_any"] = dg["start_time"] - prev_stop_any
dg_gaps = dg[["start_time", "stop_time", "gap_from_prev_dg", "gap_from_prev_any"]]

print(dg_gaps)