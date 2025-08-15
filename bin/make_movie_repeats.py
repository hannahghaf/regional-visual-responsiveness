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
stim_all = pd.read_csv(sesh_path / "stimulus_presentations.csv", low_memory=False)

# Keep only columns we need and sort by time once
stim_all = stim_all[["stimulus_name","start_time","stop_time"]].sort_values("start_time").reset_index(drop=True)

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