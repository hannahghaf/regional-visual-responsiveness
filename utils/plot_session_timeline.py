import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ===== Plotting full session timeline with ALL stimuli + gaps =====

session_id = sys.argv[1]

project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"
sesh_path = data_root / f"session_{session_id}"

raw = pd.read_csv(sesh_path / "stimulus_presentations.csv", low_memory=False)
raw = raw.sort_values("start_time").reset_index(drop=True)

# --- compute gaps across the whole session ---
st = raw["start_time"].to_numpy(dtype=float)
et = raw["stop_time"].to_numpy(dtype=float)
gap_start = et[:-1]
gap_end   = st[1:]
gaps = pd.DataFrame({"gap_start": gap_start, "gap_end": gap_end})
gaps["gap_dur"] = gaps["gap_end"] - gaps["gap_start"]
gaps = gaps[gaps["gap_dur"] > 0].reset_index(drop=True)  # only positive gaps

# --- condense segments per stimulus so brief breaks merge into blocks ---
GAP_THR = 0.5  # seconds: merge if the gap <= this
rows = []
for name, g in raw.groupby("stimulus_name"):
    g = g.sort_values("start_time")
    cur_s = cur_e = None
    for s, e in zip(g["start_time"], g["stop_time"]):
        s = float(s); e = float(e)
        if cur_s is None:
            cur_s, cur_e = s, e
        else:
            gap = s - cur_e
            if gap <= GAP_THR:
                cur_e = max(cur_e, e)
            else:
                rows.append((name, cur_s, cur_e))
                cur_s, cur_e = s, e
    if cur_s is not None:
        rows.append((name, cur_s, cur_e))

condensed = pd.DataFrame(rows, columns=["stimulus_name","start","end"])
condensed["dur"] = condensed["end"] - condensed["start"]

# --- build lanes dynamically, ordered by total duration (longest on top) ---
order = (condensed.groupby("stimulus_name")["dur"].sum()
         .sort_values(ascending=False).index.tolist())
y0, step, h = 8, 10, 6
lane_y = {name: y0 + i*step for i, name in enumerate(order)}

# --- colors per stimulus (cycle if many) ---
from itertools import cycle
palette = cycle(plt.cm.tab20.colors)
colors = {name: next(palette) for name in order}

# --- plot ---
fig, ax = plt.subplots(figsize=(14, 4 + 0.25*len(order)))

# draw gaps as gray horizontal bands spanning all lanes
if not gaps.empty:
    y_min = y0 - 3
    y_max = y0 + (len(order)-1)*step + h + 3
    for gs, ge, gd in gaps[["gap_start","gap_end","gap_dur"]].itertuples(index=False):
        ax.add_patch(plt.Rectangle((gs, y_min), ge-gs, y_max-y_min,
                                   facecolor="lightgray", alpha=0.25, edgecolor="none"))

# draw condensed stimulus segments per lane
for name in order:
    segs = condensed[condensed["stimulus_name"] == name]
    spans = [(float(s), float(d)) for s, d in zip(segs["start"], segs["dur"])]
    if spans:
        ax.broken_barh(spans, (lane_y[name], h), facecolors=colors[name], edgecolors="none")
        # outline for readability
        ax.broken_barh(spans, (lane_y[name], h), facecolors="none", edgecolors="k", linewidth=0.3)

ax.set_xlabel("Time (s)")
ax.set_yticks([lane_y[n] + h/2 for n in order])
ax.set_yticklabels(order)
ax.set_title(f"Full stimulus timeline (condensed) + gaps — session {session_id}")
ax.grid(True, axis="x", linestyle=":", linewidth=0.6)

out_dir = sesh_path / f"session_{session_id}_plots"
out_dir.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
out_path = out_dir / f"s{session_id}_stim_timeline_ALL_with_gaps.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved full-session timeline with gaps to: {out_path}")

# quick text summary of gaps
print("\nLargest gaps (top 20):")
print(gaps.sort_values("gap_dur", ascending=False).head(20))