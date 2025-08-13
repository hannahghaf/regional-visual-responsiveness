import sys
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

print(channel_unit_t_struct.head())
print(channel_unit_t_struct.columns)
print(channel_unit_t_struct.shape)

# ---- Build baseline & evoked windows from stimulus_presentations.csv ----