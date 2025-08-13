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
spikes = pd.read_csv(sesh_path / "spike_times.csv")
units = pd.read_csv(data_root / "units.csv")
channels = pd.read_csv(data_root / "channels.csv")

# ---- Merge spike times w unit and channel metadata ----
units_min = units[['id', 'ecephys_channel_id']]
channels_with_rate = spikes.merge(units_min, left_on="unit", right_on="id", how="left")

channels_with_rate.drop("id", axis=1, inplace=True)
channels_with_rate.rename(columns={'unit':'unit_id', 't':'spike_timestamp'}, inplace=True)
channels_with_rate = channels_with_rate[['ecephys_channel_id', 'unit_id', 'spike_timestamp']] ##########rename to channel_id

channel_unit_t_struct = channels_with_rate.merge(channels, left_on='ecephys_channel_id', right_on='id', how="left")
channel_unit_t_struct.drop("id", axis=1, inplace=True)
channel_unit_t_struct.rename(columns={'ecephys_channel_id':'channel_id', 'ecephys_probe_id':'probe_id'}, inplace=True)
channel_unit_t_struct = channel_unit_t_struct[['probe_id', 'channel_id', 'unit_id', 'spike_timestamp', 'ecephys_structure_acronym',
                                               'local_index', 'probe_horizontal_position', 'probe_vertical_position', 'anterior_posterior_ccf_coordinate',
                                               'dorsal_ventral_ccf_coordinate', 'left_right_ccf_coordinate', 'ecephys_structure_id']]
channel_unit_t_struct.sort_values(by="spike_timestamp", ascending=True, inplace=True)

# ---- Save output w session_id ----
master_output = sesh_path / f"s{session_id}_probe_chan_unit_spikes_struct.csv"
channel_unit_t_struct.to_csv(master_output, index=False)

print(channel_unit_t_struct.head())
print(channel_unit_t_struct.columns)

print("=================")
print(f"Saved output to: {master_output}")


# count spikes per unit
#spike_counts = spikes.groupby("unit").size()

# spike rate = total spikes / total recording time=
#spike_rates = spike_counts / duration

#print("Spike rates... ")
#print(spike_rates.head())