import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"
session_path = Path("data/session_715093703")
sesh_path = project_root / session_path

spikes = pd.read_csv(sesh_path / "spike_times.csv")
channels = pd.read_csv(sesh_path / "channels.csv")
total_channels = pd.read_csv(data_root / "channels.csv")
units = pd.read_csv(data_root / "units.csv")

# count spikes per unit
#spike_counts = spikes.groupby("unit").size()

# spike rate = total spikes / total recording time
#spike_rates = spike_counts / duration

#print("Spike rates... ")
#print(spike_rates.head())

# merge spike rates w channel ids CHANGE!!!!
units_min = units[['id', 'ecephys_channel_id']]
channels_with_rate = spikes.merge(units_min, left_on="unit", right_on="id", how="left")
print(channels_with_rate.tail())