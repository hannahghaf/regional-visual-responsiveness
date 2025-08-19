import os
import sys
from pathlib import Path
import pandas as pd
import csv
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# check CLI args
if len(sys.argv) != 4:
    print("Usage: python download_data.py <session_id> <manifest_path> <output_dir>")
    sys.exit(1)

session_id = int(sys.argv[1])
manifest_path = Path(sys.argv[2])
data_out = Path(sys.argv[3])


def process_session_csvs(session_id, cache, data_out):
    '''
    downloads and processes spike times and channels for given session
    - session_id: int or str (Allen session ID)
    - cache: initialized AllenSDK EcephysProjectCache object
    - data_out: session data output dir
    '''

    sess = cache.get_session_data(session_id)

    sess_path = data_out / f"session_{session_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    # save spike times
    spike_csv = sess_path / "spike_times.csv"
    with open(spike_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["unit", "t"])
        for unit_id, times in sess.spike_times.items():
            for t in times:
                w.writerow([unit_id, float(t)])

    # save channels & stimulus presentation data
    sess.channels.to_csv(sess_path / "channels.csv", index=False)
    sess.stimulus_presentations.to_csv(sess_path / "stimulus_presentations.csv", index=False)

    return spike_csv


def load_or_process(session_id, cache, data_out):
    '''
    loads a DataFrame for given session
    - if .pkl exists: load directly
    - if not: process data & save .pkl
    '''

    pickle_path = data_out / "pickle" / f"session_{session_id}.pkl"

    if pickle_path.exists():
        print(f"[INFO] Loading cached DataFrame for session {session_id} from {pickle_path}")
        df = pd.read_pickle(pickle_path) # fast load
    else:
        print(f"[INFO] No pickle found. Processing raw data for session {session_id}...")
        df = process_session_csvs(session_id, cache, data_out)
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(pickle_path)
        print(f"[INFO] Saved processed DataFrame to {pickle_path}")
    return df


# initialize cache and run
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
df = load_or_process(session_id, cache, data_out)
print(df.head())
print("done")