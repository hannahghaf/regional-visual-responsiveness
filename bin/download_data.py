import os
from pathlib import Path
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

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
    spikes = pd.DataFrame([
        {"unit": unit_id, "t": t}
        for unit_id, spike_times in sess.spike_times.items()
        for t in spike_times
    ])
    spikes.to_csv(f"{sess_path}/spike_times.csv", index=False)

    # save channels
    sess.channels.to_csv(sess_path / "channels.csv", index=False)
    ################################################ delete these csvs once have pkl? ######### if dont use 

    return spikes


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

project_root = Path(__file__).resolve().parent.parent
data_out = project_root / "data"

cache = EcephysProjectCache.from_warehouse(manifest=os.path.join(data_out, "manifest.json"))

sessions = cache.get_session_table().index[:5]

for sid in sessions:
    df = load_or_process(sid, cache, data_out)
    print(df.head())

print("done")