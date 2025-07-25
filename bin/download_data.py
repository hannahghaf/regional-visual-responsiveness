import os
from pathlib import Path
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

project_root = Path(__file__).resolve().parent.parent
data_out = project_root / "data"

cache = EcephysProjectCache.from_warehouse(manifest=os.path.join(data_out, "manifest.json"))
sessions = cache.get_session_table().index[:5]
print(data_out)
'''
for sid in sessions:
    sess = cache.get_session_data(sid)
    path = os.path.join(data_out, f"session_{sid}")
    os.makedirs(path, exist_ok=True)
    pd.DataFrame([
        {"unit": u, "t": t} for u, ts in sess.spike_times.items() for t in ts
    ]).to_csv(f"{path}/spike_times.csv", index=False)
    sess.channels.to_csv(f"{path}/channels.csv", index=False)
    '''