# check how often each stimulus type appears
# count how many presentations of each stimulus type were shown

import os
from pathlib import Path
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

project_root = Path(__file__).resolve().parent.parent
ecephys_root = project_root / "ecephys_cache_dir"
outdir = ecephys_root

manifest_path = ecephys_root / "manifest.json"
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

session_table = cache.get_session_table()
stim_counts = {}

for session_id in session_table.index[:10]:
    # WARNING: this will download data for 10 sessions (~2GB each)
    session = cache.get_session_data(session_id)
    stim_names = session.stimulus_presentations['stimulus_name'].value_counts()
    stim_counts[session_id] = stim_names.to_dict()

stim_df = pd.DataFrame.from_dict(stim_counts, orient='index').fillna(0).astype(int)
stim_df.index.name = 'session_id'

stim_df.to_csv("stimulus_counts_by_session.csv")
print(stim_df.head())