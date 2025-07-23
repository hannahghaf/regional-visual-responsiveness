# determine most common visual cortex regions per session
# figure out which brain regions are frequent and worth analyzing

import os
from pathlib import Path
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# reduce output noise
import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
#print(os.getcwd())

manifest_path = project_root / "ecephys_cache_dir" / "manifest.json"
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
session = cache.get_session_data(sessions.index[0]) # pick first for now

# get all channels info
channels = session.channels
channels = channels.dropna(subset=['ecephys_structure_acronym'])
visual_channels = channels[channels['ecephys_structure_acronym'].str.startswith('VIS')]
region_counts = visual_channels['ecephys_structure_acronym'].value_counts()
print(region_counts.head(10))

# print session ID
session_id = session.ecephys_session_id
print(session_id)