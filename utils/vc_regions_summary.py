# determine most common visual cortex regions per session
# figure out which brain regions are frequent and worth analyzing

import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

manifest_path = '/Users/hannahghaffari/Documents/regional-visual-responsiveness/ecephys_cache_dir/manifest.json'
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
session = cache.get_session_data(sessions.index[0]) # pick first for now

# get all channels info
channels = session.channels
visual_channels = channels[channels['structure_acronym'].str.startswith('VIS')]
region_counts = visual_channels['structure_acronym'].value_counts()
print(region_counts.head(10))