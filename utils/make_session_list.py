# make session_list.txt input of first 5 session IDs
import os
from pathlib import Path
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

project_root = Path(__file__).resolve().parent.parent
data_out = project_root / "data"

cache = EcephysProjectCache.from_warehouse(manifest=os.path.join(data_out, "manifest.json"))

sessions = cache.get_session_table().index[:5]
sessions = pd.Series(sessions)
sessions.to_csv(project_root / "session_list.txt", index=False, header=False)