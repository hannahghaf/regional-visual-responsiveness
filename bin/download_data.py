import os
from pathlib import Path
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


project_root = Path(__file__).resolve().parent.parent
print(project_root)
out = os.getenv("DATA_DIR", "/data")
print(out)