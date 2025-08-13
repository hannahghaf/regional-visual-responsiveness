from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
session_list = project_root / 'session_list.txt'
output_file = project_root / 'utils' / "summary_output.txt"

with open(session_list) as f:
    lines = [line.strip() for line in f]
    print(lines)

with open(output_file, "w") as out:
    for sesh_id in lines:
        stim_csv = project_root / 'data' / f"session_{sesh_id}" / "stimulus_presentations.csv"
        df = pd.read_csv(stim_csv)
        
        summary = (
            df[df["stimulus_name"].isin(["natural_scenes", "drifting_gratings"])]
            .groupby("stimulus_name")["duration"]
            .agg(["mean", "min", "max"])
        )

        print(f"Session: {sesh_id}", file=out)
        print(summary, file=out)

        print("", file=out)
        print("===============", file=out)
        print("", file=out)

print("done")