# Visual Responsiveness Pipeline (Nextflow)

Measure how strongly neurons in mouse visual cortex respond when visual stimuli appear, using a reproducible, containerized Nextflow pipeline. 

The pipeline:
1. downloads / organizes the Allen Brain Obesrvatory Neuropixels data for chosen sessions,

2. preprocesses and builds analysis windows (baseline/evoked), maps spikes to windows, and writes an analysis table,

3. analyzes & plots regionxstimulus deltas and saves figures and summary tables,

4. emits per-process resource logs (CPU, mean, runtimee) for each session.


## What the pipeline does:
* Input: list of session_IDs of interest from the Allen dataset
* Compute: for each session, count spikes in a short baseline window right before stimulus onset and an evoked window right after onset, for each unit (neuron). Compute delta rate (Hz) = evoked - baseline.
* Aggregate: summarize by brain region and stimulus (drifting_gratings vs. natural_movies).
* Outptu: a per-sesison analysis table, publication-ready plots, summary CSVs (medians/means, unit-weighted), and logs

## Inputs you can specify
--session_list path => *required; text file with one session_id per line
--manifest path => *; download manifest used by download_data.py
--outdir path => where all final outputs are written (default: results/)
-profile docker => use Docker profile (Dockerfile provided in repo)

### session_list sessions.txt (example):
```
715093703
719161530
721123822
732592105
```

## How to run:
### Start Docker/Colima
Ensure Nextflow and Docker (or Colima) are installed.
Start Colima with enough memory.
```
colima start --memory 6
```

Run:
```
nextflow run main.nf \
    -profile docker \
    --session_list "session_list.txt"
```

## Troubleshooting
* No plots? Check results/logs/session_<ID>/stderr.txt
* FileNotFoundError similar to this:
```
FileNotFoundError: [Errno 2] No such file or directory: '/app/data/session_737581020/spike_times.csv'
```
Delete the pickle folder in data/ and rerun the pipeline
* Process terminated with an error exit status (137) => this is due to resource limitations inside of the container, most likely out-of-memory conditions. Stop and restart Colima.



## Citation (dataset)
Allen Institute for Brain Science. Allen Brain Observatory: Visual Coding (Neuropixels). Documentation: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html 
