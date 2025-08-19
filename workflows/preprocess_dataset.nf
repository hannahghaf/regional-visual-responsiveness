nextflow.enable.dsl=2

workflow preprocess {
    take:
    downloaded_ch   // tuple: session_id and path session_<SID> from download_data

    main:
    prepped = preprocess_session(downloaded_ch)

    emit:
    prepped     // emits tuple(val(session_id), path(pivot_csv))
}

process preprocess_session {
    cpus { 2 + 1*(task.attempt - 1) }
    memory { 4.GB + 1.GB*(task.attempt - 1) }
    maxRetries 3
    errorStrategy 'retry'

    container 'visresp:latest'
    tag { "preprocess_${session_id}" }
    publishDir "results/logs", mode: 'copy',
        saveAs: { name -> "session_${session_id}/${task.process}" }

    input:
    tuple val(session_id), path(session_dir)

    output:
    tuple val(session_id), path("pivot/s${session_id}_pivot_counts_rates_by_region.csv")

    script:
    """
    # run script; it writes to /app/data/session_<SID>/
    # output per-task resource logs
    /usr/bin/time -v -o resource_usage.txt \
        python /app/bin/stimulus_spike_counts.py "${session_id}" \
        > stdout.txt 2> stderr.txt

    # stage pivot CSV where Nextflow can collect it
    mkdir -p pivot
    cp "/app/data/session_${session_id}/s${session_id}_pivot_counts_rates_by_region.csv" \
        "pivot/s${session_id}_pivot_counts_rates_by_region.csv"
    """
}