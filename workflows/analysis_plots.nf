nextflow.enable.dsl=2

workflow analysis_plots {
    take:
    pivot_ch    // tuple: session_id, pivot_csv

    main:
    plotted = plot_session(pivot_ch)

    emit:
    plotted     // path("plots/session_<SID>_plots")
}

process plot_session {
    cpus { 2 + 1*(task.attempt - 1) }
    memory { 3.GB + 1.GB*(task.attempt - 1) }
    maxRetries 3
    errorStrategy 'retry'

    container 'visresp:latest'
    tag { "plot_${session_id}" }
    publishDir "results/logs", mode: 'copy',
        saveAs: { name -> "session_${session_id}/${task.process}" }

    input:
    tuple val(session_id), path(pivot_csv)

    output:
    path "plots/session_${session_id}_plots"

    script:
    """
    # script will read pivot + raw bits from /app/data and write outputs
    # output per-task resource logs
    /usr/bin/time -v -o resource_usage.txt \
        python /app/bin/pivot_analysis_plots.py "${session_id}" \
        > stdout.txt 2> stderr.txt

    # collect plots & summaries to a single export folder
    mkdir -p plots
    cp -r "/app/data/session_${session_id}/session_${session_id}_plots" \
          "plots/session_${session_id}_plots"
    """
}