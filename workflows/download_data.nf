nextflow.enable.dsl=2

workflow download_data {
    take:
    session_manifest_ch

    main:
    downloaded = download_session_data(session_manifest_ch)

    emit:
    downloaded
}

process download_session_data {
    cpus { 2 + 1*(task.attempt - 1) }
    memory { 8.GB + 1.GB*(task.attempt - 1) }
    maxForks 1
    maxRetries 3
    errorStrategy 'retry'

    container 'visresp:latest'
    tag "session_${session_id}"
    publishDir "results/logs", mode: 'copy',
        saveAs: { name -> "session_${session_id}/${task.process}" }

    input:
    tuple val(session_id), val(manifest_path)

    output:
    tuple val(session_id), path("session_${session_id}")

    script:
    """
    mkdir -p session_${session_id}
    # paths in docker container
    # output per-task resource logs
    /usr/bin/time -v -o resource_usage.txt \
        python /app/bin/download_data.py "${session_id}" /app/data/manifest.json /app/data \
        > stdout.txt 2> stderr.txt

    # copy results into process workdir so NF can grab it
    cp -r /app/data/session_${session_id} ./ || true
    """
}