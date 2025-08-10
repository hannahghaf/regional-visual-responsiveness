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
    maxRetries 2
    errorStrategy 'retry'

    container 'visresp:latest'
    tag "session_${session_id}"

    input:
    tuple val(session_id), val(manifest_path)

    output:
    path "session_${session_id}"

    script:
    """
    mkdir -p session_${session_id}
    python ${projectDir}/bin/download_data.py ${session_id} ${manifest_path} session_${session_id}
    """
}