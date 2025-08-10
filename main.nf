nextflow.enable.dsl=2

include { download_data } from './workflows/download_data.nf'

workflow {
    def session_ids_ch = Channel.fromPath(params.session_list)
                                .splitText()
                                .map { it.trim() }
                                .filter { it }
                                .map { it as Integer }

    def manifest_path = Channel.fromPath(params.manifest)

    def session_manifest_ch = session_ids_ch.map{ sid -> tuple(sid, file(params.manifest)) }

    // run workflow to execute download_data.py
    def downloaded_ch = download_data(session_manifest_ch)
}