nextflow.enable.dsl=2

include { download_data } from './workflows/download_data.nf'
include { preprocess } from './workflows/preprocess_dataset.nf'
include { analysis_plots } from './workflows/analysis_plots.nf'


workflow {
    def session_ids_ch = Channel.fromPath(params.session_list)
                                .splitText()
                                .map { it.trim() }
                                .filter { it }
                                .map { it as Integer }

    def manifest_path = Channel.fromPath(params.manifest)

    def session_manifest_ch = session_ids_ch.map{ sid -> tuple(sid, file(params.manifest)) }

    // 1) Download - execute download_data.py
    def downloaded_ch = download_data(session_manifest_ch)

    // 2) Preprocess - emits (session_id, pivot_csv)
    def prepped_ch = preprocess(downloaded_ch)

    // 3) Plots + summaries - emits plots folder path
    def plotted_ch = analysis_plots(prepped_ch)
}