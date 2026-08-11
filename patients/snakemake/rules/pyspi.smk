# Mass computation of pairwise interaction statistics from BOLD time series 

from os.path import join

rule compute_spis:
    """Compute pairwise connectivity matrices from 284 pairwise interaction statistics using pyspi"""
    input: 
        vep_timeseries_labelled = join(config["patient_dir"], 'data/{subject}/{subject}_run-{run}_vep_timeseries_labelled.csv'), 
    output: 
        spi_matrix = join(config["results_dir"], 'run-{run}/pyspi/spis/{subject}/{subject}_spi_matrix.pkl'),
        log_summary = join(config["results_dir"], 'run-{run}/pyspi/spis/{subject}/{subject}_spi_matrix_log.txt'),
        done = 'logs/{subject}/{subject}_run-{run}_compute_spis.done'
    group: "pyspi"
    script: join(config["snakemake"], 'scripts/pyspi/compute_spis.py') 
    