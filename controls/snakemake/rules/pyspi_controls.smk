# Mass computation of pairwise interaction statistics from BOLD time series (CONTROLS)

from os.path import join

rule compute_spis:
    """Compute pairwise connectivity matrices from 284 pairwise interaction statistics using PySpi"""
    input: 
        vep_timeseries = join(config["control_dir"], 'data/{subject}/{subject}_run-{run}_vep_timeseries.csv'), 
    output: 
        spi_matrix = join(config["results_dir"], 'run-{run}/pyspi/spis_controls/{subject}/{subject}_spi_matrix.pkl'),
        log_summary = join(config["results_dir"], 'run-{run}/pyspi/spis_controls/{subject}/{subject}_spi_matrix_log.txt'),
        done = 'logs/{subject}/{subject}_run-{run}_compute_spis.done'
    group: "pyspi"
    script: join(config["snakemake"], 'scripts/pyspi/compute_spis.py') 