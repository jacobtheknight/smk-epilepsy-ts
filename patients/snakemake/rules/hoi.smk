# Compute measures of multivariate information and derive Z-scores for comparative analysis

from os.path import join

rule compute_hois:
  """Compute measures of multivariate information of epileptic networks with HOI"""
    input:
        vep_timeseries_labelled = join(config["patient_dir"], 'data/{subject}/{subject}_run-{run}_vep_timeseries_labelled.csv')
    output:
        tc = join(config["results_dir"], 'run-{run}/hoi/tc/{network}/{subject}_{network}_tc.csv'),
        dtc = join(config["results_dir"], 'run-{run}/hoi/dtc/{network}/{subject}_{network}_dtc.csv'),
        sinfo = join(config["results_dir"], 'run-{run}/hoi/sinfo/{network}/{subject}_{network}_sinfo.csv'),
        oinfo = join(config["results_dir"], 'run-{run}/hoi/oinfo/{network}/{subject}_{network}_oinfo.csv'),
        done = 'logs/{subject}/{subject}_run-{run}_{network}_hoi.done'
    group: "hoi"
    script: join(config["snakemake"], 'scripts/hoi/compute_hois.py')

rule within_subject_zscore:
    """Compare measures against null distribution"""
    input: 
        hoi_done = 'logs/{subject}/{subject}_run-{run}_{network}_hoi.done', 
        tc = join(config["results_dir"], 'run-{run}/hoi/tc/{network}/{subject}_{network}_tc.csv'),
        dtc = join(config["results_dir"], 'run-{run}/hoi/dtc/{network}/{subject}_{network}_dtc.csv'),
        sinfo = join(config["results_dir"], 'run-{run}/hoi/sinfo/{network}/{subject}_{network}_sinfo.csv'),
        oinfo = join(config["results_dir"], 'run-{run}/hoi/oinfo/{network}/{subject}_{network}_oinfo.csv'),
        vep_timeseries_labelled = join(config["patient_dir"], 'data/{subject}/{subject}_run-{run}_vep_timeseries_labelled.csv'), 
        lut = join(config["resources"], 'VepFreeSurferColorLut.txt'), 
        atlas_mni = join(config["patient_dir"], 'data/{subject}/{subject}_run-{run}_space-MNI152NLin2009cAsym_desc-vep_dseg.nii.gz')
    output: 
        result = join(config["results_dir"], 'run-{run}/hoi/zscores/{network}/patient_within_subject/{subject}_{network}_within_subject_zscores.csv'),
        null_values_tc = join(config["results_dir"], 'run-{run}/hoi/zscores/distributions_tc/within_subject/{network}/{subject}_{network}_null_values_tc.csv'), 
        null_values_dtc = join(config["results_dir"], 'run-{run}/hoi/zscores/distributions_dtc/within_subject/{network}/{subject}_{network}_null_values_dtc.csv'), 
        null_values_sinfo = join(config["results_dir"], 'run-{run}/hoi/zscores/distributions_sinfo/within_subject/{network}/{subject}_{network}_null_values_sinfo.csv'),  
        null_values_oinfo = join(config["results_dir"], 'run-{run}/hoi/zscores/distributions_oinfo/within_subject/{network}/{subject}_{network}_null_values_oinfo.csv'),  
        done = 'logs/{subject}/{subject}_run-{run}_{network}_within_subject_zscore.done'
    group: "zscore"
    script: join(config["snakemake"], 'scripts/zscore/within_subject_patients.py')

rule between_subject_zscore:
  """Compare measures against healthy control distribution"""
    input: 
        hoi_done = 'logs/{subject}/{subject}_run-{run}_{network}_hoi.done', 
        tc = join(config["results_dir"], 'run-{run}/hoi/tc/{network}/{subject}_{network}_tc.csv'),
        dtc = join(config["results_dir"], 'run-{run}/hoi/dtc/{network}/{subject}_{network}_dtc.csv'),
        oinfo = join(config["results_dir"], 'run-{run}/hoi/oinfo/{network}/{subject}_{network}_oinfo.csv'),
        sinfo = join(config["results_dir"], 'run-{run}/hoi/sinfo/{network}/{subject}_{network}_sinfo.csv')
    params:
        control_timeseries_dir = join(config["control_dir"], 'data')
    output:
        result = join(config["results_dir"], 'run-{run}/hoi/zscores/{network}/between_subject/{subject}_{network}_between_subject_zscores.csv'),
        bootstrapped_control_values_tc = join(config["results_dir"], 'run-{run}/hoi/zscores/distributions_tc/between_subject_bootstrapped/{network}/{subject}_{network}_bootstrapped_control_values_tc.csv'),
        bootstrapped_control_values_dtc = join(config["results_dir"], 'run-{run}/hoi/zscores/distributions_dtc/between_subject_bootstrapped/{network}/{subject}_{network}_bootstrapped_control_values_dtc.csv'),
        bootstrapped_control_values_sinfo = join(config["results_dir"], 'run-{run}/hoi/zscores/distributions_sinfo/between_subject_bootstrapped/{network}/{subject}_{network}_bootstrapped_control_values_sinfo.csv'),
        bootstrapped_control_values_oinfo = join(config["results_dir"], 'run-{run}/hoi/zscores/distributions_oinfo/between_subject_bootstrapped/{network}/{subject}_{network}_bootstrapped_control_values_oinfo.csv'),
        done = 'logs/{subject}/{subject}_run-{run}_{network}_between_subject_zscore.done'
    group: "zscore"
    script: join(config["snakemake"], 'scripts/zscore/between_subject.py')

rule comparative_analysis_complete:
    input: 
      within_subject_done = 'logs/{subject}/{subject}_run-{run}_ez+pz_within_subject_zscore.done',
        between_subject_done = 'logs/{subject}/{subject}_run-{run}_ez+pz_between_subject_zscore.done'   
    output: 
       done = 'logs/{subject}/{subject}_run-{run}_comparative_analysis_complete.done'
    group: 'zscore'
    shell:
        """
       mkdir -p logs/{wildcards.subject}
        touch {output.done}
       """

rule within_subject_zscore_controls:
    """Compare measures against null distribution (healthy controls)"""
    input:   
        vep_timeseries_labelled = join(config["patient_dir"], 'data/{subject}/{subject}_run-{run}_vep_timeseries_labelled.csv'),
        control_timeseries = join(config["control_dir"], 'data/{control}/{control}_run-{run}_vep_timeseries.csv'),
        control_atlas_mni = join(config["control_dir"], 'data/{control}/{control}_run-{run}_space-MNI152NLin2009cAsym_desc-vep_dseg.nii.gz'),
        lut = join(config["resources"], 'VepFreeSurferColorLut.txt')
    output:
        result = join(config["results_dir"], 'run-{run}/hoi/zscores/{network}/control_within_subject/{subject}/{control}_{network}_within_subject_zscores.csv'),
        done = 'logs/{subject}/{control}_run-{run}_{network}_within_subject_zscore.done'
    params:
        control_data_dir = join(config["control_dir"], 'data'),
        results_dir      = config["results_dir"]
    script: join(config["snakemake"], 'scripts/zscore/within_subject_controls.py') 
        