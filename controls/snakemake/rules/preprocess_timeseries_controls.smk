# Sample VEP atlas and preprocess control BOLD time series

from os.path import join

rule convert_vep_atlas: 
     """Convert VEP atlas from MGZ to NIfTI"""
     input:
         atlas_mgz = join(config["freesurfer_dir"], '{subject}/mri/aparc+aseg.vep.mgz'),
         fs_license = join(config["fs_license"])
     output:
         atlas_nii = join(config["freesurfer_dir"], '{subject}/mri/aparc+aseg.vep.nii.gz'),
         done = 'logs/{subject}/{subject}_vep_atlas_conversion.done'
     params: config["fs_container"]
     group: "vep_atlas"
     shell:
         """
         mkdir -p $(dirname {output.done})

         singularity run --cleanenv \
         -B {input.fs_license}:/license.txt \
         --env FS_LICENSE=/license.txt \
         {params} \
         mri_convert {input.atlas_mgz} {output.atlas_nii}   

         touch {output.done}
         """

rule resample_vep_atlas_to_mni:
    """Transform patient VEP atlas from FreeSurfer native space to MNI"""
    input:
        atlas_nii = join(config["freesurfer_dir"], '{subject}/mri/aparc+aseg.vep.nii.gz'),
        mni_template = join(config["fmriprep_dir"], '{subject}/func/{subject}_task-rest_dir-{dir}_run-{run}_space-MNI152NLin2009cAsym_boldref.nii.gz'), 
        t1w_to_mni = join(config["fmriprep_dir"], '{subject}/anat/{subject}_acq-MP2RAGE_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'),
        fsnative_to_t1w = join(config["fmriprep_dir"], '{subject}/anat/{subject}_acq-MP2RAGE_from-fsnative_to-T1w_mode-image_xfm.txt')
    output:
        atlas_mni = join(config["control_dir"], 'data/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_space-MNI152NLin2009cAsym_desc-vep_dseg.nii.gz'),
        done = 'logs/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_vep_atlas_resample.done'
    singularity: config["fmriprep_container"]
    group: "vep_atlas"
    shell:
        """
        antsApplyTransforms \
            -d 3 \
            -i {input.atlas_nii} \
            -r {input.mni_template} \
            -o {output.atlas_mni} \
            -t {input.t1w_to_mni} \
            -t {input.fsnative_to_t1w} \
            -n NearestNeighbor \
            -v 1

        touch {output.done}
        """

rule check_atlas_alignment:
    """Create RGB overlay for inspection of atlas-to-BOLD alignment"""
    input:
        atlas_mni = join(config["control_dir"], 'data/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_space-MNI152NLin2009cAsym_desc-vep_dseg.nii.gz'),
        bold_ref = join(config["fmriprep_dir"], '{subject}/func/{subject}_task-rest_dir-{dir}_run-{run}_space-MNI152NLin2009cAsym_boldref.nii.gz'),
    output:
        validation_nifti = join(config["control_dir"], 'qc/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_atlas_alignment.nii.gz'),
        validation_png = join(config["control_dir"], 'qc/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_atlas_alignment.png'),
        done = 'logs/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_atlas_alignment_bold.done'
    group: "qc"
    shell:
        """
        mkdir -p $(dirname {output.validation_nifti})
         
        overlay 1 1 {input.bold_ref} -a {input.atlas_mni} 0 0.5 {output.validation_nifti}
        slicer {output.validation_nifti} -A 800 {output.validation_png}
        
        touch {output.done}
        """

rule preprocess_timeseries_vep:
    """Preprocess VEP timeseries"""
    input:
         bold_image = join(config["fmriprep_dir"], '{subject}/func/{subject}_task-rest_dir-{dir}_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
         bold_ref = join(config["fmriprep_dir"], '{subject}/func/{subject}_task-rest_dir-{dir}_run-{run}_space-MNI152NLin2009cAsym_boldref.nii.gz'),
         bold_json = join(config["fmriprep_dir"], '{subject}/func/{subject}_task-rest_dir-{dir}_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.json'),
         confounds = join(config["fmriprep_dir"],'{subject}/func/{subject}_task-rest_dir-{dir}_run-{run}_desc-confounds_timeseries.tsv'),
         vep_atlas = join(config["control_dir"], 'data/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_space-MNI152NLin2009cAsym_desc-vep_dseg.nii.gz'),
         lut = join(config["vep_lut"]),
         validation = join(config["control_dir"], 'qc/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_atlas_alignment.nii.gz'), 
    output:
         checkpoint_timeseries = join(config["control_dir"], 'data/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_timeseries_checkpoint.csv'),
         vep_timeseries = join(config["control_dir"], 'data/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_vep_timeseries.csv'),
         done = 'logs/{subject}/{subject}_task-rest_dir-{dir}_run-{run}_preproc_bold_vep_timeseries.done'    
    group: "preprocess_timeseries"
    script: join(config["snakemake"], 'scripts/preproc/preproc_bold_vep_timeseries.py')
