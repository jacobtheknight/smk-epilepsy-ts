# Parcellate streamline-weighted connectomes from pre-computed SIFT anatomically-constrained tractography using MRtrix3 (CONTROLS)

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

rule convert_subject_atlas:
  """Convert subject VEP atlas to MRtrix3 format"""
  input:
    atlas_nii = join(config["freesurfer_dir"], '{subject}/mri/aparc+aseg.vep.nii.gz'),
    lut = join(config["vep_lut"]), 
    lut_mrtrix = join(config["vep_lut_mrtrix"])
  output:
    atlas_mif = join(config["connectome_dir"], '{subject}/work/aparc+aseg.vep.mif')
  group: "vep_atlas"
  shell:
    """
    mkdir -p $(dirname {output.atlas_mif})
    labelconvert {input.atlas_nii} \
      {input.lut} \
      {input.lut_mrtrix} \
      {output.atlas_mif}
    """

rule convert_subject_t1w:
  """Convert subject T1-weighted image from MGZ to NIfTI"""
  input: 
    t1w_mgz = join(config["freesurfer_dir"], '{subject}/mri/T1.mgz'),
    fs_license = join(config["fs_license"])
  output: 
    t1w_nii = join(config["freesurfer_dir"], '{subject}/mri/T1.nii.gz')
  params: config["fs_container"]
  group: "vep_atlas"
  shell:
    """
    singularity run --cleanenv \
    -B {input.fs_license}:/license.txt \
    --env FS_LICENSE=/license.txt \
    {params} \
    mri_convert {input.t1w_mgz} {output.t1w_nii}   
    """

rule transform_t1w_to_dwi:
  """Transform T1-weighted images from FreeSurfer native space to DWI"""
  input:
    t1w_nii = join(config["freesurfer_dir"], '{subject}/mri/T1.nii.gz'), 
    dwi_nii = join(config["diffusion_dir_hbp"], '{subject}/ses-mri7T/{subject}_b0_mean_blip.nii.gz'), 
  output:
    t1w_in_dwi = join(config["connectome_dir"], '{subject}/work/T1_to_diff.nii.gz'), 
    fsl_matrix = join(config["connectome_dir"], '{subject}/work/T1_to_diff_fsl.mat')
  group: "vep_atlas"
  shell: 
    """
    flirt -in {input.t1w_nii}\
      -ref {input.dwi_nii} \
      -out {output.t1w_in_dwi} \
      -omat {output.fsl_matrix}\
      -dof 6 \
      -cost normmi \
      -searchcost normmi
    """

rule convert_fsl_matrix:
  """Convert FSL transform to MRtrix3 format"""
  input: 
    fsl_matrix = join(config["connectome_dir"], '{subject}/work/T1_to_diff_fsl.mat'),
    t1w_nii = join(config["freesurfer_dir"], '{subject}/mri/T1.nii.gz'), 
    dwi_nii = join(config["diffusion_dir_hbp"], '{subject}/ses-mri7T/{subject}_b0_mean_blip.nii.gz'),
  output:
    mrtrix_out = join(config["connectome_dir"], '{subject}/work/T1_to_diff_mrtrix.mat')
  group: "vep_atlas"
  shell: 
    """
    transformconvert {input.fsl_matrix} \
    {input.t1w_nii} \
    {input.dwi_nii} \
    flirt_import {output.mrtrix_out}
    """

rule back_to_mif:
  """Convert DWI file to MIF for final transformation"""
  input:
    dwi_nii = join(config["diffusion_dir_hbp"], '{subject}/ses-mri7T/{subject}_b0_mean_blip.nii.gz'),
  output:
    dwi_mif = join(config["connectome_dir"], '{subject}/work/{subject}_b0_mean_blip.mif')
  group: "vep_atlas"
  shell:
    """
    mrconvert {input.dwi_nii} {output.dwi_mif}
    """

rule fit_atlas_to_dwi: 
  """Transform VEP atlas to diffusion space for tractography parcellation"""
  input: 
    atlas_mif = join(config["connectome_dir"], '{subject}/work/aparc+aseg.vep.mif'),
    mrtrix_out = join(config["connectome_dir"], '{subject}/work/T1_to_diff_mrtrix.mat')
  output:
    vep_atlas_diff = join(config["connectome_dir"], '{subject}/work/aparc+aseg.vep_diff.mif'),
    done = 'logs/{subject}/{subject}_transform_atlas.done'
  group: "vep_atlas"
  shell: 
    """
    mkdir -p $(dirname {output.done})
    mrtransform {input.atlas_mif} \
    -linear {input.mrtrix_out} \
    {output.vep_atlas_diff}

    touch {output.done}
    """

rule check_atlas_alignment:
  """Get tractography visualisation and NIfTI version of transformed atlas for quality control"""
  input:
    tracto_vis = join(config["diffusion_dir_hbp"], '{subject}/ses-mri7T/tractography/vis.nii.gz'), 
    vep_atlas_diff = join(config["connectome_dir"], '{subject}/work/aparc+aseg.vep_diff.mif')
  output:
    tracto_vis_copy = join(config["connectome_dir"], '{subject}/work/vis_tracto.nii.gz'), 
    vep_atlas_diff_nifti = join(config["connectome_dir"], '{subject}/work/aparc+aseg.vep_diff.nii.gz'),
    done = 'logs/{subject}/{subject}_check_atlas_alignment_diff.done'
  group: "qc"
  shell:
    """
    cp {input.tracto_vis} {output.tracto_vis_copy}
    mrconvert {input.vep_atlas_diff} {output.vep_atlas_diff_nifti}

    touch {output.done}
    """

rule parcellate_tracks: 
  """Parcellate tracks and compute weighted connectomes (number of streamlines)"""
  input: 
    tracks = join(config["diffusion_dir_hbp"], '{subject}/ses-mri7T/tractography/{subject}_10M_act_tracks_sift.tck'), 
    vep_atlas_diff = join(config["connectome_dir"], '{subject}/work/aparc+aseg.vep_diff.mif')
  output:
    vep_connectome = join(config["connectome_dir"], '{subject}/{subject}_connectome_N.csv'),
    vep_connectome_assigned = join(config["connectome_dir"], '{subject}/{subject}_connectome_assignment_N.csv'),
    done = 'logs/{subject}/{subject}_compute_connectome_N.done'
  log:
    all = join(config["connectome_dir"], '{subject}/logs/{subject}_compute_connectome_N.log')
  group: "connectome"
  shell: 
    """ 
    mkdir -p $(dirname {output.done})
    mkdir -p $(dirname {log.all})
    tck2connectome –symmetric –zero_diagonal -scale_invnodevol \
    {input.tracks} {input.vep_atlas_diff} {output.vep_connectome} \
    -out_assignment {output.vep_connectome_assigned} \
    &> {log.all}
    
    touch {output.done}
    """

rule parcellate_tracks_wiring_costs:
  """Parcellate tracks and compute weighted connectomes (length of streamlines)"""
  input: 
    tracks = join(config["diffusion_dir_hbp"], '{subject}/ses-mri7T/tractography/{subject}_10M_act_tracks_sift.tck'), 
    vep_atlas_diff = join(config["connectome_dir"], '{subject}/work/aparc+aseg.vep_diff.mif')
  output:
    vep_connectome = join(config["connectome_dir"], '{subject}/{subject}_connectome_L.csv'),
    vep_connectome_assigned = join(config["connectome_dir"], '{subject}/{subject}_connectome_assignment_L.csv'),
    done = 'logs/{subject}/{subject}_compute_connectome_L.done'
  log:
    all = join(config["connectome_dir"], '{subject}/logs/{subject}_compute_connectome_L.log')
  group: "connectome"
  shell: 
    """ 
    mkdir -p $(dirname {output.done})
    mkdir -p $(dirname {log.all})
    tck2connectome –symmetric –zero_diagonal -scale_length \
    {input.tracks} {input.vep_atlas_diff} {output.vep_connectome} \
    -out_assignment {output.vep_connectome_assigned} \
    &> {log.all}
    
    touch {output.done}
    """
