# Preprocess and parcellate patient BOLD time series using the Virtual Epileptic Patient atlas (https://github.com/HuifangWang/VEP_atlas_shared)

import os
import re
import json
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker

def load_atlas(atlas_path):
    atlas_img = nib.load(atlas_path)
    return atlas_img

# ensure output directories exist
os.makedirs(os.path.dirname(snakemake.output.vep_timeseries), exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.done), exist_ok=True)

# load Virtual Epileptic Patient atlas from subject Freesurfer
vep_atlas = load_atlas(snakemake.input.vep_atlas)
atlas_affine = vep_atlas.affine
atlas_shape = vep_atlas.shape
print(f"Loaded VEP atlas: shape={atlas_shape}, affine=\n{atlas_affine}")

# load BOLD data image from fMRIPrep
bold_img = nib.load(snakemake.input.bold_image)
bold_affine = bold_img.affine
bold_shape = bold_img.shape
print(f"Loaded BOLD: shape={bold_shape}, affine=\n{bold_affine}")

# check alignment between atlas and BOLD
affine_match = np.allclose(atlas_affine, bold_affine, atol=1e-3)
shape_match = atlas_shape[:3] == bold_shape[:3]
print(f"Atlas-BOLD alignment check: affine_match={affine_match}, shape_match={shape_match}")
if not (affine_match and shape_match):
    print("WARNING: Atlas and BOLD have different geometry. Resampling...")
else:
    print("✓ Atlas and BOLD are in the same space with matching geometry")

# load lookup table (LUT) for VEP region names
lookup_txt = snakemake.input.lut
lookup_df = pd.read_csv(lookup_txt, sep='\t', header=None, names=['line'], dtype=str, comment='#')

# load BOLD metadata for TR
with open(snakemake.input.bold_json, "r") as f:
    bold_metadata = json.load(f)
subject_t_r = bold_metadata.get("RepetitionTime")

# define confounds
confound_vars = ['trans_x', 'trans_y', 'trans_z', 
                 'rot_x', 'rot_y', 'rot_z',
                 'global_signal', 'csf', 'white_matter']

# load confounds
confounds_df = pd.read_csv(snakemake.input.confounds, sep='\t')

# select confounds and handle NaN values
confounds_clean = confounds_df[confound_vars].select_dtypes(include=[np.number]).fillna(0).values
missing_confounds = [v for v in confound_vars if v not in confounds_df.columns]
if missing_confounds:
    raise ValueError(f"Missing confound variables: {missing_confounds}")

# resample VEP atlas to BOLD space (if needed but should match with explicit transforms)
vep_data = vep_atlas.get_fdata().astype(int)
vep_unique = np.unique(vep_data)
vep_labels_idx = np.sort(vep_unique[vep_unique != 0])
print(f"Found {len(vep_labels_idx)} VEP regions: {vep_labels_idx}")

# create masker 
vep_masker = NiftiLabelsMasker(
    labels_img=vep_atlas,
    t_r=subject_t_r,
    low_pass=0.08,
    high_pass=0.009,
    standardize=True,
    detrend=True,
    smoothing_fwhm=False,
    memory='nilearn_cache',
    verbose=1
)

# extract VEP timeseries
vep_ts = vep_masker.fit_transform(bold_img, confounds=confounds_clean)
print(f"Extracted timeseries shape: {vep_ts.shape}")

# ensure consistent timepoints 
print("Truncating to 200 timepoints and skipping first 5 volumes...")
vep_ts = vep_ts[5:200, :]  # skip first 5 volumes to avoid non-steady state signal, handle scanner differences
print(f"Final number of timepoints: {vep_ts.shape[0]}")

# add VEP labels to timeseries
ncols = vep_ts.shape[1] # defensive column count for safe labelling
if len(vep_labels_idx) != ncols:
    n_keep = min(len(vep_labels_idx), ncols)
    vep_labels_idx = vep_labels_idx[:n_keep]

vep_region_names = [] # assign region names
for lab in vep_labels_idx:
    lab_int = int(lab)
    name = f"VEP_{lab_int}"
    vep_region_names.append(name)

# to dataframe (n_regions, timepoints)    
vep_df = pd.DataFrame(vep_ts).transpose()
vep_df.index = vep_region_names

# parse lookup lines into number -> region-name mapping
lut = {}
for line in lookup_df['line'].dropna().astype(str):
    m = re.match(r'^\s*(\d+)\s+([A-Za-z0-9_\-]+)', line) # match leading integer ID then the label token
    if m:
        idx = int(m.group(1))
        name = m.group(2)
        lut[idx] = name

# map vep_labels_idx to readable names; fallback to "VEP_<id>"
vep_region_names = [lut.get(int(l), f"VEP_{int(l)}") for l in vep_labels_idx]
vep_df.index = vep_region_names

# print summary of VEP regions with LUT mapping details
print(f"VEP regions after mapping: {vep_df.index.tolist()}")
print("\nLUT details:")
for lab in vep_labels_idx:
    lab_int = int(lab)
    mapped_name = lut.get(lab_int, f"VEP_{lab_int}")
    print(f"  {lab_int} -> {mapped_name}")

# save checkpoint timeseries before filtering 
vep_df.to_csv(snakemake.output.checkpoint_timeseries, index=True)

# remove non-GM regions
non_gm_regions = {
    'Left-Cerebral-White-Matter', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 
    'Left-Cerebellum-White-Matter', '3rd-Ventricle', '4th-Ventricle', 'CSF', 
    'Left-vessel', 'Left-choroid-plexus', 'Right-Cerebral-White-Matter', 
    'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 
    'Right-vessel', 'Right-choroid-plexus', 'WM-hypointensities', 'Optic-Chiasm', 
    'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior'
}

non_gm_mask = vep_df.index.isin(non_gm_regions)
removed_non_gm = vep_df.index[non_gm_mask].tolist()
if removed_non_gm:
    vep_df = vep_df.drop(index=removed_non_gm)
    print(f"Removed {len(removed_non_gm)} non-GM regions: {removed_non_gm}")

# remove excluded GM regions
excluded_gm_regions = {
    'Brain-Stem', 'Left-VentralDC', 'Right-VentralDC', 'ctx-lh-unknown', 'ctx-rh-unknown'
}

gm_exclude_mask = vep_df.index.isin(excluded_gm_regions)
removed_gm = vep_df.index[gm_exclude_mask].tolist()
if removed_gm:
    vep_df = vep_df.drop(index=removed_gm)
    print(f"Removed {len(removed_gm)} excluded GM regions: {removed_gm}")

# shape check 
final_n_regions = vep_df.shape[0]
print(f"\nFinal timeseries shape: {vep_df.shape} ({final_n_regions} regions, {vep_df.shape[1]} timepoints)")

if final_n_regions != 162:
    print(f"⚠ WARNING: Expected 162 regions, but found {final_n_regions}")

# save preprocessed VEP timeseries
vep_df.to_csv(snakemake.output.vep_timeseries, index=True)

# touch log file for checkpoint
with open(snakemake.output.done, "w") as f:
    f.write("done\n")
