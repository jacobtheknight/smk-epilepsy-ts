# Shared helpers for null-model comparison scripts (control_hoi_null_comp.py, patient_hoi_null_comp.py)

import numpy as np

# load VEP LUT
def load_vep_lut(lut_path):
    lut_dict = {}
    with open(lut_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    label_idx   = int(parts[0])
                    region_name = " ".join(parts[1:-4])
                    lut_dict[region_name] = label_idx
                except (ValueError, IndexError):
                    continue
    return lut_dict

# centroid helper
def get_region_centroid(atlas_data, affine, label_idx):
    mask = atlas_data == label_idx
    if not mask.any():
        return None
    vox = np.argwhere(mask).mean(axis=0)
    return (affine @ np.append(vox, 1))[:3]

# compute mean Pearson's correlation between network nodes
def compute_mean_pairwise_correlation(ts):
    if ts.shape[0] < 2:
        return 0.0
    corr_matrix = np.corrcoef(ts)
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    return float(np.mean(corr_matrix[upper_tri_indices]))
