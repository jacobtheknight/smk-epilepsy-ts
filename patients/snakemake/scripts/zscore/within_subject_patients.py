# Compare measures of multivariate information in epileptic networks to null distributions sampled from NI/NE regions
# Null networks matched on number of nodes, mean Euclidean distance, and mean Pearson's correlation

import os
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter
from scipy.spatial.distance import pdist
from hoi.metrics import TC, DTC, Sinfo, Oinfo
from null_utils import load_vep_lut, get_region_centroid, compute_mean_pairwise_correlation

# configure null model
null_model_cfg          = snakemake.config["null_model"]
DISTANCE_THRESHOLD_MM   = null_model_cfg["distance_threshold_mm"]
CORRELATION_THRESHOLD_R = null_model_cfg["correlation_threshold_r"]
NULL_ITERATIONS         = null_model_cfg["null_iterations"]
SAFETY_LIMIT            = null_model_cfg["safety_limit"]

network    = snakemake.wildcards.network
subject_id = snakemake.wildcards.subject

os.makedirs(os.path.dirname(snakemake.output.result),            exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.null_values_tc),    exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.null_values_dtc),   exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.null_values_sinfo), exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.null_values_oinfo), exist_ok=True)

print(f"Processing subject : {subject_id}")
print(f"Network partition  : {network}")
print(f"Null model params  : distance threshold = {DISTANCE_THRESHOLD_MM}mm, correlation threshold = {CORRELATION_THRESHOLD_R} r, iterations = {NULL_ITERATIONS}")

# load epileptic network O-information result (and extract number of regions and dominance) 
oinfo_df      = pd.read_csv(snakemake.input.oinfo)
subject_oinfo = float(oinfo_df["o_info"].iloc[0])
epi_regions   = oinfo_df["regions"].iloc[0].split(",")
n_epi_regions = int(oinfo_df["n_regions"].iloc[0])
dominance     = oinfo_df["dominance"].iloc[0]
print(f"Loaded subject O-info: {subject_oinfo:.6f}  ({dominance})  [{n_epi_regions} regions]")

# load epileptic network TC result
tc_df         = pd.read_csv(snakemake.input.tc)
subject_tc    = float(tc_df["tc"].iloc[0])
print(f"Loaded subject TC: {subject_tc:.6f}  [{n_epi_regions} regions]")

# load epileptic network DTC result
dtc_df         = pd.read_csv(snakemake.input.dtc)
subject_dtc    = float(dtc_df["dtc"].iloc[0])
print(f"Loaded subject DTC: {subject_dtc:.6f}  [{n_epi_regions} regions]")

# load epileptic network S-information result
sinfo_df      = pd.read_csv(snakemake.input.sinfo)
subject_sinfo = float(sinfo_df["s_info"].iloc[0])
print(f"Loaded subject S-info: {subject_sinfo:.6f}  [{n_epi_regions} regions]")

# load labelled patient timeseries 
file_path = snakemake.input.vep_timeseries_labelled
full_ts = pd.read_csv(file_path, header=None).iloc[1:, 0:]
print(f"Loaded timeseries shape: {full_ts.shape}")

# write NaN result for snakemake =======================================================================================
_empty_tc_null = pd.DataFrame({"iteration": [], "null_tc": [], "is_finite": []})
_empty_dtc_null = pd.DataFrame({"iteration": [], "null_dtc": [], "is_finite": []})
_empty_sinfo_null = pd.DataFrame({"iteration": [], "null_sinfo": [], "is_finite": []})
_empty_oinfo_null = pd.DataFrame({"iteration": [], "null_oinfo": [], "is_finite": []})

def _write_skipped(reason, extra=None):
    base = {
        "subject_id": subject_id, 
        "network": network, 
        "n_regions": n_epi_regions, 
        "regions": ",".join(epi_regions),
        "s_info": subject_sinfo, 
        "o_info": subject_oinfo, 
        "tc": subject_tc, 
        "dtc": subject_dtc, 
        "dominance": dominance,
        "null_method": reason, 
        "mean_network_dist": np.nan, 
        "mean_network_corr": np.nan,
        "total_iterations": np.nan, 
        "rejection_rate": np.nan,
        "unique_combinations": np.nan,
        "n_unique_regions_sampled": np.nan, 
        "sampling_entropy": np.nan,
        "s_info_zscore": np.nan,
        "o_info_zscore": np.nan,
        "tc_zscore": np.nan,
        "dtc_zscore": np.nan,
    }
    if extra:
        base.update(extra)
    pd.DataFrame([base]).to_csv(snakemake.output.result, index=False)
    _empty_tc_null.to_csv(snakemake.output.null_values_tc, index=False)
    _empty_dtc_null.to_csv(snakemake.output.null_values_dtc, index=False)
    _empty_sinfo_null.to_csv(snakemake.output.null_values_sinfo, index=False)
    _empty_oinfo_null.to_csv(snakemake.output.null_values_oinfo, index=False)
    with open(snakemake.output.done, "w") as f:
        f.write("done\n")

# guard: metrics skipped or invalid
if n_epi_regions < 3 or not np.isfinite(subject_tc) or not np.isfinite(subject_dtc) or not np.isfinite(subject_sinfo) or not np.isfinite(subject_oinfo):
    print(f"WARNING: HOI metrics were skipped or invalid for {subject_id} [{network}]. Writing null result.")
    _write_skipped("skipped_invalid_hoi")
    raise SystemExit(0)
# ======================================================================================================================

vep_lut = load_vep_lut(snakemake.input.lut)
print(f"Loaded VEP LUT")

# load VEP atlas
vep_atlas  = nib.load(snakemake.input.atlas_mni)
atlas_data = vep_atlas.get_fdata().astype(int)
atlas_aff  = vep_atlas.affine
print(f"Loaded VEP atlas shape: {atlas_data.shape}")

# compute centroids for all regions in epileptic network from VEP coordinates in MNI space
all_coords = {}
for region_name in full_ts[0].values:
    label_idx = vep_lut.get(region_name)
    if label_idx is None:
        print(f"Warning: '{region_name}' not in LUT")
        all_coords[region_name] = np.full(3, np.nan)
        continue
    centroid = get_region_centroid(atlas_data, atlas_aff, label_idx)
    all_coords[region_name] = centroid if centroid is not None else np.full(3, np.nan)

# compute mean Euclidean distance between epileptic network nodes
epi_network_coords = np.array([all_coords[r] for r in epi_regions])
valid_net      = ~np.isnan(epi_network_coords).any(axis=1)
epi_network_coords = epi_network_coords[valid_net]
epi_network_dists  = pdist(epi_network_coords, metric="euclidean")
print(f"Epileptic network ({network}) Euclidean distance: mean = {epi_network_dists.mean():.2f}mm, std = {epi_network_dists.std():.2f}mm")

# compute mean Pearson's correlation between epileptic network nodes
epi_ts = full_ts[full_ts[0].isin(epi_regions)].drop(columns=[0, 1]).to_numpy().astype(np.float64)
network_mean_corr = compute_mean_pairwise_correlation(epi_ts)
print(f"Epileptic network ({network}) Pearson's correlation: mean r = {network_mean_corr:.4f}, std = {network_mean_corr:.4f}")

# NI and NE region pool
healthy_mask   = ~full_ts[1].isin(["EZ", "PZ"])
healthy_df     = full_ts[healthy_mask]
healthy_ts_arr = healthy_df.drop(columns=[0, 1]).to_numpy().astype(np.float64)  # (N, T)
n_healthy      = healthy_ts_arr.shape[0]
print(f"NI/NE pool: {n_healthy} regions")

if n_healthy < n_epi_regions:
    print(f"WARNING: insufficient NI/NE regions ({n_healthy}) for null model (need {n_epi_regions}).")
    _write_skipped("insufficient_null_pool")
    raise SystemExit(0)

healthy_coords_list = []
valid_indices       = []
for idx, name in enumerate(healthy_df[0].values):
    coord = all_coords.get(name, np.full(3, np.nan))
    if not np.isnan(coord).any():
        healthy_coords_list.append(coord)
        valid_indices.append(idx)

healthy_coords = np.array(healthy_coords_list)
print(f"Valid NI/NE regions with coordinates: {len(valid_indices)}/{n_healthy}")

# prepare null distribution for the TC, DTC, S-information and O-information
null_tc_values = []
null_dtc_values = []
null_sinfo_values = []
null_oinfo_values    = []
accepted_combos_indices = set()
accepted_combos_names   = []
healthy_region_names    = {i: healthy_df[0].values[valid_indices[i]] for i in range(len(valid_indices))}
successful              = 0
failed                  = 0
distance_rejected       = 0
correlation_rejected    = 0
duplicates              = 0
total                   = 0
np.random.seed(42)

print(f"Generating null distribution ({NULL_ITERATIONS} iterations)...")
while successful < NULL_ITERATIONS:
    total += 1
    if total > SAFETY_LIMIT:
        print(f"ERROR: Safety limit reached ({total} iterations). Stopping.")
        break

    # match on number of nodes
    sample_idx = np.random.choice(len(valid_indices), size=n_epi_regions, replace=False)
    combo_id   = tuple(sorted(valid_indices[j] for j in sample_idx))
    if combo_id in accepted_combos_indices:
        duplicates += 1
        continue

    # match on mean Euclidean distance 
    rand_coords = healthy_coords[sample_idx]
    rand_dists  = pdist(rand_coords, metric="euclidean")
    dist_diff   = np.mean(np.abs(np.sort(epi_network_dists) - np.sort(rand_dists)))
    if dist_diff >= DISTANCE_THRESHOLD_MM:
        distance_rejected += 1
        continue

    # match on mean Pearson's correlation
    rand_ts_candidate = healthy_ts_arr[[valid_indices[j] for j in sample_idx], :] 
    rand_mean_corr = compute_mean_pairwise_correlation(rand_ts_candidate)
    corr_diff = np.abs(network_mean_corr - rand_mean_corr)
    if corr_diff >= CORRELATION_THRESHOLD_R:
        correlation_rejected += 1
        continue

    # if all checks cleared, accept null network into distribution
    accepted_combos_indices.add(combo_id)
    combo_names = tuple(sorted([healthy_region_names[j] for j in sample_idx]))
    accepted_combos_names.append(combo_names)
    
    rand_ts = healthy_ts_arr[[valid_indices[j] for j in sample_idx], :] # (N, T)
    rand_hoi = rand_ts.T                                                # (T, N)

    # compute measures of multivariate information for the null network
    try:
        null_tc = TC(rand_hoi).fit(method="gc", minsize=n_epi_regions, maxsize=n_epi_regions)
        null_tc_values.append(float(null_tc[0]))
        null_dtc = DTC(rand_hoi).fit(method="gc", minsize=n_epi_regions, maxsize=n_epi_regions)
        null_dtc_values.append(float(null_dtc[0]))
        null_sinfo = Sinfo(rand_hoi).fit(method="gc", minsize=n_epi_regions, maxsize=n_epi_regions)
        null_sinfo_values.append(float(null_sinfo[0]))
        null_oinfo = Oinfo(rand_hoi).fit(method="gc", minsize=n_epi_regions, maxsize=n_epi_regions)
        null_oinfo_values.append(float(null_oinfo[0]))
        successful += 1
        if successful % 100 == 0:
            print(f"Successful iterations: {successful}/{NULL_ITERATIONS} ({subject_id})")
    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"Warning: Failed iteration {successful + failed}: {e}")

print(f"\nNull distribution complete:")
print(f"  Distance rejections: {distance_rejected}")
print(f"  Correlation rejections: {correlation_rejected}")
print(f"  Rejection rate: {(distance_rejected + correlation_rejected) / total * 100:.2f}%")
print(f"  Unique combinations: {len(accepted_combos_indices)}/{NULL_ITERATIONS}")

# sampling analysis
all_sampled_regions = []
for combo_names in accepted_combos_names:
    all_sampled_regions.extend(combo_names)

region_counts = Counter(all_sampled_regions)
n_unique_regions = len(region_counts)
total_region_samples = len(all_sampled_regions)

# entropy of sampling distribution (normalised)
if n_unique_regions > 1:
    p = np.array(list(region_counts.values())) / total_region_samples
    sampling_entropy = -np.sum(p * np.log(p)) / np.log(n_unique_regions)
else:
    sampling_entropy = 0.0

print(f"\nRegion sampling analysis:")
print(f"  Unique regions sampled: {n_unique_regions}/{n_healthy}")
print(f"  Sampling entropy (0=biased, 1=uniform): {sampling_entropy:.3f}")

# z-score helper function
def compute_zscore(observed, null_values):
    finite = [x for x in null_values if np.isfinite(x)]
    if not finite:
        return float("nan"), len(finite)

    arr = np.array(finite)
    mean, std = float(np.mean(arr)), float(np.std(arr))

    z = (observed - mean) / std if std > 0 else 0.0

    return z, len(finite)

# z-score TC
tc_zscore, n_finite_tc = compute_zscore(subject_tc, null_tc_values)
print(f"Finite null TC values: {n_finite_tc}/{len(null_tc_values)}")

# z-score DTC
dtc_zscore, n_finite_dtc = compute_zscore(subject_dtc, null_dtc_values)
print(f"Finite null DTC values: {n_finite_dtc}/{len(null_dtc_values)}")

# z-score S-information
sinfo_zscore, n_finite_sinfo = compute_zscore(subject_sinfo, null_sinfo_values)
print(f"Finite null S-info values: {n_finite_sinfo}/{len(null_sinfo_values)}")

# z-score O-information
oinfo_zscore, n_finite_oinfo = compute_zscore(subject_oinfo, null_oinfo_values)
print(f"Finite null O-info values: {n_finite_oinfo}/{len(null_oinfo_values)}")

print(f"TC null:  z={tc_zscore:.4f}")
print(f"DTC null:  z={dtc_zscore:.4f}")
print(f"S-info null:  z={sinfo_zscore:.4f}")
print(f"O-info null:  z={oinfo_zscore:.4f}")

# save null distributions to CSV

pd.DataFrame({ # TC null distribution
    "iteration": range(len(null_tc_values)),
    "null_tc": null_tc_values,
    "is_finite":  [np.isfinite(x) for x in null_tc_values],
}).to_csv(snakemake.output.null_values_tc, index=False)
print(f"Saved null TC values -> {snakemake.output.null_values_tc}")

pd.DataFrame({ # DTC null distribution
    "iteration": range(len(null_dtc_values)),
    "null_dtc": null_dtc_values,
    "is_finite":  [np.isfinite(x) for x in null_dtc_values],
}).to_csv(snakemake.output.null_values_dtc, index=False)
print(f"Saved null DTC values -> {snakemake.output.null_values_dtc}")

pd.DataFrame({ # S-info null distribution
    "iteration": range(len(null_sinfo_values)),
    "null_sinfo": null_sinfo_values,
    "is_finite":  [np.isfinite(x) for x in null_sinfo_values],
}).to_csv(snakemake.output.null_values_sinfo, index=False)
print(f"Saved null S-info values -> {snakemake.output.null_values_sinfo}")

pd.DataFrame({ # O-info null distribution
    "iteration": range(len(null_oinfo_values)),
    "null_oinfo": null_oinfo_values,
    "is_finite":  [np.isfinite(x) for x in null_oinfo_values],
}).to_csv(snakemake.output.null_values_oinfo, index=False)
print(f"Saved null O-info values -> {snakemake.output.null_values_oinfo}")

# save combined z-score result for epileptic network
result = {
    "subject_id": subject_id,
    "network": network,
    "n_regions": n_epi_regions,
    "regions": ",".join(epi_regions),
    "tc": subject_tc,
    "dtc": subject_dtc,
    "s_info": subject_sinfo,
    "o_info": subject_oinfo,
    "dominance": dominance,
    "null_method": "within-subject",
    "mean_network_dist": np.mean(epi_network_dists),
    "mean_network_corr": network_mean_corr,
    "total_iterations": total,
    "distance_rejections": distance_rejected,
    "correlation_rejections": correlation_rejected,
    "rejection_rate": (distance_rejected + correlation_rejected) / total if total > 0 else np.nan,
    "unique_combinations": len(accepted_combos_indices),
    "n_unique_regions_sampled": n_unique_regions,
    "sampling_entropy": sampling_entropy,
    "tc_zscore": tc_zscore,
    "dtc_zscore": dtc_zscore,
    "s_info_zscore": sinfo_zscore,
    "o_info_zscore": oinfo_zscore,
}
pd.DataFrame([result]).to_csv(snakemake.output.result, index=False)
print(f"Saved within-subject patient z-scores -> {snakemake.output.result}")

# touch log file for checkpoint
with open(snakemake.output.done, "w") as f:
    f.write("done\n")
print(f"Done: {subject_id} [{network}]")
