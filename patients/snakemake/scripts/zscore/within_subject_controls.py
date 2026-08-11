# Compare measures of multivariate information in homologous control networks to null distributions sampled from all other regions
# Null networks matched on number of nodes, mean Euclidean distance, and mean Pearson's correlation

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import pdist
from hoi.metrics import TC, DTC, Sinfo, Oinfo
from null_utils import load_vep_lut, get_region_centroid, compute_mean_pairwise_correlation

# configure null model
null_model_cfg          = snakemake.config["null_model"]
DISTANCE_THRESHOLD_MM   = null_model_cfg["distance_threshold_mm"]
CORRELATION_THRESHOLD_R = null_model_cfg["correlation_threshold_r"]
NULL_ITERATIONS         = null_model_cfg["null_iterations"]
SAFETY_LIMIT            = null_model_cfg["safety_limit"]
NETWORK_LABELS = {"ez": ["EZ"], "pz": ["PZ"], "ez+pz": ["EZ", "PZ"]}

network = snakemake.wildcards.network
subject_id = snakemake.wildcards.subject
control_id = snakemake.wildcards.control
epi_labels = NETWORK_LABELS[network]

os.makedirs(os.path.dirname(snakemake.output.result), exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.done), exist_ok=True)

print(f"Processing subject: {subject_id}")
print(f"Network: {network} | Control: {control_id}")
print(f"Null model params: distance threshold = {DISTANCE_THRESHOLD_MM}mm, correlation threshold = {CORRELATION_THRESHOLD_R}, iterations = {NULL_ITERATIONS}")

# load patient labelled timeseries and extract epileptic network regions
pt_df = pd.read_csv(snakemake.input.vep_timeseries_labelled, header=None).iloc[1:].reset_index(drop=True)
epi_regions = pt_df[pt_df[1].isin(epi_labels)][0].tolist()
n_epi = len(epi_regions)
print(f"Epileptic network regions ({n_epi}): {epi_regions}")

# write NaN result for snakemake ===================================================================================
def write_skipped(reason, regions, n_regions, extra_error=""):
    write_result({
        "subject_id": subject_id,
        "control_id": control_id,
        "network": network,
        "n_regions": n_regions,
        "regions": ",".join(regions),
        "tc": np.nan,
        "dtc": np.nan,
        "s_info": np.nan,
        "o_info": np.nan,
        "dominance": "",
        "null_method": reason,
        "mean_network_dist": np.nan,
        "mean_network_corr": np.nan,
        "total_iterations": np.nan,
        "distance_rejections": 0,
        "correlation_rejections": 0,
        "rejection_rate": np.nan,
        "unique_combinations": 0,
        "tc_zscore": np.nan,
        "dtc_zscore": np.nan,
        "s_info_zscore": np.nan,
        "o_info_zscore": np.nan,
        "error": extra_error,
    })

# guard: metrics skipped or invalid 
if n_epi < 3:
    print("WARNING: fewer than 3 network regions. Writing skipped output.")
    write_skipped("skipped_invalid_network", epi_regions, n_epi, "fewer than 3 network regions")
    raise SystemExit(0)
# ==================================================================================================================

vep_lut = load_vep_lut(snakemake.input.lut)
print(f"Loaded VEP LUT ({len(vep_lut)} entries)")

# load control time series data
ctrl_ts = pd.read_csv(snakemake.input.control_timeseries, index_col=0)
ctrl_names = list(ctrl_ts.index)
ctrl_arr = ctrl_ts.to_numpy().astype(np.float64)
print(f"Loaded control timeseries shape: {ctrl_arr.shape}")

# guard against missing homologue regions in control timeseries
missing = [r for r in epi_regions if r not in ctrl_ts.index]
if missing:
    msg = f"missing homologue region(s) in control: {','.join(missing)}"
    print(f"WARNING: {msg}")
    write_skipped("missing_homologue_regions", epi_regions, n_epi, msg)
    raise SystemExit(0)

# load VEP atlas
ctrl_atlas = nib.load(snakemake.input.control_atlas_mni)
atlas_data = ctrl_atlas.get_fdata().astype(int)
atlas_aff = ctrl_atlas.affine
print(f"Loaded control VEP atlas shape: {atlas_data.shape}")

# compute region centroids for all homologue control network regions
all_coords = {}
for name in ctrl_names:
    label_idx = vep_lut.get(name)
    if label_idx is None:
        all_coords[name] = np.full(3, np.nan)
        continue
    centroid = get_region_centroid(atlas_data, atlas_aff, label_idx)
    all_coords[name] = centroid if centroid is not None else np.full(3, np.nan)

# compute TC, DTC, S-information and O-information of control homologue network
homo_ts = ctrl_ts.loc[epi_regions].to_numpy().astype(np.float64)
homo_hoi = homo_ts.T
tc_val = float(TC(homo_hoi).fit(method="gc", minsize=n_epi, maxsize=n_epi).item())
dtc_val = float(DTC(homo_hoi).fit(method="gc", minsize=n_epi, maxsize=n_epi).item())
sinfo_val = float(Sinfo(homo_hoi).fit(method="gc", minsize=n_epi, maxsize=n_epi).item())
oinfo_val = float(Oinfo(homo_hoi).fit(method="gc", minsize=n_epi, maxsize=n_epi).item())
dominance = "Syn" if oinfo_val < 0 else "Red"

# define homologue control network nodes
homo_coords = np.array([all_coords[r] for r in epi_regions])
valid_homo = ~np.isnan(homo_coords).any(axis=1)

# compute mean Euclidean distance between control homologue network nodes
mean_net_dist = float(np.mean(pdist(homo_coords[valid_homo]))) if valid_homo.sum() >= 2 else np.nan

# compute mean Pearson's correlation between control homologue network nodes
mean_net_corr = compute_mean_pairwise_correlation(homo_ts)

# set null pool of all control regions except homologue control network regions
epi_set = set(epi_regions)
pool_mask = np.array([name not in epi_set for name in ctrl_names])
pool_arr = ctrl_arr[pool_mask]
pool_names = [n for n, m in zip(ctrl_names, pool_mask) if m]

valid_pool_idx = []
valid_pool_coords = []
for i, name in enumerate(pool_names):
    c = all_coords.get(name, np.full(3, np.nan))
    if not np.isnan(c).any():
        valid_pool_idx.append(i)
        valid_pool_coords.append(c)

valid_pool_idx = np.array(valid_pool_idx)
valid_pool_coords = np.array(valid_pool_coords) if valid_pool_coords else np.empty((0, 3))
n_valid_pool = len(valid_pool_idx)
print(f"Valid null pool size: {n_valid_pool}")

# guard: insufficient null pool regions
if n_valid_pool < n_epi or not np.isfinite(mean_net_dist) or not np.isfinite(mean_net_corr):
    msg = f"insufficient null pool ({n_valid_pool} valid regions, need {n_epi})"
    print(f"WARNING: {msg}")
    write_skipped("insufficient_null_pool", epi_regions, n_epi, msg)
    raise SystemExit(0)

# observed pairwise distances for elementwise null matching
homo_dists = pdist(homo_coords[valid_homo], metric="euclidean")

# prepare null distribution for the TC, DTC, S-information and O-information
null_tc = []
null_dtc = []
null_sinfo = []
null_oinfo = []
accepted = set()
successful = 0
failed = 0
duplicates = 0
total = 0
dist_rej = 0
corr_rej = 0
np.random.seed(42)

print(f"Generating null distribution ({NULL_ITERATIONS} iterations)...")
while successful < NULL_ITERATIONS:
    total += 1
    if total > SAFETY_LIMIT:
        print(f"ERROR: Safety limit reached ({total} iterations). Stopping.")
        break

    # match on number of nodes
    chosen = np.random.choice(n_valid_pool, size=n_epi, replace=False)
    combo = tuple(sorted(chosen.tolist()))
    if combo in accepted:
        duplicates += 1
        continue

    # match on mean Euclidean distance 
    chosen_coords = valid_pool_coords[chosen]
    chosen_dists = pdist(chosen_coords, metric="euclidean")
    dist_diff = np.mean(np.abs(np.sort(homo_dists) - np.sort(chosen_dists)))
    if dist_diff >= DISTANCE_THRESHOLD_MM:
        dist_rej += 1
        continue

    # match on mean Pearson's correlation
    chosen_ts = pool_arr[valid_pool_idx[chosen]]
    null_corr = compute_mean_pairwise_correlation(chosen_ts)
    if abs(null_corr - mean_net_corr) >= CORRELATION_THRESHOLD_R:
        corr_rej += 1
        continue

    # if all checks cleared, accept null network into distribution
    accepted.add(combo)
    null_hoi = chosen_ts.T

    # compute TC, DTC, S-information and O-information for the null network
    try:
        n_tc = float(TC(null_hoi).fit(method="gc", minsize=n_epi, maxsize=n_epi).item())
        n_dtc = float(DTC(null_hoi).fit(method="gc", minsize=n_epi, maxsize=n_epi).item())
        n_sinfo = float(Sinfo(null_hoi).fit(method="gc", minsize=n_epi, maxsize=n_epi).item())
        n_oinfo = float(Oinfo(null_hoi).fit(method="gc", minsize=n_epi, maxsize=n_epi).item())
        null_sinfo.append(n_sinfo)
        null_oinfo.append(n_oinfo)
        null_tc.append(n_tc)
        null_dtc.append(n_dtc)
        successful += 1
        if successful % 100 == 0:
            print(f"Successful null samples: {successful}/{NULL_ITERATIONS}")
    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"Warning: Failed iteration {successful + failed}: {e}")

# z-score helper
def compute_zscore(observed, null_values):
    finite = [x for x in null_values if np.isfinite(x)]
    if not finite:
        return np.nan, 0
    arr = np.array(finite)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    z = (observed - mean_val) / std_val if std_val > 0 else 0.0
    return z, len(finite)

# compute z-scores for TC, DTC, S-information and O-information
n_null = len(null_oinfo)
tc_z, _ = compute_zscore(tc_val, null_tc)
dtc_z, _ = compute_zscore(dtc_val, null_dtc)
sinfo_z, _ = compute_zscore(sinfo_val, null_sinfo)
oinfo_z, _ = compute_zscore(oinfo_val, null_oinfo)

result = {
    "subject_id": subject_id,
    "control_id": control_id,
    "network": network,
    "n_regions": n_epi,
    "regions": ",".join(epi_regions),
    "tc": tc_val,
    "dtc": dtc_val,
    "s_info": sinfo_val,
    "o_info": oinfo_val,
    "dominance": dominance,
    "null_method": "within-subject",
    "mean_network_dist": mean_net_dist,
    "mean_network_corr": mean_net_corr,
    "total_iterations": total,
    "distance_rejections": dist_rej,
    "correlation_rejections": corr_rej,
    "rejection_rate": (dist_rej + corr_rej) / total if total > 0 else np.nan,
    "unique_combinations": len(accepted),
    "tc_zscore": tc_z,
    "dtc_zscore": dtc_z,
    "s_info_zscore": sinfo_z,
    "o_info_zscore": oinfo_z,
}
pd.DataFrame([result]).to_csv(snakemake.output.result, index=False)
print(f"Saved within-subject control z-scores -> {snakemake.output.result}")

# touch log file for checkpoint
with open(snakemake.output.done, "w") as f:
    f.write("done\n")
print(f"Done: {subject_id} {network} {control_id}")