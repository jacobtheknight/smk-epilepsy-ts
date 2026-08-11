# Compare measures of multivariate information in epileptic networks to a healthy control distribution
# Homologous control networks matched on same anatomical regions

import os
import glob
import numpy as np
import pandas as pd
from hoi.metrics import TC, DTC, Sinfo, Oinfo

network    = snakemake.wildcards.network
subject_id = snakemake.wildcards.subject

os.makedirs(os.path.dirname(snakemake.output.result),                               exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.bootstrapped_control_values_tc),       exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.bootstrapped_control_values_dtc),      exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.bootstrapped_control_values_sinfo),    exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.bootstrapped_control_values_oinfo),    exist_ok=True)

print(f"Processing subject : {subject_id}")
print(f"Network partition  : {network}")

# load epileptic network O-information result (and extract number of regions and dominance) 
oinfo_df      = pd.read_csv(snakemake.input.oinfo)
subject_oinfo = float(oinfo_df["o_info"].iloc[0])
epi_regions       = oinfo_df["regions"].iloc[0].split(",")
n_epi_regions = int(oinfo_df["n_regions"].iloc[0])
dominance     = oinfo_df["dominance"].iloc[0]
print(f"Loaded subject O-info: {subject_oinfo:.6f}  ({dominance})  [{n_epi_regions} regions]")

# load subject TC result
tc_df      = pd.read_csv(snakemake.input.tc)
subject_tc = float(tc_df["tc"].iloc[0])
print(f"Loaded subject TC: {subject_tc:.6f}  [{n_epi_regions} regions]")

# load subject DTC result
dtc_df      = pd.read_csv(snakemake.input.dtc)
subject_dtc = float(dtc_df["dtc"].iloc[0])
print(f"Loaded subject DTC: {subject_dtc:.6f}  [{n_epi_regions} regions]")

# load subject S-information result
sinfo_df      = pd.read_csv(snakemake.input.sinfo)
subject_sinfo = float(sinfo_df["s_info"].iloc[0])
print(f"Loaded subject S-info: {subject_sinfo:.6f}  [{n_epi_regions} regions]")

# write NaN result for snakemake ===================================================================================
_empty_bootstrap_tc = pd.DataFrame({"bootstrap_median": [], "bootstrap_mad": []})
_empty_bootstrap_dtc = pd.DataFrame({"bootstrap_median": [], "bootstrap_mad": []})
_empty_bootstrap_sinfo = pd.DataFrame({"bootstrap_median": [], "bootstrap_mad": []})
_empty_bootstrap_oinfo = pd.DataFrame({"bootstrap_median": [], "bootstrap_mad": []})

def _write_skipped(reason):
    base = {
        "subject_id": subject_id, "network": network,
        "n_regions": n_epi_regions, "regions": ",".join(epi_regions),
        "tc": subject_tc, "dtc": subject_dtc, "s_info": subject_sinfo, "o_info": subject_oinfo, "dominance": dominance,
        "null_method": reason,
        "n_controls_attempted": np.nan, "n_controls_used": np.nan,
        "tc_zscore": np.nan,
        "dtc_zscore": np.nan,
        "s_info_zscore": np.nan,
        "o_info_zscore": np.nan,
    }
    pd.DataFrame([base]).to_csv(snakemake.output.result, index=False)
    _empty_bootstrap_tc.to_csv(snakemake.output.bootstrapped_control_values_tc, index=False)
    _empty_bootstrap_dtc.to_csv(snakemake.output.bootstrapped_control_values_dtc, index=False)
    _empty_bootstrap_sinfo.to_csv(snakemake.output.bootstrapped_control_values_sinfo, index=False)
    _empty_bootstrap_oinfo.to_csv(snakemake.output.bootstrapped_control_values_oinfo, index=False)
    with open(snakemake.output.done, "w") as f:
        f.write("done\n")

# guard: metrics skipped or invalid
if n_epi_regions < 3 or not np.isfinite(subject_tc) or not np.isfinite(subject_dtc) or not np.isfinite(subject_sinfo) or not np.isfinite(subject_oinfo):
    print(f"WARNING: HOI metrics were skipped or invalid for {subject_id} [{network}]. Writing null result.")
    _write_skipped("skipped_invalid_hoi")
    raise SystemExit(0)
# ==================================================================================================================

# discover control time series database
control_dir   = snakemake.params.control_timeseries_dir
control_files = sorted(glob.glob(
    os.path.join(control_dir, "**", "*_vep_timeseries.csv"), recursive=True
))
print(f"Found {len(control_files)} control timeseries files under {control_dir}")

if len(control_files) == 0:
    msg = f"no control timeseries files found under {control_dir}"
    print(f"WARNING: {msg}")
    _write_skipped("no_control_data")
    raise SystemExit(0)

# prepare control distribution for the TC, DTC, S-information and O-information
control_tc_values    = []
control_dtc_values   = []
control_oinfo_values = []
control_sinfo_values = []
skipped              = 0
np.random.seed(42)

for ctrl_file in control_files:
    ctrl_basename = os.path.basename(ctrl_file)
    try:
        ctrl_ts = pd.read_csv(ctrl_file, index_col=0)

        missing = [r for r in epi_regions if r not in ctrl_ts.index]
        if missing:
            skipped += 1
            if skipped <= 5:
                print(f"Skipping {ctrl_basename} — missing regions: {missing[:3]}...")
            continue

        # match control time series to epileptic network regions
        ctrl_region_ts = ctrl_ts.loc[epi_regions].to_numpy().astype(np.float64)  # (N, T)
        ctrl_ts_hoi    = ctrl_region_ts.T                                      # (T, N)

        ctrl_tc = TC(ctrl_ts_hoi).fit(method="gc", minsize=n_epi_regions, maxsize=n_epi_regions)
        control_tc_values.append(float(ctrl_tc[0]))
        ctrl_dtc = DTC(ctrl_ts_hoi).fit(method="gc", minsize=n_epi_regions, maxsize=n_epi_regions)
        control_dtc_values.append(float(ctrl_dtc[0]))
        ctrl_sinfo = Sinfo(ctrl_ts_hoi).fit(method="gc", minsize=n_epi_regions, maxsize=n_epi_regions)
        control_sinfo_values.append(float(ctrl_sinfo[0]))
        ctrl_oinfo = Oinfo(ctrl_ts_hoi).fit(method="gc", minsize=n_epi_regions, maxsize=n_epi_regions)
        control_oinfo_values.append(float(ctrl_oinfo[0]))

    except Exception as e:
        skipped += 1
        if skipped <= 5:
            print(f"Warning: Failed for {ctrl_basename}: {e}")
        continue

print(f"\nControl distribution:")
print(f"  Attempted : {len(control_files)}")
print(f"  Succeeded : {len(control_oinfo_values)}")
print(f"  Skipped   : {skipped}")

# z-score helper 
def compute_zscore(observed, control_values):
    finite = [x for x in control_values if np.isfinite(x)]
    if not finite:
        return np.nan, 0
    arr = np.array(finite)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    z = (observed - mean_val) / std_val if std_val > 0 else 0.0
    return z, len(finite)

# bootstrap distribution of control values for histogram visualisation 
def bootstrap_distribution(control_values, n_bootstrap=1000):
    finite = [x for x in control_values if np.isfinite(x)]
    boot_medians, boot_mads = [], []
    if not finite:
        return boot_medians, boot_mads
    arr = np.array(finite)
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_med = float(np.median(boot_sample))
        boot_mad = float(np.median(np.abs(boot_sample - boot_med)))
        boot_medians.append(boot_med)
        boot_mads.append(boot_mad)
    return boot_medians, boot_mads

# z-score TC
tc_zscore, n_finite_tc = compute_zscore(subject_tc, control_tc_values)
bootstrap_tc_medians, bootstrap_tc_mads = bootstrap_distribution(control_tc_values)
print(f"Finite control TC values: {n_finite_tc}/{len(control_tc_values)}")

# z-score DTC
dtc_zscore, n_finite_dtc = compute_zscore(subject_dtc, control_dtc_values)
bootstrap_dtc_medians, bootstrap_dtc_mads = bootstrap_distribution(control_dtc_values)
print(f"Finite control DTC values: {n_finite_dtc}/{len(control_dtc_values)}")

# z-score S-information
sinfo_zscore, n_finite_sinfo = compute_zscore(subject_sinfo, control_sinfo_values)
bootstrap_sinfo_medians, bootstrap_sinfo_mads = bootstrap_distribution(control_sinfo_values)
print(f"Finite control S-info values: {n_finite_sinfo}/{len(control_sinfo_values)}")

# z-score O-information
oinfo_zscore, n_finite_oinfo = compute_zscore(subject_oinfo, control_oinfo_values)
bootstrap_oinfo_medians, bootstrap_oinfo_mads = bootstrap_distribution(control_oinfo_values)
print(f"Finite control O-info values: {n_finite_oinfo}/{len(control_oinfo_values)}")

print(f"TC control: z={tc_zscore:.4f}")
print(f"DTC control: z={dtc_zscore:.4f}")
print(f"S-info control: z={sinfo_zscore:.4f}")
print(f"O-info control: z={oinfo_zscore:.4f}")

# save bootstrapped healthy control distributions

pd.DataFrame({ # TC bootstrap distribution
    "bootstrap_median": bootstrap_tc_medians,
    "bootstrap_mad": bootstrap_tc_mads,
}).to_csv(snakemake.output.bootstrapped_control_values_tc, index=False)
print(f"Saved bootstrap TC distribution -> {snakemake.output.bootstrapped_control_values_tc}")

pd.DataFrame({ # DTC bootstrap distribution
    "bootstrap_median": bootstrap_dtc_medians,
    "bootstrap_mad": bootstrap_dtc_mads,
}).to_csv(snakemake.output.bootstrapped_control_values_dtc, index=False)
print(f"Saved bootstrap DTC distribution -> {snakemake.output.bootstrapped_control_values_dtc}")

pd.DataFrame({ # S-info bootstrap distribution
    "bootstrap_median": bootstrap_sinfo_medians,
    "bootstrap_mad": bootstrap_sinfo_mads,
}).to_csv(snakemake.output.bootstrapped_control_values_sinfo, index=False)
print(f"Saved bootstrap S-info distribution -> {snakemake.output.bootstrapped_control_values_sinfo}")

pd.DataFrame({ # O-info bootstrap distribution
    "bootstrap_median": bootstrap_oinfo_medians,
    "bootstrap_mad": bootstrap_oinfo_mads,
}).to_csv(snakemake.output.bootstrapped_control_values_oinfo, index=False)
print(f"Saved bootstrap O-info distribution -> {snakemake.output.bootstrapped_control_values_oinfo}")

# save combined z-score result
result = {
    "subject_id":           subject_id,
    "network":              network,
    "n_regions":            n_epi_regions,
    "regions":              ",".join(epi_regions),
    "tc":                   subject_tc,
    "dtc":                  subject_dtc,
    "s_info":               subject_sinfo,
    "o_info":               subject_oinfo,
    "dominance":            dominance,
    "null_method":          "between-subject",
    "n_controls_attempted": len(control_files),
    "n_controls_used":      n_finite_oinfo,  # use O-info count as reference 
    "tc_zscore":            tc_zscore,
    "dtc_zscore":           dtc_zscore,
    "s_info_zscore":        sinfo_zscore,
    "o_info_zscore":        oinfo_zscore,
}
pd.DataFrame([result]).to_csv(snakemake.output.result, index=False)
print(f"Saved between-subject z-scores -> {snakemake.output.result}")

# touch log file for checkpoint
with open(snakemake.output.done, "w") as f:
    f.write("done\n")
print(f"Done: {subject_id} [{network}]")
