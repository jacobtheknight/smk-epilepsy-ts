# Compute measures of multivariate information for epileptic networks using HOI (https://brainets.github.io/hoi/)

import os
import numpy as np
import pandas as pd
from hoi.metrics import TC, DTC, Sinfo, Oinfo

# network wildcard → zone label(s) mapping 
NETWORK_LABELS = {
    "ez":    ["EZ"],
    "pz":    ["PZ"],
    "ez+pz": ["EZ", "PZ"],
}

network    = snakemake.wildcards.network
file_path  = snakemake.input.vep_timeseries_labelled
subject_id = os.path.basename(file_path).split("_")[0]

os.makedirs(os.path.dirname(snakemake.output.tc), exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.dtc), exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.sinfo), exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.oinfo), exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.done), exist_ok=True)

print(f"Processing subject : {subject_id}")
print(f"Network partition  : {network}")

zone_labels = NETWORK_LABELS.get(network)
if zone_labels is None:
    raise ValueError(
        f"Unrecognised network wildcard '{network}'. "
        f"Expected one of: {list(NETWORK_LABELS.keys())}"
    )

# load labelled timeseries (col 0 = region name, col 1 = zone label, cols 2+ = TRs)
full_ts = pd.read_csv(file_path, header=None).iloc[1:, 0:]
print(f"Loaded timeseries shape: {full_ts.shape}")

# filter to network partition
mask       = full_ts[1].isin(zone_labels)
regions_df = full_ts[mask]
regions    = regions_df[0].tolist()
n_regions  = len(regions)
print(f"{subject_id}: {n_regions} region(s) in '{network}': {', '.join(regions)}")

# guard: third-order cutoff
if n_regions < 3:
    print(f"WARNING: Only {n_regions} region(s). HOI metrics require >= 3; skipping...")
    skipped = {
        "subject_id": subject_id,
        "network":    network,
        "n_regions":  n_regions,
        "regions":    ",".join(regions),
        "dominance":  np.nan,
    }
    pd.DataFrame([{**skipped, "tc": np.nan}]).to_csv(snakemake.output.tc, index=False)
    pd.DataFrame([{**skipped, "dtc": np.nan}]).to_csv(snakemake.output.dtc, index=False)
    pd.DataFrame([{**skipped, "o_info": np.nan}]).to_csv(snakemake.output.oinfo, index=False)
    pd.DataFrame([{**skipped, "s_info": np.nan}]).to_csv(snakemake.output.sinfo, index=False)
    with open(snakemake.output.done, "w") as f:
        f.write("done\n")
    raise SystemExit(0)

# build (T × N) timeseries array for HOI
ts_array  = regions_df.drop(columns=[0, 1]).to_numpy().astype(np.float64)   # (N, T)
ts_hoi    = ts_array.T                                                        # (T, N)
print(f"Timeseries shape for HOI (T × N): {ts_hoi.shape}")

# compute TC for full multiplet
print(f"Computing total correlation (method=gc, minsize=maxsize={n_regions})...")
model_tc = TC(ts_hoi)
hoi_tc   = model_tc.fit(method="gc", minsize=n_regions, maxsize=n_regions)
tc_value = hoi_tc.item()
print(f"Total correlation : {tc_value:.6f}")

pd.DataFrame([{
    "subject_id": subject_id,
    "network":    network,
    "n_regions":  n_regions,
    "regions":    ",".join(regions),
    "tc":         tc_value,
    "dominance":  np.nan
}]).to_csv(snakemake.output.tc, index=False)
print(f"Saved total correlation -> {snakemake.output.tc}")

# compute DTC for full multiplet
print(f"Computing dual total correlation (method=gc, minsize=maxsize={n_regions})...")
model_dtc = DTC(ts_hoi)
hoi_dtc   = model_dtc.fit(method="gc", minsize=n_regions, maxsize=n_regions)
dtc_value = hoi_dtc.item()
print(f"Dual total correlation : {dtc_value:.6f}")

pd.DataFrame([{
    "subject_id": subject_id,
    "network":    network,
    "n_regions":  n_regions,
    "regions":    ",".join(regions),
    "dtc":        dtc_value,
    "dominance":  np.nan
}]).to_csv(snakemake.output.dtc, index=False)
print(f"Saved dual total correlation -> {snakemake.output.dtc}")

# compute S-information for full multiplet
print(f"Computing S-information (method=gc, minsize=maxsize={n_regions})...")
model_sinfo = Sinfo(ts_hoi)
hoi_sinfo   = model_sinfo.fit(method="gc", minsize=n_regions, maxsize=n_regions)
s_info_value = hoi_sinfo.item()
print(f"S-information : {s_info_value:.6f}")

pd.DataFrame([{
    "subject_id": subject_id,
    "network":    network,
    "n_regions":  n_regions,
    "regions":    ",".join(regions),
    "s_info":     s_info_value,
    "dominance":  np.nan
}]).to_csv(snakemake.output.sinfo, index=False)
print(f"Saved S-information -> {snakemake.output.sinfo}")

# compute O-information for full multiplet
print(f"Computing O-information (method=gc, minsize=maxsize={n_regions})...")
model_oinfo = Oinfo(ts_hoi)
hoi_oinfo   = model_oinfo.fit(method="gc", minsize=n_regions, maxsize=n_regions)
o_info_value = hoi_oinfo.item()
dominance    = "Syn" if o_info_value < 0 else "Red"
print(f"O-information : {o_info_value:.6f}  ({dominance})")

pd.DataFrame([{
    "subject_id": subject_id,
    "network":    network,
    "n_regions":  n_regions,
    "regions":    ",".join(regions),
    "o_info":     o_info_value,
    "dominance":  dominance
}]).to_csv(snakemake.output.oinfo, index=False)
print(f"Saved O-information -> {snakemake.output.oinfo}")

# touch log file for checkpoint
with open(snakemake.output.done, "w") as f:
    f.write("done\n")
print(f"Done: {subject_id} [{network}]")
