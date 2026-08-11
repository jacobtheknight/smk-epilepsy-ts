# Label preprocessed VEP time series with EZ/PZ/NI/NE mapping from SEEG  

import re
import numpy as np
import pandas as pd

# load preprocessed VEP BOLD timeseries
vep_df = pd.read_csv(snakemake.input.vep_timeseries, index_col=0)

# load subject EI file from SEEG
ei_file = pd.read_excel(snakemake.input.ei_file)

# remove trailing/leading whitespaces
ei_file.columns = [str(col).strip() for col in ei_file.columns]

# remove channel information
ei_file = ei_file[['Structure', 'EI', 'ROI']]

# clear existing ROI labels
ei_file['ROI'] = np.nan

# fill ROI column based on EI values 
def ei_to_roi(ei):
    if pd.isna(ei):
        return np.nan
    elif ei >= 0.4:
        return 'EZ'
    elif 0.1 <= ei < 0.4:
        return 'PZ'
    elif 0 < ei < 0.1:
        return 'NI'
    else:
        return 'NE'

# apply ei_to_roi function to fill ROI column based on EI values
ei_file['ROI'] = ei_file['EI'].apply(ei_to_roi)

# define string structure for zone assignment (EZ, PZ, NI, NE)
def zone_tokens(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())

def normalize_name(s):
    return str(s).lower()

# function to detect hemisphere
def detect_side(tokens):
    if any(t in tokens for t in ('left', 'lh')):
        return 'left'
    if any(t in tokens for t in ('right', 'rh')):
        return 'right'
    return None

# function to handle regions labelled with more than one zone 
def assign_label_priority(rois):
    priority = ['EZ', 'PZ', 'NI']  # (EZ > PZ > NI > NE [default])
    up = [str(r).strip().upper() for r in rois if pd.notna(r)]
    for p in priority:
        if p in up:
            return p
    return 'NE'

print("Building Structure to ROI mapping from SEEG...")
structure_to_roi = {}

if 'Structure' in ei_file.columns:
    structure_to_roi = {
        struct.strip(): list(dict.fromkeys(
            grp['ROI'].dropna().astype(str).str.strip().tolist()
        ))
        for struct, grp in ei_file.groupby('Structure')
        if not grp['ROI'].dropna().empty
    }

print("Normalising structure names for matching...")
normalized_to_struct = {normalize_name(s): s for s in structure_to_roi.keys()}

# extract hemisphere info
struct_info = {}
for struct, rois in structure_to_roi.items():
    toks = zone_tokens(struct)
    side = detect_side(toks)
    
    struct_info[struct] = {
        'normalized': normalize_name(struct),
        'side': side,
        'rois': rois
    }

# assign zone labels to VEP regions in BOLD timeseries
print("Mapping VEP regions to SEEG-derived zone labels...")
vep_ezn_labels = []

for vname in vep_df.index:
    # extract normalised name and hemisphere from VEP region
    name_norm = normalize_name(vname)
    name_toks = set(zone_tokens(vname))
    name_side = detect_side(name_toks)
    
    # lookup
    best_struct = None
    if name_norm in normalized_to_struct:
        best_struct = normalized_to_struct[name_norm]
        # verify hemisphere matches if structure specifies one
        if struct_info[best_struct]['side'] and struct_info[best_struct]['side'] != name_side:
            best_struct = None
    
    # assign labels
    if best_struct is None:
        vep_ezn_labels.append('NE')
    else:
        rois = struct_info[best_struct]['rois']
        label = assign_label_priority(rois)
        vep_ezn_labels.append(label)

# attach mapping to dataframe
vep_df['zone_label'] = vep_ezn_labels
vep_df = vep_df[['zone_label'] + [col for col in vep_df.columns if col != 'zone_label']]

# save 
print(f"Saving {len(vep_df)} regions with zone labels to {snakemake.output.vep_timeseries_labelled}")
vep_df.to_csv(snakemake.output.vep_timeseries_labelled, index=True)

# touch log file for checkpoint
with open(snakemake.output.done, "w") as f:
    f.write("done\n")
    