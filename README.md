# Snakemake workflow: smk-epilepsy-ts
SEEG-informed workflow for processing patient resting-state fMRI data using the Virtual Epileptic Patient (VEP) atlas, with downstream structural connectome parcellations and (beyond) pairwise comparative analyses. 

<img width="1730" height="786" alt="workflow" src="https://github.com/user-attachments/assets/115b4175-4861-49c6-90cb-682f29e31f88" />

## Overview
The workflow is organised into patient and control Snakemake-based pipelines and includes:  
- Conversion and resampling of VEP atlas 
- Atlas-to-BOLD alignment checks and quality control
- Extraction of VEP-derived BOLD time series
- Labelling of time series using epileptogenciity index (EI) outputs from SEEG
- Structural connectome generation from diffusion tractography
- Mass computation of pairwise interaction statistics with pyspi (https://github.com/DynamicsAndNeuralSystems/pyspi)
- Multivariate information theoretic analysis with HOI (https://github.com/brainets/hoi)

## Inputs
- participants.xlsx file (expects BIDS subject naming)
- fMRIPrep output data
- FreeSurfer output data (including precomputed VEP atlas https://github.com/HuifangWang/VEP_atlas_shared)
- Patient SEEG-derived EI data (expects EZ, PZ and NI labels assigned to VEP atlas regions)
- MRtrix3 tractography (.tck files)

## Singularity containers required
- fMRIPrep
- FreeSurfer
- MRtrix3
- FSL

## Core dependencies
- NumPy
- pandas
- nibabel
- nilearn
- SciPy

## Additional Python packages 
- pyspi 
- HOI

## Author
Jacob Knight
