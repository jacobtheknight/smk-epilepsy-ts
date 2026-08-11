# Snakemake workflow: smk-epilepsy-ts
SEEG-informed workflow for processing patient resting-state fMRI data using the Virtual Epileptic Patient (VEP) atlas, with downstream structural connectome parcellations and (beyond) pairwise comparative analyses. 

<img width="740" height="561" alt="Figure 1" src="https://github.com/user-attachments/assets/51a7ddf2-edd2-40bf-841f-e85374ca5a14" />

## Overview
The workflow is organised into patient and control Snakemake-based pipelines and includes:  
- Conversion and resampling of VEP atlas 
- Atlas-to-BOLD alignment checks and quality control
- Extraction of VEP-derived BOLD time series
- Labelling of time series using epileptogenciity index (EI) outputs from SEEG
- Structural connectome generation from diffusion tractography
- Mass computation of pairwise interaction statistics using pyspi (https://github.com/DynamicsAndNeuralSystems/pyspi)
- Multivariate information theoretic analysis using HOI (https://github.com/brainets/hoi)

## Required inputs
- participants.xlsx file (sub-xxx)
- fMRIPrep output data
- FreeSurfer output data (including computed VEP atlas https://github.com/HuifangWang/VEP_atlas_shared)
- SEEG-derived EI data 
- MRtrix3 tractography files (.tck)

## Singilarity containers required
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
