# Compute healthy control pairwise connectivity matrices from 284 pairwise interaction statistics using pyspi (https://github.com/DynamicsAndNeuralSystems/pyspi)

import os, shutil, sys, io
import numpy as np 
import pandas as pd
import pyspi
from pyspi.calculator import Calculator

# ensure octave and java dependencies available in path for PySpi
os.environ["PATH"] += os.pathsep + "/usr/local/octave/bin"
os.environ["PATH"] += os.pathsep + "/usr/lib/jvm/java-11-openjdk-amd64/bin"

# ensure output directories exist 
os.makedirs(os.path.dirname(snakemake.output.spi_matrix), exist_ok=True) 
os.makedirs(os.path.dirname(snakemake.output.log_summary), exist_ok=True)
os.makedirs(os.path.dirname(snakemake.output.done), exist_ok=True) 

# use custom spi set (Liu et al. 2025)
destination = snakemake.config["minimized_config"]
source = os.path.join(os.path.dirname(pyspi.__file__), "config.yaml")
shutil.copyfile(source, destination)

# load timeseries
subject_ts = pd.read_csv(snakemake.input.vep_timeseries, index_col=0)
# subject_ts.rename(columns={subject_ts.columns[0]: 'zone_label'}, inplace=True)
print(f"✓ Loaded timeseries with shape {subject_ts.shape}")

# drop zone label column for pyspi input
# subject_ts = subject_ts.drop(columns=['zone_label'])
# print(f"Timeseries shape after filtering: {subject_ts.shape}")

# check for NaN 
if subject_ts.isnull().values.any():
    raise ValueError("Input timeseries contains NaN values. Preprocess data to remove or impute NaNs before proceeding")

# initialise calculator
print("Starting PySpi...")
calc = Calculator(configfile=snakemake.config["minimized_config"], 
                  dataset=subject_ts, normalise=False, detrend=False # already preprocessed
                  )

# compute calculator and capture terminal 
log_capture = io.StringIO()
sys_stdout = sys.stdout
sys.stderr.flush()
sys.stdout.flush()
sys.stdout = log_capture
try:
    calc.compute()
finally:
    sys.stdout = sys_stdout
    
# save calculator output
calc.table.to_pickle(snakemake.output.spi_matrix)

# save terminal output
with open(snakemake.output.log_summary, "w") as logf:
    logf.write(log_capture.getvalue())

# touch snakemake log file for checkpoint
with open(snakemake.output.done, "w") as f:
    f.write("done\n")