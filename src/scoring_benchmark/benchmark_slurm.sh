#!/bin/bash

### EXAMPLE SLURM SCRIPT ###

### SLURM PARAMETERS ###

#SBATCH --account=er8813ha
#SBATCH --job-name="scoring_benchmark"
#SBATCH --output="scoring_benchmark.out"
#SBATCH --error="scoring_benchmark.err"
#SBATCH --time="72:00:00"
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --nodelist=alap759
#SBATCH --gpus-per-task=1

### RUN COMMAND ###
# Add more commands here for the main script if necessary.
/srv/data1/general/immunopeptides_data/erik-venv/bin/python /home/er8813ha/immunopeptides/src/scoring_benchmark/run_scoring_benchmark.py