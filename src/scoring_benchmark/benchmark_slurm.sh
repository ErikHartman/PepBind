#!/bin/bash

### EXAMPLE SLURM SCRIPT ###

### SLURM PARAMETERS ###

#SBATCH --account=er8813ha
#SBATCH --job-name="benchmark_embeddings"
#SBATCH --output="benchmark_embeddings.out"
#SBATCH --error="benchmark_embeddings.err"
#SBATCH --time="72:00:00"
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --nodelist=alap759
#SBATCH --gpus-per-task=3

### RUN COMMAND ###

/srv/data1/general/immunopeptides_data/erik-venv/bin/python /home/er8813ha/immunopeptides/src/scoring_benchmark/run_scoring_benchmark.py