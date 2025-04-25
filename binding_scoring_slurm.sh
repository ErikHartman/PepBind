#!/bin/bash

### SLURM PARAMETERS ###

#SBATCH --account=er8813ha
#SBATCH --job-name="binding_scoring"
#SBATCH --output="binding_scoring.out"
#SBATCH --error="binding_scoring.err"
#SBATCH --time="72:00:00"
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --nodelist=alap759
#SBATCH --gpus-per-task=2


### RUN COMMAND ###
# Run the complete benchmark pipeline with all steps

/srv/data1/general/immunopeptides_data/erik-venv/bin/python \
    /home/er8813ha/immunopeptides/src/binding_score_function/main.py --download-pdbs --process --dock --score --decoys