#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=60G   # memory
#SBATCH -J "fit"   # job name

## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source ~/miniconda3/etc/profile.d/conda.sh

module load gcc/9.2.0

conda activate stan
module load gcc/9.2.0

cd /central/groups/mobbslab/toby/protect_task

python code/model_fit.py