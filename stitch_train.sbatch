#!/bin/bash

#SBATCH --job-name=stitch
#SBATCH --time=00:30:00
#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=8
#SBATCH --mem-per-gpu=25G

echo "Running in: $(pwd)"
echo "On node: $(hostname)"
source ~/.bash_profile

# Exit immediately if any command has a non-zero return code.
set -e
set -o pipefail

CONDAENV=$1
echo "Activate $CONDAENV"
conda activate $CONDAENV

head_node=$(hostname)
CMD="src/stitch_train.py ${*:2}"
echo "Launching command: $CMD"

srun torchrun \
     --nnodes $SLURM_NTASKS \
     --nproc-per-node $SLURM_GPUS_PER_TASK \
     --rdzv_id $RANDOM \
     --rdzv_backend c10d \
     --rdzv_endpoint $head_node:29500 \
     src/stitch_train.py "${@:2}"
