#!/bin/bash

#SBATCH --job-name=stitch
#SBATCH --time=00:30:00
#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=8
#SBATCH --mem-per-gpu=25G

if [[ "$SLURM_JOB_NUM_NODES" -ne 1 && "$SLURM_JOB_NUM_NODES" -ne "$SLURM_NTASKS" ]]; then
    echo "Error: unfortunately, --ntasks ($SLURM_NTASKS) and --nodes ($SLURM_JOB_NUM_NODES) must be equal."
    exit 1
fi

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
nodes=$SLURM_NTASKS
nproc=$SLURM_GPUS_PER_TASK
CMD="src/stitch_train.py ${*:2}"
echo "Launching command with $nodes nodes, $nproc GPUs per node, head node $head_node:"
echo "    $CMD"

srun torchrun \
     --nnodes $nodes \
     --nproc-per-node $nproc \
     --rdzv_id $RANDOM \
     --rdzv_backend c10d \
     --rdzv_endpoint $head_node:29500 \
     src/stitch_train.py "${@:2}"
