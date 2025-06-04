#!/bin/bash

# To launch, run "sbatch hgtrain.sbatch stitch src/stitch_train.py -c tests/stitch-resnet18.yml" or similar.
# Ensure the job's batch size is configured appropriately for 2 GPU, or you change the number of GPU accordingly.

#SBATCH --job-name=stitch
#SBATCH --time=03:00:00
#SBATCH --partition=hgnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=2
#SBATCH --mem=200G

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
nproc=$SLURM_GPUS_ON_NODE
progname=$2
cmd="$progname -j 12 ${*:3}"
echo "Launching command with $nodes nodes, $nproc GPUs per node, head node $head_node:"
echo "    $cmd"

srun torchrun \
     --nnodes $nodes \
     --nproc-per-node $nproc \
     --rdzv_id $RANDOM \
     --rdzv_backend c10d \
     --rdzv_endpoint $head_node:29500 \
     $progname -j 12 "${@:3}"
