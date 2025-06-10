#!/bin/bash

# To launch, run "sbatch train.sbatch stitch src/stitch_train.py -c tests/stitch-resnet18.yml" or similar.
# Ensure the job's batch size is configured appropriately for 2 GPU, or you change the number of GPU accordingly.

#SBATCH --job-name=stitch
#SBATCH --time=10:00:00
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

# Find a free port in [29500, 30500).
while :; do
  PORT=$((29500 + RANDOM % 1000))
  ! lsof -iTCP:$PORT -sTCP:LISTEN -t >/dev/null && break
done
endpoint=$(hostname):$PORT
rdzv_id=$RANDOM
nodes=$SLURM_NTASKS
nproc=$SLURM_GPUS_ON_NODE
ncpu=$SLURM_CPUS_PER_TASK
nworkers=$((ncpu / nproc))
if [ $nworkers -lt 5 ]; then
    # If number of CPU is low, add a couple extra workers to ensure good parallelism.
    nworkers=$((nworkers + 2))
fi
progname=$2
cmd="$progname -j $nworkers ${*:3}"
echo "Launching command with ($nodes nodes, $nproc GPUs, $ncpu CPUs) @ endpoint $endpoint, ID $rdzv_id:"
echo "    $cmd"

srun torchrun \
     --nnodes $nodes \
     --nproc-per-node $nproc \
     --rdzv_id $rdzv_id \
     --rdzv_backend c10d \
     --rdzv_endpoint $endpoint \
     $progname -j $nworkers "${@:3}"
