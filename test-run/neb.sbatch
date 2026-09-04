#!/bin/bash
#SBATCH --account=newbridge
#SBATCH --nodes=1
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --mem=85G
#SBATCH -J mlip_neb
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err

# Usage: sbatch --chdir <workdir> neb.sbatch <task_id>
# Runs MACE NEB only (images must already exist from a prior interpolation step).
# Writes result_<task_id>.txt on success for the Hero queue manager to detect.

TASK_ID=$1

ml pytorch/2.12.0
unset PYTHONNOUSERSITE
source "$(dirname "$0")/../.venv/bin/activate"

aseneb --neb-config mace.yaml --run-neb mace

echo "0" > result_${TASK_ID}.txt
