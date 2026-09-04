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

# NOTE: MUST BE ON A RHEL 9 LOG IN NODE ($ssh kl5)

ml pytorch/2.12.0
unset PYTHONNOUSERSITE

source "$(dirname "$0")/../.venv/bin/activate"

aseneb --interpolate 11 \
       --maceopt \
       --neb-config mace.yaml \
       --run-neb mace 
