#!/bin/bash
#SBATCH --account=newbridge
#SBATCH --nodes=1
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH -J dft-sp-placeholder
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err

# =============================================================================
# PLACEHOLDER DFT JOB SCRIPT — NOT A REAL DFT CALCULATION
#
# Usage: sbatch --chdir <image_dir> run_dft.sh <task_id> <simulation_dir>
#
# Uses MACE single-point energy/forces with 1% Gaussian noise added.
# Writes result_<task_id>.txt to simulation_dir for the Hero queue manager.
# Replace this script and `in` with real DFT inputs before production use.
# =============================================================================

TASK_ID=$1
SIMULATION_DIR=$2

echo "============================================================"
echo "  WARNING: PLACEHOLDER DFT — MACE + 1% noise, NOT real DFT"
echo "  Directory:      $(pwd)"
echo "  Task ID:        ${TASK_ID}"
echo "  Simulation dir: ${SIMULATION_DIR}"
echo "============================================================"

ml pytorch/2.12.0 2>/dev/null
unset PYTHONNOUSERSITE
source "$(dirname "$0")/../../.venv/bin/activate"

# Pass shell variables into Python via environment so we can keep <<'EOF'
# (single-quoted, no shell expansion inside the heredoc).
export _TASK_ID="${TASK_ID}"
export _SIMULATION_DIR="${SIMULATION_DIR}"

python3 - <<'PYEOF'
import os
import sys
import numpy as np
from pathlib import Path
from ase.io import read
from mace.calculators import MACECalculator

task_id       = os.environ["_TASK_ID"]
simulation_dir = os.environ["_SIMULATION_DIR"]

print("[PLACEHOLDER] Running MACE single-point (stand-in for DFT)")

# Locate the .model file in the image dir or its parent (the NEB workdir).
cwd = Path.cwd()
model_path = None
for search_dir in [cwd, cwd.parent]:
    candidates = list(search_dir.glob("*.model"))
    if candidates:
        model_path = str(candidates[0])
        break

if not model_path:
    print("ERROR: No .model file found in image dir or parent dir", file=sys.stderr)
    sys.exit(1)

print(f"[PLACEHOLDER] Model: {model_path}")

atoms = read("CONTCAR", format="vasp")
calc  = MACECalculator(model_paths=model_path, device="cpu")
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

rng           = np.random.default_rng()
energy_noisy  = energy * (1.0 + rng.normal(0, 0.01))
forces_noisy  = forces * (1.0 + rng.normal(0, 0.01, size=forces.shape))

print(f"[PLACEHOLDER] MACE energy:  {energy:.6f} eV")
print(f"[PLACEHOLDER] Noisy energy: {energy_noisy:.6f} eV  (1% Gaussian noise)")

Path("energy.txt").write_text(
    f"# PLACEHOLDER: MACE + 1% noise (NOT real DFT)\n"
    f"energy_eV = {energy_noisy:.10f}\n"
    f"mace_energy_eV = {energy:.10f}\n"
)
np.savetxt(
    "forces.txt",
    forces_noisy,
    header="PLACEHOLDER: MACE + 1% noise (NOT real DFT)\nfx fy fz (eV/Ang)",
)

# Write result file for the Hero queue manager.
result_path = Path(simulation_dir) / f"result_{task_id}.txt"
result_path.write_text(f"{energy_noisy:.10f}\n")

print(f"[PLACEHOLDER] Wrote energy.txt, forces.txt, and {result_path}")
print("[PLACEHOLDER] Done — replace this script with real DFT before production use")
PYEOF
