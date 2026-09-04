# NEB Test Run

Example run directory demonstrating the aseneb LangGraph workflow on Kestrel.

Runs a full NEB pipeline (interpolation → MACE NEB → DFT single points) using
SLURM for GPU steps. DFT is currently a placeholder (MACE + 1% noise).

---

## One-time setup

### 1. Clone the required repos

```bash
# aseflow (if you don't have it yet)
git clone <aseflow-repo-url> /your/path/to/aseflow

# AdaptiveComputing — clone anywhere you like
git clone <AC-repo-url> /your/path/to/AdaptiveComputing
```

AdaptiveComputing manages the SLURM job queue. It doesn't need to be pip-installed.
The workflow finds it via the `AC_PATH` environment variable (set this in your
`.bashrc` or at the start of each session):

```bash
export AC_PATH=/your/path/to/AdaptiveComputing
```

If `AC_PATH` is not set, the workflow falls back to `~/AdaptiveComputing`.

### 2. Create the Python environment

Create a dedicated venv inside the repo. Must be done from **kl5**:

```bash
ssh kl5
ml pytorch/2.12.0
python3 -m venv /your/path/to/aseflow/.venv --system-site-packages
source /your/path/to/aseflow/.venv/bin/activate
pip install mace-torch ase langgraph langgraph-checkpoint-sqlite
pip install -e /your/path/to/aseflow
```

The venv is gitignored — each user creates their own at `aseflow/.venv/`.

---

## Running the workflow

### Every time you start a new session

```bash
ssh kl5
ml pytorch/2.12.0
source /your/path/to/aseflow/.venv/bin/activate
export AC_PATH=/your/path/to/AdaptiveComputing   # skip if already in ~/.bashrc
cd /your/path/to/aseflow/test-run
```

> **Why kl5?** The workflow starts a background manager process (tmux session)
> that handles SLURM job submission. It must run on a fixed login node — always
> use kl5.

### Clean up from a previous run (optional)

```bash
rm -rf 01 02 03 04 05 06 07 08 09 10 11
rm -f neb_hero_db.json neb_checkpoint.db manager.log
```

### Run

```bash
python ../aseneb/langgraph_workflow.py
```

The script runs the full pipeline: interpolation → MACE NEB → DFT single points.

### If the process is interrupted

Just rerun the same command. The workflow checkpoints its state after each step,
so it will resume where it left off without re-submitting completed SLURM jobs.

---

## What to expect

The workflow prints progress as it goes:

```
[interpolate] Wrote 11 images to ...
[submit_mace_neb] Queued task abc123...
[poll_mace_neb] Waiting... (state=ready)
[poll_mace_neb] MACE NEB completed
[NEB] Wrote CONTCAR into 11 directories
[DFT] Copied inputs into 11 directories
[submit_sp] Queued 11 DFT single-point tasks
[poll_sp] 11 task(s) still running...
[DFT] All single-point calculations completed.
sp_results: [-30033, -29765, ...]
```

**In `squeue -u $USER`** you will see:

| Step | Job name | Count | Partition | Duration |
|---|---|---|---|---|
| MACE NEB | `mlip_neb` | 1 | `gpu-h100` | ~30–45 s |
| DFT single points | `dft-sp-placeholder` | 11 | `gpu-h100` | ~1–2 min |

Total wall time: roughly **3–5 minutes**.

---

## Files in this directory

| File | Purpose |
|---|---|
| `initial.vasp`, `final.vasp` | Input endpoint structures |
| `mace.yaml` | MACE NEB settings |
| `mace-fenc-finetuned-mh1.model` | Fine-tuned MACE model |
| `neb.sbatch` | SLURM script for MACE NEB (called automatically) |
| `dft_inputs_placeholder/` | Placeholder DFT inputs (MACE + 1% noise) |

Files created at runtime (safe to delete):

| File | Created by |
|---|---|
| `01/`–`11/` | Interpolation |
| `neb.traj`, `neb.log` | MACE NEB |
| `neb_hero_db.json` | SLURM job queue (managed automatically) |
| `neb_checkpoint.db` | Workflow checkpoint (enables resume) |
| `manager.log` | Background manager log |
