"""
slurm_manager.py — NEB-specific LocalHPCManager daemon for SLURM integration.

Subclasses AdaptiveComputing's LocalHPCManager to handle two job types:
  - "mace_neb": submits neb.sbatch in the NEB workdir (GPU job, ~30 min)
  - "dft_sp":   submits run_dft.sh in each image directory (DFT single point)

Designed to run as a persistent daemon in a tmux session via:
    python -m aseneb.slurm_manager <work_dir> <machine_name>

The LangGraph controller (langgraph_workflow.py) adds tasks to the shared
LocalHeroClient JSON queue; this daemon picks them up, submits sbatch jobs,
polls sacct, and marks tasks done/error — independently of the controller.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make AdaptiveComputing importable whether installed via pip or from source.
# Resolution order: AC_PATH env var → ~/AdaptiveComputing → already on sys.path.
_ac_env = os.environ.get("AC_PATH")
_AC_PATH = Path(_ac_env) if _ac_env else Path.home() / "AdaptiveComputing"
if _AC_PATH.is_dir() and str(_AC_PATH) not in sys.path:
    sys.path.insert(0, str(_AC_PATH))

from adaptive_computing.hpc.local_manager import LocalHPCManager
from adaptive_computing.hpc.manager_base import TaskError
from adaptive_computing.local_hero import LocalHeroClient

MANAGER_SCRIPT = Path(__file__).resolve()
QUEUE_NAME = "neb"
APP_ID = "aseneb"


class NEBHPCManager(LocalHPCManager):
    """SLURM manager for NEB workflows.

    Task metadata fields:
        job_type (str):  ``"mace_neb"`` or ``"dft_sp"``
        workdir  (str):  Absolute path to the NEB run directory (both types)
        script   (str):  Batch script name relative to workdir (mace_neb only,
                         default ``"neb.sbatch"``)
        image_dir (str): Absolute path to one image directory (dft_sp only)
    """

    def submit_job(self, task: dict, machine_name: str, i_fidelity: int) -> str:
        meta = task["metadata"]
        job_type = meta.get("job_type")
        task_id = task["id"]
        workdir = meta.get("workdir", str(self.simulation_dir or "."))

        if job_type == "mace_neb":
            script = meta.get("script", "neb.sbatch")
            cmd = f"sbatch --chdir {workdir!r} {script} {task_id}"

        elif job_type == "dft_sp":
            image_dir = meta["image_dir"]
            cmd = (
                f"sbatch --chdir {image_dir!r} run_dft.sh "
                f"{task_id} {workdir!r}"
            )

        else:
            raise TaskError(f"Unknown job_type: {job_type!r}")

        return self._run_submit(cmd)

    def read_result(self, task_id: str) -> str:
        """Read result_{task_id}.txt from simulation_dir."""
        result_file = Path(str(self.simulation_dir or ".")) / f"result_{task_id}.txt"
        if result_file.exists():
            value = result_file.read_text().strip()
            result_file.unlink()
            return value
        return "-1"


def create_manager(
    work_dir: str,
    machine_name: str,
    hero_client: LocalHeroClient,
) -> NEBHPCManager:
    """Return a configured :class:`NEBHPCManager`.

    Args:
        work_dir:     Absolute path to the NEB run directory.
        machine_name: Logical machine name (e.g. ``"kestrel"``).
        hero_client:  Shared :class:`LocalHeroClient` instance.
    """
    return NEBHPCManager(
        machine_name=machine_name,
        batch_scripts=["neb.sbatch"],
        scheduler_type="slurm",
        simulation_dir=work_dir,
        poll_interval=30,
        hero_client=hero_client,
    )


if __name__ == "__main__":
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    machine_name = sys.argv[2] if len(sys.argv) > 2 else "kestrel"

    work_dir = str(Path(work_dir).resolve())
    hero_client = LocalHeroClient(
        db_path=str(Path(work_dir) / "neb_hero_db.json"),
        queue_name=QUEUE_NAME,
        application_id=APP_ID,
    )
    manager = create_manager(
        work_dir=work_dir,
        machine_name=machine_name,
        hero_client=hero_client,
    )
    print(f"[slurm_manager] Starting NEB manager daemon in {work_dir}")
    manager.run_forever()
