"""LangGraph wiring for the aseneb NEB workflow.

Each README step is a node so callers can run the full pipeline or any subset:

    1. interpolate         - build N intermediate images from initial/final
    2. run_mace_neb        - MACE NEB via SLURM (submits neb.sbatch, polls Hero queue)
    3. read_final_images   - pull the final band out of neb.traj
    4. write_contcars      - write CONTCAR into each NN image directory
    5. copy_dft_inputs     - copy `in` and `run_dft.sh` into each image dir
    6. run_single_points   - DFT single points via SLURM (one job per image dir)

SLURM integration:
    Steps 2 and 6 submit sbatch jobs through AdaptiveComputing's
    LocalHeroClient queue.  A persistent tmux manager (aseneb/slurm_manager.py)
    runs independently, handling submission → sacct polling → result collection.
    LangGraph state is checkpointed to SQLite so the workflow can be interrupted
    and resumed without re-submitting already-running jobs.

Requires `langgraph`, `langgraph-checkpoint-sqlite`, and `adaptive_computing`
(pip install -e ~/AdaptiveComputing).
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import List, Optional, TypedDict

from ase.io import read, write
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

# Make AdaptiveComputing importable whether installed via pip or from source.
# Resolution order: AC_PATH env var → ~/AdaptiveComputing → already on sys.path.
_ac_env = os.environ.get("AC_PATH")
_AC_PATH = Path(_ac_env) if _ac_env else Path.home() / "AdaptiveComputing"
if _AC_PATH.is_dir() and str(_AC_PATH) not in sys.path:
    sys.path.insert(0, str(_AC_PATH))

from adaptive_computing.local_hero import LocalHeroClient
from adaptive_computing.hpc.local_launcher import ensure_manager_running

from aseneb.config_io import load_config
from aseneb.slurm_manager import MANAGER_SCRIPT, APP_ID
from aseneb.workflow import Workflow

QUEUE_NAME = "neb"
SESSION_NAME = "neb-manager"
MANAGER_HOST = "kl5"       # login node where the tmux manager session lives
SP_POLL_INTERVAL = 60   # seconds between DFT single-point polls
NEB_POLL_INTERVAL = 60  # seconds between MACE NEB polls


class NEBState(TypedDict, total=False):
    # --- inputs ---
    neb_config_path: Optional[str]
    dft_inputs: Optional[str]
    workdir: str
    initial: str
    final: str
    n_images: Optional[int]
    mace_optimize: bool
    do_interpolate: bool
    do_mace_neb: bool
    do_final_sp: bool
    machine_name: str

    # --- SLURM / Hero ---
    hero_db_path: str          # path to neb_hero_db.json
    mace_neb_task_id: Optional[str]
    sp_task_ids: Optional[List[str]]

    # --- runtime ---
    image_dirs: Optional[List[str]]   # str paths, serializable by checkpointer
    sp_results: Optional[List[float]]


REQUIRED_DFT_FILES = ("in", "run_dft.sh")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_image_dirs(workdir: Path) -> List[Path]:
    folders = [
        d for d in workdir.iterdir()
        if d.is_dir()
        and re.fullmatch(r"\d{2}", d.name)
        and (d / "POSCAR").exists()
    ]
    if not folders:
        raise RuntimeError(f"No image directories found in {workdir}")
    folders.sort(key=lambda d: int(d.name))
    return folders


def _hero_engine(state: NEBState):
    """Return (hero_client, engine, queue) for the NEB queue."""
    hero_client = LocalHeroClient(db_path=state["hero_db_path"])
    engine = hero_client.TaskEngine(application_id=APP_ID)
    try:
        queue = engine.read_queue_by_name(QUEUE_NAME, state="active")
    except (ValueError, KeyError):
        queue = engine.add_queue(QUEUE_NAME)
    return hero_client, engine, queue


def _ensure_neb_manager(state: NEBState) -> None:
    """Start the NEB tmux manager if not already running."""
    workdir = str(Path(state.get("workdir", ".")).resolve())
    ensure_manager_running(
        work_dir=workdir,
        manager_script=str(MANAGER_SCRIPT),
        machine_name=state.get("machine_name", "kestrel"),
        session_name=SESSION_NAME,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def init_node(state: NEBState) -> NEBState:
    """Resolve path defaults."""
    workdir = str(Path(state.get("workdir", ".")).resolve())
    updates: dict = {"workdir": workdir}
    if not state.get("hero_db_path"):
        updates["hero_db_path"] = str(Path(workdir) / "neb_hero_db.json")
    if not state.get("machine_name"):
        updates["machine_name"] = "kestrel"
    return updates


def interpolate_node(state: NEBState) -> NEBState:
    """Step 1: interpolate N images between initial.vasp and final.vasp."""
    n = state.get("n_images")
    if n is None:
        raise ValueError("interpolate_node requires n_images")
    neb_cfg = load_config(state["neb_config_path"]) if state.get("neb_config_path") else None
    workflow = Workflow(neb_cfg=neb_cfg)
    workflow.interpolate(
        n_images=n,
        initial=state.get("initial", "initial.vasp"),
        final=state.get("final", "final.vasp"),
        mace_optimize=state.get("mace_optimize", False),
    )
    return {}


def submit_mace_neb_node(state: NEBState) -> NEBState:
    """Step 2a: submit MACE NEB task to the Hero queue and return immediately.

    Returning here checkpoints mace_neb_task_id so that if the process is
    killed during the subsequent poll node, resume will skip re-submission.
    """
    _ensure_neb_manager(state)
    _, engine, queue = _hero_engine(state)

    if state.get("mace_neb_task_id"):
        print(f"[submit_mace_neb] Already submitted — task {state['mace_neb_task_id']}")
        return {}

    task = engine.add_task(
        queue_id=queue["id"],
        name="mace-neb",
        metatype="Task",
        metadata={
            "job_type": "mace_neb",
            "workdir": state["workdir"],
            "script": "neb.sbatch",
        },
    )
    print(f"[submit_mace_neb] Queued task {task['id']}")
    return {"mace_neb_task_id": task["id"]}


def poll_mace_neb_node(state: NEBState) -> NEBState:
    """Step 2b: poll until the MACE NEB task is done or error."""
    _, engine, _ = _hero_engine(state)
    task_id = state["mace_neb_task_id"]

    while True:
        task = engine.read_task(task_id)
        if task["state"] == "done":
            print("[poll_mace_neb] MACE NEB completed")
            return {}
        if task["state"] == "error":
            raise RuntimeError(f"MACE NEB job failed (task {task_id})")
        print(f"[poll_mace_neb] Waiting... (state={task['state']})")
        time.sleep(NEB_POLL_INTERVAL)


def read_final_images_node(state: NEBState) -> NEBState:
    """Step 3: locate image directories and verify neb.traj exists."""
    workdir = Path(state.get("workdir", "."))
    image_dirs = _find_image_dirs(workdir)

    traj_file = workdir / "neb.traj"
    if not traj_file.exists():
        raise RuntimeError(f"neb.traj not found in {workdir}")

    print(f"[NEB] Found {len(image_dirs)} image dirs; neb.traj exists")
    # Store as strings so the checkpointer can serialize them.
    return {"image_dirs": [str(d) for d in image_dirs]}


def write_contcars_node(state: NEBState) -> NEBState:
    """Step 4: read final band from neb.traj and write CONTCAR into each image directory."""
    image_dirs = [Path(d) for d in state["image_dirs"]]
    n_images = len(image_dirs)

    workdir = Path(state.get("workdir", "."))
    traj = read(workdir / "neb.traj", index=":")
    if len(traj) % n_images != 0:
        raise RuntimeError(
            f"Trajectory length ({len(traj)}) not divisible by n_images ({n_images})"
        )
    final_images = traj[-n_images:]

    for img_dir, atoms in zip(image_dirs, final_images):
        write(img_dir / "CONTCAR", atoms, format="vasp", direct=True)
    print(f"[NEB] Wrote CONTCAR into {n_images} directories")
    return {}


def copy_dft_inputs_node(state: NEBState) -> NEBState:
    """Step 5: copy the DFT input files into each image directory."""
    dft_inputs = state.get("dft_inputs")
    if not dft_inputs:
        raise ValueError("copy_dft_inputs_node requires dft_inputs")
    src_dir = Path(dft_inputs)

    for d in [Path(d) for d in state["image_dirs"]]:
        for filename in REQUIRED_DFT_FILES:
            src = src_dir / filename
            if not src.exists():
                raise FileNotFoundError(f"Missing {filename} in {src_dir}")
            dst = d / filename
            shutil.copy2(src, dst)
            if filename.endswith(".sh"):
                dst.chmod(0o755)
    print(f"[DFT] Copied inputs into {len(state['image_dirs'])} directories")
    return {}


def submit_sp_node(state: NEBState) -> NEBState:
    """Step 6a: submit one DFT single-point task per image dir and return immediately.

    Returning here checkpoints sp_task_ids so that if the process is killed
    during the subsequent poll node, resume will skip re-submission.
    """
    _ensure_neb_manager(state)
    _, engine, queue = _hero_engine(state)

    if state.get("sp_task_ids"):
        print(f"[submit_sp] Already submitted — {len(state['sp_task_ids'])} tasks")
        return {}

    workdir = state["workdir"]
    task_ids = []
    for img_dir in [Path(d) for d in state["image_dirs"]]:
        task = engine.add_task(
            queue_id=queue["id"],
            name=f"dft-sp-{img_dir.name}",
            metatype="Task",
            metadata={
                "job_type": "dft_sp",
                "workdir": workdir,
                "image_dir": str(img_dir),
            },
        )
        task_ids.append(task["id"])
    print(f"[submit_sp] Queued {len(task_ids)} DFT single-point tasks")
    return {"sp_task_ids": task_ids}


def poll_sp_node(state: NEBState) -> NEBState:
    """Step 6b: poll until all DFT single-point tasks are done or error."""
    _, engine, queue = _hero_engine(state)
    task_id_set = set(state["sp_task_ids"])

    while True:
        done = engine.read_tasks(queue["id"], metatype="Task", state="done")
        error = engine.read_tasks(queue["id"], metatype="Task", state="error")
        terminal = {t["id"] for t in done + error}

        if task_id_set <= terminal:
            break

        remaining = len(task_id_set - terminal)
        print(f"[poll_sp] {remaining} task(s) still running...")
        time.sleep(SP_POLL_INTERVAL)

    error_ids = {t["id"] for t in error} & task_id_set
    if error_ids:
        raise RuntimeError(
            f"{len(error_ids)} DFT single-point job(s) failed: {error_ids}"
        )

    done_map = {t["id"]: t["metadata"].get("y_data", [-1])[0] for t in done}
    sp_results = [done_map.get(tid, -1.0) for tid in state["sp_task_ids"]]
    print("[DFT] All single-point calculations completed.")
    return {"sp_results": sp_results}


# ---------------------------------------------------------------------------
# Router helpers
# ---------------------------------------------------------------------------

def _after_init(state: NEBState) -> str:
    if state.get("do_interpolate"):
        return "interpolate"
    if state.get("do_mace_neb"):
        return "submit_mace_neb"
    if state.get("do_final_sp"):
        return "read_final_images"
    return END


def _after_interpolate(state: NEBState) -> str:
    if state.get("do_mace_neb"):
        return "submit_mace_neb"
    if state.get("do_final_sp"):
        return "read_final_images"
    return END


def _after_poll_mace_neb(state: NEBState) -> str:
    if state.get("do_final_sp"):
        return "read_final_images"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Return the uncompiled NEB StateGraph.

    Callers compile it themselves (with or without a checkpointer).
    Use :func:`run` for the standard entrypoint which handles SQLite
    checkpointing automatically.
    """
    g = StateGraph(NEBState)

    g.add_node("init", init_node)
    g.add_node("interpolate", interpolate_node)
    g.add_node("submit_mace_neb", submit_mace_neb_node)
    g.add_node("poll_mace_neb", poll_mace_neb_node)
    g.add_node("read_final_images", read_final_images_node)
    g.add_node("write_contcars", write_contcars_node)
    g.add_node("copy_dft_inputs", copy_dft_inputs_node)
    g.add_node("submit_sp", submit_sp_node)
    g.add_node("poll_sp", poll_sp_node)

    g.add_edge(START, "init")
    g.add_conditional_edges("init", _after_init, {
        "interpolate": "interpolate",
        "submit_mace_neb": "submit_mace_neb",
        "read_final_images": "read_final_images",
        END: END,
    })
    g.add_conditional_edges("interpolate", _after_interpolate, {
        "submit_mace_neb": "submit_mace_neb",
        "read_final_images": "read_final_images",
        END: END,
    })
    # submit always flows to poll; poll decides what comes next
    g.add_edge("submit_mace_neb", "poll_mace_neb")
    g.add_conditional_edges("poll_mace_neb", _after_poll_mace_neb, {
        "read_final_images": "read_final_images",
        END: END,
    })
    g.add_edge("read_final_images", "write_contcars")
    g.add_edge("write_contcars", "copy_dft_inputs")
    g.add_edge("copy_dft_inputs", "submit_sp")
    g.add_edge("submit_sp", "poll_sp")
    g.add_edge("poll_sp", END)

    return g


def run(
    *,
    neb_config_path: Optional[str] = None,
    dft_inputs: Optional[str] = None,
    workdir: str = ".",
    initial: str = "initial.vasp",
    final: str = "final.vasp",
    n_images: Optional[int] = None,
    mace_optimize: bool = False,
    do_interpolate: bool = False,
    do_mace_neb: bool = False,
    do_final_sp: bool = False,
    machine_name: str = "kestrel",
    thread_id: str = "neb-run",
) -> NEBState:
    """Compile the graph and invoke it with the supplied flags.

    Must be called from **kl5** (``ssh kl5`` first).  The tmux manager session
    that handles SLURM job submission lives on kl5; running from any other login
    node would fail to find the existing session and risk launching a duplicate.

    Uses a SQLite checkpointer keyed by *thread_id* so interrupted runs can
    be resumed by calling ``run()`` again with the same *thread_id* and
    *workdir*.

    Examples::

        # Interpolate only
        run(n_images=15, do_interpolate=True, workdir="my_neb/")

        # Interpolate + MACE NEB (submits to SLURM via tmux manager)
        run(neb_config_path="mace.yaml", n_images=15,
            do_interpolate=True, do_mace_neb=True, workdir="my_neb/")

        # Full pipeline
        run(neb_config_path="mace.yaml", dft_inputs="dft_inputs/",
            n_images=15, do_interpolate=True, do_mace_neb=True,
            do_final_sp=True, workdir="my_neb/")
    """
    host = socket.gethostname()
    if (do_mace_neb or do_final_sp) and not host.startswith(MANAGER_HOST):
        raise RuntimeError(
            f"SLURM steps must run from {MANAGER_HOST} (currently on {host!r}). "
            f"Run: ssh {MANAGER_HOST}"
        )

    workdir_abs = str(Path(workdir).resolve())
    checkpoint_db = str(Path(workdir_abs) / "neb_checkpoint.db")

    initial_state: NEBState = {
        "neb_config_path": neb_config_path,
        "dft_inputs": dft_inputs,
        "workdir": workdir_abs,
        "initial": initial,
        "final": final,
        "n_images": n_images,
        "mace_optimize": mace_optimize,
        "do_interpolate": do_interpolate,
        "do_mace_neb": do_mace_neb,
        "do_final_sp": do_final_sp,
        "machine_name": machine_name,
    }

    g = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
        graph = g.compile(checkpointer=checkpointer)
        return graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    # Run from the test-run/ directory:
    #   cd test-run/
    #   python ../aseneb/langgraph_workflow.py
    run(
        neb_config_path="mace.yaml",
        dft_inputs="dft_inputs_placeholder/",
        n_images=11,
        mace_optimize=True,
        do_interpolate=True,
        do_mace_neb=True,
        do_final_sp=True,
        workdir=".",
    )
