#!/usr/bin/env python3
"""
NEB Postprocessing — Interactive Example
=========================================
Run this file directly:
    python postprocess_example.py

Or open it as a notebook in VS Code / Jupyter (it uses # %% cell markers).

For automated/batch use, prefer the CLI:
    nebpost --base-path results/ --file-pattern "C10-ZnN4C*"
    nebpost --help
"""

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from aseneb.postprocess import (
    find_neb_folders,
    load_traj,
    extract_final_path,
    get_dft_rel_energies,
    compute_error_metrics,
    plot_structure_grid,
    plot_neb_optimization,
    plot_final_path,
    write_snapshots,
    create_movie,
    parse_system,
)

# %%
# ==============================
# USER SETTINGS — edit these
# ==============================

BASE_PATH = "4-24-26-neb-relaxation-models-individual_v3/"
# BASE_PATH = "."  # use this if running inside the results directory

# Glob pattern to find NEB folders (relative to BASE_PATH)
FILE_PATTERN = "C10-ZnN4C+FeBr1Cl1F1_c_TO_C10-FeN4C+ZnBr1Cl1F1_c"

# Replace NEB endpoint energies with MACE-optimised values from
# initial_opt.traj / final_opt.traj (if available)
MACEOPT_ENDPOINTS = True

N_IMAGES = 15
STEP_INTERVAL = 1       # informational; passed through to load functions
MAX_STEPS = 2000        # steps beyond this → marked as not converged

# Use simple (max image) barrier instead of spline-interpolated
SIMPLE_BARRIER = False

# --- Output toggles ---
PLOT_NEB_OPTIMIZATION = True   # convergence plot per run
PLOT_NEB_FINAL_PATH   = True   # final path plot
PLOT_DFT_FINAL_PATH   = True   # overlay DFT reference on path plot
PLOT_STRUCTURE_GRIDS  = False  # initial/final structure grids
WRITE_SNAPSHOTS       = False  # per-step snapshot files
WRITE_MOVIE           = True
WRITE_SUMMARY         = True

plt.rcParams.update({"font.size": 14})


# %% ============================== MAIN ==============================

folders = find_neb_folders(BASE_PATH, FILE_PATTERN)

results = []

for folder in folders:

    print(f"\nProcessing {folder}")

    traj = load_traj(folder, maceopt_endpoints=MACEOPT_ENDPOINTS, n_images=N_IMAGES)
    if traj is None:
        continue

    final_images, rel_energies, barrier, delta_E, n_steps = extract_final_path(
        traj, N_IMAGES, simple_barrier=SIMPLE_BARRIER
    )

    converged = (n_steps - 1) < MAX_STEPS

    print(f"n_steps:   {n_steps - 1}")
    print(f"converged: {converged}")

    # --- DFT comparison ---
    rel_dft = get_dft_rel_energies(folder)
    rmse = avg_bias = None
    if rel_dft is not None:
        rmse, avg_bias = compute_error_metrics(rel_energies, rel_dft)

    results.append(
        {
            "system": os.path.basename(folder),
            "barrier_eV": barrier,
            "deltaE_eV": delta_E,
            "rmse_eV": rmse,
            "avg_bias_eV": avg_bias,
            "n_optimization_steps": n_steps - 1,
            "converged": converged,
        }
    )

    if PLOT_STRUCTURE_GRIDS:
        plot_structure_grid(folder, traj[:N_IMAGES], "Initial", "initial.png")
        plot_structure_grid(folder, final_images, "Final", "final.png")

    if PLOT_NEB_OPTIMIZATION:
        plot_neb_optimization(folder, traj, N_IMAGES, barrier, delta_E, MAX_STEPS)

    if PLOT_NEB_FINAL_PATH:
        plot_final_path(
            folder,
            rel_energies,
            barrier,
            delta_E,
            rmse,
            avg_bias,
            converged=converged,
            plot_dft=PLOT_DFT_FINAL_PATH,
        )

    if WRITE_SNAPSHOTS:
        write_snapshots(folder, traj, n_steps, N_IMAGES)

    if WRITE_MOVIE:
        create_movie(folder, n_steps, N_IMAGES)

print("\nDone.")


# %% ============================== SUMMARY TABLES ==============================

if WRITE_SUMMARY and len(results) > 0:

    print(f"Writing summary for {len(results)} systems...")

    df = pd.DataFrame(results)
    df = df.sort_values("barrier_eV", ascending=True, na_position="last").reset_index(drop=True)

    barrier_tag = "simple_barriers" if SIMPLE_BARRIER else "complex_barriers"

    # Raw (flat) summary
    raw_path = os.path.join(BASE_PATH, f"neb_summary_{barrier_tag}.xlsx")
    df.to_excel(raw_path, index=False)
    print(f"Summary written to {raw_path}")

    # Add parsed system tags
    new_cols = ["rxn", "scaffold", "orient", "rxn_type", "halide_type", "halide_ids"]
    df[new_cols] = df["system"].apply(parse_system)

    sys_i = df.columns.get_loc("system") + 1
    ordered = (
        df.columns[:sys_i].tolist()
        + new_cols
        + df.columns.difference(new_cols, sort=False)[sys_i:].tolist()
    )
    df = (
        df[ordered]
        .sort_values("barrier_eV", ascending=True, na_position="last")
        .reset_index(drop=True)
    )

    # Averaged per reaction
    df_avg = (
        df.groupby("rxn", as_index=False)
        .agg({
            "scaffold": "first",
            "rxn_type": "first",
            "halide_type": "first",
            "halide_ids": "first",
            "barrier_eV": "mean",
            "deltaE_eV": "mean",
        })
        .sort_values("barrier_eV", ascending=True, na_position="last")
        .reset_index(drop=True)
    )

    # Best barrier / deltaE per reaction
    df_min_barrier = (
        df.loc[df.groupby("rxn")["barrier_eV"].idxmin()]
        .sort_values("barrier_eV", ascending=True, na_position="last")
        .reset_index(drop=True)
    )
    df_min_deltaE = (
        df.loc[df.groupby("rxn")["deltaE_eV"].idxmin()]
        .sort_values("deltaE_eV", ascending=True, na_position="last")
        .reset_index(drop=True)
    )

    processed_path = os.path.join(BASE_PATH, f"neb_summary_{barrier_tag}_processed.xlsx")
    with pd.ExcelWriter(processed_path) as writer:
        df.to_excel(writer, sheet_name="raw", index=False)
        df_avg.to_excel(writer, sheet_name="averaged", index=False)
        df_min_barrier.to_excel(writer, sheet_name="min_barriers", index=False)
        df_min_deltaE.to_excel(writer, sheet_name="min_deltaEs", index=False)

    print(f"Processed summary written to {processed_path}")
