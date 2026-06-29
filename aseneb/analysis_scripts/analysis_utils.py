#!/usr/bin/env python3
import re
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cycler
from ase.io import read, write
from vestapy import Visualizer
from pathlib import Path
from scipy.constants import physical_constants 
from collections import Counter
import ast


import copy
from ase.mep import NEBTools
from ase.utils.forcecurve import fit_images
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from ase.calculators.singlepoint import SinglePointCalculator

Ha2eV = physical_constants['Hartree energy in eV'][0]

# ============================== Core Functions ==============================

def get_energy(img):
    """Return the energy of an ASE Atoms object or np.nan if unavailable."""
    if 'energy' in img.info:
        return img.info['energy']
    if img.calc is not None and 'energy' in img.calc.results:
        return img.calc.results['energy']
    return np.nan

def find_neb_folders(base_path, file_pattern):
    candidates = glob.glob(os.path.join(base_path, file_pattern))
    folders = [p for p in candidates if os.path.isdir(p)]
    return sorted(folders)

def load_traj(folder, maceopt_endpoints, n_images):

    traj_file = os.path.join(folder, "neb.traj")

    if not os.path.exists(traj_file):
        print(f"Trajectory not found: {traj_file}")
        return None

    traj = read(traj_file, index=":")

    if not maceopt_endpoints:
        return traj

    # Replace endpoint energies
    init_file = os.path.join(folder, "initial_opt.traj")
    final_file = os.path.join(folder, "final_opt.traj")

    if not os.path.exists(init_file) or not os.path.exists(final_file):
        print(f"WARNING: Missing endpoint traj in {folder}, using NEB endpoints instead of opt energies.")
        return traj

    img0 = read(init_file, index=-1)
    imgf = read(final_file, index=-1)

    e0 = get_energy(img0)
    ef = get_energy(imgf)

    n_steps = len(traj) // n_images 

    for step in range(n_steps):

        i0 = step * n_images
        iF = step * n_images + (n_images - 1)

        #traj[i0].info["energy"] = e0
        #traj[iF].info["energy"] = ef

        # JMC: force fitting needs results dict populated
        traj[i0].calc.results = copy.deepcopy(img0.calc.results)
        traj[iF].calc.results = copy.deepcopy(imgf.calc.results)

    return traj

def compute_barrier_deltaE(rel_energies, simple_barrier):

    delta_E = rel_energies[-1]

    if simple_barrier:

        max_idx = int(np.nanargmax(rel_energies))

        barrier = rel_energies[max_idx]  # max E relative to initial

    # COMPLEX BARRIER CALCULATION
    else:
        # Initialize
        barrier = 0.0

        current_min = rel_energies[0]

        for i in range(1, len(rel_energies)):
            if rel_energies[i] >= rel_energies[i-1]:
                # Still going uphill (or flat)
                rise = rel_energies[i] - current_min
                if rise > barrier:
                    barrier = rise
            else:
                # Slope breaks → start new segment
                current_min = rel_energies[i]    

    return barrier, delta_E   

def extract_final_path(traj, n_images, simple_barrier):

    n_steps = (len(traj) // n_images)
    final_images = traj[(n_steps-1) * n_images: n_steps * n_images]
    energies = np.array([get_energy(img) for img in final_images])

    # Calculate relative energies
    ref = energies[0]
    rel_energies = energies - ref

    barrier, delta_E = compute_barrier_deltaE(rel_energies, simple_barrier)

    return (final_images,rel_energies,barrier,delta_E,n_steps)

def get_dft_rel_energies(folder):
    csv_path = os.path.join(folder, "energies.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    dft = df["F"].to_numpy()
    return (dft - dft[0]) * Ha2eV

def get_jdftx_data(folder,n_images):
#JMC: there's probably a cleaner way of reading through all of the subdirs
    """Collect JDFTx out file data into list of atoms objects using Pymatgen parser"""
    all_converged = True
    atoms_jdftx = []
    for i in range(1, n_images+1):
        d = f'{i:02}'
        filename = f'{folder}/{d}/out'
        out = JDFTXOutfile.from_file(filename)
        atoms = AseAtomsAdaptor.get_atoms(out.structure)        
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=out.e,
            forces=out.forces)
        atoms_jdftx.append(atoms)
        if not out.converged:
            all_converged = False
    return atoms_jdftx, all_converged

    return atoms_jdftx, all_converged

def compute_error_metrics(rel_ml, rel_dft):
    n = min(len(rel_ml), len(rel_dft))
    ml = rel_ml[:n]
    dft = rel_dft[:n]

    diff = ml - dft

    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    avg_bias = np.mean(diff)

    return rmse, avg_bias

# ============================== PARSING AND NAMING ==============================

def get_reaction_name(folder):

    folder_name = os.path.basename(folder)

    if "_TO_" not in folder_name:
        return folder_name

    reactant_raw, product_raw = folder_name.split("_TO_", 1)

    tag_re = re.compile(r"^(?P<body>.*)_(?P<tag>[A-Za-z]+)$")

    def split_tag(s: str):
        m = tag_re.match(s)
        if not m:
            return s, None
        return m.group("body"), m.group("tag")

    def subscript_numbers_except_C10_C12(name: str) -> str:
        """
        Subscript all numbers except when part of C10 or C12.
        Example:
            ZnN4C+FeBr3 -> ZnN$_4$C+FeBr$_3$
            C10-ZnN4C   -> C10-ZnN$_4$C
        """

        def repl(match):
            full = match.group(0)

            # skip C10 / C12
            if full in ("C10", "C12"):
                return full

            letter = match.group(1)
            number = match.group(2)

            return f"{letter}$_{number}$"

        return re.sub(r"([A-Za-z])(\d+)", repl, name)

    reactant, tag_r = split_tag(reactant_raw)
    product, tag_p = split_tag(product_raw)

    tag = tag_p or tag_r
    tag_txt = f" [{tag}]" if tag else ""

    # apply subscripting
    reactant = subscript_numbers_except_C10_C12(reactant)
    product = subscript_numbers_except_C10_C12(product)

    return rf"{reactant} $\rightarrow$ {product}{tag_txt}"

def split_orientation(name):
    m = re.match(r"(.+)_([a-z]+)$", name)
    if m:
        return m.group(1), m.group(2)
    return name, None

def get_scaffold(name):
    m = re.search(r"^(C\d+)-", name)
    return m.group(1) if m else None

def parse_stoichiometry(name):
    body = re.sub(r"^C\d+-", "", name)
    body = body.replace("+", "")

    matches = re.findall(r"([A-Z][a-z]?)(\d*)", body)

    counts = Counter()
    for el, n in matches:
        counts[el] += int(n) if n else 1

    return counts

def strip_config(name):
    # removes trailing _d, _dr, _r, _c
    return re.sub(r"_([a-z]+)$", "", name)

def make_rxn(system):
    left, right = system.split("_TO_")
    left_clean = strip_config(left)
    right_clean = strip_config(right)
    return f"{left_clean}_TO_{right_clean}"

def parse_system(system):
    left, right = system.split("_TO_")

    # --- config-aware parsing ---
    base_left, orient = split_orientation(left)

    scaffold = get_scaffold(base_left)
    stoch = parse_stoichiometry(base_left)

    # Reaction type
    rxn_type = "exchange" if "ZnN4C" in base_left else "deposition"

    # Halide classification
    halide_ids = [el for el in stoch if el in ["Cl", "Br", "F", "I"]]
    halide_type = "single" if len(halide_ids) == 1 else "mixed"

    # --- config-independent rxn ---
    left_clean = strip_config(left)
    right_clean = strip_config(right)
    rxn = f"{left_clean}_TO_{right_clean}"

    return pd.Series([rxn, scaffold, orient, rxn_type, halide_type, halide_ids])

# ============================== NEB PLOTTING ==============================

def plot_optimization(folder, traj, n_images, barrier, delta_E, max_steps, step_interval):

    name = get_reaction_name(folder)

    n_steps = len(traj) // n_images

    # --- Reference energy from final NEB path  ---
    lasttraj = [get_energy(img) for img in traj[-n_images:]]
    E_ref = lasttraj[0]

    steps_to_plot = list(range(0, n_steps, step_interval))

    # Always include the final step (even if it doesn't align with the interval)
    if (n_steps - 1) not in steps_to_plot:
        steps_to_plot.append(n_steps - 1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_to_plot)))
    plt.rc("axes", prop_cycle=cycler(color=colors))

    fig, ax = plt.subplots(figsize=(8, 4))

    for c, step in zip(colors, steps_to_plot):

        energies = [get_energy(traj[step * n_images + i]) for i in range(n_images)]

        ref_energies = np.subtract(energies, E_ref)

        ax.plot(np.arange(1, n_images + 1), ref_energies, color=c)

    if n_steps - 1 > max_steps:
        ax.plot(np.arange(1, n_images + 1), ref_energies, color='r')

    ax.set_xlabel("Reaction Coordinate")
    ax.set_ylabel("Potential Energy (eV)")
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f'NEB: {name}', fontsize=13)

    ax.text(
        0.02,
        0.95,
        f"Barrier: {barrier:.2f} eV\nΔE: {delta_E:.2f} eV",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.3", alpha=0.8),
    )

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=n_steps))

    fig.colorbar(sm, ax=ax, label="NEB optimization step")

    plt.tight_layout()
    plt.savefig(f"{folder}.png", dpi=300)
    plt.close()

def plot_final_path(folder, rel_energies, barrier, delta_E, rmse, bias, converged, plot_dft=False):

    c = 'r' if not converged else plt.cm.viridis(1.0) 

    name = get_reaction_name(folder)

    fig, ax = plt.subplots(figsize=(7, 4))

    xvals = np.arange(1, len(rel_energies) + 1)
    ax.plot(xvals, rel_energies, marker="o", color=c, label="NEB path")

    for x, y in zip(xvals, rel_energies):
        ax.text(x, y + 0.015, f"{y:.2f}", ha="center", fontsize=10)

    energies_for_limits = list(rel_energies)

    # Optionally overlay DFT energies
    if plot_dft:
        csv_path = os.path.join(folder, "energies.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # F energies relative to first entry
            dft_energies = df["F"].to_numpy()
            rel_dft = (dft_energies - dft_energies[0]) * Ha2eV
            energies_for_limits.extend(rel_dft)

            # x-axis: evenly spaced points (1-based to match NEB)
            x_dft = np.arange(1, len(rel_dft) + 1)

            # Color red if IM column is False, black otherwise
            # Adjust column name here if needed ('I' or 'IonicSteps')
            
            # Optional connecting line
            ax.plot(x_dft, rel_dft, linestyle="-", marker="o", color="#9fb6d5", label="DFT energies")

            # Plot DFT points
            im_series = df["IM"] if "IM" in df.columns else pd.Series([True] * len(df))
            im_series = im_series.fillna(True).astype(bool)

            colors = ["#9fb6d5" if val else "red" for val in im_series]
            for xi, yi, col in zip(x_dft, rel_dft, colors):
                ax.scatter(xi, yi, color=col, s=36)
                ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", fontsize=10, color="k")

            ax.text(0.8,0.95,
                f"rmse: {rmse:.2f} eV\nbias: {bias:.2f} eV",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.3", alpha=0.8),
                fontsize=12,
            )

        else:
            print(f"Warning: {csv_path} not found, skipping DFT overlay.")

    ax.set_xlabel("Reaction Coordinate")
    ax.set_ylabel("Relative Energy (eV)")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title(f"NEB: {name}", fontsize=13)

    # Extend y-axis max slightly to make room for point labels
    y_min, y_max = np.nanmin(energies_for_limits), np.nanmax(energies_for_limits)
    y_range = y_max - y_min
    pad_top = 0.10 * y_range if y_range > 0 else 0.2
    ax.set_ylim(bottom=ax.get_ylim()[0], top=y_max + pad_top)

    ax.text(
        0.02,
        0.95,
        f"Barrier: {barrier:.2f} eV\nΔE: {delta_E:.2f} eV",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.3", alpha=0.8),
    )

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{folder}_finalpath.png", dpi=300)
    plt.close()

    ''' SB Numerical Stuff and Saving '''
    # Construct the array to be saved
    folder_name = os.path.basename(folder.rstrip("/"))
    # Flatten to one row so the folder label can sit beside the data
    relative_diffs = np.concatenate([rel_energies, rel_dft], axis=0)
    # Find n_images
    n_images = len(rel_energies)

    # Compute forward numerical gradients
    grad_mace = rel_energies[1:] - rel_energies[:-1]
    grad_dft = rel_dft[1:] - rel_dft[:-1]

    # Find potential TS indices by examining consecutive grad values
    maxima_idx_mace = np.where((grad_mace[:-1] > 0) & (grad_mace[1:] < 0))[0] + 1
    n_maxima_mace = len(maxima_idx_mace)
    maxima_idx_dft = np.where((grad_dft[:-1] > 0) & (grad_dft[1:] < 0))[0] + 1
    n_maxima_dft = len(maxima_idx_dft)

    # Determine simple barrier heights from prior minima to TSs
    lhs_minima_mace = []
    lhs_minima_mace_idx = []
    mace_barriers = []

    for idx, ts in enumerate(maxima_idx_mace):
        # Determine left side minima bounds
        if idx == 0:
            bound_1_idx = 0
        else:
            bound_1_idx = maxima_idx_mace[idx - 1]

        bound_2_idx = ts

        # Region excludes current TS, includes previous TS if idx > 0
        region = rel_energies[bound_1_idx:bound_2_idx]

        # Safety check
        if len(region) == 0:
            lhs_minima_mace.append(np.nan)
            lhs_minima_mace_idx.append(-1)
            mace_barriers.append(np.nan)
            continue

        # Local minimum between bounds
        local_min_rel_idx = np.argmin(region)
        local_min_idx = bound_1_idx + local_min_rel_idx
        local_min_energy = rel_energies[local_min_idx]

        lhs_minima_mace.append(local_min_energy)
        lhs_minima_mace_idx.append(local_min_idx)

        # Barrier from left-side minimum to TS
        mace_barriers.append(rel_energies[ts] - local_min_energy)


    # Determine simple barrier heights from prior minima to DFT TSs
    lhs_minima_dft = []
    lhs_minima_dft_idx = []
    dft_barriers = []

    for idx, ts in enumerate(maxima_idx_dft):
        # Determine left side minima bounds
        if idx == 0:
            bound_1_idx = 0
        else:
            bound_1_idx = maxima_idx_dft[idx - 1]

        bound_2_idx = ts

        # Region excludes current TS, includes previous TS if idx > 0
        region = rel_dft[bound_1_idx:bound_2_idx]

        # Safety check
        if len(region) == 0:
            lhs_minima_dft.append(np.nan)
            lhs_minima_dft_idx.append(-1)
            dft_barriers.append(np.nan)
            continue

        # Local minimum between bounds
        local_min_rel_idx = np.argmin(region)
        local_min_idx = bound_1_idx + local_min_rel_idx
        local_min_energy = rel_dft[local_min_idx]

        lhs_minima_dft.append(local_min_energy)
        lhs_minima_dft_idx.append(local_min_idx)

        # Barrier from left-side minimum to TS
        dft_barriers.append(rel_dft[ts] - local_min_energy)


    # Error metrics
    energy_mae = np.mean(np.abs(rel_energies - rel_dft))
    energy_rmse = np.sqrt(np.mean((rel_energies - rel_dft)**2))
    max_abs_error = np.max(np.abs(rel_energies - rel_dft))

    # Compare MACE and DFT TS indices
    maxima_idx_mace = np.asarray(maxima_idx_mace)
    maxima_idx_dft = np.asarray(maxima_idx_dft)

    mace_barriers = np.asarray(mace_barriers)
    dft_barriers = np.asarray(dft_barriers)

    # Shared TS image indices
    matching_ts_idx = np.intersect1d(maxima_idx_mace, maxima_idx_dft)

    n_matching_ts = len(matching_ts_idx)

    # For each matching TS, compare MACE and DFT barrier heights
    matching_barrier_diffs = []

    for ts in matching_ts_idx:
        mace_pos = np.where(maxima_idx_mace == ts)[0][0]
        dft_pos = np.where(maxima_idx_dft == ts)[0][0]

        barrier_diff = mace_barriers[mace_pos] - dft_barriers[dft_pos]

        matching_barrier_diffs.append(barrier_diff)

    matching_barrier_diffs = np.asarray(matching_barrier_diffs)

    ts_match_tol = 1

    near_matches = []

    for mace_i, mace_ts in enumerate(maxima_idx_mace):
        if len(maxima_idx_dft) == 0:
            continue

        dists = np.abs(maxima_idx_dft - mace_ts)
        closest_dft_pos = np.argmin(dists)

        if dists[closest_dft_pos] <= ts_match_tol:
            dft_ts = maxima_idx_dft[closest_dft_pos]
            barrier_diff = mace_barriers[mace_i] - dft_barriers[closest_dft_pos]

            near_matches.append({
                "mace_ts": int(mace_ts),
                "dft_ts": int(dft_ts),
                "idx_diff": int(mace_ts - dft_ts),
                "barrier_diff": float(barrier_diff),
                "abs_barrier_diff": float(abs(barrier_diff)),
            })


    # Determine if energy decreases before the first maxima
    if n_maxima_mace > 0:
        start_not_minima = bool(np.any(rel_energies[:maxima_idx_mace[0]] < 0))
    else:
        start_not_minima = False

    def clean_for_json(x):
        """
        Convert numpy types/arrays and NaN/inf values into JSON-safe Python objects.
        """
        if isinstance(x, np.ndarray):
            return [clean_for_json(v) for v in x.tolist()]

        if isinstance(x, (list, tuple)):
            return [clean_for_json(v) for v in x]

        if isinstance(x, dict):
            return {k: clean_for_json(v) for k, v in x.items()}

        if isinstance(x, (np.integer,)):
            return int(x)

        if isinstance(x, (np.floating,)):
            if np.isnan(x) or np.isinf(x):
                return None
            return float(x)

        if isinstance(x, float):
            if np.isnan(x) or np.isinf(x):
                return None
            return x

        if isinstance(x, (np.bool_,)):
            return bool(x)

        return x

    record = {
    "folder": folder_name,
    "n_images": n_images,

    # Raw numerical profiles
    "rel_mace": rel_energies,
    "rel_dft": rel_dft,
    "grad_mace": grad_mace,
    "grad_dft": grad_dft,

    # Maxima / TS info
    "n_maxima_mace": n_maxima_mace,
    "maxima_idx_mace": maxima_idx_mace,
    "n_maxima_dft": n_maxima_dft,
    "maxima_idx_dft": maxima_idx_dft,

    # Left-side minima and barriers
    "lhs_minima_mace": lhs_minima_mace,
    "lhs_minima_mace_idx": lhs_minima_mace_idx,
    "mace_barriers": mace_barriers,

    "lhs_minima_dft": lhs_minima_dft,
    "lhs_minima_dft_idx": lhs_minima_dft_idx,
    "dft_barriers": dft_barriers,

    # Energy error metrics
    "energy_mae": energy_mae,
    "energy_rmse": energy_rmse,
    "max_abs_error": max_abs_error,

    # Exact TS matches
    "n_matching_ts": n_matching_ts,
    "matching_ts_idx": matching_ts_idx,
    "matching_barrier_diffs": matching_barrier_diffs,

    # Near TS matches
    "ts_match_tol": ts_match_tol,
    "n_near_matching_ts": len(near_matches),
    "near_matches": near_matches,

    # Qualitative flags
    "start_not_minima": start_not_minima,
    }

    out_path = os.path.join(folder, "neb_metrics.jsonl")

    with open(out_path, "w") as f:
        f.write(json.dumps(clean_for_json(record)) + "\n")

def plot_final_fitted_path(folder, n_images, mlip_forcefit, barrier, delta_E, plot_dft=False):
    """
    Plot NEB final path with optional overlay of DFT energies from energies.csv.

    Parameters
    ----------
    folder : str
        Path to NEB folder (should contain energies.csv if plot_dft=True)
    rel_energies : np.ndarray
        NEB energies relative to initial image (in eV)
    barrier : float
        Barrier along path (in eV)
    delta_E : float
        Energy difference (final - initial) (in eV)
    c : matplotlib color, optional
        Line color for NEB energies
    plot_dft : bool, default False
        If True, overlay DFT energies from energies.csv
    hartree_to_ev : float, default 27.2114
        Conversion factor from Hartree to eV for DFT energies
    """
    name = get_reaction_name(folder)

    fig, ax = plt.subplots(figsize=(8, 4))

    lines = []
    labels = []

    if plot_dft:

        dft_energies = []

        try:
            atoms_jdftx, all_converged = get_jdftx_data(folder, n_images)
            dft_forcefit = fit_images(atoms_jdftx)

            dft_path = dft_forcefit.path
            dft_energies = dft_forcefit.energies
            dft_fit_path = dft_forcefit.fit_path
            dft_fit_energies = dft_forcefit.fit_energies

            line1, = ax.plot(
                np.array(dft_path) + 1,
                dft_energies,
                marker='o',
                color='dimgray',
                linewidth=0,
                markersize=8,
                alpha=0.6,
                zorder=6
            )

            for x, y in dft_forcefit.lines:
                ax.plot(
                    np.array(x) + 1,
                    y,
                    color='darkgrey',
                    marker=None,
                    linewidth=1.5,
                    zorder=4
                )

            ax.plot(
                np.array(dft_fit_path) + 1,
                dft_fit_energies,
                color='dimgray',
                marker=None,
                linewidth=2,
                zorder=5
            )

            for xi, yi in zip(dft_path, dft_energies):
                ax.text(
                    np.array(xi) + 1,
                    yi + 0.015,
                    f"{yi:.2f}",
                    ha="center",
                    fontsize=10,
                    color="k",
                    zorder=8
                )

            lines.append(line1)
            labels.append('DFT')

        except np.linalg.LinAlgError as e:
            print(f"WARNING: Skipping DFT overlay for {folder}")
            print(f"         ASE fit_images failed with singular matrix: {e}")

        except Exception as e:
            print(f"WARNING: Failed to generate DFT overlay for {folder}")
            print(f"         {type(e).__name__}: {e}")

    else:
        dft_energies = []

    '''
    # Optionally overlay DFT energies
    if plot_dft:

        atoms_jdftx, all_converged = get_jdftx_data(folder, n_images)
        dft_forcefit = fit_images(atoms_jdftx)
        dft_path = dft_forcefit.path  # the points themselves (relative energy)
        dft_energies = dft_forcefit.energies
        dft_fit_path = dft_forcefit.fit_path  # the fitted interpolation (relative energy)
        dft_fit_energies = dft_forcefit.fit_energies

        line1, = ax.plot(np.array(dft_path)+1, dft_energies, marker='o', 
                         color='dimgray', linewidth=0, markersize=8, alpha=0.6, zorder=6)
        for x, y in dft_forcefit.lines:  # force tangent lines
            ax.plot(np.array(x)+1, y, color='darkgrey', marker=None, linewidth=1.5, zorder=4)
        ax.plot(np.array(dft_fit_path)+1, dft_fit_energies, color='dimgray', marker=None, linewidth=2, zorder=5)

        for xi, yi in zip(dft_path, dft_energies):
            ax.text(np.array(xi)+1, yi + 0.015, f"{yi:.2f}", ha="center", fontsize=10, color="k", zorder=8)

        lines.append(line1)
        labels.append('DFT')
    '''
        
    mlip_path = mlip_forcefit.path  # the points themselves (relative energy)
    mlip_energies = mlip_forcefit.energies
    mlip_fit_path = mlip_forcefit.fit_path  # the fitted interpolation (relative energy)
    mlip_fit_energies = mlip_forcefit.fit_energies
    line2, = ax.plot(np.array(mlip_path)+1, mlip_energies, marker='o', 
                     color='dodgerblue', linewidth=0, markersize=8, alpha=0.6, zorder=6)
    for x, y in mlip_forcefit.lines:  # force tangent lines
        ax.plot(np.array(x)+1, y, color='lightskyblue', marker=None, linewidth=1.5, zorder=4)
    ax.plot(np.array(mlip_fit_path)+1, mlip_fit_energies, color='dodgerblue', marker=None, linewidth=2, zorder=5)

    for xi, yi in zip(mlip_path, mlip_energies):
        ax.text(np.array(xi)+1, yi + 0.015, f"{yi:.2f}", ha="center", fontsize=10, color="k", zorder=8)

    lines.append(line2)
    labels.append('MLIP')
    
    ax.legend(tuple(lines), tuple(labels), 
              loc='upper right', fontsize=10, handletextpad=0.3)
    ax.set_xlabel(r'Path ($\mathrm{\AA}$)')
    ax.set_ylabel('Relative energy (eV)')
    ax.grid(True, color='gainsboro', which="major", linestyle="-", linewidth=0.6, alpha=0.5, zorder=1)
    ax.set_title(f"NEB: {name}", fontsize=13)

    # Extend y-axis max slightly to make room for point labels
    energies_for_limits = list(mlip_energies)
    energies_for_limits.extend(dft_energies)
    y_min, y_max = np.nanmin(energies_for_limits), np.nanmax(energies_for_limits)
    y_range = y_max - y_min
    pad_top = 0.10 * y_range if y_range > 0 else 0.2
    ax.set_ylim(bottom=ax.get_ylim()[0], top=y_max + pad_top)
    ax.set_xlim(left=0)
    
    ax.text(
        0.02,
        0.95,
        f"Barrier: {barrier:.2f} eV\nΔE: {delta_E:.2f} eV",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.3", alpha=0.8),
        zorder=10,
    )
    ax.axhline(y=0, color='lightgrey', linewidth=1, zorder=1)
    
    plt.savefig(f"{folder}_finalfittedpath.png", dpi=300)
    plt.close()



# ============================== XYZ SNAPSHOTS AND MOVIE ==============================
def write_structures(folder, traj, n_steps, n_images, maceopt_endpoints):
    save_dir = os.path.join(folder, "xyz_snapshots")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Pre-load optimized endpoints (with energies and forces)
    endpoints = None
    if maceopt_endpoints:
        endpoints = {
            1: read(os.path.join(folder, "initial_opt.traj")),
            n_images: read(os.path.join(folder, "final_opt.traj"))
        }

    # 2. Write the initial and final NEB iterations
    for step_idx in [0, n_steps - 1]:
        prefix = f"step{step_idx:04d}"

        for i in range(1, n_images + 1):
            filename = os.path.join(save_dir, f"{prefix}_img{i:02d}.xyz")

            if maceopt_endpoints:
                # Use optimized endpoint structures
                if i in endpoints:
                    atoms = endpoints[i]
                else:
                    # Intermediate images from the NEB trajectory
                    traj_idx = step_idx * (n_images - 2) + (i - 2)
                    atoms = traj[traj_idx]
            else:
                # Original indexing when endpoints are included in the trajectory
                if not (1 < i < n_images):
                    continue
                traj_idx = step_idx * n_images + (i - 1)
                atoms = traj[traj_idx]

            # Write energies and forces if present
            write(filename, atoms, format="extxyz")

def create_movie(folder, n_steps, n_images):
    snapshot_dir = Path(folder) / "xyz_snapshots"
    final_step = n_steps - 1

    poscar_names = [
        str(snapshot_dir / f"step{final_step:04d}_img{i:02d}.xyz")
        for i in range(1, n_images + 1)   
    ]

    # Safety check
    for f in poscar_names:
        if not Path(f).exists():
            raise FileNotFoundError(f"Missing snapshot: {f}")

    V = Visualizer(poscar_names)

    V.create_pngs(
        skip_existing_pngs=False,
        rotation=(-90, 0, 0),
        zoom=1,
        dpi_level=3,
        #png_crop_window=(0.37, 0.63, 0.3, 0.7),
        png_scale=0.5,
        vesta_render_stall=3,
    )

    V.create_movie(
        str(Path(folder) / "pathway_sideview.gif"),
        delay=500,
    )

    V.create_pngs(
        skip_existing_pngs=False,
        rotation=(0, 0, 0),
        zoom=1,
        dpi_level=3,
        #png_crop_window=(0.37, 0.63, 0.3, 0.7),
        png_scale=0.5,
        vesta_render_stall=3,
    )

    V.create_movie(
        str(Path(folder) / "pathway_topview.gif"),
        delay=500,
    )
# ============================== SUMMARY SHEETS AND PLOTS ==============================

def process_summary(base_path, max_steps):
    input_file = os.path.join(base_path, "neb_summary.xlsx")
    output_file = os.path.join(base_path, "neb_summary_processed.xlsx")

    df = pd.read_excel(input_file)

    # Drop unconverged
    df = df[df["n_optimization_steps"] != max_steps].reset_index(drop=True)

    # Add parsed columns
    new_cols = ["rxn", "scaffold", "orient", "rxn_type", "halide_type", "halide_ids"]
    df[new_cols] = df["system"].apply(parse_system)

    # Reorder columns
    sys_i = df.columns.get_loc("system") + 1
    ordered = (
        df.columns[:sys_i].tolist()
        + new_cols
        + df.columns.difference(new_cols, sort=False)[sys_i:].tolist()
    )
    df = df[ordered]

    # Sort
    df = df.sort_values(by="barrier_eV", ascending=True, na_position="last").reset_index(drop=True)

    # Aggregations
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
        .sort_values(by="barrier_eV", ascending=True)
        .reset_index(drop=True)
    )

    df_min_barrier = df.loc[df.groupby("rxn")["barrier_eV"].idxmin()] \
        .sort_values(by="barrier_eV", ascending=True).reset_index(drop=True)

    df_min_deltaE = df.loc[df.groupby("rxn")["deltaE_eV"].idxmin()] \
        .sort_values(by="deltaE_eV", ascending=True).reset_index(drop=True)

    # Write output
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name="raw", index=False)
        df_avg.to_excel(writer, sheet_name="averaged", index=False)
        df_min_barrier.to_excel(writer, sheet_name="min_barriers", index=False)
        df_min_deltaE.to_excel(writer, sheet_name="min_deltaEs", index=False)

    print(f"\nProcessed summary written to: {output_file}")

    return df, df_avg, df_min_barrier, df_min_deltaE

def plot_barriers(
    df,
    use_min=True,
    only_singles=False,
    outfile="barriers.png",
):

    halide_colors = {
    # Singles
    "F":        "#3B82F6",  # blue
    "Cl":       "#FACC15",  # yellow
    "Br":       "#EF4444",  # red
    "I":        "#8B5CF6",  # violet

    # Pairs
    "Cl+F":     "#22C55E",  # green
    "Br+Cl":    "#F97316",  # orange
    "Br+F":     "#A855F7",  # purple
    "F+I":      "#6366F1",  # indigo
    "Cl+I":     "#A3E635",  # violet+yellow
    "Br+I":     "#EC4899",  # violet+red

    # Triples
    "Br+Cl+F":  "#84CC16",  # red+yellow+blue = olive
    "Cl+F+I":   "#0EA5E9",  # violet+yellow+blue = cyan
    "Br+F+I":   "#D946EF",  # violet+red+blue = magenta
    "Br+Cl+I":  "#F59E0B",  # violet+red+yellow = orange
    }

    def halide_color(halides):
        if isinstance(halides, str):
            halides = ast.literal_eval(halides)
        key = "+".join(sorted(halides))
        return halide_colors.get(key, "#888888")

    def format_reaction_label(folder_name: str) -> str:
        """
        Convert folder name like:
        'C10-ZnN4C+FeF3_TO_C10-FeN4C+ZnF3'
        into a nicely formatted LaTeX label:
        'C10-ZnN$_4$C+FeF$_3$ → C10-FeN$_4$C+ZnF$_3$'
        """

        # Split reactant/product
        if "_TO_" in folder_name:
            reactant_raw, product_raw = folder_name.split("_TO_", 1)
        else:
            reactant_raw, product_raw = folder_name, None

        # Regex to subscript numbers except C10/C12
        def subscript_numbers_except_C10_C12(name: str) -> str:
            def repl(match):
                full = match.group(0)
                if full in ("C10", "C12"):
                    return full
                letter, number = match.group(1), match.group(2)
                return f"{letter}$_{number}$"
            return re.sub(r"([A-Za-z])(\d+)", repl, name)

        reactant = subscript_numbers_except_C10_C12(reactant_raw)
        if product_raw:
            product = subscript_numbers_except_C10_C12(product_raw)
            return rf"{reactant} $\rightarrow$ {product}"
        else:
            return reactant

    # Select dataset already handled outside ideally, but keep flexible
    data = df.copy()

    if only_singles:
        def is_single(h):
            if isinstance(h, str):
                h = ast.literal_eval(h)
            return len(h) == 1
        data = data[data["halide_ids"].apply(is_single)]

    data_sorted = data.sort_values("barrier_eV", ascending=False).reset_index(drop=True)

    labels   = data_sorted["rxn"].tolist()
    barriers = data_sorted["barrier_eV"].tolist()
    colors   = [halide_color(h) for h in data_sorted["halide_ids"]]
    is_dep   = (data_sorted["rxn_type"] == "deposition").tolist()

    fig, ax = plt.subplots(figsize=(11, 11))

    y = np.arange(len(labels))
    bars = ax.barh(y, barriers, color=colors, height=0.65)

    # Styling logic (unchanged)
    for bar, halides in zip(bars, data_sorted["halide_ids"]):
        if isinstance(halides, str):
            halides = ast.literal_eval(halides)

        if len(halides) == 1:
            bar.set_linestyle("-")
        elif len(halides) == 3:
            bar.set_linestyle(":")
        else:
            bar.set_linestyle((0, (4, 2)))

    # Labels
    ax.set_yticks(y)
    ax.set_yticklabels([format_reaction_label(r) for r in labels])

    ax.set_xlabel("Barrier (eV)")
    ax.set_xlim(0, max(barriers) + 0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

# ============================== MAIN CONTROL FUNCTION ==============================

def run_neb_analysis(
    base_path,
    file_pattern,
    n_images,
    max_steps,
    step_interval=5,
    maceopt_endpoints=True,
    make_plots=True,
    dft_overlay=True,
    simple_barrier=False,
    do_write_structures=False,
    do_write_movie=False,
    force_rerun=False,
):

    summary_file = os.path.join(base_path, "neb_summary.xlsx")

    # ANALYSIS STAGE

    results = []

    if (not os.path.exists(summary_file)) or force_rerun:

        print("\nRunning NEB analysis...")

        folders = find_neb_folders(base_path, file_pattern)

        for i, folder in enumerate(folders, 1):
            print(f"\n[{i}/{len(folders)}] Processing {folder}")

            traj = load_traj(
                folder,
                maceopt_endpoints=maceopt_endpoints,
                n_images=n_images,
            )

            if traj is None:
                continue

            final_images, rel_energies, barrier, delta_E, n_steps = extract_final_path(traj,n_images,simple_barrier)

            converged = (n_steps - 1) < max_steps

            rel_dft = get_dft_rel_energies(folder)

            rmse, avg_bias = None, None

            if rel_dft is not None:
                rmse, avg_bias = compute_error_metrics(rel_energies, rel_dft)

            results.append({
                "system": os.path.basename(folder),
                "barrier_eV": barrier,
                "deltaE_eV": delta_E,
                "rmse_eV": rmse,
                "avg_bias_eV": avg_bias,
                "n_optimization_steps": n_steps - 1,
                "converged": converged,
            })

            # ---------------- PLOTS ----------------
            if make_plots:
                plot_optimization(folder,traj,n_images,barrier,delta_E,max_steps,step_interval)

                plot_final_path(folder, rel_energies, barrier, delta_E, rmse, avg_bias, converged=converged, plot_dft=dft_overlay)

                try:
                    forcefit = fit_images(final_images)
                except np.linalg.LinAlgError as e:
                    print(f"WARNING: Skipping MLIP fit for {folder}: singular matrix ({e})")
                    forcefit = None

                if forcefit is not None:
                    plot_final_fitted_path(folder,n_images,forcefit,barrier,delta_E,plot_dft=dft_overlay)
                else:
                    print(f"Skipping fitted-path plot for {folder} (no valid forcefit)")

            # ---------------- STRUCTURES AND MOVIES ----------------
            if do_write_structures:
                write_structures(folder, traj, n_steps, n_images, maceopt_endpoints)

            if do_write_movie:
                create_movie(folder, n_steps, n_images)

# ---------------- EXCEL SUMMARIES ----------------
        if len(results) > 0:
            df = (pd.DataFrame(results).sort_values("barrier_eV", na_position="last").reset_index(drop=True))

            df.to_excel(summary_file, index=False)

            print(f"\nSummary written to {summary_file}")

        else:
            print("\nNo results generated (nothing to write).")

    else:
        print(f"\nSkipping NEB analysis (found {summary_file}). Use --force-rerun to overwrite.")

    if not os.path.exists(summary_file):
        print("No summary file found — skipping processing.")
        return

    process_summary(base_path, max_steps)
