#%%
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
from ase.visualize.plot import plot_atoms


# ==============================
# USER SETTINGS
# ==============================

BASE_PATH = "mpa0/"
# BASE_PATH = "."  # use this if running inside a directory
MACEOPT_ENDPOINTS = True  # whether to replace NEB endpoint energies with MACE-optimized energies from initial_opt.traj and final_opt.traj
FILE = "*"  # glob pattern to find NEB folders (relative to BASE_PATH)

N_IMAGES = 15
STEP_INTERVAL = 5
SAVE_INTERVAL = 100
MAX_STEPS = 1000

PLOT_NEB_OPTIMIZATION = True
PLOT_NEB_FINAL_PATH = True
PLOT_DFT_FINAL_PATH = True
SIMPLE_BARRIER = True

PLOT_STRUCTURE_GRIDS = True
WRITE_SNAPSHOTS = True
WRITE_SUMMARY = True

plt.rcParams.update({"font.size": 11})

#%% ============================== HELPER FUNCTIONS ==============================

def get_energy(img):
    """Return the energy of an ASE Atoms object or np.nan if unavailable."""
    if 'energy' in img.info:
        return img.info['energy']
    if img.calc is not None and 'energy' in img.calc.results:
        return img.calc.results['energy']
    return np.nan

def find_neb_folders(base):
    candidates = glob.glob(os.path.join(base, FILE))
    folders = [p for p in candidates if os.path.isdir(p)]
    return sorted(folders)

def load_traj(folder, maceopt_endpoints=MACEOPT_ENDPOINTS):

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

    n_steps = len(traj) // N_IMAGES 

    for step in range(n_steps):

        i0 = step * N_IMAGES
        iF = step * N_IMAGES + (N_IMAGES - 1)

        traj[i0].info["energy"] = e0
        traj[iF].info["energy"] = ef

    return traj

def extract_final_path(traj, n_images, simple_barrier=SIMPLE_BARRIER):

    n_steps = (len(traj) // n_images)
    final_images = traj[(n_steps-1) * n_images: n_steps * n_images]
    energies = np.array([get_energy(img) for img in final_images])

    # Calculate relative energies
    ref = energies[0]
    rel_energies = energies - ref

    # Barrier / deltaE
    if simple_barrier:

        max_idx = int(np.nanargmax(rel_energies))

        barrier = rel_energies[max_idx]  # max E relative to initial
        delta_E = rel_energies[-1]       # final E relative to initial

    # THIS COMPLEX BARRIER CALCULATION IS UNDER DEVELOPMENT
    else:

        max_idx = int(np.nanargmax(rel_energies))

        minE_after_max = np.nanmin(rel_energies[max_idx:])

        if max_idx == 0:

            barrier = rel_energies[max_idx]
            delta_E = minE_after_max

        else:

            minE_before_max = np.nanmin(rel_energies[:max_idx])

            barrier = (rel_energies[max_idx]- minE_before_max)

            delta_E = (minE_after_max - minE_before_max)

    return (final_images,rel_energies,barrier,delta_E,n_steps)

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

# ============================== PLOTS ==============================

def plot_neb_optimization(folder, traj, barrier, delta_E):

    name = get_reaction_name(folder)

    n_steps = len(traj) // N_IMAGES

    # --- Reference energy from final NEB path  ---
    lasttraj = [get_energy(img) for img in traj[-N_IMAGES:]]
    E_ref = lasttraj[0]

    steps_to_plot = list(range(0, n_steps, STEP_INTERVAL))

    # Always include the final step (even if it doesn't align with the interval)
    if (n_steps - 1) not in steps_to_plot:
        steps_to_plot.append(n_steps - 1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_to_plot)))
    plt.rc("axes", prop_cycle=cycler(color=colors))

    fig, ax = plt.subplots(figsize=(8, 4))

    for c, step in zip(colors, steps_to_plot):

        energies = [get_energy(traj[step * N_IMAGES + i]) for i in range(N_IMAGES)]

        ref_energies = np.subtract(energies, E_ref)

        ax.plot(range(N_IMAGES), ref_energies, color=c)

    if n_steps - 1 == MAX_STEPS:
        ax.plot(range(N_IMAGES), ref_energies, color='r')

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

    plt.savefig(f"{folder}.png", dpi=300)
    plt.show()

def plot_final_path(folder, rel_energies, barrier, delta_E, c=plt.cm.viridis(1.0), plot_dft=False, hartree_to_ev=27.2114):
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
            rel_dft = (dft_energies - dft_energies[0]) * hartree_to_ev
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
    plt.savefig(f"{folder}_finalpath.png", dpi=300)
    plt.show()

def plot_structure_grid(folder, images, title, filename):

    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    axes = axes.flatten()

    for i, img in enumerate(images):

        ax = axes[i]

        plot_atoms(
            img,
            ax=ax,
            radii=0.5,
            rotation="-90x,90y",
            show_unit_cell=False,
        )

        ax.set_title(f"Image {i}")
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.suptitle(title)

    plt.savefig(os.path.join(folder, filename), dpi=300)
    plt.show()

# ============================== XYZ SNAPSHOTS ==============================

def write_snapshots(folder, traj, n_steps):

    save_dir = os.path.join(folder, "xyz_snapshots")
    os.makedirs(save_dir, exist_ok=True)

    # Initial path
    for i in range(N_IMAGES):
        write(
            os.path.join(save_dir, f"step0000_img{i:02d}.xyz"),
            traj[i],
        )

    # Final path
    for i in range(N_IMAGES):
        idx = (n_steps - 1) * N_IMAGES + i
        write(
            os.path.join(save_dir, f"step{n_steps-1:04d}_img{i:02d}.xyz"),
            traj[idx],
        )

#%% ============================== MAIN ==============================

folders = find_neb_folders(BASE_PATH)

results = []

for folder in folders:

    print(f"\nProcessing {folder}")

    traj = load_traj(folder)
    if traj is None:
        continue

    final_images, rel_energies, barrier, delta_E, n_steps = extract_final_path(traj, N_IMAGES)

    print(f"Barrier = {barrier:.3f} eV")
    print(f"ΔE = {delta_E:.3f} eV")

    results.append(
        {
            "system": os.path.basename(folder),
            "barrier_eV": barrier,
            "deltaE_eV": delta_E,
            "n_optimization_steps": n_steps-1,
        }
    )

    if PLOT_NEB_OPTIMIZATION:
        plot_neb_optimization(folder, traj, barrier, delta_E)

    if PLOT_NEB_FINAL_PATH:
        if n_steps - 1 == MAX_STEPS:
            plot_final_path(folder, rel_energies, barrier, delta_E, c='r', plot_dft=PLOT_DFT_FINAL_PATH)
        else:
            plot_final_path(folder, rel_energies, barrier, delta_E, plot_dft=PLOT_DFT_FINAL_PATH)

    if PLOT_STRUCTURE_GRIDS:

        initial_images = traj[:N_IMAGES]

        plot_structure_grid(
            folder,
            initial_images,
            "Initial trajectory",
            "initial_images_grid.png",
        )

        plot_structure_grid(
            folder,
            final_images,
            "Final trajectory",
            "final_images_grid.png",
        )

    if WRITE_SNAPSHOTS:
        write_snapshots(folder, traj, n_steps)


# ==============================
# WRITE SUMMARY TABLE
# ==============================

if WRITE_SUMMARY and len(results) > 0:

    df = pd.DataFrame(results)

    # Order by barrier (lowest to highest)
    df = df.sort_values(by="barrier_eV", ascending=True, na_position="last").reset_index(drop=True)

    # Write Excel (and keep CSV if desired)
    df.to_excel(os.path.join(BASE_PATH, "neb_summary.xlsx"), index=False)

    print("\nSummary written to neb_summary.xlsx")


print("\nDone.")
