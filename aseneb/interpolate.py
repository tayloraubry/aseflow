from pathlib import Path
from ase.io import read, write
from ase.mep import NEB
import numpy as np

def reorder_to_match(reference, target):
    """
    Reorder `target` atoms to match the species ordering of `reference`.
    Assumes same composition.
    """
    ref_symbols = reference.get_chemical_symbols()
    tgt_symbols = target.get_chemical_symbols()

    if sorted(ref_symbols) != sorted(tgt_symbols):
        raise ValueError("Initial and final structures do not have identical composition.")

    # Build mapping from symbol -> indices in target
    symbol_to_indices = {}
    for i, s in enumerate(tgt_symbols):
        symbol_to_indices.setdefault(s, []).append(i)

    # Construct new order
    new_indices = []
    for s in ref_symbols:
        new_indices.append(symbol_to_indices[s].pop(0))

    reordered = target[new_indices]
    reordered.set_cell(target.get_cell())
    reordered.set_pbc(target.get_pbc())

    return reordered


def interpolate_images(
    initial_path: Path,
    final_path: Path,
    workdir: Path,
    n_images: int,
    method: str = "idpp",
    mic: bool = True,
    apply_constraint: bool = True,
):
    initial = read(initial_path, format="vasp")
    final = read(final_path, format="vasp")

    new_final = reorder_to_match(initial, final)

    images = [initial]
    images += [initial.copy() for _ in range(n_images - 2)]
    images += [new_final]

    neb = NEB(images)
    neb.interpolate(method=method, mic=mic, apply_constraint=apply_constraint)

    for i, image in enumerate(images):
        folder = workdir / str(i + 1).zfill(2)
        folder.mkdir(parents=True, exist_ok=True)
        image.write(folder / "POSCAR", format="vasp", direct=True)

    print(f"[interpolate] Wrote {n_images} images to {workdir.resolve()}")
