from pathlib import Path
from ase.io import read
from ase.mep import NEB
import warnings

def interpolate_images(
    initial_path: Path,
    final_path: Path,
    workdir: Path,
    n_images: int,
    method: str = "idpp",
    mic: bool = True,
):
    initial = read(initial_path, format="vasp")
    final = read(final_path, format="vasp")

    images = [initial]
    images += [initial.copy() for _ in range(n_images - 2)]
    images += [final]

    neb = NEB(images)
    neb.interpolate(method=method, mic=mic)

    for i, image in enumerate(images):
        folder = workdir / str(i + 1).zfill(2)
        folder.mkdir(parents=True, exist_ok=True)
        image.write(folder / "POSCAR", format="vasp", direct=True)

    print(f"[interpolate] Wrote {n_images} images to {Path.cwd()}")