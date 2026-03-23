import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*torch.jit.load is deprecated.*")
warnings.filterwarnings("ignore", message=".*default method has changed.*")  # the NEB one
warnings.filterwarnings("ignore", category=UserWarning, module="mace")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
from pathlib import Path
import re
import shutil
import subprocess
from ase.io import read, write

class DFTRunner:
    def __init__(self, workdir=".", run_script="run_dft.sh", dft_inputs=None):
        self.workdir = Path(workdir)
        self.run_script = Path(run_script)
        self.dft_inputs = Path(dft_inputs) if dft_inputs else None
        self.required_files = ["in", "run_dft.sh"]

    def _find_image_dirs(self):
        folders = [
            d for d in self.workdir.iterdir()
            if d.is_dir()
            and re.fullmatch(r"\d{2}", d.name)
            and (d / "POSCAR").exists()
        ]
        if not folders:
            raise RuntimeError(f"No image directories found in {self.workdir}")

        folders.sort(key=lambda d: int(d.name))
        return folders

    def run_dft_single_points(self):
        if not self.dft_inputs:
            raise ValueError("No dft_inputs directory provided to DFTRunner.")

        image_dirs = self._find_image_dirs()
        n_images = len(image_dirs)

        print(f"[DFT] Found {n_images} image directories")

        traj_file = self.workdir / "neb.traj"
        if not traj_file.exists():
            raise RuntimeError(f"neb.traj not found in {self.workdir}")

        traj = read(traj_file, index=":")

        if len(traj) % n_images != 0:
            raise RuntimeError(
                f"Trajectory length ({len(traj)}) is not divisible by number of images ({n_images})")

        final_images = traj[-n_images:]

        for img_dir, atoms in zip(image_dirs, final_images):
            contcar = img_dir / "CONTCAR"
            write(contcar, atoms, format="vasp", direct=True)

        for d in image_dirs:
            print(f"[DFT] Preparing {d}...")
            
            # Copy the required files into the image directory
            for filename in self.required_files:
                src = self.dft_inputs / filename
                dst = d / filename
                
                if not src.exists():
                    raise FileNotFoundError(f"Missing {filename} in {self.dft_inputs}")
                
                shutil.copy2(src, dst)
                if filename.endswith(".sh"):
                    dst.chmod(0o755)

            # Run the local script
            subprocess.run(["./run_dft.sh"], cwd=d, check=True)

        print("[DFT] All single point calculations completed.")
