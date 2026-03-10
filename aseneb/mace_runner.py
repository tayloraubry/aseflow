import re
from pathlib import Path
from ase.io import read
from ase.mep import NEB
from ase.optimize import BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin, FIRE
from mace.calculators import MACECalculator
from aseneb.config import RunConfig

OPTIMIZERS = {
    "BFGS": BFGS,
    "BFGSLineSearch": BFGSLineSearch,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "GPMin": GPMin,
    "MDMin": MDMin,
    "FIRE": FIRE,
}

class MACERunner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: RunConfig):
        return cls(cfg)
    
    def load_images(self):
        workdir = Path(".")
        folders = [
            d for d in workdir.iterdir()
            if d.is_dir() and re.fullmatch(r"\d{2}", d.name) and (d / "POSCAR").exists()
        ]
        if not folders:
            raise RuntimeError(f"No POSCAR files found in {workdir}")
        folders.sort(key=lambda d: int(d.name))
        print(f"Image dirs found ({len(folders)}):",", ".join(d.name for d in folders))
        images = [read(f / "POSCAR") for f in folders]
        return images

    def attach_calculators(self, images):
        for i, img in enumerate(images):
            folder = Path(str(i + 1).zfill(2))
            img.calc = MACECalculator(
                model_paths=str(self.cfg.calculator.model),
                device=self.cfg.calculator.device,
                outfile=str(folder),
                head="default",
                default_dtype="float64",
            )

    def run(self):
        images = self.load_images()
        self.attach_calculators(images)

        neb = NEB(
            images,
            climb=self.cfg.neb.climb,
            method=self.cfg.neb.method,
        )

        opt_cls = OPTIMIZERS[self.cfg.optimizer.name]
        opt_kwargs = {
            "logfile": "neb.log",
            "trajectory": "neb.traj",
        }
        if self.cfg.optimizer.a is not None:
            opt_kwargs["a"] = self.cfg.optimizer.a

        optimizer = opt_cls(neb, **opt_kwargs)
        optimizer.run(
            fmax=self.cfg.optimizer.fmax,
            steps=self.cfg.optimizer.steps,
        )

        print("NEB run completed.")