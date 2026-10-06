import re
from pathlib import Path
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin, FIRE, FIRE2
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
    "FIRE2": FIRE2,
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

    def load_images_from_traj(self, trajfile):
        n_images = len(self.load_images())

        frames = read(trajfile, index=":")

        return frames[-n_images:]

    def attach_calculators(self, images):
        for i, img in enumerate(images):
            folder = Path(str(i + 1).zfill(2))
            img.calc = MACECalculator(
                model_paths=str(self.cfg.calculator.model),
                device=self.cfg.calculator.device,
                outfile=str(folder),
                head = self.cfg.calculator.head,
                default_dtype="float64",
            )

    def optimize_structure(self, infile, outfile=None):
        infile = Path(infile)
        outfile = Path(outfile) if outfile else infile

        atoms = read(infile)

        atoms.calc = MACECalculator(
            model_paths=str(self.cfg.calculator.model),
            device=self.cfg.calculator.device,
            head = self.cfg.calculator.head,
            default_dtype="float64",
        )

        opt_cls = OPTIMIZERS[self.cfg.optimizer.name]

        opt = opt_cls(
            atoms,
            logfile=f"{infile.stem}_opt.log",
            trajectory=f"{infile.stem}_opt.traj",
        )

        opt.run(
            fmax=self.cfg.optimizer.fmax,
            steps=self.cfg.optimizer.steps,
        )

        write(outfile, atoms, format="vasp", direct=True)

        print(f"Optimized {infile} → {outfile}")

        return outfile

    def run_neb(self, images, k, climb, logfile, trajectory):

        self.attach_calculators(images)

        neb = NEB(
            images,
            k=k,
            climb=climb,
            method=self.cfg.neb.method,
        )

        opt_cls = OPTIMIZERS[self.cfg.optimizer.name]

        opt_kwargs = {
            "logfile": logfile,
            "trajectory": trajectory,
        }

        if self.cfg.optimizer.a is not None:
            opt_kwargs["a"] = self.cfg.optimizer.a

        optimizer = opt_cls(neb, **opt_kwargs)

        optimizer.run(
            fmax=self.cfg.optimizer.fmax,
            steps=self.cfg.optimizer.steps,
        )

    def run(self):

        # Original single-stage behavior
        if not self.cfg.neb_stages:

            images = self.load_images()

            self.run_neb(
                images=images,
                k=self.cfg.neb.k,
                climb=self.cfg.neb.climb,
                logfile="neb.log",
                trajectory="neb.traj",
            )

            print("NEB run completed.")
            return

        # Multi-stage behavior
        images = self.load_images()

        for i, stage in enumerate(self.cfg.neb_stages):

            print(
                f"Running NEB stage {i + 1}: "
                f"k={stage.k}, climb={stage.climb}"
            )

            trajfile = f"neb_stage{i}.traj"
            logfile = f"neb_stage{i}.log"

            self.run_neb(
                images=images,
                k=stage.k,
                climb=stage.climb,
                logfile=logfile,
                trajectory=trajfile,
            )

            # Restart next stage from final band
            if i < len(self.cfg.neb_stages) - 1:
                images = self.load_images_from_traj(trajfile)

        print("All NEB stages completed.")
