from aseneb.mace_runner import MACERunner
from aseneb.dft_runner import DFTRunner
from aseneb.interpolate import interpolate_images
from pathlib import Path
import warnings

class Workflow:
    def __init__(self, neb_cfg, dft_cfg=None):
        self.mace_runner = MACERunner.from_config(neb_cfg) 
        self.dft_runner = DFTRunner(dft_cfg) if dft_cfg else None

    def interpolate(self, n_images: int, initial="initial.vasp", final="final.vasp"):
        interpolate_images(
            initial_path=Path(initial),
            final_path=Path(final),
            workdir=Path("."),
            n_images=n_images
        )

    def run_mace_neb(self):
        self.mace_runner.run()

    def run_dft_neb(self):
        if not self.dft_runner:
            raise RuntimeError("DFT config not provided")
        self.dft_runner.run_neb()

    def run_dft_single_points(self, dft_inputs):
        # Match the argument name 'dft_inputs'
        runner = DFTRunner(dft_inputs=dft_inputs) 
        runner.run_dft_single_points()