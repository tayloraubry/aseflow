from aseneb.mace_runner import MACERunner
from aseneb.dft_runner import DFTRunner
from aseneb.interpolate import interpolate_images
from pathlib import Path
import warnings

class Workflow:
    def __init__(self, neb_cfg, dft_cfg=None):
        self.mace_runner = MACERunner.from_config(neb_cfg) 
        self.dft_runner = DFTRunner(dft_cfg) if dft_cfg else None

    def interpolate(
        self,
        n_images: int,
        initial="initial.vasp",
        final="final.vasp",
        mace_optimize=False,
    ):
        initial_path = Path(initial)
        final_path = Path(final)

        if mace_optimize:

            initial_opt = initial_path.with_name(
                initial_path.stem + "_opt.vasp"
            )

            final_opt = final_path.with_name(
                final_path.stem + "_opt.vasp"
            )

            initial_path = self.mace_runner.optimize_structure(
                infile=initial_path,
                outfile=initial_opt,
            )

            final_path = self.mace_runner.optimize_structure(
                infile=final_path,
                outfile=final_opt,
            )

        interpolate_images(
            initial_path=initial_path,
            final_path=final_path,
            workdir=Path("."),
            n_images=n_images,
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
