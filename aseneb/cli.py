#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import argparse
from aseneb.config_io import load_config
from aseneb.workflow import Workflow


def main():
    parser = argparse.ArgumentParser("NEB  MACE/DFT workflow driver")

    parser.add_argument("--neb-config")
    parser.add_argument("--dft-config")

    parser.add_argument("--interpolate", type=int, help="If present, number of NEB images to interpolate from initial.vasp and final.vasp")
    parser.add_argument("--maceopt", action="store_true",help="Optimize initial/final with MACE before interpolation")

    parser.add_argument("--run-neb", choices=["mace", "dft"], help="Which NEB backend to run")

    parser.add_argument("--final-sp", action="store_true")
    parser.add_argument("--dft-inputs", type=str, help="Directory with in and job.sbatch files")

    args = parser.parse_args()

    neb_cfg = load_config(args.neb_config) if args.neb_config else None
    dft_cfg = load_config(args.dft_config) if args.dft_config else None

    wf = Workflow(neb_cfg, dft_cfg)

    if args.maceopt and not neb_cfg:
        parser.error("--maceopt requires --neb-config")

    if args.interpolate:
        wf.interpolate(args.interpolate,mace_optimize=args.maceopt)

    if args.run_neb == "mace":
        wf.run_mace_neb()
    elif args.run_neb == "dft":
        wf.run_dft_neb()

    if args.final_sp:
        if not args.dft_inputs:
            parser.error("--final-sp requires --dft-inputs")
        wf.run_dft_single_points(args.dft_inputs)

if __name__ == "__main__":
    main()
