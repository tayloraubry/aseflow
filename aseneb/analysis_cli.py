"""
CLI entry point for NEB postprocessing.

Usage:
    aseneb-analyze --base-path folder/ --pattern "C10-ZnN4C*"
    aseneb-analyze --pattern "C10*" --no-plots --create-movie
    aseneb-analyze --help
"""

import argparse
from .analysis_utils import run_neb_analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze NEB trajectories and generate plots + summaries")

    parser.add_argument("--base-path", default=".", help="Base directory containing NEB folders")

    parser.add_argument("--pattern", default="C*", help="Glob pattern to match NEB folders, use quotes (e.g., 'C10-ZnN4C*')")

    parser.add_argument("--n-images", type=int, required=True, help="Number of images in the NEB path (including endpoints)")

    parser.add_argument("--max-steps", type=int, required=True, help="Maximum number of optimization steps to consider for plotting")

    parser.add_argument("--step-interval", type=int, default=5, help="Interval of optimization steps to plot in NEB optimization plot")

    parser.add_argument("--no-maceopt", action="store_true", help="Disable endpoint replacement with MACE optimized structures")

    parser.add_argument("--simple-barrier", action="store_true", help="Use simple barrier definition")

    parser.add_argument("--no-plots", action="store_true", help="Skip all plotting")

    parser.add_argument("--no-dft", action="store_true", help="Skip DFT overlay in final pathway plots")

    parser.add_argument("--write-snapshots", action="store_true", help="Write XYZ snapshots (note: needed for movie generation)")

    parser.add_argument("--create-movie", action="store_true", help="Enable movie generation")

    parser.add_argument("--no-summary", action="store_true", help="Disable Excel summary output")

    parser.add_argument("--force-rerun", action="store_true", help="Recompute NEB analysis even if neb_summary.xlsx already exists (overwrites existing results).")

    args = parser.parse_args()

    run_neb_analysis(
        base_path=args.base_path,
        file_pattern=args.pattern,
        n_images=args.n_images,
        step_interval=args.step_interval,
        max_steps=args.max_steps,
        maceopt_endpoints=not args.no_maceopt,
        simple_barrier=args.simple_barrier,
        make_plots=not args.no_plots,
        dft_overlay=not args.no_dft,
        do_write_structures=args.write_snapshots,
        do_write_movie=args.create_movie,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
