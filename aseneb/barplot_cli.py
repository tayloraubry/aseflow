import argparse
from .analysis import process_summary, plot_barriers


def main():
    parser = argparse.ArgumentParser(
        description="Plot NEB barrier summaries from processed data"
    )

    parser.add_argument("--base-path", default=".", help="Directory containing neb_summary.xlsx")

    parser.add_argument("--use-min", action="store_true")
    parser.add_argument("--only-singles", action="store_true")

    args = parser.parse_args()

    df, df_avg, df_min_barrier, df_min_deltaE = process_summary(
        args.base_path,
        max_steps=None,  # or make configurable
    )

    data = df_min_barrier if args.use_min else df_avg

    plot_barriers(
        data,
        use_min=args.use_min,
        only_singles=args.only_singles,
    )