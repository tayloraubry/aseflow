#!/usr/bin/env python3

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Optional override map for directory comparison mode.
HEADER_OVERRIDES = {}


SAMPLINGS = ["c", "d", "dr", "r"]

SAMPLING_PATTERN = re.compile(
    r"^(.*?)_(c|d|dr|r)_TO_(.*?)_(c|d|dr|r)_finalfittedpath$"
)

def matches_system(name: str) -> bool:
    return (
        name.lower().endswith(".png")
        and "_finalfittedpath" in name
    )

def display_name_for_dir(directory: str) -> str:
    base = os.path.basename(os.path.normpath(directory))
    return HEADER_OVERRIDES.get(base, base)

def clean_dir_name(path):
    return Path(path).resolve().name

# ==========================================================
# Mode 1: Compare directories
# ==========================================================

def collect_directory_images(method_dirs):
    per_method = defaultdict(dict)

    for method_dir in method_dirs:
        if not os.path.isdir(method_dir):
            continue

        display_method = display_name_for_dir(method_dir)

        for dirpath, _, filenames in os.walk(method_dir):
            for filename in filenames:
                if not matches_system(filename):
                    continue

                system_name = os.path.splitext(filename)[0]
                full_path = os.path.join(dirpath, filename)

                per_method[display_method][system_name] = full_path

    methods = [display_name_for_dir(d) for d in method_dirs]

    key_sets = [
        set(per_method.get(method, {}).keys())
        for method in methods
    ]

    common_systems = (
        set.intersection(*key_sets)
        if key_sets
        else set()
    )

    systems = {}

    for system_name in sorted(common_systems):
        systems[system_name] = {
            method: per_method[method][system_name]
            for method in methods
        }

    return methods, systems


# ==========================================================
# Mode 2: Compare c/d/dr/r samplings within one directory
# ==========================================================

def collect_sampling_images(directory):

    systems = defaultdict(dict)

    for dirpath, _, filenames in os.walk(directory):

        for filename in filenames:

            if not matches_system(filename):
                continue

            stem = os.path.splitext(filename)[0]

            match = SAMPLING_PATTERN.match(stem)

            if match is None:
                continue

            reactant, samp1, product, samp2 = match.groups()

            # Safety check
            if samp1 != samp2:
                continue

            system_name = f"{reactant}_TO_{product}"

            systems[system_name][samp1] = os.path.join(
                dirpath,
                filename
            )

    # Keep only complete c/d/dr/r sets
    complete_systems = {}

    for system_name, files in systems.items():

        if all(s in files for s in SAMPLINGS):

            complete_systems[system_name] = {
                s: files[s]
                for s in SAMPLINGS
            }

    return SAMPLINGS, complete_systems


# ==========================================================
# PDF rendering
# ==========================================================

def render_pdf(
    systems,
    methods,
    output_path,
    rows_per_page=6,
    page_width=16,
    row_height=2.7,
):

    system_names = sorted(systems.keys())

    if not system_names:
        raise SystemExit(
            "No complete image sets found."
        )

    with PdfPages(output_path) as pdf:

        for start in range(
            0,
            len(system_names),
            rows_per_page,
        ):

            page_names = system_names[
                start:start + rows_per_page
            ]

            nrows = len(page_names)

            fig_height = max(
                2.0,
                row_height * nrows,
            )

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=len(methods),
                figsize=(page_width, fig_height),
                constrained_layout=True,
            )

            if nrows == 1:
                axes = [axes]

            for row_idx, system_name in enumerate(page_names):

                row_axes = axes[row_idx]

                for col_idx, method in enumerate(methods):

                    ax = row_axes[col_idx]
                    ax.axis("off")

                    path = systems[system_name].get(method)

                    if path and os.path.isfile(path):

                        try:
                            img = Image.open(path)
                            ax.imshow(img)

                        except Exception:

                            ax.text(
                                0.5,
                                0.5,
                                "error",
                                ha="center",
                                va="center",
                                fontsize=8,
                            )

                            ax.set_facecolor(
                                "#f2dede"
                            )

                    else:

                        ax.text(
                            0.5,
                            0.5,
                            "missing",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

                        ax.set_facecolor(
                            "#eeeeee"
                        )

                    if row_idx == 0:
                        ax.set_title(
                            method,
                            fontsize=10,
                        )

                row_axes[0].set_ylabel(
                    system_name,
                    fontsize=8,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=60,
                )

            pdf.savefig(fig, dpi=300)
            plt.close(fig)


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "If one directory is supplied, compare "
            "c/d/dr/r samplings. If multiple "
            "directories are supplied, compare "
            "directories."
        )
    )

    parser.add_argument(
        "dirs",
        nargs="+",
        help=(
            "One directory (sampling comparison) "
            "or multiple directories "
            "(directory comparison)."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output PDF path.",
    )

    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--page-width",
        type=float,
        default=16.0,
    )

    parser.add_argument(
        "--row-height",
        type=float,
        default=2.7,
    )

    args = parser.parse_args()

    if args.output is None:

        dir_names = [
            clean_dir_name(d)
            for d in args.dirs
        ]

        args.output = (
            "finalpath_table_"
            + "_".join(dir_names)
            + ".pdf"
        )
    # ------------------------------------------------------
    # Auto-select mode
    # ------------------------------------------------------

    if len(args.dirs) == 1:

        print(
            f"Sampling comparison mode: "
            f"{args.dirs[0]}"
        )

        methods, systems = collect_sampling_images(
            args.dirs[0]
        )

    else:

        print(
            "Directory comparison mode:"
        )

        for d in args.dirs:
            print(f"  {d}")

        methods, systems = collect_directory_images(
            args.dirs
        )

    render_pdf(
        systems,
        methods,
        args.output,
        rows_per_page=args.rows_per_page,
        page_width=args.page_width,
        row_height=args.row_height,
    )

    print(
        f"Wrote PDF: {args.output}"
    )


if __name__ == "__main__":
    main()
