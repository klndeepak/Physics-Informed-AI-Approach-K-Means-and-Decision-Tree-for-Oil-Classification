#!/usr/bin/env python3
"""Generate the additional PNG figures and CSV tables."""

import argparse

from raman_analysis.supplementary.pipeline import render_saved_tables, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-only", action="store_true", help="render PNG files from existing CSV tables"
    )
    args = parser.parse_args()
    render_saved_tables() if args.figures_only else run()


if __name__ == "__main__":
    main()
