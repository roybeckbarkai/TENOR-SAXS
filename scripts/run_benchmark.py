"""Run the TENOR-SAXS end-to-end noise benchmark (clean-room port of
``run_TENOR_benchmark.m``).

Simulates a noise-free detector image per target scattering-weighted
variance ``V``, adds photon-counting noise at several flux levels over
multiple replicates, runs the full TENOR-SAXS protocol on each, and saves a
results table plus a violin-plot diagnostic figure.

Usage:
    .venv/bin/python3 scripts/run_benchmark.py [--output-root DIR]
        [--n-cases N] [--n-replicates N] [--seed N] [--overwrite-clean]

By default, generated data is written OUTSIDE the repo (a sibling
``tenor-saxs-v2-data/benchmark`` directory), matching this project's
convention of keeping generated data out of git.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenor_saxs import benchmark, plotting  # noqa: E402

_DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "benchmark"


def _build_config(args: argparse.Namespace) -> benchmark.BenchmarkConfig:
    config = benchmark.BenchmarkConfig(
        output_root=str(args.output_root),
        seed=args.seed,
        n_replicates=args.n_replicates,
    )
    if args.n_cases is not None:
        config.v_values = np.asarray(config.v_values)[: args.n_cases]
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT, help="Directory for the clean-database cache, results CSV/PKL, and violin PNG.")
    parser.add_argument("--n-cases", type=int, default=None, help="Only run the first N of the 12 target V values (smoke-test convenience). Default: all 12.")
    parser.add_argument("--n-replicates", type=int, default=30, help="Number of noise replicates per (case, photon-flux level).")
    parser.add_argument("--seed", type=int, default=314159, help="Base seed for the deterministic per-replicate seed derivation.")
    parser.add_argument("--overwrite-clean", action="store_true", help="Force regeneration of the clean (noise-free) database cache even if a matching one exists.")
    args = parser.parse_args(argv)

    config = _build_config(args)

    print(f"Running TENOR-SAXS benchmark: {len(config.v_values)} case(s), {config.n_replicates} replicate(s), output -> {config.output_root}")
    results_df, manifest = benchmark.run_benchmark(config, overwrite_clean=args.overwrite_clean)

    n_ok_cases = int((manifest["Status"] == "ok").sum())
    n_rows = len(results_df)
    # np.isfinite(BestV), NOT Status=="ok" -- the latter only means
    # tenor_protocol didn't raise, a much weaker condition: the observable
    # can still come back unusable (e.g. every branch "outside calibration
    # range") without raising anything, especially at low photon counts.
    valid_mask = np.isfinite(results_df["BestV"])
    valid_fraction = float(valid_mask.mean()) if n_rows else float("nan")

    abs_err = (results_df.loc[valid_mask, "BestV"] - results_df.loc[valid_mask, "True_V"]).abs()
    mean_abs_err = float(abs_err.mean()) if len(abs_err) else float("nan")
    median_abs_err = float(abs_err.median()) if len(abs_err) else float("nan")

    print("--- Summary ---")
    print(f"Clean cases ok: {n_ok_cases}/{len(manifest)}")
    print(f"Result rows: {n_rows}")
    print(f"Overall valid fraction (finite BestV): {valid_fraction:.3f}")
    print(f"Mean |BestV - True_V| (valid rows): {mean_abs_err:.4g}")
    print(f"Median |BestV - True_V| (valid rows): {median_abs_err:.4g}")

    # The manuscript's current benchmark figure is the (V, R0) pair -- the
    # apparent-radius panel was dropped (it has no physical interest: the
    # apparent radius is fixed by construction, so its own discrepancy is a
    # tautology -- see plotting.plot_r0_discrepancy_violin).
    fig, _axes = plotting.plot_tenor_benchmark_violins(results_df, panels=("V", "R0"))
    png_path = Path(config.output_root) / "benchmark_violin.png"
    fig.savefig(png_path, dpi=150)
    print(f"Saved violin plot: {png_path}")


if __name__ == "__main__":
    main()
