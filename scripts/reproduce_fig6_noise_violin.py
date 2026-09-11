"""Reproduce paper Fig. 6: noise-sensitivity violin plot.

Paper caption: "The discrepancy statistics of simulations of ensembles of
Gaussian chain form-factor, at different collected photon densities. The
ensemble's radius of gyration distribution is normal with a mean of
R0=3nm and relative standard deviations in the range of 0-0.55."

This runs the full noise benchmark (BenchmarkConfig's defaults already
match this setup: rg=3.0, phi2=1/18 (Gaussian chain), distribution='normal',
v_values=linspace(0.01,0.55,12)**2, 7 half-decade photon densities
1e6-1e9 photons/nm^2) via tenor_saxs.benchmark, and plots the resulting
(V, R0) violin panels -- the manuscript's current benchmark figure.

Usage:
    .venv/bin/python3 scripts/reproduce_fig6_noise_violin.py [--n-replicates N] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenor_saxs import benchmark, plotting  # noqa: E402


def main(n_replicates: int, out_dir: Path) -> None:
    data_root = Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "fig6-benchmark"
    config = benchmark.BenchmarkConfig(output_root=str(data_root), n_replicates=n_replicates)

    t0 = time.time()
    results_df, manifest = benchmark.run_benchmark(config)
    t1 = time.time()
    print(f"run_benchmark took {t1 - t0:.1f}s, {len(results_df)} rows")

    n_ok = (manifest["Status"] == "ok").sum()
    print(f"clean database: {n_ok}/{len(manifest)} cases ok")
    # np.isfinite(BestV), NOT Status=="ok" -- the latter only means
    # tenor_protocol didn't raise (a much weaker condition; see
    # plotting._percent_valid's docstring / the internal validation notes for the same
    # bug caught and fixed in the violin annotation itself).
    import numpy as np

    valid_frac = np.isfinite(results_df["BestV"]).mean()
    print(f"overall valid fraction: {valid_frac:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # The manuscript's current benchmark figure is the (V, R0) pair -- the
    # apparent-radius panel was dropped (it has no physical interest: fixed
    # by construction, so its own discrepancy is a tautology).
    fig, axes = plotting.plot_tenor_benchmark_violins(results_df, panels=("V", "R0"))
    fig.suptitle(f"Reproduction of paper Fig. 6: Gaussian chain, R0=3nm, normal, n_replicates={n_replicates}")
    out_path = out_dir / "fig6_noise_violin_reproduction.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    results_df.to_csv(out_dir / "fig6_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-replicates", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.n_replicates, args.out_dir)
