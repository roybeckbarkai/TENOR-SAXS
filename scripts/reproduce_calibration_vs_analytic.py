"""Reproduce the manuscript's calibration-vs-analytic comparison for Y_G2 / Y_M.

The manuscript prescribes a simulation-derived calibration curve as the
recommended inversion route for Y_G2 (``Yg210``) and Y_M (``Ym210``) --
their first-order analytic formulas are measurably biased, and Y_G2's
analytic curve turns over at V=0.273, capping its usable coverage at 82%
no matter how much flux is collected. This was the single most
consequential finding of the round-4 code audit (`C-01`/`C7` in
CODE_CHANGES.md): the package had NO simulation-derived calibration route
at all before ``tenor_saxs.calibration``/``tenor_protocol``'s
``calibration_override`` were added.

This script builds one calibration curve (Gaussian chain, matching this
package's own noise-benchmark defaults) and compares the analytic vs.
calibration route for Yg210/Ym210:

1. Noise-free, across the full round-4 benchmark V grid (12 values) --
   bias and coverage (does the route return a finite V at all?).
2. A modest noisy comparison (fewer replicates than the full 30-replicate
   benchmark, to keep this script's own runtime short) at two photon-flux
   levels spanning the benchmark's range.

Usage:
    .venv/bin/python3 scripts/reproduce_calibration_vs_analytic.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenor_saxs import benchmark, calibration, distributions, formfactors, protocol, psf, simulation  # noqa: E402

# Matches tenor_saxs.benchmark.BenchmarkConfig's own noise-benchmark defaults.
SD_DIST = 360.0
WAVELENGTH = 0.1
DET_SIDE = 3.5
DET_PIX = 500
PSF0 = psf.bartlett2d(3, 15)
PXN = np.array([91, 81, 117, 127])
R0 = 3.0
N_RADII = 25
DISTRIBUTION = "normal"
PHI2 = formfactors.GUINIER_TABLE["gaussian_chain"].phi2
WEIGHT_POWER = formfactors.GUINIER_TABLE["gaussian_chain"].weight_power

V_GRID = np.linspace(0.01, 0.55, 12) ** 2  # the round-4 benchmark's own V grid
OBSERVABLES = ("Yg210", "Ym210")


def _route(name: str, override) -> dict:
    return {"analytic": None, "calibration": {name: override}}


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building the simulated calibration curve (Gaussian chain, R0=3nm)...")
    t0 = time.time()
    curve = calibration.build_calibration_curve(r0=R0, phi2=PHI2, weight_power=WEIGHT_POWER, pxn=PXN)
    print(f"  {len(curve.raw_v_nodes)} nodes ok, {len(curve.failed_nodes)} failed, {time.time()-t0:.1f}s")

    # --- 1. Noise-free bias/coverage across the benchmark's own V grid -----
    rows = []
    for v_target in V_GRID:
        try:
            target = distributions.target_effective_distribution(
                target_v=float(v_target), target_observed_rg=R0, dist_name=DISTRIBUTION,
                n=N_RADII, weight_power=WEIGHT_POWER,
            )
        except distributions.VarianceTargetUnreachable:
            continue
        sim = simulation.scatter2d(
            rg=target["requested_rg"], noise=0, v_rel=target["input_variance"], phi2=PHI2,
            det_pix=DET_PIX, sd_dist=SD_DIST, wavelength=WAVELENGTH, det_side=DET_SIDE,
            psf0=PSF0, dist_type=DISTRIBUTION, n_radii=N_RADII, weight_power=WEIGHT_POWER,
        )
        v_true = target["realized_v"]
        for name in OBSERVABLES:
            for route, override in [("analytic", None), ("calibration", {name: curve.curves[name]})]:
                r = protocol.tenor_protocol(
                    sim.intensity, sim.qx, sim.qy, PHI2, pxn=PXN, wavelength=WAVELENGTH,
                    observables=(name,), calibration_override=override,
                )
                v_est = r.v_estimates[name]
                rows.append(
                    dict(
                        Observable=name, Route=route, True_V=v_true,
                        Estimated_V=v_est, Valid=bool(np.isfinite(v_est)),
                    )
                )

    import pandas as pd

    noiseless_df = pd.DataFrame(rows)
    noiseless_df.to_csv(out_dir / "calibration_vs_analytic_noiseless.csv", index=False)

    print("\n--- Noise-free bias/coverage (bias in sqrt(V), valid = finite estimate) ---")
    for name in OBSERVABLES:
        for route in ("analytic", "calibration"):
            sub = noiseless_df[(noiseless_df.Observable == name) & (noiseless_df.Route == route)]
            valid = sub[sub.Valid]
            bias = float(np.mean(np.sqrt(np.clip(valid.Estimated_V, 0, None)) - np.sqrt(valid.True_V))) if len(valid) else float("nan")
            print(f"  {name:8s} {route:11s}: valid={len(valid)}/{len(sub)}  mean sqrt(V) bias={bias:+.4f}")

    fig, axes = plt.subplots(1, len(OBSERVABLES), figsize=(6 * len(OBSERVABLES), 4.5))
    if len(OBSERVABLES) == 1:
        axes = [axes]
    for ax, name in zip(axes, OBSERVABLES):
        for route, marker in (("analytic", "x"), ("calibration", "o")):
            sub = noiseless_df[(noiseless_df.Observable == name) & (noiseless_df.Route == route) & noiseless_df.Valid]
            ax.plot(np.sqrt(sub.True_V), np.sqrt(np.clip(sub.Estimated_V, 0, None)), marker, ms=6, label=route)
        v_line = np.linspace(0, np.sqrt(V_GRID.max()), 50)
        ax.plot(v_line, v_line, "k--", lw=1, alpha=0.5, label="perfect recovery")
        ax.set_xlabel(r"true $V^{1/2}$")
        ax.set_ylabel(r"recovered $V^{1/2}$")
        ax.set_title(f"{name}: analytic vs. calibration route")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_vs_analytic_noiseless.png", dpi=150)
    print(f"Saved {out_dir / 'calibration_vs_analytic_noiseless.png'}")

    # --- 2. A modest noisy comparison at two photon-flux levels -------------
    n_replicates = 10
    peak_photon_levels = [benchmark._default_peak_photons()[i] for i in (0, 3)]  # low and mid-high
    noisy_rows = []
    rng = np.random.default_rng(20260906)
    for v_target in [0.0025, 0.09, 0.25]:  # low/mid/high V, matching the earlier study's spot-checks
        target = distributions.target_effective_distribution(
            target_v=float(v_target), target_observed_rg=R0, dist_name=DISTRIBUTION,
            n=N_RADII, weight_power=WEIGHT_POWER,
        )
        sim = simulation.scatter2d(
            rg=target["requested_rg"], noise=0, v_rel=target["input_variance"], phi2=PHI2,
            det_pix=DET_PIX, sd_dist=SD_DIST, wavelength=WAVELENGTH, det_side=DET_SIDE,
            psf0=PSF0, dist_type=DISTRIBUTION, n_radii=N_RADII, weight_power=WEIGHT_POWER,
        )
        v_true = target["realized_v"]
        for peak_photons in peak_photon_levels:
            for rep in range(n_replicates):
                noisy = benchmark.add_noise(sim.intensity, float(peak_photons), seed=int(rng.integers(1, 2**31 - 1)))
                for name in OBSERVABLES:
                    for route, override in [("analytic", None), ("calibration", {name: curve.curves[name]})]:
                        r = protocol.tenor_protocol(
                            noisy, sim.qx, sim.qy, PHI2, pxn=PXN, wavelength=WAVELENGTH,
                            observables=(name,), calibration_override=override,
                        )
                        v_est = r.v_estimates[name]
                        noisy_rows.append(
                            dict(
                                Observable=name, Route=route, True_V=v_true, PeakPhotons=float(peak_photons),
                                Estimated_V=v_est, Valid=bool(np.isfinite(v_est)),
                            )
                        )

    noisy_df = pd.DataFrame(noisy_rows)
    noisy_df.to_csv(out_dir / "calibration_vs_analytic_noisy.csv", index=False)

    print("\n--- Noisy bias/coverage (small comparison, not the full 30-replicate benchmark) ---")
    for name in OBSERVABLES:
        for peak_photons in peak_photon_levels:
            for route in ("analytic", "calibration"):
                sub = noisy_df[
                    (noisy_df.Observable == name) & (noisy_df.Route == route) & (noisy_df.PeakPhotons == peak_photons)
                ]
                valid = sub[sub.Valid]
                bias = (
                    float(np.mean(np.sqrt(np.clip(valid.Estimated_V, 0, None)) - np.sqrt(valid.True_V)))
                    if len(valid)
                    else float("nan")
                )
                print(
                    f"  {name:8s} peak_photons={peak_photons:9.1f} {route:11s}: "
                    f"valid={len(valid)}/{len(sub)}  mean sqrt(V) bias={bias:+.4f}"
                )

    print(f"\nSaved {out_dir / 'calibration_vs_analytic_noiseless.csv'} and _noisy.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.out_dir)
