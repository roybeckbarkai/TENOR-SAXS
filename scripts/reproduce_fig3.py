"""Reproduce paper Fig. 3: sensitivity of Y_G to distribution shape, in
weighted vs. unweighted variance.

Paper caption: "The sensitivity to distribution type, in weighted and
unweighted cases. Top left: Dependence of the Y_G observable on the
polydispersity for different ensemble distribution functions (solid
sphere's form-factor, R0=5nm). The horizontal span of the curves' braid
indicates the uncertainty of variance extraction on the distribution
function ... Right panels present the same data scaled by the scattering
strength weighted normalized variance, V."

Central claim being tested: Y_G should be a near-single-valued function of
the SCATTERING-weighted V regardless of distribution shape (a narrow
"braid" on the right/weighted panel), but a much broader, distribution-
shape-dependent braid when plotted against the NUMBER-weighted variance
(left/unweighted panel).

Usage:
    .venv/bin/python3 scripts/reproduce_fig3.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenor_saxs import distributions, formfactors, protocol, psf, simulation  # noqa: E402

SD_DIST = 360.0
WAVELENGTH = 0.1
DET_SIDE = 3.5
DET_PIX = 500
PSF0 = psf.bartlett2d(3, 15)
PXN = np.array([87, 85, 125, 123])
R0 = 5.0  # matches the manuscript's stated baseline and every other reproduce_fig*.py script
N_RADII = 11
PHI2 = formfactors.GUINIER_TABLE["solid_sphere"].phi2
WEIGHT_POWER = formfactors.GUINIER_TABLE["solid_sphere"].weight_power

DISTRIBUTIONS = ["normal", "lognormal", "uniform", "triangular", "boltzmann", "schulz"]
V_NUMBER_GRID = np.linspace(0.01, 0.30, 12)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    v_theory = np.linspace(0.0, 0.30, 200)
    theory_yg100 = protocol.analytical_theory(v_theory, PHI2)["Yg100"]
    axes[0].plot(np.sqrt(v_theory), theory_yg100, "k-", lw=2, label="theory (vs weighted V, for reference)", alpha=0.4)
    axes[1].plot(np.sqrt(v_theory), theory_yg100, "k-", lw=2, label="theory")

    for dist_name in DISTRIBUTIONS:
        v_number_used, v_weighted_used, yg100_vals = [], [], []
        for v_num in V_NUMBER_GRID:
            try:
                x, p = distributions.discretize_ensemble(N_RADII, v_num, dist_name, weight_power=WEIGHT_POWER)
                _, v_weighted, _ = distributions.scattering_weighted_stats(x, p, WEIGHT_POWER)
            except Exception:
                continue
            result = simulation.scatter2d(
                rg=R0, noise=0, v_rel=v_num, phi2=PHI2, det_pix=DET_PIX, sd_dist=SD_DIST,
                wavelength=WAVELENGTH, det_side=DET_SIDE, psf0=PSF0,
                dist_type=dist_name, n_radii=N_RADII, weight_power=WEIGHT_POWER,
            )
            r = protocol.tenor_protocol(
                result.intensity, result.qx, result.qy, PHI2, pxn=PXN,
                wavelength=WAVELENGTH, observables=("Yg100",),
            )
            v_number_used.append(v_num)
            v_weighted_used.append(v_weighted)
            yg100_vals.append(r.observed["Yg100"])

        axes[0].plot(np.sqrt(v_number_used), yg100_vals, "o--", ms=4, label=dist_name)
        axes[1].plot(np.sqrt(v_weighted_used), yg100_vals, "o--", ms=4, label=dist_name)
        print(f"{dist_name}: done ({len(v_number_used)} points)")

    axes[0].set_xlabel(r"$V_{number}^{1/2}$")
    axes[0].set_ylabel("Y_G")
    axes[0].set_title("vs. NUMBER-weighted V^1/2 (broad braid expected)")
    axes[1].set_xlabel(r"$V_{scattering}^{1/2}$")
    axes[1].set_ylabel("Y_G")
    axes[1].set_title("vs. SCATTERING-weighted V^1/2 (narrow braid expected)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Reproduction of paper Fig. 3: distribution-shape sensitivity (solid sphere, R0=3nm)")
    fig.tight_layout()
    out_path = out_dir / "fig3_reproduction.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.out_dir)
