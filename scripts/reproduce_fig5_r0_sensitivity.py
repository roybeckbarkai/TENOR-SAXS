"""Reproduce paper Appendix C sensitivity study (Fig. sensitiVT / Yg_Rg.eps):
sensitivity of the Yg100 observable to the ensemble's mean radius of
gyration R0.

Paper: "Variable-Resolution Scattering Reveals Ensemble Properties"
(Steinitz & Beck), Appendix "Degree of independence on ensemble and
instrumental parameters" (label sensitivityAnalisys). Baseline parameters
per Tab. beam_simul_param: Diamond B21-like instrument, 11 normally
distributed ensemble members, R0=5nm baseline, inspected span 2.5-7nm,
noise-free. The paper's claim: "the radius of gyration, R0, hardly plays a
role in the spherical and Guinier form-factors" (Fig. sensitiVT).

Usage:
    .venv/bin/python3 scripts/reproduce_fig5_r0_sensitivity.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenor_saxs import formfactors, protocol, psf, simulation  # noqa: E402

# Diamond B21-like instrument setup (init_TENOR_params.m defaults), matching
# reproduce_fig2.py's baseline.
SD_DIST = 360.0
WAVELENGTH = 0.1
DET_SIDE = 3.5
DET_PIX = 500
PSF0 = psf.bartlett2d(3, 15)
PXN = np.array([87, 85, 125, 123])
N_RADII = 11
DISTRIBUTION = "normal"

SHAPES = [
    ("spherical_shell", formfactors.GUINIER_TABLE["spherical_shell"].phi2, formfactors.GUINIER_TABLE["spherical_shell"].weight_power),
    ("solid_sphere", formfactors.GUINIER_TABLE["solid_sphere"].phi2, formfactors.GUINIER_TABLE["solid_sphere"].weight_power),
    ("thin_rod", formfactors.GUINIER_TABLE["thin_rod"].phi2, formfactors.GUINIER_TABLE["thin_rod"].weight_power),
    ("gaussian_chain", formfactors.GUINIER_TABLE["gaussian_chain"].phi2, formfactors.GUINIER_TABLE["gaussian_chain"].weight_power),
    ("thin_disk", 0.000666, formfactors.weight_power_for_name("thin_disk")),
]

# Table tab:beam_simul_param: "Mean radius of gyration: R0=5nm, inspected span 2.5-7nm".
R0_GRID = np.linspace(2.5, 7.0, 7)
V_VALUES = [0.10, 0.20]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(V_VALUES), figsize=(6 * len(V_VALUES), 4.5), sharey=False)

    for name, phi2, weight_power in SHAPES:
        for v, ax in zip(V_VALUES, axes):
            yg100 = []
            for r0 in R0_GRID:
                result = simulation.scatter2d(
                    rg=r0, noise=0, v_rel=v, phi2=phi2, det_pix=DET_PIX, sd_dist=SD_DIST,
                    wavelength=WAVELENGTH, det_side=DET_SIDE, psf0=PSF0,
                    dist_type=DISTRIBUTION, n_radii=N_RADII, weight_power=weight_power,
                )
                r = protocol.tenor_protocol(
                    result.intensity, result.qx, result.qy, phi2, pxn=PXN,
                    wavelength=WAVELENGTH, observables=("Yg100",),
                    # Matches the manuscript's stated cubic (3rd-order) G/M fit for
                    # these Appendix sensitivity figures (main-text/noise-benchmark
                    # results stay quadratic, the tenor_protocol default).
                    use_r3=True, use_g3=True,
                )
                yg100.append(r.observed["Yg100"])
            ax.plot(R0_GRID, yg100, "-o", ms=4, label=name)
        print(f"{name}: done ({len(R0_GRID)} R0 values x {len(V_VALUES)} V values)")

    for ax, v in zip(axes, V_VALUES):
        ax.set_xlabel("R0 (nm)")
        ax.set_ylabel("Yg100")
        ax.set_title(f"V = {v:.2f}")
        ax.axvline(5.0, color="gray", ls=":", lw=1, label="baseline R0=5nm")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Sensitivity of Yg100 to ensemble mean radius of gyration R0 (baseline R0=5nm, 11 normal members, no noise)")
    fig.tight_layout()
    out_path = out_dir / "fig5_r0_sensitivity.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.out_dir)
