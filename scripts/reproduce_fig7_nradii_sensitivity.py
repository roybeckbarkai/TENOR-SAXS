"""Reproduce paper Appendix C sensitivity study (Fig. sensitiVT1 / Yg_N_rg.eps):
sensitivity of the Yg100 observable to the ensemble discretization
resolution (number of discrete radii used to represent the ensemble).

Paper: "Variable-Resolution Scattering Reveals Ensemble Properties"
(Steinitz & Beck), Appendix "Degree of independence on ensemble and
instrumental parameters" (label sensitivityAnalisys). Tab.
beam_simul_param's "Number of discrete radius values used to represent the
ensemble" row gives baseline 11, inspected span 11-61. The paper's claim:
"the number of different radii (distribution resolution) does not affect
the method's performance" (Fig. sensitiVT1).

Performance note: simulating the thin_disk shape uses its exact form
factor (scipy.special.jv), which is measured to cost ~0.14s per radius per
scatter2d call (vs. a small fraction of that for the other four shapes,
whose exact/Guinier form factors are much cheaper). To keep this script's
total runtime in the few-minutes range, thin_disk's n_radii sweep is
CAPPED at 41 (instead of the paper's full 61) -- timed beforehand at ~5.8s
per scatter2d call at n_radii=41 vs. ~8.5s at n_radii=61; capping at 41
keeps the thin_disk portion of the sweep comfortably fast while still
showing the qualitative trend across most of the paper's inspected span.
The other four shapes use the full 11-61 span.

Usage:
    .venv/bin/python3 scripts/reproduce_fig7_nradii_sensitivity.py [--out-dir DIR]
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
R0 = 5.0
DISTRIBUTION = "normal"

# Paper's full inspected span: 11-61. thin_disk is capped at 41 -- see
# module docstring for the timing rationale.
N_RADII_GRID_FULL = [11, 21, 31, 41, 51, 61]
N_RADII_GRID_THIN_DISK = [11, 21, 31, 41]

SHAPES = [
    ("spherical_shell", formfactors.GUINIER_TABLE["spherical_shell"].phi2, formfactors.GUINIER_TABLE["spherical_shell"].weight_power, N_RADII_GRID_FULL),
    ("solid_sphere", formfactors.GUINIER_TABLE["solid_sphere"].phi2, formfactors.GUINIER_TABLE["solid_sphere"].weight_power, N_RADII_GRID_FULL),
    ("thin_rod", formfactors.GUINIER_TABLE["thin_rod"].phi2, formfactors.GUINIER_TABLE["thin_rod"].weight_power, N_RADII_GRID_FULL),
    ("gaussian_chain", formfactors.GUINIER_TABLE["gaussian_chain"].phi2, formfactors.GUINIER_TABLE["gaussian_chain"].weight_power, N_RADII_GRID_FULL),
    ("thin_disk", 0.000666, formfactors.weight_power_for_name("thin_disk"), N_RADII_GRID_THIN_DISK),
]

V_VALUES = [0.10, 0.20]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(V_VALUES), figsize=(6 * len(V_VALUES), 4.5), sharey=False)

    for name, phi2, weight_power, n_radii_grid in SHAPES:
        for v, ax in zip(V_VALUES, axes):
            yg100 = []
            for n_radii in n_radii_grid:
                result = simulation.scatter2d(
                    rg=R0, noise=0, v_rel=v, phi2=phi2, det_pix=DET_PIX, sd_dist=SD_DIST,
                    wavelength=WAVELENGTH, det_side=DET_SIDE, psf0=PSF0,
                    dist_type=DISTRIBUTION, n_radii=n_radii, weight_power=weight_power,
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
            label = name if name != "thin_disk" else "thin_disk (capped at 41)"
            ax.plot(n_radii_grid, yg100, "-o", ms=4, label=label)
        print(f"{name}: done ({len(n_radii_grid)} n_radii values x {len(V_VALUES)} V values)")

    for ax, v in zip(axes, V_VALUES):
        ax.set_xlabel("Number of discrete radii (n_radii)")
        ax.set_ylabel("Yg100")
        ax.set_title(f"V = {v:.2f}")
        ax.axvline(11, color="gray", ls=":", lw=1, label="baseline n_radii=11")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Sensitivity of Yg100 to ensemble discretization resolution "
        "(R0=5nm, normal, no noise; thin_disk sweep capped at 41 -- see script docstring)"
    )
    fig.tight_layout()
    out_path = out_dir / "fig7_nradii_sensitivity.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.out_dir)
