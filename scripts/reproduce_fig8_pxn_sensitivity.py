"""Reproduce paper Appendix C sensitivity study (Fig. sensitiVT2 / Yg_Pxn.eps):
sensitivity of the Yg100 observable to the TENOR-SAXS digital-smearing
pixel quartet (Pxn) used at analysis time.

Paper: "Variable-Resolution Scattering Reveals Ensemble Properties"
(Steinitz & Beck), Appendix "Degree of independence on ensemble and
instrumental parameters" (label sensitivityAnalisys). Tab.
beam_simul_param's "Smearing PSFs (pixels of 4-sigma kernel)" row gives
baseline [11x17, 3x5], inspected span [11x17,3x5] - [101x117,113x115]. The
paper's claim: "within the wide span of smearing kernel sizes, the effect
on the observables is indistinct" (Fig. sensitiVT2) -- i.e. Pxn is a
free analysis choice, not something the ground-truth calibration is
sensitive to.

Unlike the other three sensitivity scripts, Pxn only affects the ANALYSIS
(protocol.tenor_protocol), not the simulated detector image -- so per
(shape, V) pair we simulate ONE detector image and re-run tenor_protocol
on the same (intensity, qx, qy) for every Pxn quartet, which is very fast.

Usage:
    .venv/bin/python3 scripts/reproduce_fig8_pxn_sensitivity.py [--out-dir DIR]
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
R0 = 5.0
N_RADII = 11
DISTRIBUTION = "normal"
BASELINE_PXN = np.array([87, 85, 125, 123])

# Pxn quartets [nx1, ny1, nx2, ny2] spanning the paper's low end
# ([11x17, 3x5]-style) to its high end ([101x117, 113x115]-style), all
# positive odd integers, each pair differing as a set from the other and
# not internally symmetric (psf.validate_pxn's rules).
PXN_LIST = [
    np.array([3, 5, 11, 17]),
    np.array([9, 15, 21, 25]),
    np.array([25, 31, 45, 53]),
    np.array([87, 85, 125, 123]),
    np.array([101, 117, 113, 115]),
]
for _pxn in PXN_LIST:
    psf.validate_pxn(_pxn)

SHAPES = [
    ("spherical_shell", formfactors.GUINIER_TABLE["spherical_shell"].phi2, formfactors.GUINIER_TABLE["spherical_shell"].weight_power),
    ("solid_sphere", formfactors.GUINIER_TABLE["solid_sphere"].phi2, formfactors.GUINIER_TABLE["solid_sphere"].weight_power),
    ("thin_rod", formfactors.GUINIER_TABLE["thin_rod"].phi2, formfactors.GUINIER_TABLE["thin_rod"].weight_power),
    ("gaussian_chain", formfactors.GUINIER_TABLE["gaussian_chain"].phi2, formfactors.GUINIER_TABLE["gaussian_chain"].weight_power),
    ("thin_disk", 0.000666, formfactors.weight_power_for_name("thin_disk")),
]

V_VALUES = [0.10, 0.20]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(V_VALUES), figsize=(6 * len(V_VALUES), 4.5), sharey=False)
    pxn_mean = [float(np.mean(pxn)) for pxn in PXN_LIST]

    for name, phi2, weight_power in SHAPES:
        for v, ax in zip(V_VALUES, axes):
            # Simulate ONCE per (shape, V); Pxn only affects the analysis step.
            result = simulation.scatter2d(
                rg=R0, noise=0, v_rel=v, phi2=phi2, det_pix=DET_PIX, sd_dist=SD_DIST,
                wavelength=WAVELENGTH, det_side=DET_SIDE, psf0=PSF0,
                dist_type=DISTRIBUTION, n_radii=N_RADII, weight_power=weight_power,
            )
            yg100 = []
            for pxn in PXN_LIST:
                r = protocol.tenor_protocol(
                    result.intensity, result.qx, result.qy, phi2, pxn=pxn,
                    wavelength=WAVELENGTH, observables=("Yg100",),
                    # Matches the manuscript's stated cubic (3rd-order) G/M fit for
                    # these Appendix sensitivity figures (main-text/noise-benchmark
                    # results stay quadratic, the tenor_protocol default).
                    use_r3=True, use_g3=True,
                )
                yg100.append(r.observed["Yg100"])
            ax.plot(pxn_mean, yg100, "-o", ms=4, label=name)
        print(f"{name}: done ({len(PXN_LIST)} Pxn quartets x {len(V_VALUES)} V values)")

    baseline_mean = float(np.mean(BASELINE_PXN))
    for ax, v in zip(axes, V_VALUES):
        ax.set_xscale("log")
        ax.set_xlabel("Pxn quartet mean pixel count (proxy for kernel size)")
        ax.set_ylabel("Yg100")
        ax.set_title(f"V = {v:.2f}")
        ax.axvline(baseline_mean, color="gray", ls=":", lw=1, label="baseline [87,85,125,123]")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Sensitivity of Yg100 to the analysis-time smearing Pxn quartet "
        "(fixed simulated image per shape/V, R0=5nm, 11 normal members, no noise)"
    )
    fig.tight_layout()
    out_path = out_dir / "fig8_pxn_sensitivity.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.out_dir)
