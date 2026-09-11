"""Reproduce paper Appendix C sensitivity study (Fig. sensitiVT3 / Yg_PSF0.eps):
sensitivity of the Yg100 observable to the instrumental point-spread
function (beam divergence / slit) size.

Paper: "Variable-Resolution Scattering Reveals Ensemble Properties"
(Steinitz & Beck), Appendix "Degree of independence on ensemble and
instrumental parameters" (label sensitivityAnalisys). Tab.
beam_simul_param's "Beam divergence (slit size in pixels)" row gives
baseline 3x15 and inspected span "3x3 - 45x25"; we also probe a near-delta
(1,1) kernel below that span and the MATLAB reference's own commented-out
"lab saxs" example bartlett2d(25,25) (matlab/init_TENOR_params.m)
to cover "below typical synchrotron beam sizes to above laboratory X-ray
setups". The paper's claim: "the instrumental PSF size hardly affects the
observables" (Fig. sensitiVT3) -- this is the method's headline robustness
claim (the digital-smearing-comparison method is largely insensitive to the
native instrument PSF, unlike a naive single-kernel deconvolution would be).

Usage:
    .venv/bin/python3 scripts/reproduce_fig6_psf0_sensitivity.py [--out-dir DIR]
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
PXN = np.array([87, 85, 125, 123])
R0 = 5.0
N_RADII = 11
DISTRIBUTION = "normal"

SHAPES = [
    ("spherical_shell", formfactors.GUINIER_TABLE["spherical_shell"].phi2, formfactors.GUINIER_TABLE["spherical_shell"].weight_power),
    ("solid_sphere", formfactors.GUINIER_TABLE["solid_sphere"].phi2, formfactors.GUINIER_TABLE["solid_sphere"].weight_power),
    ("thin_rod", formfactors.GUINIER_TABLE["thin_rod"].phi2, formfactors.GUINIER_TABLE["thin_rod"].weight_power),
    ("gaussian_chain", formfactors.GUINIER_TABLE["gaussian_chain"].phi2, formfactors.GUINIER_TABLE["gaussian_chain"].weight_power),
    ("thin_disk", 0.000666, formfactors.weight_power_for_name("thin_disk")),
]

# (n, m) kernel footprints for psf.bartlett2d, spanning below-synchrotron
# near-delta up to the matlab reference's "lab saxs" 25x25 example and the
# table's high end 45x25. Baseline is (3, 15).
PSF_SHAPES = [(1, 1), (3, 3), (3, 15), (9, 9), (15, 15), (25, 25), (45, 25)]
BASELINE_PSF_SHAPE = (3, 15)
V_VALUES = [0.10, 0.20]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(V_VALUES), figsize=(6 * len(V_VALUES), 4.5), sharey=False)
    footprints = [n * m for n, m in PSF_SHAPES]

    for name, phi2, weight_power in SHAPES:
        for v, ax in zip(V_VALUES, axes):
            yg100 = []
            for n, m in PSF_SHAPES:
                psf0 = psf.bartlett2d(n, m)
                result = simulation.scatter2d(
                    rg=R0, noise=0, v_rel=v, phi2=phi2, det_pix=DET_PIX, sd_dist=SD_DIST,
                    wavelength=WAVELENGTH, det_side=DET_SIDE, psf0=psf0,
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
            ax.plot(footprints, yg100, "-o", ms=4, label=name)
        print(f"{name}: done ({len(PSF_SHAPES)} PSF footprints x {len(V_VALUES)} V values)")

    baseline_footprint = BASELINE_PSF_SHAPE[0] * BASELINE_PSF_SHAPE[1]
    for ax, v in zip(axes, V_VALUES):
        ax.set_xscale("log")
        ax.set_xlabel("Instrument PSF footprint (n x m pixels)")
        ax.set_ylabel("Yg100")
        ax.set_title(f"V = {v:.2f}")
        ax.axvline(baseline_footprint, color="gray", ls=":", lw=1, label="baseline 3x15")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Sensitivity of Yg100 to instrumental PSF size "
        "(bartlett2d(n,m), R0=5nm, 11 normal members, no noise)"
    )
    fig.tight_layout()
    out_path = out_dir / "fig6_psf0_sensitivity.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.out_dir)
