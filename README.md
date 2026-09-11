# TENOR-SAXS

Python port and validation of **TENOR-SAXS** (Technique for ENsemble
Observation by Resolution variation in SAXS), the variable-resolution
scattering method introduced in *"Variable-Resolution Scattering Reveals
Ensemble Properties"* (Steinitz & Beck).

The method recovers a nanoparticle ensemble's polydispersity (the
scattering-weighted relative variance of the radius of gyration, `V`)
directly from a single 2D SAXS image: digitally smear the image with two
different anisotropic-Gaussian point-spread functions, fit the log-ratio of
the two smeared images to a low-order polynomial, and invert the result
against a closed-form (or simulated) calibration curve — without needing to
know the instrument's native resolution function.

This is a **clean-room, independent re-derivation** from the paper and its
MATLAB reference implementation (`matlab/`), validated against both
directly (see the "Validation summary" section below).

**Try it interactively, no install required:**
[Open the demo notebook in Google Colab](https://colab.research.google.com/drive/1JyB62BvTwV7SjoPmVrqMNHJUzpvn9Hoj?usp=sharing)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Quickstart

```python
import numpy as np
from tenor_saxs import psf, simulation, protocol

# Simulate a polydisperse Gaussian-chain ensemble (R0=3nm, V=0.05, normal distribution).
result = simulation.scatter2d(
    rg=3.0, noise=-1e4, v_rel=0.05, phi2=1/18,
    det_pix=500, sd_dist=360.0, wavelength=0.1, det_side=3.5,
    psf0=psf.bartlett2d(3, 15), dist_type="normal", n_radii=25,
)

# Recover V from the simulated image.
r = protocol.tenor_protocol(result.intensity, result.qx, result.qy, phi2=1/18)
print(f"True V=0.05, recovered V={r.best_v:.4f} +/- {r.best_v_se:.4f}, Rg={r.rg:.3f}")
```

## Repository layout

```
tenor_saxs/    the Python package (primary implementation)
matlab/           the original MATLAB reference implementation, plus an
                  Octave export harness used to validate this Python port
                  numerically against it
scripts/          CLI entry points and paper-figure reproductions
tests/            pytest suite
```

### `tenor_saxs/` — Python package

- `formfactors.py` — canonical monodisperse form factors (spherical shell,
  solid sphere, thin rod, Gaussian chain; exact + Guinier expansion),
  Table 1 curvatures.
- `distributions.py` — ensemble discretization and the scattering-weighted
  variance targeting search.
- `psf.py` — digital smearing kernels and PSF convolution, including a full
  port of `filter2_ungridded` for curved/non-uniform q-grids (large/
  wide-angle detectors).
- `simulation.py` — forward 2D SAXS image simulation.
- `mg_extract.py` — the core log-ratio polynomial fit engine.
- `protocol.py` — observable construction, analytical V-inversion,
  multi-observable combination.
- `calibration.py` — simulation-derived calibration curves for `Yg210`/
  `Ym210`, an alternative to the analytic inversion (drop-in via
  `tenor_protocol(..., calibration_override=...)`).
- `benchmark.py` — clean-database caching + noise-sweep benchmark harness.
- `plotting.py` — violin-plot discrepancy visualization.

### `matlab/` — reference implementation

The original MATLAB source this package is a Python port of, plus
`run_reference_export.m`, an Octave-compatible export harness used to
generate the numeric reference data this port was validated against.

### `scripts/`

- `run_benchmark.py` — general-purpose CLI for the clean-database +
  noise-sweep benchmark harness.
- `reproduce_fig2.py` — theory vs. simulation, across the 5 canonical form
  factors and a range of ensemble variances.
- `reproduce_fig3.py` — sensitivity of the recovered variance to the
  ensemble's assumed distribution shape (weighted vs. unweighted).
- `reproduce_fig5_r0_sensitivity.py`, `reproduce_fig6_psf0_sensitivity.py`,
  `reproduce_fig7_nradii_sensitivity.py`, `reproduce_fig8_pxn_sensitivity.py`
  — the paper's Appendix C sensitivity studies (mean radius of gyration,
  instrument point-spread function size, ensemble discretization
  resolution, and digital-smearing kernel size, respectively).
- `reproduce_fig6_noise_violin.py` — the noise-sensitivity violin plot
  (recovered variance/radius-of-gyration discrepancy vs. photon flux).
- `reproduce_calibration_vs_analytic.py` — the simulated-calibration-vs-
  analytic inversion comparison for Yg210/Ym210.

## Validation summary

Validated against the vendored MATLAB reference code (run under GNU
Octave) and, separately, against a real MATLAB benchmark run:

- Simulated clean images match Octave to ~1e-12 (including exact NaN
  pattern agreement) and a real MATLAB run to <1%.
- The full analysis pipeline's recovered `Rg` matches MATLAB to ~4e-8, and
  the final combined `V` estimate to ~1e-4, across a 144-case grid spanning
  4 form factors × 8 variances × 3 distributions.
- A from-scratch rerun of the full noise benchmark (2,520 analyses) matches
  the manuscript's own reported per-photon-density bias/RMS/valid-fraction
  numbers to within Monte-Carlo noise at every level.
- The simulation-derived calibration route for Yg210/Ym210 (see
  `calibration.py`) independently confirms the manuscript's claimed bias
  reduction over the analytic inversion, and covers V values the analytic
  route cannot invert at all.
- 101/101 unit and parity tests pass.

## Reproducing the paper's figures

```bash
.venv/bin/python3 scripts/reproduce_fig2.py                     # theory vs. simulation, 5 form factors
.venv/bin/python3 scripts/reproduce_fig3.py                     # distribution-shape sensitivity
.venv/bin/python3 scripts/reproduce_fig5_r0_sensitivity.py       # sensitivity to mean Rg
.venv/bin/python3 scripts/reproduce_fig6_psf0_sensitivity.py     # sensitivity to instrument PSF size
.venv/bin/python3 scripts/reproduce_fig6_noise_violin.py         # noise-sensitivity violin plot
.venv/bin/python3 scripts/reproduce_fig7_nradii_sensitivity.py   # sensitivity to ensemble discretization
.venv/bin/python3 scripts/reproduce_fig8_pxn_sensitivity.py      # sensitivity to smearing-kernel size
.venv/bin/python3 scripts/reproduce_calibration_vs_analytic.py   # calibration vs. analytic inversion, Yg210/Ym210
```

Figures are written to `../tenor-saxs-v2-data/figures/` (outside the repo,
keeping generated data out of git).

## Running the test suite

```bash
.venv/bin/python3 -m pytest tests/ -v
```

MATLAB-parity tests skip gracefully if the Octave-exported reference data
(`../tenor-saxs-v2-data/octave-reference/`) isn't present — regenerate it
with `octave --no-gui --eval "run_reference_export"` from `matlab/`.

## Provenance

Independent from, and cross-checked against, a pre-existing package (from
the same research group) implementing an earlier, empirically-calibrated
version of a closely related method.

## License

MIT — see [`LICENSE`](LICENSE).
