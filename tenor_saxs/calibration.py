"""Simulation-derived calibration curves for Y_G2 / Y_M.

The manuscript prescribes a simulation-derived calibration curve as the
recommended inversion route for Y_G2 (``Yg210``) and Y_M (``Ym210``): their
first-order analytic formulas (:func:`tenor_saxs.protocol.analytical_theory`)
are measurably biased at this package's own default geometry (analytic bias
-0.085 / -0.119 on a noise-free grid, vs. +0.016 / +0.019 for the simulated
calibration route), and ``Yg210``'s analytic curve turns over at ``V=0.273``,
capping its usable (monotonic-branch) coverage at 82% no matter how much
flux is collected -- see the package's internal validation notes. The package previously had no
calibration route at all: every observable was inverted only against
``analytical_theory()``.

This module builds that calibration curve by actually simulating the known
form-factor over a grid of ``V`` values at the ensemble's own apparent
radius/weighting power, extracting each observable through the SAME
simulate -> smear -> fit pipeline used on a real image
(:func:`tenor_saxs.simulation.scatter2d` -> :func:`tenor_saxs.mg_extract.mg_extract`),
and fitting a smooth low-degree polynomial through the (noise-free)
simulated nodes -- exactly the recipe the manuscript describes ("one may
simulate the ideal scattering for various V to find the relevant
coefficients ... used to produce a calibration curve").

:func:`build_calibration_curve` returns a :class:`CalibrationCurve` whose
``.curves`` has the same keys/shape as
:func:`tenor_saxs.protocol.analytical_theory`, so it is a drop-in
replacement anywhere that dict is used, e.g.
``tenor_protocol(..., calibration_override=curve.curves)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import psf as psf_module
from .distributions import target_effective_distribution
from .simulation import scatter2d

__all__ = ["CalibrationCurve", "build_calibration_curve"]

_OBSERVABLE_NAMES = ("Yg100", "Yg210", "Ym210", "Jg10", "Jg21", "Jm")

# Per an internal design review's own evidence: "a degree-5 fit to 33 noise-free
# nodes (RMS residual <0.0014)" -- the raw look-up has a small
# non-monotonic discretizer wiggle near V~=0.28 that a smooth fit removes.
DEFAULT_N_NODES = 33
DEFAULT_DEGREE = 5
DEFAULT_V_NODE_RANGE = (1.0e-4, 0.32)


@dataclass(slots=True)
class CalibrationCurve:
    """A simulated ``(V, observable)`` calibration table, polynomial-smoothed.

    ``v_grid``/``curves`` are the SMOOTHED curves evaluated on a dense grid
    -- suitable to hand directly to
    ``tenor_protocol(..., calibration_override=curve.curves)`` (see that
    function's docstring for the grid-alignment requirement) or to
    :func:`tenor_saxs.protocol.invert_lookup` directly. ``raw_v_nodes``/
    ``raw_observed`` are the actual simulated data points before smoothing,
    kept for diagnostics/plots. ``failed_nodes`` lists the requested ``V``
    values that could not be simulated (e.g. an unreachable discretizer
    target), with the reason.
    """

    v_grid: np.ndarray
    curves: dict[str, np.ndarray]
    raw_v_nodes: np.ndarray
    raw_observed: dict[str, np.ndarray]
    degree: int
    failed_nodes: list[tuple[float, str]]


def build_calibration_curve(
    r0: float,
    phi2: float,
    weight_power: float = 0.0,
    v_nodes: np.ndarray | None = None,
    pxn: np.ndarray = np.array([87, 85, 125, 123]),
    sd_dist: float = 360.0,
    wavelength: float = 0.1,
    det_side: float = 3.5,
    det_pix: int = 500,
    psf0: np.ndarray | None = None,
    signum: float = 4.0,
    use_r3: bool = False,
    use_g3: bool = False,
    weight_mode: str = "intensity",
    qrg_max: float | None = None,
    deadpix_factor: float | None = None,
    distribution: str = "normal",
    n_radii: int = 25,
    p_minimum: float = 1e-5,
    p_maximum: float = 5.0,
    variance_tolerance: float = 2e-4,
    max_iterations: int = 50,
    generator_coverage: float = 0.995,
    apply_weight_again: bool = True,
    expansion_factor: float = 1.6,
    degree: int = DEFAULT_DEGREE,
    v_grid_range: tuple[float, float] = (-0.05, 0.35),
    v_grid_n: int = 4001,
) -> CalibrationCurve:
    """Simulate a noise-free ``V``-calibration curve for all six observables.

    ``v_grid_range``/``v_grid_n`` default to :func:`tenor_saxs.protocol.tenor_protocol`'s
    own ``v_range``/``v_grid_n`` defaults, so the common case (both left at
    default) produces a curve directly usable as
    ``tenor_protocol(..., calibration_override=curve.curves)`` with no
    further alignment step.

    ``r0``/``phi2``/``weight_power`` should match the real ensemble the
    calibration curve is meant to stand in for (the apparent/target radius,
    the form-factor curvature, and the scattering-strength weighting
    power); ``pxn``/instrument geometry/window/fit-order arguments should
    match the analysis being calibrated. Per the manuscript's own finding
    (Fig. Yg_Pxn / ``fig:sensitiVT2``), the calibration curve is
    insensitive to which smearing quartet is used to generate it, so the
    default ``pxn`` need not match the analysis quartet exactly -- but
    matching it removes that as a variable.
    """
    from . import mg_extract as mg_extract_module
    from .protocol import _six_observables_from_mg_result

    if psf0 is None:
        psf0 = psf_module.bartlett2d(3, 15)
    if v_nodes is None:
        v_nodes = np.linspace(*DEFAULT_V_NODE_RANGE, DEFAULT_N_NODES)
    v_nodes = np.asarray(v_nodes, dtype=float)

    raw_v: list[float] = []
    raw_observed: dict[str, list[float]] = {name: [] for name in _OBSERVABLE_NAMES}
    failed_nodes: list[tuple[float, str]] = []

    for v_target in v_nodes:
        try:
            target = target_effective_distribution(
                target_v=float(v_target),
                target_observed_rg=r0,
                dist_name=distribution,
                n=n_radii,
                weight_power=weight_power,
                p_minimum=p_minimum,
                p_maximum=p_maximum,
                variance_tolerance=variance_tolerance,
                max_iterations=max_iterations,
                generator_coverage=generator_coverage,
                apply_weight_again=apply_weight_again,
                expansion_factor=expansion_factor,
            )
            result = scatter2d(
                rg=target["requested_rg"],
                noise=0,
                v_rel=target["input_variance"],
                phi2=phi2,
                det_pix=det_pix,
                sd_dist=sd_dist,
                wavelength=wavelength,
                det_side=det_side,
                psf0=psf0,
                dist_type=distribution,
                n_radii=n_radii,
                weight_power=weight_power,
            )
            mg_result = mg_extract_module.mg_extract(
                pxn,
                result.qx,
                result.qy,
                result.intensity,
                signum=signum,
                rg2=None,
                use_r3=use_r3,
                use_g3=use_g3,
                weight_mode=weight_mode,
                wavelength=wavelength,
                **({"qrg_max": qrg_max} if qrg_max is not None else {}),
                **({"deadpix_factor": deadpix_factor} if deadpix_factor is not None else {}),
            )
            observed, _sigma0, _delta0 = _six_observables_from_mg_result(mg_result, use_g3)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: record and continue
            # (VarianceTargetUnreachable is the expected failure mode near
            # the achievable-V ceiling for a given distribution/weight_power;
            # anything else surfaces in failed_nodes for the caller to see.)
            failed_nodes.append((float(v_target), repr(exc)))
            continue

        raw_v.append(float(target["realized_v"]))
        for name in _OBSERVABLE_NAMES:
            raw_observed[name].append(float(observed[name]))

    if len(raw_v) < degree + 1:
        raise RuntimeError(
            f"only {len(raw_v)}/{len(v_nodes)} calibration nodes succeeded, need at "
            f"least {degree + 1} for a degree-{degree} fit. First failures: "
            f"{failed_nodes[:3]!r}"
        )

    raw_v_arr = np.array(raw_v, dtype=float)
    v_grid = np.linspace(v_grid_range[0], v_grid_range[1], v_grid_n)
    curves: dict[str, np.ndarray] = {}
    raw_observed_arr: dict[str, np.ndarray] = {}
    for name in _OBSERVABLE_NAMES:
        y = np.array(raw_observed[name], dtype=float)
        raw_observed_arr[name] = y
        coeffs = np.polyfit(raw_v_arr, y, degree)
        curves[name] = np.polyval(coeffs, v_grid)

    return CalibrationCurve(
        v_grid=v_grid,
        curves=curves,
        raw_v_nodes=raw_v_arr,
        raw_observed=raw_observed_arr,
        degree=degree,
        failed_nodes=failed_nodes,
    )
