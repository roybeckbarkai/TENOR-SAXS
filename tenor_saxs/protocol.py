"""TENOR-SAXS protocol orchestration and V-inversion layer.

Clean-room Python port of ``TENOR_protocol.m`` (see
``TENOR-SAXS/Update_Sep2_2026/TENOR_protocol.m`` in the sibling repository for
the authoritative MATLAB reference; line numbers cited below refer to that
file).

This module takes the polynomial-fit output produced by
:mod:`tenor_saxs.mg_extract` for a single 2D detector image and:

1. Builds the six physically-named "observables" (``Yg100``, ``Yg210``,
   ``Ym210``, ``Jg10``, ``Jg21``, ``Jm``) from the fitted coefficients and the
   PSF-difference scale factor ``AG``.
2. Propagates the polynomial-fit covariance to each observable via the
   delta method (:func:`observable_gradients`).
3. Inverts each observable against an analytical calibration curve
   (:func:`analytical_theory`) over a ``V`` grid to recover a per-observable
   estimate of the ensemble's scattering-weighted relative variance ``V``
   (:func:`invert_lookup`).
4. Combines the individual per-observable ``V`` estimates into one final
   answer (:func:`combine_estimates`).

IMPORTANT: ``V`` is the SCATTERING-WEIGHTED relative variance,
``V = Var_w(Rg) / mean_w(Rg)^2`` -- not generally the number-weighted
variance. Values ``V < 0`` are permitted only as modest analytical
extrapolation diagnostics and are not physical negative variances (mirrors
the MATLAB docstring, lines 18-25).

Coefficient layout in ``p`` / ``cov_p`` (matches ``mg_extract.py``):
index 0 = g0, 1 = g1, 2 = g2, [3 = g3 if ``use_g3``], then
``m_offset = 3 + int(use_g3)``: index ``m_offset`` = m1, ``m_offset + 1`` =
m2, [``m_offset + 2`` = m3 if ``use_r3``]. There is no m0.

Sign/PSF convention (MATLAB lines 76-88, 138-144): ``mg_extract`` fits
``log(F_wide / F_narrow)`` (kernel 2 over kernel 1), so the PSF-difference
scale factor uses PSF2 minus PSF1, i.e. "wide minus narrow":
``d_sx2 = sigma_x2**2 - sigma_x1**2``, ``d_sy2 = sigma_y2**2 - sigma_y1**2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from .mg_extract import DEADPIX_FACTOR as _DEADPIX_FACTOR_DEFAULT
from .mg_extract import QRG_MAX as _QRG_MAX_DEFAULT

# Re-exported so callers can do `protocol.QRG_MAX` / `protocol.DEADPIX_FACTOR`
# without reaching into mg_extract directly. mg_extract does not import
# protocol, so this is not circular.
QRG_MAX = _QRG_MAX_DEFAULT
DEADPIX_FACTOR = _DEADPIX_FACTOR_DEFAULT

__all__ = [
    "InvertResult",
    "TenorProtocolResult",
    "analytical_theory",
    "observable_gradients",
    "y_t_gradient",
    "invert_lookup",
    "combine_estimates",
    "tenor_protocol",
    "OBSERVABLE_NAMES",
    "OBSERVABLE_NAMES_WITH_YT",
    "QRG_MAX",
    "DEADPIX_FACTOR",
]

# The six MATLAB-backed observables (default set, unchanged from before Y_T
# existed, so existing callers' behavior is untouched by its addition).
_OBSERVABLE_NAMES = ("Yg100", "Yg210", "Ym210", "Jg10", "Jg21", "Jm")
OBSERVABLE_NAMES = _OBSERVABLE_NAMES
# Convenience superset including Y_T (opt-in: pass this explicitly as
# `observables=protocol.OBSERVABLE_NAMES_WITH_YT`, e.g. for the
# Y_T-vs-Yg100 comparison study). Y_T is NOT in the default set because it
# requires an extra fit (a separate single-image quadratic Guinier fit,
# mg_extract.fit_quadratic_weighted_centered) on top of the six
# mg_extract-derived observables, which existing callers didn't ask for.
OBSERVABLE_NAMES_WITH_YT = _OBSERVABLE_NAMES + ("Y_T",)

# Statuses for which invert_lookup does not produce a usable (V, slope) pair.
_UNUSABLE_STATUSES = (
    "outside calibration range or no monotonic branch",
    "multiple analytical roots",
)


# ---------------------------------------------------------------------------
# 1. Analytical calibration curves (TENOR_protocol.m: analytical_theory,
#    lines ~341-354)
# ---------------------------------------------------------------------------


def analytical_theory(v, phi2) -> dict[str, Any]:
    """Table-2 closed-form analytical observable-vs-V relations.

    Works elementwise for ``v`` as a NumPy array or a plain scalar; ``phi2``
    is the scalar monodisperse form-factor curvature (``nu`` / ``phi''`` in
    the MATLAB source).

    Returns a dict with keys ``A, NG, B, NM, Jg10, Jg21, Jm, Yg100, Yg210,
    Ym210`` in whatever type ``v`` was (array in, array out).
    """
    a = 1 + 18 * phi2 + 10 * v + 108 * v * phi2
    ng = 18 * phi2 + 8 * v + 450 * v * phi2
    b = 1 + 9 * phi2 + 6 * v + 54 * v * phi2
    nm = 18 * phi2 + 8 * v + 342 * v * phi2

    jg10 = a / (-3 * (1 + v) ** 2)
    jg21 = ng / (-3 * (1 + v) * a)
    jm = nm / (-3 * (1 + v) * b)

    yg100 = a / (4 * (1 + v) ** 2)
    yg210 = ng / (4 * (1 + v) * a)
    ym210 = nm / (4 * (1 + v) * b)

    # Y_T = t2/t1**2 from a direct single-image quadratic Guinier fit
    # (ln I(q) ~= t0 + t1*q**2 + t2*q**4), paper Eq. "Y_T definition"
    # (main text, following Eq. lnI_dimless_). Has NO MATLAB reference
    # implementation at all (the internal validation notes item 2) -- included here as
    # a from-scratch addition alongside the six MATLAB-backed observables.
    y_t = (9 * phi2 + 4 * v + 54 * v * phi2) / (2 * (1 + v) ** 2)

    return {
        "A": a,
        "NG": ng,
        "B": b,
        "NM": nm,
        "Jg10": jg10,
        "Jg21": jg21,
        "Jm": jm,
        "Yg100": yg100,
        "Yg210": yg210,
        "Ym210": ym210,
        "Y_T": y_t,
    }


# ---------------------------------------------------------------------------
# 2. Delta-method gradients (TENOR_protocol.m: observable_gradients,
#    lines ~356-404)
# ---------------------------------------------------------------------------


def observable_gradients(p, rg2: float, ag: float, use_g3: bool) -> dict[str, np.ndarray]:
    """Analytic gradients of the six observables w.r.t. the coefficient vector ``p``.

    Each returned gradient is a length-``len(p)`` array, zero everywhere
    except at the 2-3 coefficient entries the observable actually depends
    on. Intended for delta-method SE propagation:
    ``SE = sqrt(max(grad @ cov_p @ grad, 0))``.
    """
    n = len(p)
    g0, g1, g2 = p[0], p[1], p[2]
    m_offset = 3 + int(use_g3)
    m1, m2 = p[m_offset], p[m_offset + 1]

    grads: dict[str, np.ndarray] = {}

    # Jg10 = (g1/g0)/rg2
    d = np.zeros(n)
    d[0] = -g1 / (g0 ** 2 * rg2)
    d[1] = 1.0 / (g0 * rg2)
    grads["Jg10"] = d

    # Jg21 = (g2/g1)/rg2
    d = np.zeros(n)
    d[1] = -g2 / (g1 ** 2 * rg2)
    d[2] = 1.0 / (g1 * rg2)
    grads["Jg21"] = d

    # Jm = (m2/m1)/rg2
    d = np.zeros(n)
    d[m_offset] = -m2 / (m1 ** 2 * rg2)
    d[m_offset + 1] = 1.0 / (m1 * rg2)
    grads["Jm"] = d

    # Yg100 = ag*g1/g0**2
    d = np.zeros(n)
    d[0] = -2 * ag * g1 / g0 ** 3
    d[1] = ag / g0 ** 2
    grads["Yg100"] = d

    # Yg210 = ag*g2/(g1*g0)
    d = np.zeros(n)
    d[0] = -ag * g2 / (g1 * g0 ** 2)
    d[1] = -ag * g2 / (g1 ** 2 * g0)
    d[2] = ag / (g1 * g0)
    grads["Yg210"] = d

    # Ym210 = ag*m2/(m1*g0)
    d = np.zeros(n)
    d[0] = -ag * m2 / (m1 * g0 ** 2)
    d[m_offset] = -ag * m2 / (g0 * m1 ** 2)
    d[m_offset + 1] = ag / (g0 * m1)
    grads["Ym210"] = d

    return grads


def y_t_gradient(t1: float, t2: float) -> np.ndarray:
    """Gradient of ``Y_T = t2/t1**2`` w.r.t. its OWN fit coefficients ``[t0, t1, t2]``.

    Unlike the other six observables (which all depend on
    ``mg_extract``'s log-ratio coefficient vector ``p``), ``Y_T`` comes
    from a separate single-image fit
    (:func:`tenor_saxs.mg_extract.fit_quadratic_weighted_centered`), so
    it needs its own small gradient rather than an entry in
    :func:`observable_gradients`. ``d/dt0=0`` (Y_T doesn't depend on the
    intercept); ``d/dt1 = -2*t2/t1**3``; ``d/dt2 = 1/t1**2``.
    """
    return np.array([0.0, -2.0 * t2 / t1**3, 1.0 / t1**2])


# ---------------------------------------------------------------------------
# 3. Analytical-curve inversion (TENOR_protocol.m: invert_lookup,
#    lines ~406-466)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InvertResult:
    v_estimate: float
    local_slope: float
    status: str  # "ok" | "calibration locally flat" |
    # "outside calibration range or no monotonic branch" |
    # "multiple analytical roots"


def invert_lookup(v_grid, y_grid, y_observed, min_slope: float = 1e-8) -> InvertResult:
    """Invert one observable's calibration curve against an observed value.

    Splits ``v_grid``/``y_grid`` into monotonic branches at sign changes (or
    near-zero crossings) of the numerical derivative ``dY/dV``, then, for
    each branch whose ``y`` range brackets ``y_observed``, inverts via a
    monotone cubic (PCHIP) interpolant. Zero matching branches or more than
    one matching branch are both treated as failures (no branch is silently
    preferred) -- this mirrors MATLAB's reject-on-ambiguity philosophy
    rather than a "primary + alternatives" reporting style.
    """
    v_grid = np.asarray(v_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    valid = np.isfinite(v_grid) & np.isfinite(y_grid)
    v_grid = v_grid[valid]
    y_grid = y_grid[valid]

    if not np.isfinite(y_observed) or v_grid.size < 3:
        return InvertResult(
            float("nan"), float("nan"), "outside calibration range or no monotonic branch"
        )

    d_y = np.gradient(y_grid, v_grid)
    sign_slope = np.sign(d_y)
    sign_slope[np.abs(d_y) < min_slope] = 0.0

    prod = sign_slope[1:] * sign_slope[:-1]
    break_starts = np.where(prod <= 0)[0] + 1
    breaks = np.concatenate(([0], break_starts, [v_grid.size]))

    candidates: list[float] = []
    candidate_slopes: list[float] = []

    for b in range(len(breaks) - 1):
        idx = np.arange(breaks[b], breaks[b + 1])
        if idx.size < 3:
            continue
        v_branch = v_grid[idx]
        y_branch = y_grid[idx]

        order = np.argsort(y_branch)
        y_sorted = y_branch[order]
        v_sorted = v_branch[order]

        y_unique, unique_idx = np.unique(y_sorted, return_index=True)
        v_unique = v_sorted[unique_idx]

        if y_unique.size < 3 or y_observed < y_unique[0] or y_observed > y_unique[-1]:
            continue

        interpolator = PchipInterpolator(y_unique, v_unique)
        value = float(interpolator(y_observed))
        local_slope = float(np.interp(value, v_grid, d_y))

        if np.isfinite(value) and np.isfinite(local_slope):
            candidates.append(value)
            candidate_slopes.append(local_slope)

    if len(candidates) == 0:
        return InvertResult(
            float("nan"), float("nan"), "outside calibration range or no monotonic branch"
        )
    if len(candidates) > 1:
        return InvertResult(float("nan"), float("nan"), "multiple analytical roots")

    v_estimate = candidates[0]
    local_slope = candidate_slopes[0]
    status = "calibration locally flat" if abs(local_slope) < min_slope else "ok"
    return InvertResult(v_estimate, local_slope, status)


# ---------------------------------------------------------------------------
# 4. Combining multiple observable-based estimates (TENOR_protocol.m:
#    combine_estimates, lines ~468-514)
# ---------------------------------------------------------------------------


def combine_estimates(
    v_values, dv_values, usable_mask, strategy: str = "inverseVariance"
) -> tuple[float, float]:
    """Combine per-observable ``(V, SE)`` pairs into one ``(best_v, best_se)``.

    ``strategy`` is one of ``"bestSingle"``, ``"mean"``, ``"median"``,
    ``"robust"``, or ``"inverseVariance"`` (case-insensitive); an unknown
    strategy raises ``ValueError``. Returns ``(nan, nan)`` if no entries are
    usable.
    """
    valid_strategies = {"bestsingle", "mean", "median", "robust", "inversevariance"}
    strategy_key = strategy.lower()
    if strategy_key not in valid_strategies:
        raise ValueError(f"Unknown combination strategy: {strategy!r}")

    v_values = np.asarray(v_values, dtype=float)
    dv_values = np.asarray(dv_values, dtype=float)
    usable_mask = np.asarray(usable_mask, dtype=bool)

    indices = np.where(usable_mask)[0]
    if indices.size == 0:
        return float("nan"), float("nan")

    v = v_values[indices]
    dv = dv_values[indices]

    if strategy_key == "bestsingle":
        j = int(np.argmin(dv))
        return float(v[j]), float(dv[j])

    if strategy_key == "mean":
        n = indices.size
        best_v = float(np.mean(v))
        best_se = float(np.sqrt(np.sum(dv ** 2)) / n)
        return best_v, best_se

    if strategy_key == "median":
        n = indices.size
        best_v = float(np.median(v))
        best_se = float(np.sqrt(np.pi / 2) * np.median(dv) / np.sqrt(n))
        return best_v, best_se

    if strategy_key == "robust":
        centre = np.median(v)
        mad = np.median(np.abs(v - centre))
        consistency = 1.0 / norm.ppf(0.75)
        if mad > 0:
            keep = np.abs(v - centre) <= 3 * consistency * mad
            v = v[keep]
            dv = dv[keep]
        if v.size == 0:
            return float("nan"), float("nan")
        weights = 1.0 / dv ** 2
        best_v = float(np.sum(weights * v) / np.sum(weights))
        best_se = float(np.sqrt(1.0 / np.sum(weights)))
        return best_v, best_se

    # inverseVariance (default)
    weights = 1.0 / dv ** 2
    best_v = float(np.sum(weights * v) / np.sum(weights))
    best_se = float(np.sqrt(1.0 / np.sum(weights)))
    return best_v, best_se


# ---------------------------------------------------------------------------
# 5. Orchestrator (TENOR_protocol.m: top-level TENOR_protocol, lines ~1-243)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TenorProtocolResult:
    """Result of :func:`tenor_protocol`.

    ``rg`` is the raw APPARENT Guinier radius of gyration -- ``sqrt(RG2)``
    from ``mg_extract``, i.e. ``R0*sqrt(1+V)`` per the paper's Eq. 5. It is
    biased upward by polydispersity broadening and should not be reported
    as "the" ensemble Rg on its own.

    ``rg_corrected`` undoes that broadening using the recovered ``best_v``:
    ``rg_corrected = rg / sqrt(1 + best_v)``, i.e. the scattering-weighted
    mean R0 itself. This is the SAME first-order-in-V relationship
    (Eq. 5/Eq. 3 of the paper) that underlies every Table-2 formula this
    module already uses elsewhere -- it requires no assumption about the
    ensemble's distribution SHAPE (only its first two scattering-weighted
    moments, R0 and V, matter to this order), which is the same
    distribution-shape-weak-dependence property the paper highlights for
    the observables themselves. It is therefore NOT the kind of
    distribution-family-specific empirical bias correction the older,
    unrelated ``tenor_saxs_protocol`` package's
    ``correct_rg_for_polydispersity`` performs (a numerically-fitted
    per-distribution lookup table) -- this is a direct analytical
    consequence of the same approximation already used throughout this
    package, valid in the same regime the rest of the method is valid in,
    and ``NaN`` whenever ``best_v`` itself is not finite or ``best_v<=-1``.
    """

    observed: dict[str, float]
    observed_se: dict[str, float]
    v_estimates: dict[str, float]
    v_se: dict[str, float]
    status: dict[str, str]
    best_v: float
    best_v_se: float
    strategy_used: str
    rg: float
    rg_corrected: float
    actual_psf: np.ndarray
    sigma0: float
    delta0: float
    ag: float
    am: float
    v_grid: np.ndarray
    negative_best_v_is_extrapolation: bool
    mg_result: object


def _six_observables_from_mg_result(mg_result, use_g3: bool) -> tuple[dict[str, float], float, float]:
    """Build the six MATLAB-backed observables from one image's mg_extract fit.

    Shared by :func:`tenor_protocol` (the single-image analysis path) and
    :func:`simulate_calibration_curve` (which calls this once per simulated
    ``V`` node) so the observable formulas live in exactly one place.
    Returns ``(observables, sigma0, delta0)`` -- the caller derives
    ``ag = sigma0/2`` / ``am = delta0/2`` itself if it needs them (only
    ``tenor_protocol`` does, for its result dataclass).
    """
    sx1, sy1, sx2, sy2 = mg_result.actual_psf
    # mg_extract fits log(F_wide / F_narrow) (kernel 2 over kernel 1), so the
    # signed PSF difference is "wide minus narrow" (PSF2 - PSF1).
    d_sx2 = sx2 ** 2 - sx1 ** 2
    d_sy2 = sy2 ** 2 - sy1 ** 2
    sigma0 = 0.5 * (d_sx2 + d_sy2)
    delta0 = 0.5 * (d_sx2 - d_sy2)
    ag = 0.5 * sigma0

    p = mg_result.p
    rg2 = mg_result.rg2
    g0, g1, g2 = p[0], p[1], p[2]
    m_offset = 3 + int(use_g3)
    m1, m2 = p[m_offset], p[m_offset + 1]

    raw_yg100 = g1 / g0 ** 2
    raw_yg210 = g2 / (g1 * g0)
    raw_ym210 = m2 / (m1 * g0)

    observed = {
        "Yg100": ag * raw_yg100,
        "Yg210": ag * raw_yg210,
        "Ym210": ag * raw_ym210,
        "Jg10": (g1 / g0) / rg2,
        "Jg21": (g2 / g1) / rg2,
        "Jm": (m2 / m1) / rg2,
    }
    return observed, sigma0, delta0


def tenor_protocol(
    intensity: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    phi2: float,
    pxn: np.ndarray = np.array([87, 85, 125, 123]),
    signum: float = 4.0,
    use_r3: bool = False,
    use_g3: bool = False,
    weight_mode: str = "intensity",
    wavelength: float = 0.1,
    use_single: bool = False,
    v_range: tuple[float, float] = (-0.05, 0.35),
    v_grid_n: int = 4001,
    min_slope: float = 1e-8,
    strategy: str = "inverseVariance",
    observables: tuple[str, ...] = _OBSERVABLE_NAMES,
    qrg_max: float = _QRG_MAX_DEFAULT,
    deadpix_factor: float = _DEADPIX_FACTOR_DEFAULT,
    calibration_override: dict[str, np.ndarray] | None = None,
) -> TenorProtocolResult:
    """Run the full TENOR-SAXS extraction protocol on one 2D detector image.

    Fits the polynomial coefficients via ``mg_extract.mg_extract``, builds
    the requested physically-named observables and their delta-method SEs,
    inverts each against a calibration curve (by default the analytical one,
    :func:`analytical_theory`) to get a per-observable ``V`` estimate, and
    combines them into one final answer.

    ``qrg_max``/``deadpix_factor`` are passed straight through to
    :func:`tenor_saxs.mg_extract.mg_extract` -- see its docstring and the
    ``QRG_MAX``/``DEADPIX_FACTOR`` module constants for what they mean and
    how they interact.

    ``calibration_override``:
        Optional per-observable replacement for the analytic calibration
        curve, e.g. ``{"Yg210": ..., "Ym210": ...}`` -- only the named
        observables are overridden; any observable not present still uses
        ``analytical_theory``. The manuscript recommends this simulation-
        derived route specifically for ``Yg210``/``Ym210`` (see
        :func:`tenor_saxs.calibration.build_calibration_curve`), whose
        analytic formulas are measurably biased and, for ``Yg210``, non-
        invertible past ``V=0.273``. Each array MUST be evaluated on
        exactly this call's own ``v_grid`` (``np.linspace(*v_range,
        v_grid_n)``) -- :func:`tenor_saxs.calibration.build_calibration_curve`'s
        ``v_grid_range``/``v_grid_n`` default to this function's own
        ``v_range``/``v_grid_n`` defaults for exactly this reason. Default
        ``None`` leaves every observable on the analytic curve, i.e.
        today's exact behavior.
    """
    from tenor_saxs import mg_extract as mg_extract_module

    mg_result = mg_extract_module.mg_extract(
        pxn,
        qx,
        qy,
        intensity,
        signum=signum,
        rg2=None,
        use_r3=use_r3,
        use_g3=use_g3,
        weight_mode=weight_mode,
        wavelength=wavelength,
        use_single=use_single,
        qrg_max=qrg_max,
        deadpix_factor=deadpix_factor,
    )

    all_observed, sigma0, delta0 = _six_observables_from_mg_result(mg_result, use_g3)
    ag = 0.5 * sigma0
    am = 0.5 * delta0
    p = mg_result.p
    rg2 = mg_result.rg2

    y_t_se: float | None = None
    if "Y_T" in observables:
        # Y_T comes from a SEPARATE single-image fit (no wide/narrow
        # smearing comparison at all -- just ln(I) vs q**2 over the same
        # Guinier-region mask mg_extract already established from this
        # image's own RG2 estimate), not from mg_result.p/cov_p. See
        # the internal validation notes item 2 and mg_extract.fit_quadratic_weighted_centered.
        qvr = np.hypot(qx, qy)
        q_lower, q_upper = mg_result.guinier_qrange
        mask_yt = (qvr > q_lower) & (qvr < q_upper)
        intensity_masked = np.asarray(intensity)[mask_yt]
        if weight_mode == "sqrt_intensity":
            weight_yt = np.sqrt(intensity_masked)
        else:
            weight_yt = intensity_masked
        with np.errstate(divide="ignore", invalid="ignore"):
            log_intensity_yt = np.log(intensity_masked)
        yt_fit = mg_extract_module.fit_quadratic_weighted_centered(qvr[mask_yt] ** 2, log_intensity_yt, weight_yt)
        t0_yt, t1_yt, t2_yt = yt_fit.p
        all_observed["Y_T"] = t2_yt / t1_yt ** 2
        yt_grad = y_t_gradient(t1_yt, t2_yt)
        y_t_se = float(np.sqrt(max(yt_grad @ yt_fit.cov_p @ yt_grad, 0.0)))

    observed = {name: all_observed[name] for name in observables}

    grads = observable_gradients(p, rg2, ag, use_g3)
    observed_se: dict[str, float] = {}
    for name in observables:
        if name == "Y_T":
            observed_se[name] = y_t_se
            continue
        grad = grads[name]
        variance = grad @ mg_result.cov_p @ grad
        observed_se[name] = float(np.sqrt(max(variance, 0.0)))

    v_grid = np.linspace(v_range[0], v_range[1], v_grid_n)
    calibration = analytical_theory(v_grid, phi2)
    if calibration_override is not None:
        for name, values in calibration_override.items():
            values = np.asarray(values, dtype=float)
            if values.shape != v_grid.shape:
                raise ValueError(
                    f"calibration_override[{name!r}] has shape {values.shape}, expected "
                    f"{v_grid.shape} to match this call's v_grid (v_range={v_range}, "
                    f"v_grid_n={v_grid_n}) -- build the override curve with "
                    "calibration.build_calibration_curve(..., v_grid_range=v_range, "
                    "v_grid_n=v_grid_n) so the grids align."
                )
            calibration[name] = values

    v_estimates: dict[str, float] = {}
    v_se: dict[str, float] = {}
    status: dict[str, str] = {}
    for name in observables:
        result = invert_lookup(v_grid, calibration[name], observed[name], min_slope)
        v_estimates[name] = result.v_estimate
        status[name] = result.status
        # Matches MATLAB exactly (line 184-187): the SE is only propagated
        # when the local slope clears min_slope, i.e. NOT for
        # "calibration locally flat" either -- that status is excluded from
        # the combination step just like the two rejection statuses, rather
        # than being merely down-weighted by a very large SE.
        if (
            np.isfinite(result.v_estimate)
            and np.isfinite(result.local_slope)
            and abs(result.local_slope) >= min_slope
        ):
            v_se[name] = observed_se[name] / abs(result.local_slope)
        else:
            v_se[name] = float("nan")

    v_array = np.array([v_estimates[name] for name in observables], dtype=float)
    se_array = np.array([v_se[name] for name in observables], dtype=float)
    usable_mask = np.isfinite(v_array) & np.isfinite(se_array) & (se_array > 0)

    best_v, best_v_se = combine_estimates(v_array, se_array, usable_mask, strategy)

    rg = float(np.sqrt(rg2))
    if np.isfinite(best_v) and (1.0 + best_v) > 0:
        rg_corrected = float(rg / np.sqrt(1.0 + best_v))
    else:
        rg_corrected = float("nan")
    negative_best_v_is_extrapolation = bool(np.isfinite(best_v) and best_v < 0)

    return TenorProtocolResult(
        observed=observed,
        observed_se=observed_se,
        v_estimates=v_estimates,
        v_se=v_se,
        status=status,
        best_v=best_v,
        best_v_se=best_v_se,
        strategy_used=strategy,
        rg=rg,
        rg_corrected=rg_corrected,
        actual_psf=mg_result.actual_psf,
        sigma0=sigma0,
        delta0=delta0,
        ag=ag,
        am=am,
        v_grid=v_grid,
        negative_best_v_is_extrapolation=negative_best_v_is_extrapolation,
        mg_result=mg_result,
    )
