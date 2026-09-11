"""MG_extract: the log-ratio polynomial-fitting engine of TENOR-SAXS.

Port of the top-level ``MG_extract`` function (MATLAB ``MG_extract.m:1-231``)
and its local weighted-least-squares helper
``fit_I_r_theta_ratios_weighted_centered`` (``MG_extract.m:290-585``).

The method fits the log-ratio of two differently-smeared copies of the same
2D SAXS detector image,

    log(F_wide / F_narrow)(r, theta) = G(r) + M(r) * cos(2*theta)

where ``r = q**2`` and ``F_wide``/``F_narrow`` are the input intensity
convolved with a wide/narrow anisotropic Gaussian smearing kernel pair, to a
low-order polynomial model in ``r``. This module's job stops at returning
the fitted polynomial coefficients and their covariance matrix -- it
deliberately knows nothing about V-inversion, calibration curves, or named
physical observables (``Yg100``/``Yg210``/``Ym210``/etc); those live in
``protocol.py``, which consumes ``MgExtractResult.p``/``.cov_p`` directly and
computes its own gradients for whatever ratios it needs. This mirrors the
MATLAB architecture: ``TENOR_protocol.m`` never reads ``MG_extract.m``'s
internal ratio fields -- it recomputes gradients itself from ``res.p``/
``res.covP``.

Numerics (QR-based weighted least squares, the weighted-centering /
binomial-decentering transform, the ``0.79`` Guinier cutoff, the literal
``z95 = 1.95996398454005`` MATLAB uses for a 95% CI -- unused in this file
but kept as the same constant for cross-file consistency) are adapted from
the already-validated port in
``tenor_saxs_protocol/tenor_analysis.py:248-513``. Unlike that port, this
module does not apply a PSF-difference scaling prefactor and does not
compute any named ratio fields (``g_ratio``, ``g100_ratio``, ``g210_ratio``,
``m210_ratio`` or their confidence intervals) -- see the module docstring
above and ``protocol.py`` for where that now lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from . import psf

# MATLAB's literal 95%-CI z-score (MG_extract.m:458). Not used for any
# computation in this file (no CIs are computed here), kept only so that any
# downstream code cross-checking against MATLAB literals has one canonical
# source. protocol.py should import this rather than re-declaring it.
Z95 = 1.95996398454005

# --- Analysis-window tuning parameters (see the module docstring) ---------
# The Guinier fit window is the intersection of two conditions
# (MG_extract.m:186-194):
#   (i)  an upper Guinier edge   q < QRG_MAX / sqrt(rg2)   -- i.e. q*Rg_app < QRG_MAX
#   (ii) a detector-border margin excluding pixels within
#        DEADPIX_FACTOR * max(pxn) pixels of the outer edge.
#
# Round-4 update: the manuscript's final choice, after the round-3/4 window
# sweeps, is q*R0_tilde < 1.0 and a one-kernel-width border margin -- a
# bias-variance compromise (a small, well-characterized, monotone-in-V
# underestimate of V in exchange for a >3x reduction in the photon density
# needed for RMS(V^0.5) < 0.1), not a validity threshold. Round 1-3 shipped
# QRG_MAX=0.79, DEADPIX_FACTOR=2.0.
QRG_MAX = 1.0
DEADPIX_FACTOR = 1.0


@dataclass(slots=True)
class WlsFitResult:
    """Result of the weighted, centered log-ratio polynomial fit.

    ``p`` and ``cov_p`` are expressed in the ORIGINAL (uncentered) ``r``
    basis, ordered ``[g0, g1, g2, (g3), m1, m2, (m3)]`` -- the trailing
    ``g3``/``m3`` entries are present only when ``use_g3``/``use_r3`` was
    requested. There is no ``m0`` term anywhere: the physical constraint
    ``m0 = 0`` is structural, never fitted (MG_extract.m:294-295,340-351).
    """

    p: np.ndarray
    cov_p: np.ndarray
    mu_r: float
    rank_x: int
    sse: float
    sst_w: float
    dof: int
    n_used: int


def fit_i_r_theta_ratios_weighted_centered(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    use_r3: bool = False,
    use_g3: bool = False,
) -> WlsFitResult:
    """Weighted least-squares fit of ``y(r,theta) = G(r) + M(r)*cos(2*theta)``.

    Port of ``fit_I_r_theta_ratios_weighted_centered`` (MG_extract.m:290-585).

    ``r`` is ALREADY ``q**2`` -- the caller's responsibility, this function
    never scales it. ``G(r) = g0 + g1*r + g2*r**2`` (``+ g3*r**3`` if
    ``use_g3``); ``M(r) = m1*r + m2*r**2`` (``+ m3*r**3`` if ``use_r3``),
    with no ``m0`` (constant) term in ``M`` at all.

    Numerical procedure (MG_extract.m:316-437):

    1. Filter to finite ``r``, ``theta``, ``y``, ``weight``.
    2. Weighted-center the G-block only: ``mu_r = sum(w*r)/sum(w)``,
       ``rc = r - mu_r``. The M-block columns (``r*cos(2*theta)``, ...) are
       built from the UNcentered ``r`` -- centering them would introduce a
       spurious pure-``cos(2*theta)`` column and change the model class
       (MG_extract.m:323-325).
    3. Weighted QR solve of ``[Gcols, Mcols] @ p_c = y`` in the centered
       basis (``sw = sqrt(w)``; economy QR of ``X*sw``), falling back to an
       ``lstsq`` minimum-norm solve if ``R`` is rank-deficient
       (MG_extract.m:362-376) -- a numerically-equivalent substitute for
       MATLAB's manual SVD + tolerance fallback.
    4. Residual variance/covariance in the CENTERED basis via ``R`` (never
       forming the normal equations): ``s2 = SSE/dof``,
       ``cov_p_centered = s2 * (R^-1)(R^-1)^T`` (MG_extract.m:378-389).
    5. Map back to the ORIGINAL (uncentered) ``r`` basis with a
       block-diagonal transform ``T``: the G-block sub-matrix re-expands
       the binomial ``(rc + mu_r)**n`` into powers of ``r`` (undoing the
       centering algebraically); the M-block sub-matrix is the identity,
       since the M columns were never centered (MG_extract.m:408-437).
       ``p = T @ p_c``, ``cov_p = T @ cov_p_centered @ T.T``.

    Also returns ``sst_w`` (weighted total sum of squares, an optional R^2
    diagnostic never consumed by anything downstream of this file).
    """
    r = np.asarray(r, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    weight = np.asarray(weight, dtype=float).ravel()
    cos2t = np.cos(2.0 * theta)

    valid = np.isfinite(r) & np.isfinite(theta) & np.isfinite(y) & np.isfinite(weight)
    if not np.any(valid):
        raise ValueError("No valid samples.")
    r = r[valid]
    cos2t = cos2t[valid]
    y = y[valid]
    w = weight[valid]
    n = r.size

    # -------- weighted centering for the G-block only -------------------
    w_sum = np.sum(w)
    mu_r = float(np.sum(w * r) / w_sum)
    rc = r - mu_r

    if use_g3:
        g_cols = [np.ones_like(r), rc, rc**2, rc**3]
    else:
        g_cols = [np.ones_like(r), rc, rc**2]
    if use_r3:
        m_cols = [r * cos2t, (r**2) * cos2t, (r**3) * cos2t]
    else:
        m_cols = [r * cos2t, (r**2) * cos2t]

    k_g = len(g_cols)
    k_m = len(m_cols)
    X = np.column_stack(g_cols + m_cols)
    k = X.shape[1]

    # -------- stable weighted least squares via QR -----------------------
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    Q, R = np.linalg.qr(Xw, mode="reduced")
    rank_x = int(np.linalg.matrix_rank(R))
    if rank_x < k:
        # SVD-based minimum-norm least-squares fallback for rank-deficient R
        # (numerically equivalent to MATLAB's manual SVD + tolerance
        # fallback at MG_extract.m:368-373).
        p_c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    else:
        p_c = solve_triangular(R, Q.T @ yw)

    # -------- residuals / variance & covariance (centered basis) --------
    residual = y - X @ p_c
    sse = float(np.sum(w * residual**2))
    dof = max(n - k, 1)
    s2 = sse / dof

    # Matches MG_extract.m:388-389, which computes this unconditionally
    # from R regardless of which branch produced p_c above.
    r_inv = solve_triangular(R, np.eye(k))
    cov_p_centered = s2 * (r_inv @ r_inv.T)

    # -------- map back to the original (uncentered) r-basis --------------
    # G-block: re-expand (rc + mu_r)**n into powers of r, n = 0..3.
    t_full = np.eye(4)
    t_full[0, :] = [1.0, -mu_r, mu_r**2, -(mu_r**3)]
    t_full[1, :] = [0.0, 1.0, -2.0 * mu_r, 3.0 * mu_r**2]
    t_full[2, :] = [0.0, 0.0, 1.0, -3.0 * mu_r]
    t_full[3, :] = [0.0, 0.0, 0.0, 1.0]

    T = np.zeros((k, k))
    T[:k_g, :k_g] = t_full[:k_g, :k_g]
    T[k_g:, k_g:] = np.eye(k_m)

    p = T @ p_c
    cov_p = T @ cov_p_centered @ T.T

    # -------- weighted total sum of squares (diagnostic only) -----------
    ybar_w = float(np.sum(w * y) / np.sum(w))
    sst_w = float(np.sum(w * (y - ybar_w) ** 2))

    return WlsFitResult(
        p=p,
        cov_p=cov_p,
        mu_r=mu_r,
        rank_x=rank_x,
        sse=sse,
        sst_w=sst_w,
        dof=dof,
        n_used=n,
    )


def fit_quadratic_weighted_centered(r: np.ndarray, y: np.ndarray, weight: np.ndarray) -> WlsFitResult:
    """Weighted least-squares fit of ``y(r) = t0 + t1*r + t2*r**2`` (no MATLAB counterpart).

    This is the single-image analogue of
    :func:`fit_i_r_theta_ratios_weighted_centered`, needed for the paper's
    ``Y_T`` observable (``Y_T = t2/t1**2`` from a direct Guinier-region
    quadratic fit of ``ln I(q)`` vs. ``q**2`` on ONE image -- no log-ratio
    of two differently-smeared images, no ``cos(2*theta)`` M-block at all).
    There is no MATLAB reference to port here: ``Y_T`` and the raw cumulant
    ``a,b,c`` triple (paper Appendix B) are derived and discussed in the
    paper's text but implemented nowhere in the vendored MATLAB code (see
    the internal validation notes item 2) -- this function and its use in
    :func:`tenor_saxs.protocol.tenor_protocol` are this package's own
    from-scratch implementation of that gap, built with the same
    weighted-centering / QR / binomial-decentering numerics already
    validated for the G/M fit above (this is intentionally a strict
    subset of that machinery -- a single 3-column block, no M-block split
    -- rather than a new numerical method).

    Returns a :class:`WlsFitResult` with ``p = [t0, t1, t2]`` in the
    ORIGINAL (uncentered) ``r`` basis and its ``3x3`` covariance.
    """
    r = np.asarray(r, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    weight = np.asarray(weight, dtype=float).ravel()

    valid = np.isfinite(r) & np.isfinite(y) & np.isfinite(weight)
    if not np.any(valid):
        raise ValueError("No valid samples.")
    r = r[valid]
    y = y[valid]
    w = weight[valid]
    n = r.size
    k = 3
    if n <= k:
        raise ValueError(f"fit_quadratic_weighted_centered: need more than {k} valid samples, got {n}")

    w_sum = np.sum(w)
    mu_r = float(np.sum(w * r) / w_sum)
    rc = r - mu_r

    X = np.column_stack([np.ones_like(r), rc, rc**2])
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    Q, R = np.linalg.qr(Xw, mode="reduced")
    rank_x = int(np.linalg.matrix_rank(R))
    if rank_x < k:
        p_c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    else:
        p_c = solve_triangular(R, Q.T @ yw)

    residual = y - X @ p_c
    sse = float(np.sum(w * residual**2))
    dof = max(n - k, 1)
    s2 = sse / dof

    r_inv = solve_triangular(R, np.eye(k))
    cov_p_centered = s2 * (r_inv @ r_inv.T)

    # Re-expand the quadratic (t0_c + t1_c*rc + t2_c*rc**2), rc = r - mu_r,
    # back into powers of the original r (same binomial-decentering idea as
    # the G-block transform above, truncated to degree 2).
    T = np.array(
        [
            [1.0, -mu_r, mu_r**2],
            [0.0, 1.0, -2.0 * mu_r],
            [0.0, 0.0, 1.0],
        ]
    )
    p = T @ p_c
    cov_p = T @ cov_p_centered @ T.T

    ybar_w = float(np.sum(w * y) / np.sum(w))
    sst_w = float(np.sum(w * (y - ybar_w) ** 2))

    return WlsFitResult(p=p, cov_p=cov_p, mu_r=mu_r, rank_x=rank_x, sse=sse, sst_w=sst_w, dof=dof, n_used=n)


@dataclass(slots=True)
class MgExtractResult:
    """Result of :func:`mg_extract`.

    ``actual_psf`` is ``[sigma_x1, sigma_y1, sigma_x2, sigma_y2]`` in
    q-UNITS, where kernel 1 is the NARROW kernel (built from ``pxn[0:2]``)
    and kernel 2 is the WIDE kernel (built from ``pxn[2:4]``), per
    ``MG_extract.m:75-97,118-138``.
    """

    p: np.ndarray
    cov_p: np.ndarray
    rg2: float
    actual_psf: np.ndarray
    pxn: np.ndarray
    fit: WlsFitResult
    guinier_qrange: tuple[float, float]


def mg_extract(
    pxn: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    intensity: np.ndarray,
    signum: float = 4.0,
    rg2: float | None = None,
    use_r3: bool = False,
    use_g3: bool = False,
    weight_mode: str = "intensity",
    wavelength: float = 0.1,
    use_single: bool = False,
    qrg_max: float = QRG_MAX,
    deadpix_factor: float = DEADPIX_FACTOR,
) -> MgExtractResult:
    """Fit the log-ratio of a wide/narrow-smeared image pair to extract MG coefficients.

    Port of the top-level ``MG_extract`` function (MG_extract.m:1-231),
    scoped down to returning only the raw fitted polynomial coefficients and
    their covariance -- no V-inversion, calibration curves, or named
    physical-ratio fields (those belong to ``protocol.py``).

    Parameters
    ----------
    pxn:
        4-element ``[nx1, ny1, nx2, ny2]`` smearing pixel-count quartet,
        validated via :func:`psf.validate_pxn`. ``(nx1, ny1)`` builds the
        NARROW kernel, ``(nx2, ny2)`` builds the WIDE kernel
        (MG_extract.m:75-82 for wide, 118-122 for narrow).
    qx, qy:
        Meshgrids of detector q-coordinates (1/nm), same shape as
        ``intensity``.
    intensity:
        Raw (unfiltered) 2D detector intensity.
    signum:
        Number of standard deviations spanned by each Gaussian smearing
        kernel (MG_extract.m default: 4).
    rg2:
        ``r_g**2 * (1+V)`` estimate. If ``None``/non-finite/non-positive, it
        is estimated from the raw, unweighted, unmasked intensity via
        :func:`best_origin_quad_b_faster_bins` as ``-3 * best_b``
        (MG_extract.m:44-51).
    use_r3, use_g3:
        Include a cubic term in ``M(r)``/``G(r)`` respectively. Note this
        port's defaults (``False``, ``False``) intentionally differ from
        MATLAB's ``MG_extract`` defaults (``true``, ``true``,
        MG_extract.m:29-35); callers that want MATLAB-parity should pass
        ``use_r3=True, use_g3=True`` explicitly.
    weight_mode:
        ``"intensity"`` (default: weight = I, matching MATLAB's actual
        active code at MG_extract.m:220 -- and, per an empirical study of
        the two candidate weightings focused on the high-noise regime
        (an internal study script, see the package's internal validation notes), the
        theoretically variance-optimal choice under the photon-counting
        noise model used throughout this package) or ``"sqrt_intensity"``
        (weight = sqrt(I); an earlier draft of the paper's text described
        this instead, before the paper was updated to match the code and
        this finding). Any other value raises ``ValueError``.
    wavelength:
        Passed through to :func:`psf.filter2_ungridded` (MG_extract.m's
        ``WAVELENGTH`` argument, default ``0.1`` matching
        ``init_TENOR_params.m``). Needed for the center-equivalent angular
        coordinate transform, not just the fast/regular-grid path.
    use_single:
        Passed through to :func:`psf.filter2_ungridded` -- see its
        docstring. ``False`` (this port's default) keeps the convolution in
        float64; ``True`` replicates MATLAB's internal ``single`` cast for
        MATLAB-parity comparisons.
    qrg_max:
        Upper Guinier edge, as ``q * sqrt(rg2) < qrg_max``. Together with
        ``deadpix_factor`` this sets the analysis window -- see the
        ``QRG_MAX``/``DEADPIX_FACTOR`` module constants above.
    deadpix_factor:
        Detector-border margin, in units of ``max(pxn)`` pixels, excluded
        from the fit on each side.

    Returns
    -------
    MgExtractResult
    """
    pxn = psf.validate_pxn(pxn)
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    qvr = np.hypot(qx, qy)

    # -------- Guinier r_g^2*(1+V) estimate (raw, unfiltered, unweighted) --
    if rg2 is None or not np.isfinite(rg2) or rg2 <= 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            guinier_fit = best_origin_quad_b_faster_bins(qvr**2, np.log(intensity))
        rg2 = -3.0 * guinier_fit.best_b
    rg2 = float(rg2)

    # -------- narrow/wide smearing kernels and convolutions ---------------
    # Narrow kernel from pxn[0:2] (MG_extract.m:118-122), wide from pxn[2:4]
    # (MG_extract.m:79-82). MG_extract ALWAYS filters via filter2_ungridded
    # (never a bare filter2/smear call) -- MG_extract.m:92,130:
    # `filter2_ungridded(H,I_mat,qvx,qvy,1,[],[],0,WAVELENGTH)`, i.e.
    # allow_shortcut=1, dx_h/dy_h auto, rel_cutoff=0.
    kernel_narrow = psf.gaussian_smearing_kernel(int(pxn[0]), int(pxn[1]), signum)
    kernel_wide = psf.gaussian_smearing_kernel(int(pxn[2]), int(pxn[3]), signum)
    f_narrow = psf.filter2_ungridded(
        kernel_narrow, intensity, qx, qy, wavelength, allow_shortcut=True, rel_cutoff=0.0, use_single=use_single
    )
    f_wide = psf.filter2_ungridded(
        kernel_wide, intensity, qx, qy, wavelength, allow_shortcut=True, rel_cutoff=0.0, use_single=use_single
    )

    # -------- recover each kernel's own sigma (pixels -> q-units) --------
    sigma_x1 = psf.fit_kernel_sigma_pixels(kernel_narrow, axis=1)
    sigma_y1 = psf.fit_kernel_sigma_pixels(kernel_narrow, axis=0)
    sigma_x2 = psf.fit_kernel_sigma_pixels(kernel_wide, axis=1)
    sigma_y2 = psf.fit_kernel_sigma_pixels(kernel_wide, axis=0)

    dqpix = float(np.mean(np.diff(qy[:, 0])))
    actual_psf = np.array([sigma_x1, sigma_y1, sigma_x2, sigma_y2], dtype=float) * dqpix

    # -------- log-ratio of wide- to narrow-smeared images ------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(f_wide / f_narrow)

    # -------- Guinier-region mask (MG_extract.m:183-201) -------------------
    if not np.isfinite(deadpix_factor) or deadpix_factor < 0:
        raise ValueError(f"deadpix_factor must be finite and >= 0, got {deadpix_factor!r}")
    if not np.isfinite(qrg_max) or qrg_max <= 0:
        raise ValueError(f"qrg_max must be finite and > 0, got {qrg_max!r}")
    deadpix = float(deadpix_factor) * float(np.max(pxn))
    max_q = float(np.max(qvr))
    q_upper = min(max_q - deadpix * dqpix, float(qrg_max) / np.sqrt(rg2))
    if q_upper <= 0:
        raise ValueError(
            "not enough pixels - consider using a smaller slit "
            f"(computed Guinier q-window upper bound {q_upper!r} <= 0)"
        )
    mask = (qvr > 0) & (qvr < q_upper)

    theta = np.arctan2(qy, qx)

    if weight_mode == "sqrt_intensity":
        weight = np.sqrt(intensity[mask])
    elif weight_mode == "intensity":
        weight = intensity[mask]
    else:
        raise ValueError(
            f"weight_mode must be 'sqrt_intensity' or 'intensity', got {weight_mode!r}"
        )

    fit = fit_i_r_theta_ratios_weighted_centered(
        qvr[mask] ** 2,
        theta[mask],
        log_ratio[mask],
        weight,
        use_r3=use_r3,
        use_g3=use_g3,
    )

    return MgExtractResult(
        p=fit.p,
        cov_p=fit.cov_p,
        rg2=rg2,
        actual_psf=actual_psf,
        pxn=pxn,
        fit=fit,
        guinier_qrange=(0.0, q_upper),
    )


def _percentile_fast(v: np.ndarray, p_low: float, p_high: float) -> tuple[float, float]:
    """MATLAB's ``percentile_fast`` (MG_extract.m:2590-2595): plain linear-interpolation
    percentile -- numerically identical to ``numpy.percentile(v, [p_low, p_high])``
    (the default ``method='linear'``), so no hand-rolled sort/interpolate is needed.
    """
    lo, hi = np.percentile(v, [p_low, p_high])
    return float(lo), float(hi)


@dataclass(slots=True)
class GuinierFitResult:
    """Full result of :func:`best_origin_quad_b_faster_bins`, matching MATLAB's
    ``[best_b, best_b_CI, best_xmax, best_y_at_xmax_CI, best_coef, best_coef_CI, info]``.
    """

    best_b: float
    best_b_ci: tuple[float, float]
    best_xmax: float
    best_y_at_xmax_ci: tuple[float, float]
    best_coef: np.ndarray  # [c0, c1, c2] of the accepted window's quadratic fit
    best_coef_ci: np.ndarray  # (3, 2): [lower, upper] per coefficient
    stopped_early: bool
    best_idx: int  # number of points in the accepted window


def best_origin_quad_b_faster_bins(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 200,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    min_pts: int = 10,
    tol_pct: float = 5.0,
    abs_tol: float = 1e-3,
    threshold: float = -1.0 / 6.0,
    rng: np.random.Generator | None = None,
) -> GuinierFitResult:
    """Robust auto-windowed quadratic Guinier fit with a residual bootstrap CI.

    Full port of ``best_origin_quad_b_faster_bins`` (MG_extract.m:2453-2587).
    Given ``x = q**2`` and ``y = log(intensity)`` (both raw/unmasked/
    unweighted), this expands a quadratic-fit window from the origin
    outward over quantile-spaced candidate cutoffs, stopping once the fit
    looks stable, then runs a 100-replicate residual bootstrap on the
    accepted window to report confidence intervals. Only ``best_b`` (via
    ``RG2 = -3*best_b``) is consumed elsewhere in this package, but every
    other MATLAB output is computed and returned for full fidelity.

    Search phase, per candidate window (all points with ``x`` below a
    quantile cutoff, requiring at least ``min_pts`` points):

    1. Unweighted quadratic fit ``y = c0 + c1*x + c2*x**2`` via the O(1)
       cumulative-sum trick MATLAB uses (MG_extract.m:2478-2488).
    2. Predicted ``y`` at the window's edge ``x``-value and its standard
       error from the fit's own residual MSE (MG_extract.m:2521-2540), plus
       the linear coefficient's relative standard error.
    3. Expansion stops (the window is accepted) once the predicted-``y``
       lower ~95% bound drops below ``threshold`` AND the linear
       coefficient's relative standard error is under ``tol_pct``
       (MG_extract.m:2546-2551); otherwise the window keeps expanding to the
       last candidate, and the LAST successfully-fit window is used.

    Bootstrap phase (MG_extract.m:2554-2587), on the accepted window only:
    residual bootstrap via ``beta_b = beta + M*(resid(R) - mean(resid))``
    with ``M = (X'X)^-1 X'`` and ``R`` = ``n_bootstrap`` columns of
    with-replacement resample indices into the residuals; percentile CIs
    (``alpha``, default 95%) are taken over the ``n_bootstrap`` replicate
    coefficient vectors and the replicate predicted-y-at-``x_max`` values.

    If ``rng`` is ``None``, a fresh ``numpy.random.default_rng()`` is used
    for the bootstrap resampling -- this does NOT reproduce MATLAB's
    Mersenne-Twister ``randi`` stream bit-for-bit (expected, documented at
    the package level; only the algorithm's structure needs to match).
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = (x >= 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n = x.size
    if n < min_pts:
        raise ValueError(
            f"best_origin_quad_b_faster_bins: only {n} valid (x>=0, finite) samples, "
            f"need at least min_pts={min_pts}"
        )

    s1 = np.cumsum(np.ones(n))
    sx = np.cumsum(x)
    sx2 = np.cumsum(x**2)
    sx3 = np.cumsum(x**3)
    sx4 = np.cumsum(x**4)
    sy = np.cumsum(y)
    sxy = np.cumsum(x * y)
    sx2y = np.cumsum((x**2) * y)
    sy2 = np.cumsum(y**2)

    # bin_idx[k] = COUNT of x-points at or below the k-th quantile edge
    # (MATLAB's `[~, bin_idx] = histc(edges(2:end), x)` -- binning the quantile
    # edges into x-defined bins is the same lookup as searchsorted here).
    p_edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, p_edges)  # equivalent to MATLAB's quantile_fast
    bin_idx = np.searchsorted(x, edges[1:], side="right")
    bin_idx[bin_idx == 0] = n  # MATLAB's defensive edge-case handling (MG_extract.m:2495)

    z_val = 1.96  # MATLAB's own "approx for 95% CI without stats toolbox" (MG_extract.m:2499)
    stopped_early = False
    best_idx = int(bin_idx[0])

    for k in range(bin_idx.size):
        idx = int(bin_idx[k])
        if idx < min_pts:
            continue
        i = idx - 1  # 0-based index of the window's last point into the cumulative sums
        xt_x = np.array(
            [
                [s1[i], sx[i], sx2[i]],
                [sx[i], sx2[i], sx3[i]],
                [sx2[i], sx3[i], sx4[i]],
            ]
        )
        xt_y = np.array([sy[i], sxy[i], sx2y[i]])
        beta = np.linalg.solve(xt_x, xt_y)
        mse = max(float(sy2[i] - beta @ xt_y) / (idx - 3), 0.0)
        inv_xtx = np.linalg.solve(xt_x, np.eye(3))
        se = np.sqrt(np.diag(inv_xtx) * mse)

        xk = x[i]
        xvec = np.array([1.0, xk, xk**2])
        y_val = float(xvec @ beta)
        y_se = np.sqrt(max(float(xvec @ inv_xtx @ xvec) * mse, 0.0))
        y_low = y_val - z_val * y_se

        b_est = float(beta[1])
        rel_err = (z_val * float(se[1]) / max(abs(b_est), abs_tol)) * 100.0

        best_idx = idx
        if (y_low < threshold) and (rel_err <= tol_pct):
            stopped_early = True
            break

    # -------- final residual bootstrap on the accepted window only (MG_extract.m:2554-2587) --
    final_idx = best_idx
    xn = x[:final_idx]
    yn = y[:final_idx]
    X = np.column_stack([np.ones(final_idx), xn, xn**2])
    xtx_f = X.T @ X
    beta_f = np.linalg.solve(xtx_f, X.T @ yn)
    M = np.linalg.solve(xtx_f, X.T)  # (3, final_idx)
    resid = yn - X @ beta_f

    r_idx = rng.integers(0, final_idx, size=(final_idx, n_bootstrap))
    mean_resid = float(resid.mean())  # ~0 exactly for OLS-with-intercept residuals; kept for fidelity
    beta_boot = beta_f[:, None] + M @ (resid[r_idx] - mean_resid)  # (3, n_bootstrap)

    lower_p = 100.0 * (alpha / 2.0)
    upper_p = 100.0 * (1.0 - alpha / 2.0)

    best_xmax = float(x[final_idx - 1])
    best_coef_ci = np.array(
        [
            _percentile_fast(beta_boot[0, :], lower_p, upper_p),
            _percentile_fast(beta_boot[1, :], lower_p, upper_p),
            _percentile_fast(beta_boot[2, :], lower_p, upper_p),
        ]
    )
    y_at_xmax_boot = np.array([1.0, best_xmax, best_xmax**2]) @ beta_boot  # (n_bootstrap,)
    best_y_at_xmax_ci = _percentile_fast(y_at_xmax_boot, lower_p, upper_p)

    return GuinierFitResult(
        best_b=float(beta_f[1]),
        best_b_ci=tuple(best_coef_ci[1]),
        best_xmax=best_xmax,
        best_y_at_xmax_ci=best_y_at_xmax_ci,
        best_coef=beta_f,
        best_coef_ci=best_coef_ci,
        stopped_early=stopped_early,
        best_idx=final_idx,
    )
