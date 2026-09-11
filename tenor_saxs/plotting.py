"""Violin-style diagnostic plots for TENOR-SAXS benchmark results.

Clean-room port of ``plot_tenor_violin_1dGT.m`` (see
``TENOR-SAXS/Update_Sep2_2026/plot_tenor_violin_1dGT.m`` in the sibling
``TENOR-SAXS`` repository for the authoritative MATLAB reference), which
draws three per-noise-level violin panels from a
:func:`tenor_saxs.benchmark.run_benchmark` results table: the ``V``
(equivalently ``p = sqrt(V)``) discrepancy and the relative ``Rg``
discrepancy.

Substitutions from the MATLAB original
-----------------------------------------
- MATLAB builds each violin manually via ``histcounts`` + a hand-drawn
  ``patch``. Here, :func:`matplotlib.axes.Axes.violinplot` is used instead --
  a standard, modern replacement producing an equivalent horizontal-density
  shape per noise level, without reimplementing histogram-to-polygon
  plumbing.
- MATLAB's jitter is proportional to each point's own ``True_p`` relative to
  the noise-level's mean ``True_p`` (``plot_tenor_violin_1dGT.m:146``) --
  meaningful there because a single noise level can mix multiple ``V``
  cases. Here, jitter is a plain uniform random offset
  (``matplotlib``/NumPy's own RNG), which is simpler and visually
  equivalent for the purpose of showing point density without overlap.
- MATLAB's x-axis is a derived "photon density" (photons/nm^-2, via the
  instrument's pixel pitch); here the x-axis is directly ``PeakPhotons``
  (log-scaled, since the six levels are geometrically spaced), which is
  simpler and avoids re-deriving instrument-specific units.

Two bugs fixed after initial review (see the package's internal validation notes):
- ``_percent_valid`` originally checked ``Status == "ok"`` (meaning
  ``tenor_protocol`` did not raise -- a much weaker condition than a
  usable result, since ``combine_estimates`` can legitimately return
  ``(nan, nan)`` without raising), always reporting close to 100% valid
  even when most rows had an unusable ``NaN`` ``BestV``. Now checks
  ``np.isfinite(BestV)`` directly, matching MATLAB's own definition.
- The per-level mean/std annotation now uses a 5%-outlier-trimmed
  mean/std (:func:`_trimmed_mean_std`), matching the paper's Fig. 6
  caption ("5% outliers were omitted from the analysis") -- previously a
  plain, non-robust mean/std.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats as _stats

__all__ = [
    "plot_v_discrepancy_violin",
    "plot_rg_discrepancy_violin",
    "plot_r0_discrepancy_violin",
    "plot_tenor_benchmark_violins",
]

_JITTER_WIDTH = 0.2
_VIOLIN_HALF_WIDTH = 0.35
_OUTLIER_TRIM_FRACTION = 0.05  # matches the paper's "5% outliers were omitted from the analysis"


def _trimmed_mean_std(values: np.ndarray, trim_fraction: float = _OUTLIER_TRIM_FRACTION) -> tuple[float, float]:
    """Robust (outlier-trimmed) mean and std, matching the paper's stated convention.

    The paper's Fig. 6 caption: "5% outliers were omitted from the
    analysis. The mean bias is represented by a solid line, while the
    discrepancy standard deviation margin is indicated by the dashed
    line." Symmetric two-sided trim (``trim_fraction/2`` from each tail,
    via :func:`scipy.stats.trim_mean`) drops the most extreme
    ``trim_fraction`` of points in total before computing the mean; the
    std is computed on that same trimmed subset for consistency. Only
    these SUMMARY statistics are robustified -- the violin/scatter still
    shows every finite point, matching the paper's own figure (which
    plots the full point cloud but annotates a trimmed mean+/-std).
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size < 3 or trim_fraction <= 0:
        return float(np.mean(values)), float(np.std(values))
    trimmed_mean = float(_stats.trim_mean(values, trim_fraction / 2.0))
    lo, hi = np.percentile(values, [100 * trim_fraction / 2.0, 100 * (1 - trim_fraction / 2.0)])
    kept = values[(values >= lo) & (values <= hi)]
    trimmed_std = float(np.std(kept)) if kept.size > 0 else float("nan")
    return trimmed_mean, trimmed_std


def _filter_noiseless(results_df: pd.DataFrame) -> pd.DataFrame:
    """Drop any exact zero-noise rows, defensively.

    Mirrors ``plot_tenor_violin_1dGT.m:25``'s ``data_tab(data_tab.Noise ~= 0,
    :)`` filter. :func:`tenor_saxs.benchmark.run_noise_benchmark` never
    emits a zero-``PeakPhotons`` row (every noise level in
    ``BenchmarkConfig.peak_photons`` is strictly positive), so this filter is
    a no-op in normal use -- kept only as a defensive guard against a
    caller-constructed table that happens to include one.
    """
    if "PeakPhotons" in results_df.columns:
        return results_df[results_df["PeakPhotons"] != 0]
    return results_df


def _x_axis_column(results_df: pd.DataFrame) -> str:
    """``"PhotonQDensity"`` if present, else fall back to ``"PeakPhotons"``.

    Per an internal design review: the photon Q-DENSITY
    (:func:`tenor_saxs.benchmark.photon_q_density`, photons/nm^-2) is
    the more physically meaningful noise axis -- it depends only on
    instrument geometry, not on an arbitrary pixel-binning choice, unlike
    the raw peak-photons-per-pixel count. Falls back to ``"PeakPhotons"``
    for any caller-constructed table that doesn't have the column (e.g. a
    synthetic table built directly rather than via
    :func:`tenor_saxs.benchmark.run_noise_benchmark`).
    """
    return "PhotonQDensity" if "PhotonQDensity" in results_df.columns else "PeakPhotons"


def _x_axis_label(results_df: pd.DataFrame) -> str:
    return "Photon Q-density (photons/nm$^{-2}$)" if _x_axis_column(results_df) == "PhotonQDensity" else "Peak photons"


def _noise_levels(results_df: pd.DataFrame) -> np.ndarray:
    return np.sort(results_df[_x_axis_column(results_df)].unique())


def _percent_valid(sub: pd.DataFrame) -> float:
    """Percentage of rows at this noise level with a finite recovered ``BestV``.

    Matches MATLAB's own "% valid" definition (plot_tenor_violin_1dGT.m):
    the fraction of rows where the observable was NOT outside the valid
    analytical range, i.e. a finite ``V_est`` was produced. Deliberately
    NOT ``Status == "ok"`` -- that only means ``tenor_protocol`` did not
    raise an exception, which is a much weaker condition: ``combine_estimates``
    can legitimately return ``(nan, nan)`` (e.g. every observable landed
    "outside calibration range" or "multiple analytical roots") without
    raising anything, especially at low photon counts. Checking ``Status``
    alone silently overcounted validity (found showing 100% even when a
    majority of rows had a NaN ``BestV`` -- caught the same way the
    weighting-scheme study's own paired-test bug was caught, see
    the internal validation notes).
    """
    if len(sub) == 0:
        return float("nan")
    return 100.0 * float(np.isfinite(sub["BestV"]).mean())


def _draw_violin_panel(
    ax: matplotlib.axes.Axes,
    results_df: pd.DataFrame,
    value_fn,
    ylabel: str,
    rng: np.random.Generator,
) -> matplotlib.axes.Axes:
    """Shared drawing logic for one discrepancy panel across noise levels.

    For each unique ``PeakPhotons`` level (an x position at its own value, on
    a log-scaled axis): draws a violin of ``value_fn(sub)`` over rows with a
    finite value, a jittered scatter of the individual points, a red mean
    marker with a mean+/-std line, and a "<pct>% valid" text annotation
    (``Status=="ok"`` fraction).

    Positions are the actual ``PeakPhotons`` values (not a 1..n categorical
    index) so that ``ax.set_xscale("log")`` reflects their true geometric
    spacing; widths/jitter/mean-marker half-widths scale multiplicatively
    with each position rather than additively, since additive offsets on a
    log axis would look inconsistent across widely-spaced levels.
    """
    x_col = _x_axis_column(results_df)
    noise_levels = _noise_levels(results_df)
    positions = noise_levels

    # First pass: gather per-level values/stats so the "% valid" annotation
    # can be placed relative to the GLOBAL data range across all levels,
    # rather than an incrementally-autoscaled (and therefore order-dependent)
    # y-limit.
    per_level_values: list[np.ndarray] = []
    per_level_pct_valid: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for noise in noise_levels:
        sub = results_df[results_df[x_col] == noise]
        per_level_pct_valid.append(_percent_valid(sub))
        values = value_fn(sub)
        values = values[np.isfinite(values)]
        per_level_values.append(values)
        mean, std = _trimmed_mean_std(values)
        means.append(mean)
        stds.append(std)

    all_values = np.concatenate([v for v in per_level_values if v.size > 0]) if any(v.size for v in per_level_values) else np.array([0.0])
    data_min, data_max = float(np.min(all_values)), float(np.max(all_values))
    data_span = max(data_max - data_min, 1e-12)
    y_text = data_min - 0.05 * data_span

    violin_data: list[np.ndarray] = []
    violin_positions: list[float] = []
    violin_widths: list[float] = []
    for i, (pos, values, pct_valid) in enumerate(zip(positions, per_level_values, per_level_pct_valid)):
        if values.size > 0:
            violin_data.append(values)
            violin_positions.append(pos)
            violin_widths.append(2 * _VIOLIN_HALF_WIDTH * pos)
            mean = means[i]

            # Multiplicative jitter/mean-marker half-width: an additive
            # offset would look wildly inconsistent across the (typically
            # geometrically-spaced) noise levels on this log-scaled axis.
            jitter = 10.0 ** ((rng.random(values.size) - 0.5) * _JITTER_WIDTH)
            ax.plot(pos * jitter, values, ".", color="0.5", markersize=4, zorder=2)
            ax.plot([pos * 0.85, pos * 1.15], [mean, mean], color="red", linewidth=2, zorder=3)

        ax.text(
            pos,
            y_text,
            f"{0 if np.isnan(pct_valid) else round(pct_valid)}%\nvalid",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    if violin_data:
        parts = ax.violinplot(violin_data, positions=violin_positions, widths=violin_widths, showmeans=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("tab:blue")
            body.set_edgecolor("tab:blue")
            body.set_alpha(0.25)

    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)
    if len(positions) > 1:
        ax.plot(positions, means_arr, "r-", linewidth=1.2, label=f"Trimmed mean ({100*_OUTLIER_TRIM_FRACTION:.0f}% outliers omitted)")
        ax.plot(positions, means_arr + stds_arr, "k--", linewidth=1, label="Trimmed mean +/- 1 STD")
        ax.plot(positions, means_arr - stds_arr, "k--", linewidth=1)
        ax.legend(loc="best", fontsize=8)

    ax.set_xscale("log")
    ax.set_xticks(positions)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{n:.3g}" for n in noise_levels]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(_x_axis_label(results_df))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax


def _v_discrepancy(sub: pd.DataFrame) -> np.ndarray:
    """``sqrt(max(0, BestV)) - sqrt(True_V)`` -- identical to the ``p`` discrepancy.

    ``plot_tenor_violin_1dGT.m``'s own V-mode (line 88) and p-mode (line 96)
    panels compute numerically identical quantities (``p_est = sqrt(max(0,
    BestV))`` and ``True_p = sqrt(True_V)`` at data-generation time), a
    redundancy the MATLAB source itself notes in its structure -- this
    module exposes only the one panel.
    """
    best_v = np.clip(sub["BestV"].to_numpy(dtype=float), 0.0, None)
    true_v = sub["True_V"].to_numpy(dtype=float)
    return np.sqrt(best_v) - np.sqrt(true_v)


def _rg_relative_discrepancy(sub: pd.DataFrame) -> np.ndarray:
    """``(Rg - True_Rg) / True_Rg``."""
    rg = sub["Rg"].to_numpy(dtype=float)
    true_rg = sub["True_Rg"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rg - true_rg) / true_rg


def _r0_relative_discrepancy(sub: pd.DataFrame) -> np.ndarray:
    """``(RgCorrected - True_R0) / True_R0``.

    Unlike :func:`_rg_relative_discrepancy` (the apparent, Guinier-biased
    radius against its own fixed-by-construction target), this compares the
    polydispersity-CORRECTED radius against the true scattering-weighted
    mean -- the manuscript's current Fig. 5, replacing the earlier
    apparent-radius panel (which has no physical interest: the apparent
    radius is fixed by construction, so its own discrepancy is a
    tautology). Requires a results table produced after ``True_R0`` was
    added to :func:`tenor_saxs.benchmark.run_noise_benchmark` -- raises a
    descriptive ``KeyError`` on an older table rather than silently doing
    the wrong comparison.
    """
    missing = [c for c in ("RgCorrected", "True_R0") if c not in sub.columns]
    if missing:
        raise KeyError(
            f"results table is missing {missing!r} -- regenerate it with the "
            "current tenor_saxs.benchmark.run_noise_benchmark (True_R0 was "
            "added alongside RgCorrected; an older results table predates it)"
        )
    rg_corrected = sub["RgCorrected"].to_numpy(dtype=float)
    true_r0 = sub["True_R0"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rg_corrected - true_r0) / true_r0


def plot_v_discrepancy_violin(
    results_df: pd.DataFrame, ax: matplotlib.axes.Axes | None = None, rng: np.random.Generator | None = None
) -> matplotlib.axes.Axes:
    """Per-noise-level violin of the ``sqrt(V)`` (equivalently ``p``) discrepancy.

    Port of ``plot_tenor_violin_1dGT.m``'s Figure-1/Figure-3 panels (V and p
    discrepancy, which are numerically identical -- see
    :func:`_v_discrepancy`). Accepts/returns a matplotlib ``Axes`` rather
    than calling ``plt.show()``, so callers can save or further customize
    the figure.
    """
    results_df = _filter_noiseless(results_df)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    if rng is None:
        rng = np.random.default_rng()
    return _draw_violin_panel(ax, results_df, _v_discrepancy, r"$\sqrt{V}$ discrepancy", rng)


def plot_rg_discrepancy_violin(
    results_df: pd.DataFrame, ax: matplotlib.axes.Axes | None = None, rng: np.random.Generator | None = None
) -> matplotlib.axes.Axes:
    """Per-noise-level violin of the relative ``Rg`` discrepancy ``(Rg - True_Rg)/True_Rg``.

    Port of ``plot_tenor_violin_1dGT.m``'s Figure-2 panel (``hasTrueRg``
    branch, lines 105-111).
    """
    results_df = _filter_noiseless(results_df)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    if rng is None:
        rng = np.random.default_rng()
    ax = _draw_violin_panel(ax, results_df, _rg_relative_discrepancy, r"$\Delta R_g / R_{g,\mathrm{true}}$", rng)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.2)
    return ax


def plot_r0_discrepancy_violin(
    results_df: pd.DataFrame, ax: matplotlib.axes.Axes | None = None, rng: np.random.Generator | None = None
) -> matplotlib.axes.Axes:
    """Per-noise-level violin of the corrected-radius discrepancy ``(RgCorrected - True_R0)/True_R0``.

    The manuscript's current Fig. 5 -- see :func:`_r0_relative_discrepancy`.
    """
    results_df = _filter_noiseless(results_df)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    if rng is None:
        rng = np.random.default_rng()
    ax = _draw_violin_panel(ax, results_df, _r0_relative_discrepancy, r"$\Delta R_0 / R_{0,\mathrm{true}}$", rng)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.2)
    return ax


_PANEL_SPECS: dict[str, tuple] = {
    "V": (_v_discrepancy, r"$\sqrt{V}$ discrepancy", "V discrepancy", False),
    "p": (_v_discrepancy, r"$p$ discrepancy", "p discrepancy (identical to V)", False),
    "Rg": (_rg_relative_discrepancy, r"$\Delta R_g / R_{g,\mathrm{true}}$", "Rg relative discrepancy", True),
    "R0": (_r0_relative_discrepancy, r"$\Delta R_0 / R_{0,\mathrm{true}}$", "R0 relative discrepancy (corrected)", True),
}


def plot_tenor_benchmark_violins(
    results_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
    panels: Sequence[str] = ("V", "p", "Rg"),
) -> tuple[matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]]:
    """Multi-panel figure of the requested discrepancy panels.

    Port of ``plot_tenor_violin_1dGT``'s top-level multi-figure layout,
    combined into one ``matplotlib.pyplot.subplots`` figure rather than
    separate MATLAB figure windows. Returns ``(fig, axes)`` without calling
    ``plt.show()``.

    ``panels`` selects which of ``"V"`` (sqrt(V) discrepancy), ``"p"``
    (identical data, kept for MATLAB-figure parity), ``"Rg"`` (apparent-
    radius relative discrepancy) and ``"R0"`` (corrected-radius relative
    discrepancy, the manuscript's current Fig. 5 -- see
    :func:`plot_r0_discrepancy_violin`) to draw, and in what order. Default
    is unchanged from before this parameter existed (``("V", "p", "Rg")``);
    the manuscript's current pair is ``panels=("V", "R0")``. Raises
    ``ValueError`` on an unknown panel name.
    """
    unknown = [p for p in panels if p not in _PANEL_SPECS]
    if unknown:
        raise ValueError(f"unknown panel name(s) {unknown!r}; choose from {sorted(_PANEL_SPECS)}")

    results_df = _filter_noiseless(results_df)
    if rng is None:
        rng = np.random.default_rng()

    fig, axes = plt.subplots(1, len(panels), figsize=(16 * len(panels) / 3, 4.5), squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, panels):
        value_fn, ylabel, title, zero_line = _PANEL_SPECS[name]
        _draw_violin_panel(ax, results_df, value_fn, ylabel, rng)
        if zero_line:
            ax.axhline(0.0, color="k", linestyle="--", linewidth=1.2)
        ax.set_title(title)

    fig.tight_layout()
    return fig, axes
