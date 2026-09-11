"""End-to-end TENOR-SAXS benchmark: clean database generation, noise
injection, and protocol extraction, reproducing the Figure-4/Appendix-C
benchmark driven by ``run_TENOR_benchmark.m``.

Clean-room port of ``run_TENOR_benchmark.m`` (see
``matlab/`` -- specifically
``TENOR-SAXS/Update_Sep2_2026/run_TENOR_benchmark.m`` in the sibling
``TENOR-SAXS`` repository -- for the authoritative MATLAB reference; line
numbers cited below refer to that file). The benchmark has two stages:

1. **Clean database generation** (:func:`generate_clean_database` /
   :func:`resolve_clean_database`): for each target scattering-weighted
   variance ``V`` in ``BenchmarkConfig.v_values``, find the discretizer
   inputs that realize it (:func:`tenor_saxs.distributions.target_effective_distribution`)
   and simulate the corresponding noise-free 2D detector image
   (:func:`tenor_saxs.simulation.scatter2d` with ``noise=0``). This stage
   is cached to disk (``clean_database.h5``) and only regenerated when a
   clean-relevant configuration parameter changes, mirroring MATLAB's
   ``can_reuse_database``/``extract_clean_parameters``.
2. **Noise benchmark** (:func:`run_noise_benchmark`): for every ok clean
   case, every photon-flux level, and every replicate, add photon-counting
   noise (:func:`add_noise`) and run the full TENOR protocol
   (:func:`tenor_saxs.protocol.tenor_protocol`) to recover an estimated
   ``V``/``Rg``, mirroring MATLAB's ``TENOR_run_noise_benchmark``.

:func:`run_benchmark` is the top-level orchestrator combining both stages.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from . import psf as psf_module
from .distributions import target_effective_distribution
from .mg_extract import DEADPIX_FACTOR as _DEADPIX_FACTOR_DEFAULT
from .mg_extract import QRG_MAX as _QRG_MAX_DEFAULT
from .protocol import tenor_protocol
from .simulation import scatter2d

__all__ = [
    "BenchmarkConfig",
    "generate_clean_database",
    "resolve_clean_database",
    "add_noise",
    "run_noise_benchmark",
    "run_benchmark",
]

_CLEAN_DB_FILENAME = "clean_database.h5"
_RESULTS_CSV_FILENAME = "benchmark_results.csv"
_RESULTS_PKL_FILENAME = "benchmark_results.pkl"


def _default_v_values() -> np.ndarray:
    # [uri2x-15]: linspace(0.01,0.55,12).^2 -- 12 values, 1e-4..0.3025.
    # (was (0.01:.05:.55).^2, 11 values, before the manuscript's round-4
    # benchmark rerun.)
    return np.linspace(0.01, 0.55, 12) ** 2


def _dq(sd_dist: float, wavelength: float, det_side: float, det_pix: int) -> float:
    """Detector q-space pixel pitch (1/nm), shared by :func:`photon_q_density`
    and :func:`_default_peak_photons` so the two stay consistent by
    construction rather than via a duplicated/hand-computed literal."""
    return 4.0 * np.pi / wavelength * det_side / sd_dist / (2.0 * round(det_pix / 2) + 1.0)


def _default_peak_photons() -> np.ndarray:
    # [uri2x-15]: seven half-decade photon Q-densities, N_q2 = 1e6..1e9
    # photons/nm^-2 (photons per unit q-space area -- see
    # photon_q_density -- instrument-geometry-only, not an arbitrary
    # pixel-binning-dependent count), converted to this config's own
    # default peak-photons-per-pixel via peak_photons = N_q2 * dq**2 at
    # the default geometry (sd_dist=360, wavelength=0.1, det_side=3.5,
    # det_pix=500). (Was 6 geometrically-spaced levels via
    # 10.^(2.5:.5:5)/1.65, corresponding to N_q2 = 3.2e7..1.0e10, before
    # the manuscript's round-4 benchmark rerun.)
    dq2 = _dq(360.0, 0.1, 3.5, 500) ** 2
    n_q2 = np.logspace(6.0, 9.0, 7)
    return n_q2 * dq2


def _default_pxn() -> np.ndarray:
    # [uri2x-15]'s chosen quartet, [91 81 117 127] -- NOTE this differs
    # from protocol.tenor_protocol's own (unrelated, unchanged) default
    # [87,85,125,123]. (Was [85 75 111 125] before the manuscript's
    # round-4 benchmark rerun.)
    return np.array([91, 81, 117, 127])


@dataclass(slots=True)
class BenchmarkConfig:
    """Bundled configuration for one TENOR-SAXS benchmark run.

    Field groups mirror ``TENOR_benchmark_setup`` (run_TENOR_benchmark.m:22-44):
    instrument, ensemble (+ target-search options), noise, and TENOR-analysis
    settings, plus the run-level ``output_root``/``seed``/``n_replicates``.
    """

    output_root: str
    seed: int = 314159
    n_replicates: int = 30

    # --- instrument ---
    sd_dist: float = 360.0
    wavelength: float = 0.1
    det_side: float = 3.5
    det_pix: int = 500
    psf0: np.ndarray | None = None  # defaults to psf.bartlett2d(3, 15)

    # --- ensemble ---
    rg: float = 3.0
    v_values: np.ndarray = field(default_factory=_default_v_values)
    phi2: float = 1.0 / 18.0
    distribution: str = "normal"
    n_radii: int = 25
    weight_power: float = 0.0

    # --- target-search options (target_effective_distribution kwargs) ---
    p_minimum: float = 1e-5
    p_maximum: float = 5.0
    variance_tolerance: float = 2e-4
    max_iterations: int = 50
    generator_coverage: float = 0.995
    apply_weight_again: bool = True
    expansion_factor: float = 1.6

    # --- noise ---
    peak_photons: np.ndarray = field(default_factory=_default_peak_photons)
    clip_negative: bool = True

    # --- tenor-analysis (protocol.tenor_protocol kwargs) ---
    pxn: np.ndarray = field(default_factory=_default_pxn)
    signum: float = 4.0
    use_r3: bool = False
    use_g3: bool = False
    weight_mode: str = "intensity"
    v_range: tuple[float, float] = (-0.05, 0.35)
    v_grid_n: int = 4001
    min_slope: float = 1e-8
    strategy: str = "inverseVariance"
    observables: tuple[str, ...] = ("Yg100",)
    # --- analysis window (see mg_extract.QRG_MAX / mg_extract.DEADPIX_FACTOR) ---
    qrg_max: float = _QRG_MAX_DEFAULT
    deadpix_factor: float = _DEADPIX_FACTOR_DEFAULT

    def __post_init__(self) -> None:
        if self.psf0 is None:
            self.psf0 = psf_module.bartlett2d(3, 15)
        else:
            self.psf0 = np.asarray(self.psf0, dtype=float)
        self.v_values = np.asarray(self.v_values, dtype=float)
        self.peak_photons = np.asarray(self.peak_photons, dtype=float)
        self.pxn = np.asarray(self.pxn, dtype=int)

    def clean_parameters(self) -> dict[str, Any]:
        """JSON-serializable dict of every field affecting clean-map generation.

        Mirrors ``extract_clean_parameters`` (run_TENOR_benchmark.m:215-240):
        instrument geometry/PSF, ensemble shape/discretization settings, the
        target-search options, and the per-case ``True_V``/``Phi2`` columns.
        Deliberately EXCLUDES ``seed``, ``n_replicates``, noise settings, and
        tenor-analysis settings -- those may change without invalidating the
        cached clean database.
        """
        return {
            "schema_version": 1,
            "instrument": {
                "sd_dist": float(self.sd_dist),
                "wavelength": float(self.wavelength),
                "det_side": float(self.det_side),
                "det_pix": int(self.det_pix),
                "psf0": np.asarray(self.psf0, dtype=float).tolist(),
            },
            "ensemble": {
                "rg": float(self.rg),
                "distribution": str(self.distribution),
                "n_radii": int(self.n_radii),
                "weight_power": float(self.weight_power),
                "target_options": {
                    "p_minimum": float(self.p_minimum),
                    "p_maximum": float(self.p_maximum),
                    "variance_tolerance": float(self.variance_tolerance),
                    "max_iterations": int(self.max_iterations),
                    "generator_coverage": float(self.generator_coverage),
                    "apply_weight_again": bool(self.apply_weight_again),
                    "expansion_factor": float(self.expansion_factor),
                },
            },
            "cases": {
                "true_v": np.asarray(self.v_values, dtype=float).tolist(),
                "phi2": float(self.phi2),
            },
        }


# ---------------------------------------------------------------------------
# Atomic-write helpers (same-directory temp file + os.replace, matching
# tenor_saxs_protocol/io_utils.py's save_simulation_h5 pattern -- avoids
# cross-filesystem rename failures).
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, write_fn) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


_MANIFEST_STRING_COLUMNS = {"Status", "Error"}


def _write_manifest_group(h5_group: h5py.Group, manifest: pd.DataFrame) -> None:
    for col in manifest.columns:
        values = manifest[col]
        if col in _MANIFEST_STRING_COLUMNS or values.dtype == object:
            data = np.asarray([("" if v is None else str(v)) for v in values.tolist()], dtype=object)
            h5_group.create_dataset(col, data=data, dtype=h5py.string_dtype())
        else:
            h5_group.create_dataset(col, data=np.asarray(values, dtype=float))


def _read_manifest_group(h5_group: h5py.Group) -> pd.DataFrame:
    data = {}
    for col in h5_group.keys():
        values = h5_group[col][()]
        if values.dtype.kind in ("O", "S"):
            values = [v.decode() if isinstance(v, bytes) else v for v in values]
        data[col] = values
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Stage 1: clean (noise-free) database generation
# ---------------------------------------------------------------------------


def generate_clean_database(
    config: BenchmarkConfig, out_dir: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Simulate one noise-free 2D detector image per target ``V`` in ``config.v_values``.

    Port of ``TENOR_generate_clean_database``'s per-case loop
    (run_TENOR_benchmark.m:109-213, minus the caching wrapper -- see
    :func:`resolve_clean_database` for that). For each case:

    1. :func:`tenor_saxs.distributions.target_effective_distribution` finds
       the discretizer inputs (``requested_rg``, ``input_variance``) that
       realize the target scattering-weighted ``V`` at the target observed
       ``Rg``.
    2. :func:`tenor_saxs.simulation.scatter2d` simulates the corresponding
       noise-free (``noise=0``) detector image.

    Any exception raised by either step is caught PER CASE: that case is
    recorded with ``Status="skipped"`` and the exception message in
    ``Error``, and the loop continues to the remaining cases, exactly as
    MATLAB's ``try``/``catch`` around each case (run_TENOR_benchmark.m:133-197)
    does -- one bad case never aborts the whole clean-database build.

    Returns
    -------
    qx, qy : np.ndarray
        The (shared) q-space grid, from the first successfully simulated case.
    I_clean : np.ndarray
        Shape ``(n_cases, *qx.shape)``. Rows for skipped cases are left as
        zero (matching MATLAB's zero-preallocated, never-overwritten slots).
    manifest : pd.DataFrame
        One row per case with columns ``CaseID, True_V, Phi2, RequestedRg,
        InputVariance, RealizedV, PredictedObservedRg, WeightedRmsFactor,
        Status, Error`` (plus ``True_p``/``PNumerical`` bonus columns).

    Raises
    ------
    RuntimeError
        If every single case fails (mirrors MATLAB's "TENOR:NoValidCases").
    """
    v_values = np.asarray(config.v_values, dtype=float)
    n_cases = v_values.size

    qx: np.ndarray | None = None
    qy: np.ndarray | None = None
    i_clean: np.ndarray | None = None
    records: list[dict[str, Any]] = []

    for idx, true_v in enumerate(v_values):
        case_id = idx + 1
        record: dict[str, Any] = {
            "CaseID": case_id,
            "True_V": float(true_v),
            "Phi2": float(config.phi2),
            "True_p": float(np.sqrt(max(true_v, 0.0))),
        }
        try:
            target = target_effective_distribution(
                target_v=float(true_v),
                target_observed_rg=config.rg,
                dist_name=config.distribution,
                n=config.n_radii,
                weight_power=config.weight_power,
                p_minimum=config.p_minimum,
                p_maximum=config.p_maximum,
                variance_tolerance=config.variance_tolerance,
                max_iterations=config.max_iterations,
                generator_coverage=config.generator_coverage,
                apply_weight_again=config.apply_weight_again,
                expansion_factor=config.expansion_factor,
            )
            result = scatter2d(
                rg=target["requested_rg"],
                noise=0,
                v_rel=target["input_variance"],
                phi2=config.phi2,
                det_pix=config.det_pix,
                sd_dist=config.sd_dist,
                wavelength=config.wavelength,
                det_side=config.det_side,
                psf0=config.psf0,
                dist_type=config.distribution,
                n_radii=config.n_radii,
                weight_power=config.weight_power,
            )
            if qx is None:
                qx, qy = result.qx, result.qy
                i_clean = np.zeros((n_cases,) + qx.shape, dtype=float)
            elif result.qx.shape != qx.shape:
                raise ValueError("The q-grid size changed between clean cases.")
            i_clean[idx] = result.intensity

            record.update(
                Status="ok",
                Error="",
                RequestedRg=float(target["requested_rg"]),
                InputVariance=float(target["input_variance"]),
                RealizedV=float(target["realized_v"]),
                PredictedObservedRg=float(target["predicted_observed_rg"]),
                WeightedRmsFactor=float(target["weighted_rms_factor"]),
                PNumerical=float(target["p_numerical"]),
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, mirrors MATLAB's try/catch
            record.update(
                Status="skipped",
                Error=str(exc),
                RequestedRg=np.nan,
                InputVariance=np.nan,
                RealizedV=np.nan,
                PredictedObservedRg=np.nan,
                WeightedRmsFactor=np.nan,
                PNumerical=np.nan,
            )
        records.append(record)

    if qx is None:
        raise RuntimeError(
            "No case produced a valid distribution and clean intensity map "
            "(mirrors MATLAB's TENOR:NoValidCases)."
        )

    manifest = pd.DataFrame(records)
    column_order = [
        "CaseID",
        "True_V",
        "Phi2",
        "RequestedRg",
        "InputVariance",
        "RealizedV",
        "PredictedObservedRg",
        "WeightedRmsFactor",
        "Status",
        "Error",
        "True_p",
        "PNumerical",
    ]
    manifest = manifest[column_order]
    return qx, qy, i_clean, manifest


def _save_clean_database(path: Path, config: BenchmarkConfig, qx: np.ndarray, qy: np.ndarray, i_clean: np.ndarray, manifest: pd.DataFrame) -> None:
    clean_params_json = json.dumps(config.clean_parameters(), sort_keys=True)

    def _write(tmp_path: Path) -> None:
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset("qx", data=qx, compression="gzip", compression_opts=5)
            h5.create_dataset("qy", data=qy, compression="gzip", compression_opts=5)
            h5.create_dataset("I_clean", data=i_clean, compression="gzip", compression_opts=5)
            manifest_group = h5.create_group("manifest")
            _write_manifest_group(manifest_group, manifest)
            h5.attrs["clean_parameters_json"] = clean_params_json

    _atomic_write(path, _write)


def _load_clean_database(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, str]:
    with h5py.File(path, "r") as h5:
        qx = h5["qx"][()]
        qy = h5["qy"][()]
        i_clean = h5["I_clean"][()]
        manifest = _read_manifest_group(h5["manifest"])
        clean_params_json = h5.attrs["clean_parameters_json"]
    return qx, qy, i_clean, manifest, clean_params_json


def resolve_clean_database(
    config: BenchmarkConfig, out_dir: str | Path, overwrite: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a cached clean database if its parameters match, else (re)generate it.

    Port of ``TENOR_generate_clean_database``'s caching wrapper around the
    per-case loop, specifically ``can_reuse_database``
    (run_TENOR_benchmark.m:49-108, 242-307): the cache at
    ``out_dir/clean_database.h5`` is reused only if it exists, ``overwrite``
    is ``False``, and its stored clean-parameters blob is byte-for-byte
    identical (via JSON serialization) to ``config.clean_parameters()`` --
    any difference (instrument, ensemble, target-search settings, or the
    ``True_V``/``Phi2`` case list) triggers a full regeneration, exactly as
    MATLAB's ``isequaln`` comparison does. Noise and tenor-analysis settings
    are excluded from this comparison by construction (see
    :meth:`BenchmarkConfig.clean_parameters`).
    """
    out_dir = Path(out_dir)
    db_path = out_dir / _CLEAN_DB_FILENAME
    requested_params_json = json.dumps(config.clean_parameters(), sort_keys=True)

    if not overwrite and db_path.exists():
        try:
            qx, qy, i_clean, manifest, stored_params_json = _load_clean_database(db_path)
            if stored_params_json == requested_params_json and len(manifest) == len(config.v_values):
                return qx, qy, i_clean, manifest
        except Exception:
            pass  # fall through to regeneration on any read/parse failure

    qx, qy, i_clean, manifest = generate_clean_database(config, out_dir)
    _save_clean_database(db_path, config, qx, qy, i_clean, manifest)
    return qx, qy, i_clean, manifest


# ---------------------------------------------------------------------------
# Stage 2: noise injection + TENOR protocol extraction
# ---------------------------------------------------------------------------


def add_noise(
    intensity_clean: np.ndarray,
    peak_photons: float,
    seed: int | None = None,
    clip_negative: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add photon-counting (shot) noise to a clean intensity map.

    Port of ``TENOR_add_noise`` (run_TENOR_benchmark.m:393-404) called ALWAYS
    with a negative ``level`` (``level = -peak_photons``) -- the benchmark
    never exercises MATLAB's flat-Gaussian branch (``level >= 0``):
    ``I = I0 + sqrt(max(I0,0)/peak_photons) * randn(size(I0))``, optionally
    clipping negative finite results to zero.

    NOTE ON REPRODUCIBILITY: MATLAB seeds a global Mersenne-Twister stream
    (``rng(seed,'twister')``) while this uses NumPy's PCG64
    (``np.random.default_rng(seed)``). Even with an identical ``seed``, the
    actual noise realization will NOT match MATLAB's bit-for-bit -- only the
    deterministic seed-DERIVATION formula in :func:`run_noise_benchmark`
    matches MATLAB's, giving structural/reproducibility parity (same case
    always gets the same seed value across runs of this Python port), not
    numerical parity with the original MATLAB study.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    i0 = np.asarray(intensity_clean, dtype=float)
    noisy = i0 + np.sqrt(np.clip(i0, 0.0, None) / peak_photons) * rng.standard_normal(i0.shape)
    if clip_negative:
        bad = np.isfinite(noisy) & (noisy < 0)
        noisy[bad] = 0.0
    return noisy


def photon_q_density(peak_photons: float, sd_dist: float, wavelength: float, det_side: float, det_pix: int) -> float:
    """Peak photon Q-DENSITY (photons per unit q-space area), not raw photon count.

    Per an internal design review: the peak-photons-per-pixel count
    alone depends on pixel size, which is an arbitrary detector-binning
    choice; ``peak_photons / dq**2`` (units "photons/nm^-2", i.e. per unit
    AREA of q-space measured in inverse-nm) does not, making it the more
    physically meaningful x-axis for a noise-sensitivity comparison across
    different instrument configurations. Matches
    ``plot_tenor_violin_1dGT.m:13``'s
    ``dq = 4*pi/lambda*det_side/SD_dist/(2*round(DETpix/2)+1)`` exactly
    (an isotropic, single pixel-pitch value -- the detector is square).

    Note this quantity depends ONLY on instrument geometry, not on the
    ensemble (R0, distribution, etc.) -- the more physically complete
    quantity would be "photons actually collected within the Guinier
    region", which DOES depend on the ensemble (a larger R0 means a
    smaller accessible q-range), but per the user's own instruction this
    simpler instrument-only quantity is what's plotted, using R0=3nm
    (matching this package's own noise-benchmark default) for consistency
    with the reference violin plot it's being compared against.
    """
    dq = _dq(sd_dist, wavelength, det_side, det_pix)
    return float(peak_photons) / dq**2


def run_noise_benchmark(
    config: BenchmarkConfig,
    qx: np.ndarray,
    qy: np.ndarray,
    i_clean: np.ndarray,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Add fresh noise to every ok clean case and run the TENOR protocol on it.

    Port of ``TENOR_run_noise_benchmark`` (run_TENOR_benchmark.m:406-450).
    For each case with ``Status=="ok"`` in ``manifest``, each of
    ``config.peak_photons`` (1-based index ``j``), and each replicate ``r``
    from 1 to ``config.n_replicates``, a deterministic seed is derived as::

        sd = (config.seed + 104729*case_id + 1009*j + 37*r) % (2**32 - 1)
        sd = 1 if sd == 0 else sd

    IMPORTANT / DISCREPANCY FROM AN EARLIER DRAFT OF THIS SPEC: ``case_id``
    here is each case's RAW, 1-based ``CaseID`` from the FULL (unfiltered)
    case list -- NOT that case's position within just the "ok" subset. In
    MATLAB, ``ok = find(manifest.Status == "ok")`` and the loop body uses
    ``ok(c)`` directly in the seed formula; since MATLAB's ``CaseID`` column
    is assigned sequentially ``1:N`` before any filtering, ``ok(c)`` equals
    the ORIGINAL row index (== CaseID) of the c-th ok case, not a compacted
    ``1..num_ok`` counter. The two notions coincide only when no case was
    skipped; whenever an earlier case was skipped, the raw CaseID and the
    ok-subset position diverge, and MATLAB's seed formula follows the raw
    CaseID. This port matches that raw-CaseID behavior.

    Each triple's noisy image goes through
    :func:`tenor_saxs.protocol.tenor_protocol` with the case's own
    ``Phi2`` and the config's tenor-analysis settings. Exceptions are caught
    PER (case, noise level, replicate) triple: that row is recorded with
    ``Status="failed"`` and the exception message, and the sweep continues.

    NOTE ON REPRODUCIBILITY: see :func:`add_noise`'s docstring -- matching
    ``seed`` values do NOT reproduce MATLAB's specific noise realizations
    bit-for-bit (different RNG algorithms/streams). Only the deterministic
    seed-DERIVATION formula matches, for structural/reproducibility parity
    across repeated runs of this Python port, not numerical parity with the
    original MATLAB study.

    Returns
    -------
    pd.DataFrame
        Columns: ``CaseID, Replicate, NoiseLevelIndex, PeakPhotons,
        PhotonQDensity, Seed, True_V, True_Rg, True_R0, Phi2, BestV,
        BestV_SE, Rg, RgCorrected, Status, Error``. ``True_Rg`` is the
        apparent (Guinier) radius, fixed by construction; ``True_R0`` is the
        true scattering-weighted mean radius that ``RgCorrected`` estimates.
        ``PhotonQDensity`` (see
        :func:`photon_q_density`) is the instrument-geometry-only photon
        density in photons/nm^-2, independent of pixel binning -- the
        more physically meaningful noise axis for cross-instrument
        comparison (an internal design review).
    """
    ok_manifest = manifest[manifest["Status"] == "ok"].reset_index(drop=True)
    records: list[dict[str, Any]] = []

    for row in ok_manifest.itertuples(index=False):
        case_id = int(row.CaseID)
        idx = case_id - 1
        i0 = i_clean[idx]
        true_v = float(row.True_V)
        # PredictedObservedRg is the actually-realized target Rg for this
        # case (matches MATLAB's use of cfg.ensemble.rg as True_Rg, which for
        # this port is the realized/predicted observed Rg from the target
        # search rather than the nominal input config.rg).
        true_rg = float(row.PredictedObservedRg)
        # RequestedRg is the true SCATTERING-WEIGHTED mean radius the target
        # search aimed for -- what RgCorrected estimates. True_Rg (above) is
        # the APPARENT (Guinier) radius, biased upward by polydispersity and
        # fixed by construction for this case; the two coincide only at
        # V=0. True_R0 is what the corrected-radius discrepancy (Rg
        # violin) should be compared against, not True_Rg.
        true_r0 = float(row.RequestedRg)
        phi2_case = float(row.Phi2)

        for j, peak_photons in enumerate(config.peak_photons, start=1):
            for r in range(1, config.n_replicates + 1):
                seed = (config.seed + 104729 * case_id + 1009 * j + 37 * r) % (2**32 - 1)
                if seed == 0:
                    seed = 1

                record: dict[str, Any] = {
                    "CaseID": case_id,
                    "Replicate": r,
                    "NoiseLevelIndex": j,
                    "PeakPhotons": float(peak_photons),
                    "PhotonQDensity": photon_q_density(
                        peak_photons, config.sd_dist, config.wavelength, config.det_side, config.det_pix
                    ),
                    "Seed": int(seed),
                    "True_V": true_v,
                    "True_Rg": true_rg,
                    "True_R0": true_r0,
                    "Phi2": phi2_case,
                }
                try:
                    noisy = add_noise(i0, float(peak_photons), seed=seed, clip_negative=config.clip_negative)
                    result = tenor_protocol(
                        noisy,
                        qx,
                        qy,
                        phi2=phi2_case,
                        pxn=config.pxn,
                        signum=config.signum,
                        use_r3=config.use_r3,
                        use_g3=config.use_g3,
                        weight_mode=config.weight_mode,
                        wavelength=config.wavelength,
                        v_range=config.v_range,
                        v_grid_n=config.v_grid_n,
                        min_slope=config.min_slope,
                        strategy=config.strategy,
                        observables=config.observables,
                        qrg_max=config.qrg_max,
                        deadpix_factor=config.deadpix_factor,
                    )
                    record.update(
                        BestV=float(result.best_v),
                        BestV_SE=float(result.best_v_se),
                        Rg=float(result.rg),
                        RgCorrected=float(result.rg_corrected),
                        Status="ok",
                        Error="",
                    )
                except Exception as exc:  # noqa: BLE001 - deliberately broad, mirrors MATLAB's try/catch
                    record.update(
                        BestV=np.nan,
                        BestV_SE=np.nan,
                        Rg=np.nan,
                        RgCorrected=np.nan,
                        Status="failed",
                        Error=str(exc),
                    )
                records.append(record)

    column_order = [
        "CaseID",
        "Replicate",
        "NoiseLevelIndex",
        "PeakPhotons",
        "PhotonQDensity",
        "Seed",
        "True_V",
        "True_Rg",
        "True_R0",
        "Phi2",
        "BestV",
        "BestV_SE",
        "Rg",
        "RgCorrected",
        "Status",
        "Error",
    ]
    if not records:
        return pd.DataFrame(columns=column_order)
    return pd.DataFrame(records)[column_order]


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_benchmark(config: BenchmarkConfig, overwrite_clean: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full TENOR-SAXS benchmark: resolve the clean database, then noise-sweep it.

    Port of the top-level ``run_TENOR_benchmark`` (run_TENOR_benchmark.m:1-20).
    Saves ``results_df`` to ``config.output_root/benchmark_results.csv`` (and
    a companion ``.pkl`` preserving dtypes/NaNs exactly) via the same
    same-directory-temp-file-then-rename atomic-write pattern used for the
    clean database cache.

    Returns
    -------
    (results_df, manifest) : tuple[pd.DataFrame, pd.DataFrame]
    """
    out_dir = Path(config.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    qx, qy, i_clean, manifest = resolve_clean_database(config, out_dir, overwrite=overwrite_clean)
    results_df = run_noise_benchmark(config, qx, qy, i_clean, manifest)

    csv_path = out_dir / _RESULTS_CSV_FILENAME
    pkl_path = out_dir / _RESULTS_PKL_FILENAME
    _atomic_write(csv_path, lambda tmp: results_df.to_csv(tmp, index=False))
    _atomic_write(pkl_path, lambda tmp: results_df.to_pickle(tmp))

    return results_df, manifest
