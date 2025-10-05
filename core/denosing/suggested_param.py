
# -*- coding: utf-8 -*-
"""
suggested_param.py
------------------
Light-curve driven suggestion of (transit_duration_hours, rho_mult) for GP detrending.

Public API:
- suggest_from_df(df, time_col="time", flux_col="flux", error_col=None) -> dict
- suggest_from_csv(path, time_col=None, flux_col=None, error_col=None) -> dict

Both return:
{"transit_duration_hours": float, "rho_mult": float}
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd

__all__ = [
    "suggest_from_df",
    "suggest_from_csv",
    "estimate_transit_duration_hours",
    "estimate_rho_mult",
]

# ---------- Core estimation helpers ----------

def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    out = np.convolve(ypad, kernel, mode="valid")
    return out  # same length as input due to padding choice

def _local_minima(y: np.ndarray) -> np.ndarray:
    return np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1

def _half_depth_width(time: np.ndarray, flux: np.ndarray, i: int,
                      w_left: int = 50, w_right: int = 50) -> float:
    n = len(flux)
    L = max(0, i - w_left)
    R = min(n, i + w_right)
    baseline = np.percentile(flux[L:R], 75) if R > L else np.median(flux)
    depth = baseline - flux[i]
    if not np.isfinite(depth) or depth <= 0:
        return np.nan

    half = baseline - 0.5 * depth

    # Left crossing
    li = i
    while li > 0 and flux[li] < half:
        li -= 1
    if li == 0:
        t_left = time[0]
    else:
        f0, f1 = flux[li], flux[li + 1]
        t0, t1 = time[li], time[li + 1]
        if f1 == f0:
            t_left = t0
        else:
            t_left = t0 + (half - f0) * (t1 - t0) / (f1 - f0)

    # Right crossing
    ri = i
    while ri < n - 1 and flux[ri] < half:
        ri += 1
    if ri == n - 1:
        t_right = time[-1]
    else:
        f0, f1 = flux[ri - 1], flux[ri]
        t0, t1 = time[ri - 1], time[ri]
        if f1 == f0:
            t_right = t1
        else:
            t_right = t0 + (half - f0) * (t1 - t0) / (f1 - f0)

    width_days = max(0.0, t_right - t_left)
    return width_days

def estimate_transit_duration_hours(time, flux, error=None,
                                    smooth_frac: float = 0.02,
                                    max_dips: int = 7) -> float:
    """Return estimated transit duration (hours) from (time, flux[, error])."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    n = len(time)
    if n < 20:
        return 2.5  # fallback

    # Light smoothing to suppress high-freq noise
    window = int(max(11, round(n * smooth_frac)))
    window = min(window, 301)
    sm = _moving_average(flux, window)

    minima = _local_minima(sm)
    widths = []

    if len(minima) == 0:
        i = int(np.argmin(sm))
        w = _half_depth_width(time, sm, i, w_left=window, w_right=window)
        if np.isfinite(w) and w > 0:
            widths.append(w)
    else:
        idx_sorted = minima[np.argsort(sm[minima])][:max_dips]
        for i in idx_sorted:
            w = _half_depth_width(time, sm, i, w_left=window, w_right=window)
            if np.isfinite(w) and w > 0:
                widths.append(w)

    if len(widths) == 0:
        width_hours = 2.5
    else:
        width_hours = float(np.median(widths) * 24.0)

    # Guard: ensure >= a few cadences or 15 minutes
    dt = np.median(np.diff(time))
    min_hours = max(0.25, 5.0 * dt * 24.0)
    return float(max(width_hours, min_hours))

def estimate_rho_mult(time, flux, duration_hours: float) -> float:
    """Map slow-trend strength to rho_mult in {1.5, 3.5, 5, 7}."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    n = len(time)
    if n < 50:
        return 5.0

    flux0 = flux - np.nanmedian(flux)

    # Slow trend proxy with 10% window
    window = int(max(31, round(n * 0.10)))
    window = min(window, 1001)
    trend = _moving_average(flux0, window)

    total_std = np.nanstd(flux0)
    trend_std = np.nanstd(trend)
    ratio = 0.0 if total_std == 0 else trend_std / total_std

    if ratio > 0.9:
        rho_mult = 1.5
    elif ratio > 0.6:
        rho_mult = 3.5
    elif ratio > 0.3:
        rho_mult = 5.0
    else:
        rho_mult = 7.0

    # Duration min-guard is handled in estimate_transit_duration_hours
    return float(rho_mult)

# ---------- Public API ----------

def suggest_from_df(df: "pd.DataFrame",
                    time_col: str = "time",
                    flux_col: str = "flux",
                    error_col: Optional[str] = None,
                    **rho_kwargs) -> Dict[str, float]:
    """Suggest parameters from a pandas DataFrame with the given column names."""
    time = df[time_col].to_numpy()
    flux = df[flux_col].to_numpy()
    error = df[error_col].to_numpy() if error_col else None
    duration_hours = estimate_transit_duration_hours(time, flux, error)
    rho_mult = estimate_rho_mult(time, flux, duration_hours, **rho_kwargs)
    return {"transit_duration_hours": float(duration_hours),
            "rho_mult": float(rho_mult)}

def suggest_from_csv(path: str,
                     time_col: Optional[str] = None,
                     flux_col: Optional[str] = None,
                     error_col: Optional[str] = None,
                     **rho_kwargs) -> Dict[str, float]:
    """Suggest parameters from a CSV; if column names are None, try common defaults."""
    df = pd.read_csv(path)
    # Column detection (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    def pick(name, cands):
        if name: return name
        for k in cands:
            if k in cols: return cols[k]
        return None
    tcol = pick(time_col, ["time","t","bjd","bjd_tdb","jd","mjd"])
    fcol = pick(flux_col, ["flux","pdcsap_flux","sap_flux","f","flux_norm","flux_corr"])
    ecol = pick(error_col,["error","flux_err","flux_error","eflux","err","yerr"])
    if tcol is None or fcol is None:
        raise ValueError(f"Could not detect time/flux columns. columns={list(df.columns)}")
    return suggest_from_df(df, tcol, fcol, ecol, **rho_kwargs)
