# === Cell 3B: detrending & ensemble helpers (flux_mat 없어도 동작) ===

import numpy as np
from numpy.linalg import lstsq

def pick_comps_rms_aware_general(ti, series_mat, bright_vec, xy, bright_tol=0.25, k=20):
    assert series_mat.ndim == 2
    assert bright_vec.ndim == 1 and series_mat.shape[1] == bright_vec.shape[0] == xy.shape[0]
    tflux = bright_vec[ti]
    lo, hi = (1.0 - bright_tol)*tflux, (1.0 + bright_tol)*tflux
    cand = [j for j in range(series_mat.shape[1])
            if j != ti and np.isfinite(bright_vec[j]) and lo <= bright_vec[j] <= hi]
    if not cand:
        return []
    rms_list = []
    for j in cand:
        s = series_mat[:, j]
        med = np.nanmedian(s)
        if not (np.isfinite(med) and med != 0):
            continue
        s_norm = s / med
        rms = np.nanstd(s_norm)
        rms_list.append((j, rms))
    if not rms_list:
        return []
    rms_list.sort(key=lambda t: t[1])
    return [j for j,_ in rms_list[:k]]

def weighted_reference(series_mat_comps):
    norm = series_mat_comps / np.nanmedian(series_mat_comps, axis=0)
    var  = np.nanvar(norm, axis=0)
    w = 1.0 / np.clip(var, 1e-8, None)
    w /= np.nansum(w)
    ref = np.nansum(series_mat_comps * w, axis=1)
    ref /= np.nanmedian(ref)
    return ref, w

def detrend_by_covariates(y, covs, max_iter=3, clip=3.0):
    X = np.column_stack([np.ones_like(y)] + [c for c in covs])
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        beta, *_ = lstsq(X[good], y[good], rcond=None)
        model = X @ beta
        resid = y - model
        s = np.nanstd(resid[good])
        if not (np.isfinite(s) and s > 0):
            break
        good = good & (np.abs(resid) < clip*s)
    baseline = X @ beta
    corr = y / baseline
    corr /= np.nanmedian(corr[good])
    return baseline, corr, good