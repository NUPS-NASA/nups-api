from config import (
 LIGHT_DIR, BIAS_DIR, DARK_DIR, FLAT_DIR, OUTPUT_DIR,
 USE_BIAS, USE_DARK, USE_FLAT,
DO_ALIGNMENT, SAVE_ALIGNED_FITS, ALIGNED_DIR,
PLOT_DIR, SAVE_WIDE_CSV, WIDE_CSV_PATH, TIME_CSV_PATH,
FWHM_PIX, THRESH_SIGMA, MAX_STARS_DETECT, EDGE_MARGIN,
R_AP, R_IN, R_OUT, K_COMPS, BRIGHT_TOL_FRAC, MIN_SEP_PIX, CLIP_SIGMA,
N_LABELS_PREVIEW, PREVIEW_PATH
)
# ====== 기본 임포트 및 전역 설정 ======
import os, glob, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus

import astroalign as aa

# 경고 과다 출력 방지
warnings.filterwarnings("ignore", category=UserWarning)

# matplotlib 한글 폰트/유니코드 마이너스 대응(환경에 따라 필요시)
plt.rcParams['axes.unicode_minus'] = False

# ====== 폴더 준비 ======
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
if SAVE_ALIGNED_FITS:
    os.makedirs(ALIGNED_DIR, exist_ok=True)

print("Paths ready.")
# 이거 main으로 빼는게 좋을듯?

# ====== 파일 검색 ======
def list_fits_in(dirpath):
    if not os.path.isdir(dirpath):
        return []
    files = sorted(glob.glob(os.path.join(dirpath, "*.fits")) + 
                   glob.glob(os.path.join(dirpath, "*.fit")))
    return files

# ====== 헤더에서 시간(JD/BJD/HJD/MJD/DATE-OBS) 읽기 ======
def read_time_from_header(hdr):
    # (1) JD/BJD/HJD
    for key in ["JD", "BJD", "HJD"]:
        if key in hdr:
            try:
                val = float(hdr[key])
                if np.isfinite(val):
                    return val
            except Exception:
                pass
    # (2) MJD
    if "MJD" in hdr:
        try:
            val = float(hdr["MJD"])
            if np.isfinite(val):
                return val + 2400000.5  # MJD -> JD
        except Exception:
            pass
    # (3) DATE-OBS (ISO 혹은 일반)
    if "DATE-OBS" in hdr:
        for fmt in ["isot", None]:
            try:
                if fmt == "isot":
                    return Time(hdr["DATE-OBS"], format="isot", scale="utc").jd
                else:
                    return Time(hdr["DATE-OBS"], scale="utc").jd
            except Exception:
                continue
    return np.nan

# ====== FITS 로딩 ======
def load_fits_data(path):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)
        hdr  = hdul[0].header
    return data, hdr

# ====== 다중 프레임 중앙값 결합 ======
def median_combine(files):
    if not files:
        return None, None
    stack = []
    hdr0 = None
    for p in files:
        dat, hdr = load_fits_data(p)
        if hdr0 is None:
            hdr0 = hdr
        stack.append(dat.astype(float))
    master = np.nanmedian(np.stack(stack, axis=0), axis=0)
    return master, hdr0

# ====== 마스터 바이어스 ======
def build_master_bias(bias_dir):
    files = list_fits_in(bias_dir)
    if not files:
        return None
    mbias, _ = median_combine(files)
    return mbias

# ====== 노출시간 추출 ======
def extract_exptime(hdr):
    for key in ["EXPTIME", "EXPOSURE", "EXP_TIME"]:
        if key in hdr:
            try:
                val = float(hdr[key])
                if np.isfinite(val):
                    return val
            except Exception:
                pass
    return None

# ====== 마스터 다크(노출시간별) ======
def build_master_dark_by_exptime(dark_dir):
    files = list_fits_in(dark_dir)
    if not files:
        return {}
    by_exp = {}
    for p in files:
        _, hdr = load_fits_data(p)
        expt = extract_exptime(hdr)
        if expt is None:
            continue
        by_exp.setdefault(expt, []).append(p)
    out = {}
    for expt, flist in by_exp.items():
        mdark, _ = median_combine(flist)
        out[expt] = mdark
    return out

# ====== 마스터 플랫(바이어스/다크 보정 후 정규화) ======
def build_master_flat(flat_dir, master_bias=None, dark_dict=None):
    files = list_fits_in(flat_dir)
    if not files:
        return None
    cal_stack = []
    for p in files:
        dat, hdr = load_fits_data(p)
        if master_bias is not None:
            dat = dat - master_bias
        if dark_dict is not None and len(dark_dict) > 0:
            expt = extract_exptime(hdr)
            if expt is not None:
                nearest = min(dark_dict.keys(), key=lambda k: abs(k - expt))
                scale = expt / nearest if nearest and nearest != 0 else 1.0
                dat = dat - dark_dict[nearest] * scale
        cal_stack.append(dat)
    mflat = np.nanmedian(np.stack(cal_stack, axis=0), axis=0)
    # 중앙값으로 정규화
    finite = np.isfinite(mflat)
    med = np.nanmedian(mflat[finite]) if np.any(finite) else None
    if med and np.isfinite(med) and med != 0:
        mflat = mflat / med
    return mflat

# ====== 프레임 보정(바이어스/다크/플랫) ======
def calibrate_frame(data, hdr, master_bias=None, dark_dict=None, flat_norm=None):
    out = data.astype(float).copy()
    if master_bias is not None:
        out = out - master_bias
    if dark_dict is not None and len(dark_dict) > 0:
        expt = extract_exptime(hdr)
        if expt is not None:
            nearest = min(dark_dict.keys(), key=lambda k: abs(k - expt))
            scale = expt / nearest if nearest and nearest != 0 else 1.0
            out = out - dark_dict[nearest] * scale
    if flat_norm is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = out / flat_norm
    return out

# ====== 정렬(astroalign) ======
def align_to_reference(src_img, ref_img):
    try:
        aligned, tf = aa.register(src_img, ref_img, detection_sigma=3.0, max_control_points=50)
        return aligned.astype(float), tf
    except aa.MaxIterError as e:
        raise RuntimeError(f"Alignment failed (MaxIterError): {e}")
    except Exception as e:
        raise RuntimeError(f"Alignment failed: {e}")

# ====== 별 검출(DAOStarFinder) ======
def detect_stars(ref_img):
    mean, med, std = sigma_clipped_stats(ref_img, sigma=3.0, maxiters=5)
    dao = DAOStarFinder(fwhm=FWHM_PIX, threshold=THRESH_SIGMA * std)
    tbl = dao(ref_img - med)
    if tbl is None or len(tbl) == 0:
        raise RuntimeError("No stars detected. Adjust FWHM/THRESH_SIGMA/FWHM_PIX.")
    # 밝은 순 정렬
    tbl.sort("flux")
    tbl = tbl[::-1]
    if len(tbl) > MAX_STARS_DETECT:
        tbl = tbl[:MAX_STARS_DETECT]
    xyf = np.vstack([tbl["xcentroid"].data, tbl["ycentroid"].data, tbl["flux"].data]).T
    H, W = ref_img.shape
    m = (xyf[:,0] > EDGE_MARGIN) & (xyf[:,0] < W-EDGE_MARGIN) & (xyf[:,1] > EDGE_MARGIN) & (xyf[:,1] < H-EDGE_MARGIN)
    return xyf[m]

# ====== 원형 aperture/annulus를 이용한 배경차감 순수 플럭스 ======
def measure_frame_photometry(img, xy):
    apert = CircularAperture(xy, r=R_AP)
    ann   = CircularAnnulus(xy, r_in=R_IN, r_out=R_OUT)
    ap_masks  = apert.to_mask(method="exact")
    ann_masks = ann.to_mask(method="exact")

    # (1) 배경(annulus) 중앙값
    sky_vals = []
    for m in ann_masks:
        ann_data = m.multiply(img)
        mask = (ann_data == 0) | ~np.isfinite(ann_data)
        sky_vals.append(np.nanmedian(ann_data[~mask]) if np.any(~mask) else 0.0)
    sky_vals = np.array(sky_vals, dtype=float)

    # (2) aperture 총합 - (배경 * 면적)
    fluxes = []
    for (m, sky) in zip(ap_masks, sky_vals):
        ap_data = m.multiply(img)
        mask = (ap_data == 0) | ~np.isfinite(ap_data)
        pix = ap_data[~mask]
        area = np.sum(~mask)
        if area == 0:
            fluxes.append(np.nan)
        else:
            fluxes.append(np.nansum(pix) - sky * area)
    return np.array(fluxes, dtype=float)

# ====== 비교성 선택(밝기 유사 & 최소 거리) ======
def pick_comps_for_target(target_idx, med_flux, xy, k=K_COMPS):
    tflux = med_flux[target_idx]
    tx, ty = xy[target_idx, 0], xy[target_idx, 1]
    lower, upper = (1.0 - BRIGHT_TOL_FRAC) * tflux, (1.0 + BRIGHT_TOL_FRAC) * tflux
    cand = []
    for j in range(len(med_flux)):
        if j == target_idx:
            continue
        if not np.isfinite(med_flux[j]):
            continue
        if (med_flux[j] >= lower) and (med_flux[j] <= upper):
            dx = xy[j,0] - tx
            dy = xy[j,1] - ty
            if math.hypot(dx, dy) >= MIN_SEP_PIX:
                cand.append((j, abs(med_flux[j] - tflux)))
    cand.sort(key=lambda t: t[1])
    return [c[0] for c in cand[:k]]

# ====== 강건한 상대광도(엔상블/시그마클리핑) ======
def robust_rel_flux(target_series, comps_series):
    denom = np.nansum(comps_series, axis=1)  # 비교성 합
    rel = target_series / denom              # 상대값
    med = np.nanmedian(rel)                  # 중앙값 정규화
    reln = rel / med if np.isfinite(med) and med != 0 else rel
    mu, sig = np.nanmedian(reln), np.nanstd(reln)
    ok = np.abs(reln - mu) < CLIP_SIGMA * sig if np.isfinite(sig) and sig > 0 else np.isfinite(reln)
    return reln, ok

# ====== 미리보기용 스트레치 ======
def _stretch(img, p_lo=1, p_hi=99):
    finite = np.isfinite(img)
    if not np.any(finite):
        return img
    v1, v2 = np.percentile(img[finite], [p_lo, p_hi])
    v1, v2 = float(v1), float(v2)
    out = np.clip((img - v1) / max(v2 - v1, 1e-9), 0, 1)
    return out

# ====== 검출 미리보기 저장(반지 & 라벨) ======
def save_detection_preview(ref_img, xy, path=PREVIEW_PATH, n_labels=N_LABELS_PREVIEW):
    disp = _stretch(ref_img)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(disp, cmap="gray", origin="lower")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("stretched intensity")

    # 앞쪽 n개만 도식화(라벨 난잡 방지)
    n = min(n_labels, xy.shape[0])
    apert = CircularAperture(xy[:n], r=R_AP)
    ann   = CircularAnnulus(xy[:n], r_in=R_IN, r_out=R_OUT)
    try:
        apert.plot(ax=ax, lw=1.2, color="cyan")
        ann.plot(ax=ax, lw=1.0, color="lime")
    except Exception:
        apert.plot(ax=ax, lw=1.2, color="cyan")

    # 라벨(스타 인덱스)
    for i in range(n):
        x, y = xy[i]
        ax.text(x+5, y+5, f"{i}", color="yellow", fontsize=9, weight="bold", ha="left", va="bottom")

    ax.set_title(f"Detected stars (N={xy.shape[0]}), aperture/annulus rings")
    ax.set_xlim(0, ref_img.shape[1])
    ax.set_ylim(0, ref_img.shape[0])
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path

# ====== 상대광도 플로팅(클리핑점 X마커) ======
def plot_lightcurve(times, rel_flux, ok_mask, title, outpath, comps_ids=None):
    t0 = np.nanmin(times)
    xh = (times - t0) * 24.0  # 시간(시간 단위)
    plt.figure(figsize=(7.2, 4.2))
    plt.scatter(xh[~ok_mask], rel_flux[~ok_mask], s=14, marker='x', alpha=0.6, label="clipped")
    plt.plot(xh[ok_mask], rel_flux[ok_mask], 'o', ms=3, label="data")
    plt.xlabel("Time since first frame [hr]")
    plt.ylabel("Relative flux (ensemble norm.)")
    if comps_ids is not None:
        sub = f" / comps: {','.join(map(str, comps_ids))}"
    else:
        sub = ""
    plt.title(f"{title}{sub}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

print("Utility functions ready.")
