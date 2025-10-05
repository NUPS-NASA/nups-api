from astropy.modeling.models import Gaussian2D
import numpy as np
from PrepareFits import fits, load_fits_data, read_time_from_header

def get_header_airmass_2(hdr):
    # AIRMASS 또는 SECZ(=airmass 근사) 사용
    for k in ("AIRMASS", "SECZ"):
        if k in hdr:
            try:
                v = float(hdr[k])
                if np.isfinite(v) and v>0:
                    return v
            except Exception:
                pass
    return np.nan

def estimate_frame_fwhm_2(img, xy_ref, n=40, box=11):
    # 밝은 순 n개만 사용
    idx = np.arange(len(xy_ref))
    # ref_img 밝기 대신 현재 frame photometry가 있으면 그걸 쓰는게 더 좋지만,
    # 간단히 중심 근처 n개 사용
    use = idx[:min(n, len(idx))]
    sigmas = []
    h, w = img.shape
    r = box//2
    for i in use:
        x0, y0 = xy_ref[i]
        xi, yi = int(round(x0)), int(round(y0))
        if xi-r<0 or yi-r<0 or xi+r>=w or yi+r>=h: 
            continue
        cut = img[yi-r:yi+r+1, xi-r:xi+r+1]
        if not np.all(np.isfinite(cut)):
            continue
        # 2차 모멘트 기반 σ 추정
        yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]
        xbar = (cut*xx).sum()/cut.sum()
        ybar = (cut*yy).sum()/cut.sum()
        varx = (cut*((xx-xbar)**2)).sum()/cut.sum()
        vary = (cut*((yy-ybar)**2)).sum()/cut.sum()
        if varx>0 and vary>0:
            sigma = np.sqrt(0.5*(varx+vary))
            sigmas.append(float(sigma))
    if not sigmas:
        return np.nan
    sigma_pix = np.median(sigmas)
    return 2.3548 * sigma_pix  # FWHM = 2.3548 σ

# === Cell 8B: extract covariates (airmass, FWHM, sky) ===
def get_header_airmass(h):
    for k in ("AIRMASS","SECZ"):
        if k in h:
            try:
                v = float(h[k]); 
                if np.isfinite(v) and v>0: return v
            except: pass
    return np.nan

def estimate_frame_fwhm(img, xy_ref, n=50, box=11):
    idx = np.arange(len(xy_ref))
    use = idx[:min(n, len(idx))]
    h, w = img.shape; r = box//2; sigmas = []
    yy, xx = np.mgrid[0:box, 0:box]
    for i in use:
        x0, y0 = xy_ref[i]; 
        xi, yi = int(round(x0)), int(round(y0))
        if xi-r<0 or yi-r<0 or xi+r>=w or yi+r>=h: 
            continue
        cut = img[yi-r:yi+r+1, xi-r:xi+r+1]
        if not np.all(np.isfinite(cut)): 
            continue
        s = cut.sum(); 
        if s<=0: continue
        xbar = (cut*xx).sum()/s; ybar = (cut*yy).sum()/s
        varx = (cut*((xx-xbar)**2)).sum()/s; vary = (cut*((yy-ybar)**2)).sum()/s
        if varx>0 and vary>0:
            sigmas.append(float(np.sqrt(0.5*(varx+vary))))
    if not sigmas: return np.nan
    return 2.3548*np.median(sigmas)
