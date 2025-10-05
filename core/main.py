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

from PrepareFits import (
 list_fits_in, build_master_bias, build_master_dark_by_exptime, build_master_flat, load_fits_data, calibrate_frame, _stretch, detect_stars, save_detection_preview, read_time_from_header, align_to_reference, measure_frame_photometry,
)
from FrameLoad import (
    brief_array, brief_dark_dict_summary,
)
from FWHM import (
    get_header_airmass, estimate_frame_fwhm, get_header_airmass_2, estimate_frame_fwhm_2
)
from Detrending import (
    pick_comps_rms_aware_general, weighted_reference, detrend_by_covariates
)

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

# ====== 라이트 프레임 로드 ======
light_files = list_fits_in(LIGHT_DIR)
if not light_files:
    raise FileNotFoundError(f"No light frames in {LIGHT_DIR}")
print(f"Found {len(light_files)} light frames.")

# ====== 마스터 프레임 생성 ======
mbias = build_master_bias(BIAS_DIR) if USE_BIAS else None
dark_dict = build_master_dark_by_exptime(DARK_DIR) if USE_DARK else {}
mflat = build_master_flat(FLAT_DIR, master_bias=mbias, dark_dict=dark_dict) if USE_FLAT else None

print(brief_array("Master Bias", mbias))
print(brief_dark_dict_summary(dark_dict))      # ← 여기가 핵심 교정
print(brief_array("Master Flat(norm)", mflat))

# ====== 기준 프레임(첫 라이트) 보정 ======
ref_raw, ref_hdr = load_fits_data(light_files[0])
ref_cal = calibrate_frame(ref_raw, ref_hdr, master_bias=mbias, dark_dict=dark_dict, flat_norm=mflat)
ref_img = ref_cal

# (선택) 보정 전/후 비교 이미지를 저장해두면 디버깅이 편리합니다.
before_path = os.path.join(OUTPUT_DIR, "preview_ref_before.png")
after_path  = os.path.join(OUTPUT_DIR, "preview_ref_after.png")

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(_stretch(ref_raw), cmap="gray", origin="lower"); plt.title("Ref raw")
plt.subplot(1,2,2); plt.imshow(_stretch(ref_img), cmap="gray", origin="lower"); plt.title("Ref calibrated")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "preview_ref_compare.png"), dpi=160); plt.close()

plt.imsave(before_path, _stretch(ref_raw), cmap="gray", origin="lower")
plt.imsave(after_path,  _stretch(ref_img), cmap="gray", origin="lower")

print("Saved:", before_path, "and", after_path)

# 기준 프레임을 정렬결과 폴더에 FITS로도 보관(아이덴티티 정렬)
if SAVE_ALIGNED_FITS:
    h = ref_hdr.copy()
    h["HISTORY"] = "calibrated; reference frame; aligned identity"
    fits.writeto(os.path.join(ALIGNED_DIR, os.path.basename(light_files[0])),
                 ref_img.astype(np.float32), h, overwrite=True)
    print("Reference FITS saved to aligned_fits.")

xyf = detect_stars(ref_img)     # [x, y, flux]
xy  = xyf[:, :2]
print(f"Detected {len(xy)} stars on reference frame.")

prev_path = save_detection_preview(ref_img, xy, PREVIEW_PATH, N_LABELS_PREVIEW)
print(f"Detection preview saved:", prev_path)


airmass_list = []
fwhm_list    = []
sky_list     = []

for idx, p in enumerate(light_files):
    _, hdr = load_fits_data(p)              # 헤더만 재활용
    airmass_list.append(get_header_airmass_2(hdr))

# 정렬된 프레임(또는 보정된 프레임)을 다시 열어 FWHM/sky 계산
# 저장된 aligned_fits가 있으면 그걸 쓰는게 가장 정확
aligned_paths = sorted(glob.glob(os.path.join(ALIGNED_DIR, "*.fit*"))) if SAVE_ALIGNED_FITS else []
if aligned_paths and len(aligned_paths)==len(light_files):
    to_iter = aligned_paths
    open_data = lambda path: fits.getdata(path).astype(float)
else:
    # 정렬본이 없으면 다시 한 번 보정해서 추정(느릴 수 있음)
    to_iter = light_files
    open_data = lambda path: calibrate_frame(load_fits_data(path)[0], load_fits_data(path)[1],
                                             master_bias=mbias, dark_dict=dark_dict, flat_norm=mflat)

for path in to_iter:
    img = open_data(path)
    fwhm_list.append(estimate_frame_fwhm_2(img, xy, n=50, box=11))
    # 전역 하늘 밝기(중앙값) – 프레임마다 배경 변화 추적
    finite = np.isfinite(img)
    sky_list.append(np.nanmedian(img[finite]) if np.any(finite) else np.nan)

airmass_arr = np.array(airmass_list, dtype=float)
fwhm_arr    = np.array(fwhm_list,    dtype=float)
sky_arr     = np.array(sky_list,     dtype=float)

print("Covariates ready:",
      f"\n  Airmass:  n={np.sum(np.isfinite(airmass_arr))}/{len(airmass_arr)}",
      f"\n  FWHM(px): n={np.sum(np.isfinite(fwhm_arr))}/{len(fwhm_arr)}  median={np.nanmedian(fwhm_arr):.2f}",
      f"\n  Sky:      n={np.sum(np.isfinite(sky_arr))}/{len(sky_arr)}  median={np.nanmedian(sky_arr):.2f}")


airmass_list, fwhm_list, sky_list = [], [], []

# airmass from headers of original light files
for p in list_fits_in(LIGHT_DIR):
    _, hdr = load_fits_data(p)
    airmass_list.append(get_header_airmass(hdr))

# choose image source
img_paths = sorted(glob.glob(os.path.join(ALIGNED_DIR, "*.fit*"))) if (SAVE_ALIGNED_FITS and os.path.isdir(ALIGNED_DIR)) else list_fits_in(LIGHT_DIR)
def _open_img(path):
    if SAVE_ALIGNED_FITS and os.path.dirname(path).endswith(os.path.basename(ALIGNED_DIR)):
        return fits.getdata(path).astype(float)
    dat, hdr = load_fits_data(path)
    return dat.astype(float)

for p in img_paths:
    img = _open_img(p)
    fwhm_list.append(estimate_frame_fwhm(img, xy, n=50, box=11))
    finite = np.isfinite(img); sky_list.append(np.nanmedian(img[finite]) if np.any(finite) else np.nan)

airmass_arr = np.array(airmass_list, float)
fwhm_arr    = np.array(fwhm_list,    float)
sky_arr     = np.array(sky_list,     float)
print("covariates →",
      f"airmass n={np.sum(np.isfinite(airmass_arr))}",
      f"fwhm n={np.sum(np.isfinite(fwhm_arr))}, med={np.nanmedian(fwhm_arr):.2f}",
      f"sky n={np.sum(np.isfinite(sky_arr))}, med={np.nanmedian(sky_arr):.2f}")

times = []
rows = []
skipped = 0

for idx, p in enumerate(light_files):
    data, hdr = load_fits_data(p)
    cal = calibrate_frame(data, hdr, master_bias=mbias, dark_dict=dark_dict, flat_norm=mflat)

    if DO_ALIGNMENT and idx > 0:
        try:
            aligned, tf = align_to_reference(cal, ref_img)
        except Exception as e:
            print(f"[WARN] Align failed at {os.path.basename(p)}: {e}")
            skipped += 1
            continue
    else:
        aligned = cal

    t_jd = read_time_from_header(hdr)
    if not np.isfinite(t_jd):
        t_jd = np.nan
    times.append(t_jd)

    fluxes = measure_frame_photometry(aligned, xy)  # 길이 = 검출된 별 수
    rows.append(fluxes)

    if SAVE_ALIGNED_FITS:
        h = hdr.copy()
        h["HISTORY"] = "calibrated & aligned"
        fits.writeto(os.path.join(ALIGNED_DIR, os.path.basename(p)),
                     aligned.astype(np.float32), h, overwrite=True)

    if (idx+1) % 10 == 0:
        print(f" processed {idx+1}/{len(light_files)} frames...")

if skipped > 0:
    print(f"Alignment skipped {skipped} frames due to errors.")

print("Loop done. Measured photometry on", len(rows), "frames.")

# ====== 행렬화 ======
flux_mat = np.vstack(rows)           # (N_frames, N_stars)
times = np.array(times, dtype=float) # (N_frames,)

# ====== 시간 결측 시 대체 ======
if np.any(~np.isfinite(times)):
    # 헤더에 시간이 없을 경우 프레임 인덱스를 시간축으로 사용
    times = np.arange(len(times), dtype=float)

# ====== 품질 필터(유효 비율 50% 초과 별만 유지) ======
valid_ratio = np.mean(np.isfinite(flux_mat), axis=0)
keep = valid_ratio > 0.5
xy, flux_mat = xy[keep], flux_mat[:, keep]

print(f"Kept {xy.shape[0]} stars after quality mask (>{0.5*100:.0f}% valid).")

# --- series_mat 준비: 사용 가능한 소스에서 자동 선택 ---
def build_series_and_brightness():
    # 1) 이미 메모리에 절대 플럭스가 있으면 제일 좋음
    if 'flux_mat' in globals():
        series_mat = flux_mat
        bright_vec = np.nanmedian(flux_mat, axis=0)  # 절대 플럭스 중앙값 = 밝기 수준
        return series_mat, bright_vec

    # 2) rows 리스트(프레임별 포토메트리)가 있으면 재구성
    if 'rows' in globals() and isinstance(rows, list) and len(rows) > 0:
        series_mat = np.vstack(rows)
        bright_vec = np.nanmedian(series_mat, axis=0)
        return series_mat, bright_vec

    # 3) Wide CSV가 저장돼 있다면 로드 (JD 컬럼 제외)
    try:
        import pandas as pd, os
        if 'WIDE_CSV_PATH' in globals() and os.path.isfile(WIDE_CSV_PATH):
            df = pd.read_csv(WIDE_CSV_PATH)
            cols = [c for c in df.columns if c.lower() != 'jd']
            series_mat = df[cols].to_numpy(dtype=float)
            # 상대광도 CSV라면 모든 별의 중앙값이 ~1이므로 밝기유사 판별이 무의미
            # → ref 프레임에서 한 번 포토메트리하여 밝기 벡터를 만들거나,
            #   없으면 series_mat 중앙값을 fallback으로 사용
            if 'ref_img' in globals() and 'xy' in globals():
                bright_vec = measure_frame_photometry(ref_img, xy)
            else:
                bright_vec = np.nanmedian(series_mat, axis=0)
            return series_mat, bright_vec
    except Exception as e:
        print("[WARN] Failed to load WIDE_CSV:", e)

    raise RuntimeError("series_mat 소스를 찾지 못했습니다. flux_mat/rows/WIDE_CSV 중 하나가 필요합니다.")

series_mat, bright_vec = build_series_and_brightness()

# 무결성 체크
assert series_mat.ndim == 2
assert 'xy' in globals()
assert series_mat.shape[1] == xy.shape[0], f"shape mismatch: series_mat={series_mat.shape}, stars={xy.shape[0]}"

# === Sync frame paths with the matrix you are using ===
import os, glob
from astropy.io import fits
import numpy as np

def ensure_frame_paths_for_series(n_frames):
    """
    series_mat/raw_rel과 '같은 순서, 같은 개수'의 프레임 경로 리스트를 확보한다.
    우선순위: FRAME_PATHS(이미 있음) > aligned_paths > LIGHT_DIR
    """
    global FRAME_PATHS
    if 'FRAME_PATHS' in globals() and len(FRAME_PATHS) == n_frames:
        return FRAME_PATHS
    if 'aligned_paths' in globals() and len(aligned_paths) >= n_frames:
        FRAME_PATHS = aligned_paths[:n_frames]
        return FRAME_PATHS
    files = list_fits_in(LIGHT_DIR)
    assert len(files) >= n_frames, "LIGHT_DIR에 프레임이 부족합니다."
    FRAME_PATHS = files[:n_frames]
    return FRAME_PATHS

# series_mat/raw_rel과 같은 프레임 수로 맞추기
N_frames = series_mat.shape[0]
FRAME_PATHS = ensure_frame_paths_for_series(N_frames)


def open_img_and_hdr(path):
    # aligned_fits면 header도 같이 읽고, 아니면 원본에서 보정 전 이미지로 추정(속도 우선)
    try:
        data = fits.getdata(path).astype(float)
        hdr  = fits.getheader(path)
    except Exception:
        data, hdr = load_fits_data(path)
    return data, hdr

def build_covariates_from_paths(paths):
    airmass, fwhm, sky, tt = [], [], [], []
    for p in paths:
        img, hdr = open_img_and_hdr(p)
        airmass.append(get_header_airmass(hdr))
        fwhm.append(estimate_frame_fwhm(img, xy, n=50, box=11))
        finite = np.isfinite(img)
        sky.append(np.nanmedian(img[finite]) if np.any(finite) else np.nan)
        t = read_time_from_header(hdr)
        tt.append(t if np.isfinite(t) else np.nan)
    airmass_arr = np.array(airmass, float)
    fwhm_arr    = np.array(fwhm,    float)
    sky_arr     = np.array(sky,     float)
    times_arr   = np.array(tt,      float)
    if np.any(~np.isfinite(times_arr)):
        times_arr = np.arange(len(paths), dtype=float)  # 시간 없으면 인덱스로 대체
    return airmass_arr, fwhm_arr, sky_arr, times_arr

airmass_arr, fwhm_arr, sky_arr, times_synced = build_covariates_from_paths(FRAME_PATHS)

# times 길이도 맞추기 (있다면 교체)
if 'times' not in globals() or len(times) != len(times_synced):
    times = times_synced

# (디버깅 출력)
print("synced lengths:",
      "y/raw_rel" if 'raw_rel' in globals() else "series_mat", 
      series_mat.shape[0],
      "| airmass", len(airmass_arr), "| fwhm", len(fwhm_arr), "| sky", len(sky_arr), "| times", len(times))

# === Cell 9B: choose series_mat automatically & analyze star 0053 ===
import pandas as pd, os

def build_series_and_brightness():
    # 1) 절대 플럭스 행렬이 메모리에 있으면 최상
    if 'flux_mat' in globals():
        return flux_mat, np.nanmedian(flux_mat, axis=0)
    # 2) rows 리스트로부터 복원
    if 'rows' in globals() and isinstance(rows, list) and len(rows) > 0:
        m = np.vstack(rows); return m, np.nanmedian(m, axis=0)
    # 3) wide CSV (상대광도일 수도 있음)
    if 'WIDE_CSV_PATH' in globals() and os.path.isfile(WIDE_CSV_PATH):
        df = pd.read_csv(WIDE_CSV_PATH)
        cols = [c for c in df.columns if c.lower() != 'jd']
        mat = df[cols].to_numpy(float)
        # 상대광도 CSV일 경우, ref 프레임에서 포토메트리로 밝기 벡터를 만드는 편이 낫다
        if 'ref_img' in globals() and 'xy' in globals():
            bv = measure_frame_photometry(ref_img, xy)
        else:
            bv = np.nanmedian(mat, axis=0)
        return mat, bv
    raise RuntimeError("series_mat source not found (need flux_mat OR rows OR WIDE_CSV_PATH).")

series_mat, bright_vec = build_series_and_brightness()
assert series_mat.shape[1] == xy.shape[0], f"shape mismatch: {series_mat.shape} vs stars={xy.shape[0]}"

# --- Star 0053 ---
TI, K = 7, 20
comp_ids = pick_comps_rms_aware_general(TI, series_mat, bright_vec, xy, bright_tol=0.25, k=K)
assert len(comp_ids) > 1, "비교성이 너무 적습니다. bright_tol↑ 혹은 K↑ 해보세요."

ref, w = weighted_reference(series_mat[:, comp_ids])
raw_rel = series_mat[:, TI] / ref
raw_rel /= np.nanmedian(raw_rel)

cov_list, cov_names = [], []
if 'airmass_arr' in globals() and np.any(np.isfinite(airmass_arr)): cov_list.append(airmass_arr); cov_names.append("airmass")
if 'fwhm_arr'    in globals() and np.any(np.isfinite(fwhm_arr)):    cov_list.append(fwhm_arr);    cov_names.append("fwhm")
if 'sky_arr'     in globals() and np.any(np.isfinite(sky_arr)):     cov_list.append(sky_arr);     cov_names.append("sky")

if cov_list:
    baseline, rel_corr, good = detrend_by_covariates(raw_rel, cov_list, max_iter=4, clip=3.0)
else:
    baseline = np.ones_like(raw_rel); rel_corr = raw_rel; good = np.isfinite(raw_rel)

t0 = np.nanmin(times) if 'times' in globals() else 0.0
xh = ((times - t0)*24.0) if 'times' in globals() else np.arange(len(raw_rel), float)

plt.figure(figsize=(8.2,4.6))
plt.plot(xh, raw_rel, '.', ms=3, alpha=0.45, label="raw relative")
if cov_list:
    plt.plot(xh, baseline/np.nanmedian(baseline), '-', lw=2, alpha=0.9, label="baseline (fit)")
plt.plot(xh, rel_corr, '.', ms=4, label="detrended")
plt.xlabel("Time since first frame [hr]" if 'times' in globals() else "Frame index")
plt.ylabel("Relative flux")
plt.title(f"Star {TI:04d} (K={len(comp_ids)} comps; cov={','.join(cov_names) if cov_names else 'none'})")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); 
# plt.show()

print("comp_ids:", comp_ids[:12], f"... total {len(comp_ids)}")

# 0) 출력 폴더 준비 (덮어쓰기 싫으면 폴더명을 바꾸세요)
PLOT_DIR_DETREND = os.path.join(OUTPUT_DIR, "plots_allstars_lc_detrended")
os.makedirs(PLOT_DIR_DETREND, exist_ok=True)
CSV_DETREND_PATH = os.path.join(OUTPUT_DIR, "allstars_relflux_detrended_wide.csv")

# >>> 추가: 별별 CSV 저장 폴더
CSV_DIR_PER_STAR = os.path.join(OUTPUT_DIR, "detrended_csv_per_star")
os.makedirs(CSV_DIR_PER_STAR, exist_ok=True)

# LIGHT_DIR 안의 object-*.fits를 시간(=파일명) 순서로 정렬
OBJECT_FILES = sorted(glob.glob(os.path.join(LIGHT_DIR, "object-*.fits")))
assert len(OBJECT_FILES) == N_frames, f"OBJECT_FILES({len(OBJECT_FILES)}) != N_frames({N_frames})"

# 1) 분석에 사용할 행렬(series_mat)과 밝기 벡터(bright_vec) 확보
#    (이미 위 셀에서 build_series_and_brightness()가 정의되어 있다고 가정)
series_mat, bright_vec = build_series_and_brightness()
assert series_mat.shape[1] == xy.shape[0], \
    f"shape mismatch: series_mat={series_mat.shape} vs stars={xy.shape[0]}"

N_frames, N_stars = series_mat.shape

# 2) 공변량 준비(있으면 사용). 길이 N_frames로 강제 동기화
cov_list, cov_names = [], []
def _sync_len(a):
    return a[:N_frames] if len(a) >= N_frames else np.pad(a, (0, N_frames-len(a)), constant_values=np.nan)

if 'airmass_arr' in globals() and np.any(np.isfinite(airmass_arr)):
    cov_list.append(_sync_len(np.asarray(airmass_arr, float))); cov_names.append("airmass")
if 'fwhm_arr' in globals() and np.any(np.isfinite(fwhm_arr)):
    cov_list.append(_sync_len(np.asarray(fwhm_arr, float)));    cov_names.append("fwhm")
if 'sky_arr' in globals() and np.any(np.isfinite(sky_arr)):
    cov_list.append(_sync_len(np.asarray(sky_arr, float)));     cov_names.append("sky")

# 3) 시간축 준비(있으면 시간→hours, 없으면 프레임 인덱스)
if 'times' in globals() and len(times) == N_frames:
    t0 = np.nanmin(times)
    xh = (times - t0) * 24.0
    x_label = "Time since first frame [hr]"
else:
    xh = np.arange(N_frames, dtype=float)
    x_label = "Frame index"

# 4) 보정 결과 누적(옵션: CSV 저장용)
rel_wide_det = {}
K = 20               # 비교성 개수(권장 15~25)
BRIGHT_TOL = 0.25    # 밝기 유사도 허용치(±25%)
MIN_COMPS  = 5       # 최소 비교성 수(부족하면 skip)

def plot_detrended_only(x, y, title, outpath):
    plt.figure(figsize=(7.6, 4.4))
    plt.plot(x, y, '.', ms=4)
    plt.xlabel(x_label)
    plt.ylabel("Relative flux (detrended)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

saved = 0
skipped = 0

for ti in range(N_stars):
    # 4-1) 비교성 선택(밝기 유사 + 낮은 RMS)
    comp_ids = pick_comps_rms_aware_general(
        ti, series_mat, bright_vec, xy, bright_tol=BRIGHT_TOL, k=K
    )
    if len(comp_ids) < MIN_COMPS:
        skipped += 1
        continue

    # 4-2) 가중 참조곡선
    ref, _ = weighted_reference(series_mat[:, comp_ids])

    # 4-3) 상대광도 & 공변량 보정
    raw_rel = series_mat[:, ti] / ref
    raw_rel /= np.nanmedian(raw_rel)

    if cov_list:  # 공변량 있으면 detrend
        baseline, rel_corr, good = detrend_by_covariates(raw_rel, cov_list, max_iter=4, clip=3.0)
    else:
        rel_corr = raw_rel

    # 4-4) 저장(플롯 + CSV 누적)
    outpng = os.path.join(PLOT_DIR_DETREND, f"lc_star{ti:04d}_det.png")
    title  = f"Star {ti:04d} @ (x={xy[ti,0]:.1f}, y={xy[ti,1]:.1f})"
    plot_detrended_only(xh, rel_corr, title, outpng)

    # >>> 추가: 별별 CSV 저장
    csv_star = os.path.join(CSV_DIR_PER_STAR, f"lc_star{ti:04d}_det.csv")
    if 'times' in globals() and len(times) == N_frames:
        # 시간 축이 있으면 JD 포함
        pd.DataFrame(
            {"time": times, "flux": rel_corr}
        ).to_csv(csv_star, index=False)
    else:
        # 없으면 프레임 인덱스 저장
        pd.DataFrame(
            {"frame": np.arange(N_frames, dtype=int), "rel_flux_det": rel_corr}
        ).to_csv(csv_star, index=False)

    rel_wide_det[f"star{ti:04d}"] = rel_corr
    saved += 1

    if saved % 25 == 0:
        print(f"  saved {saved} / processed {ti+1} stars...")


print(f"Done. Saved {saved} detrended PNGs to: {PLOT_DIR_DETREND} (skipped {skipped} stars)")

# 5) (옵션) detrended wide CSV 저장
if saved > 0:
    df = pd.DataFrame(rel_wide_det)
    # times가 있으면 함께 저장
    if 'times' in globals() and len(times) == N_frames:
        df.insert(0, "JD", times)
    df.to_csv(CSV_DETREND_PATH, index=False)
    print(f"Detrended wide CSV written: {CSV_DETREND_PATH}")
