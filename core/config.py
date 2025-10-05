import os

exoplanet_name = "kepler-845b"

# ====== 사용자 파라미터 ======
LIGHT_DIR   = f"./{exoplanet_name} data/object"   # 관측(light) 프레임 폴더
BIAS_DIR    = f"./{exoplanet_name} data/bias"     # 바이어스 폴더 (없으면 None 또는 빈 폴더)
DARK_DIR    = f"./{exoplanet_name} data/dark"     # 다크 폴더
FLAT_DIR    = f"./{exoplanet_name} data/flat"     # 플랫 폴더
OUTPUT_DIR  = f"./{exoplanet_name}_Basic_out5"    # 결과 저장 폴더

USE_BIAS = True
USE_DARK = True
USE_FLAT = True

# 정렬/저장 옵션
DO_ALIGNMENT        = True
SAVE_ALIGNED_FITS   = True
ALIGNED_DIR         = os.path.join(OUTPUT_DIR, "aligned_fits")

# 결과 플롯/CSV 경로
PLOT_DIR            = os.path.join(OUTPUT_DIR, "plots_allstars_lc")
SAVE_WIDE_CSV       = True
WIDE_CSV_PATH       = os.path.join(OUTPUT_DIR, "allstars_relflux_wide.csv")
TIME_CSV_PATH       = os.path.join(OUTPUT_DIR, "times_jd.csv")

# 검출/광도측정 파라미터
FWHM_PIX           = 3.5
THRESH_SIGMA       = 5.0
MAX_STARS_DETECT   = 2000
EDGE_MARGIN        = 12
R_AP               = 3.0 * FWHM_PIX
R_IN, R_OUT        = 6.0 * FWHM_PIX, 10.0 * FWHM_PIX
K_COMPS            = 3
BRIGHT_TOL_FRAC    = 0.30
MIN_SEP_PIX        = 3.0 * FWHM_PIX
CLIP_SIGMA         = 4.0

# 미리보기(검출표시) 설정
N_LABELS_PREVIEW   = 100   # 밝은 순서 최대 N개 라벨
PREVIEW_PATH       = os.path.join(OUTPUT_DIR, "detected_stars_preview.png")