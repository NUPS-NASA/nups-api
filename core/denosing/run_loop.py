# minimal_batch_loop.py
# 딱 반복문만: lc_star000x_det.csv들을 읽어서 보정 → batch_outputs에 저장

from pathlib import Path
import pandas as pd
import os
import re
from config import exoplanet_name

# project_root / Denoising / run_loop.py 라고 가정
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]        # project_root
ALIGN_DIR = PROJECT_ROOT / "Alignment"
DENOISE_DIR = PROJECT_ROOT / "Denoising"

# === I/O 경로 ===
INPUT_DIR  = ALIGN_DIR / f"{exoplanet_name}_Basic_out5/detrended_csv_per_star"  # lc_star*_det.csv들이 있는 폴더
OUTPUT_DIR = DENOISE_DIR / f"{exoplanet_name}_batch_outputs"  # 결과 저장 폴더
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 정규식 패턴 예: lc_star0008_det.csv ~ lc_star0201_det.csv
pattern = re.compile(r"lc_star\d{4}_det\.csv$")

# 패턴에 맞는 파일만 가져오기
matched_files = [
    f for f in os.listdir(INPUT_DIR)
    if pattern.match(f)
]

# 전체 경로 포함해서 출력
full_paths = [INPUT_DIR / f for f in matched_files]

# 파이프라인 함수 import (이미 갖고 있는 파일들)
from suggested_param import suggest_from_csv
from gp_lightcurve_detrend import detrend_df

# 옵션(원하면 숫자만 바꾸세요, SAMPLES, WARMUP, CHAINS는 너무 크면 느림)
SAMPLES = 300   # 300, 400, 600
WARMUP  = 300   # 300, 400, 600
CHAINS  = 1     # 1 OR 2
SEED    = 42

for csv_path in full_paths:
    out_path = OUTPUT_DIR / f"{csv_path.stem}_corrected.csv"
    if out_path.exists():
        print(f"[SKIP] {csv_path.name} -> already exists")
        continue

    print(f"[RUN ] {csv_path.name}")
    df = pd.read_csv(csv_path)  # 컬럼명은 time, flux, error 라고 가정

    # 추천 파라미터
    params = suggest_from_csv(csv_path)
    duration = float(params["transit_duration_hours"])
    rho_mult = float(params["rho_mult"])

    # 디트렌드
    out_df, gp_mean, _ = detrend_df(
        df, time="time", flux="flux", err="error",
        unit="days", center_flux=True, mean_const=True,
        samples=SAMPLES, warmup=WARMUP, chains=CHAINS, seed=SEED,
        transit_duration_hours=duration, rho_mult=rho_mult
    )

    out_df.to_csv(out_path, index=False)
    print(f"[OK  ] -> {out_path.name}")

print("Done.")