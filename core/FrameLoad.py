from config import (
 LIGHT_DIR, BIAS_DIR, DARK_DIR, FLAT_DIR,
 USE_BIAS, USE_DARK, USE_FLAT,
 )
import numpy as np
from PrepareFits import (
 list_fits_in, build_master_bias, build_master_dark_by_exptime, build_master_flat,
)

# ====== 라이트 프레임 로드 ======
light_files = list_fits_in(LIGHT_DIR)
if not light_files:
    raise FileNotFoundError(f"No light frames in {LIGHT_DIR}")
print(f"Found {len(light_files)} light frames.")

# ====== 마스터 프레임 생성 ======
mbias = build_master_bias(BIAS_DIR) if USE_BIAS else None
dark_dict = build_master_dark_by_exptime(DARK_DIR) if USE_DARK else {}
mflat = build_master_flat(FLAT_DIR, master_bias=mbias, dark_dict=dark_dict) if USE_FLAT else None

def brief_array(name, arr):
    """넘파이 배열을 요약해서 문자열로 반환."""
    if arr is None:
        return f"{name}: None"
    finite = np.isfinite(arr)
    if not np.any(finite):
        return f"{name}: shape={arr.shape}, all-NaN"
    med  = np.nanmedian(arr[finite])
    mean = np.nanmean(arr[finite])
    std  = np.nanstd(arr[finite])
    return f"{name}: shape={arr.shape}, med={med:.3f}, mean={mean:.3f}, std={std:.3f}"

def brief_dark_dict_summary(dark_dict):
    """노출시간별 마스터다크를 요약해서 여러 줄 문자열로 반환."""
    if not dark_dict:
        return "Master Dark(dict): None"
    lines = ["Master Dark(dict):"]
    for expt, mdark in sorted(dark_dict.items(), key=lambda kv: kv[0]):
        if mdark is None:
            lines.append(f"  - {expt:g}s: None")
            continue
        finite = np.isfinite(mdark)
        med  = np.nanmedian(mdark[finite]) if np.any(finite) else np.nan
        mean = np.nanmean(mdark[finite]) if np.any(finite) else np.nan
        std  = np.nanstd(mdark[finite])  if np.any(finite) else np.nan
        lines.append(
            f"  - {expt:g}s: shape={mdark.shape}, med={med:.3f}, mean={mean:.3f}, std={std:.3f}"
        )
    return "\n".join(lines)

print(brief_array("Master Bias", mbias))
print(brief_dark_dict_summary(dark_dict))      # ← 여기가 핵심 교정
print(brief_array("Master Flat(norm)", mflat))