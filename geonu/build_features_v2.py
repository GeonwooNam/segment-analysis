# %% [셋업]
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR     = Path("/Users/namgeon-u/Desktop/claude/segment analysis")
FEATURES_DIR = BASE_DIR / "features"

print(f"BASE_DIR     : {BASE_DIR}")
print(f"FEATURES_DIR : {FEATURES_DIR}")

# 이용금액 종합 컬럼 (mean / std 쌍)
AMT_MEAN_COL = "이용금액_신용합계_B0M_mean_6m"
AMT_STD_COL  = "이용금액_신용합계_B0M_std_6m"

# %% [로드] train/test features v1 로드
print("\n[로드] train_features.parquet ...")
train = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
print(f"  train shape : {train.shape}")

print("[로드] test_features.parquet ...")
test  = pd.read_parquet(FEATURES_DIR / "test_features.parquet")
print(f"  test  shape : {test.shape}")

# 실제 컬럼 존재 여부 검증
required_cols = [
    "카드이용한도금액_last",
    "CL이자율_할인전_last",
    "입회경과개월수_신용_last",
    "회원여부_연체_ever",
    "청구금액_R6M_last",
    AMT_MEAN_COL,
    AMT_STD_COL,
]
missing = [c for c in required_cols if c not in train.columns]
if missing:
    print(f"  [경고] train에 없는 컬럼: {missing}")
else:
    print("  필수 컬럼 전부 존재 ✓")


# %% [파생 피처] B·A 특화 파생 피처 추가
# quantile 임계값은 train 기준으로 계산 → test에 동일 값 적용 (누수 방지)

print("\n[파생 피처] 임계값 계산 (train 기준) ...")

# --- 임계값 사전 계산 ---
q99_한도     = train["카드이용한도금액_last"].quantile(0.99)
q05_이자율   = train["CL이자율_할인전_last"].quantile(0.05)

print(f"  q99 카드이용한도금액_last  : {q99_한도:,.0f}")
print(f"  q05 CL이자율_할인전_last   : {q05_이자율:.6f}")


def add_derived_features(df: pd.DataFrame, q99_limit: float, q05_rate: float) -> pd.DataFrame:
    """B·A 특화 파생 피처 8개 추가 (in-place 방지 위해 복사본 반환)"""
    df = df.copy()

    # 1. 초고한도 플래그 (상위 1%) — A·B = 극단적 고한도
    df["초고한도_flag"] = (df["카드이용한도금액_last"] > q99_limit).astype(int)

    # 2. VIP 복합 지수 — 한도 × 이용금액 (둘 다 높아야 진짜 VIP)
    df["vip_score"] = df["카드이용한도금액_last"] * df[AMT_MEAN_COL]

    # 3. 극저금리 플래그 (하위 5%) — 최우량 신용
    df["극저금리_flag"] = (df["CL이자율_할인전_last"] < q05_rate).astype(int)

    # 4. 장기 무연체 플래그 — 60개월 이상 회원이면서 연체 이력 없음
    df["장기무연체_flag"] = (
        (df["입회경과개월수_신용_last"] > 60) &
        (df["회원여부_연체_ever"] == 0)
    ).astype(int)

    # 5. 한도소진율 — 이용금액 / 한도 (활용 강도)
    df["한도소진율"] = df[AMT_MEAN_COL] / (df["카드이용한도금액_last"] + 1)

    # 6. 이용금액 변동계수 — std / mean (이용 패턴 안정성)
    df["이용금액_변동계수"] = df[AMT_STD_COL] / (df[AMT_MEAN_COL] + 1)

    # 7. 청구금액 대비 한도 비율
    df["청구한도비율"] = df["청구금액_R6M_last"] / (df["카드이용한도금액_last"] + 1)

    # 8. 이자율 × 이용금액 역수 — 낮을수록 우량 (낮은 이자율에 높은 이용)
    df["이자율_이용역수"] = df["CL이자율_할인전_last"] / (df[AMT_MEAN_COL] + 1)

    return df


print("[파생 피처] train 적용 ...")
train = add_derived_features(train, q99_한도, q05_이자율)
print(f"  train shape after : {train.shape}")

print("[파생 피처] test 적용 (train 기준 임계값 사용) ...")
test  = add_derived_features(test, q99_한도, q05_이자율)
print(f"  test  shape after : {test.shape}")


# %% [검증] 추가된 피처 확인
DERIVED_COLS = [
    "초고한도_flag", "vip_score", "극저금리_flag", "장기무연체_flag",
    "한도소진율", "이용금액_변동계수", "청구한도비율", "이자율_이용역수",
]

print("\n" + "=" * 60)
print("파생 피처 기초 통계 (train)")
print("=" * 60)
print(train[DERIVED_COLS].describe().to_string())

if "Segment" in train.columns:
    print("\n" + "=" * 60)
    print("Segment별 초고한도_flag 비율")
    print("=" * 60)
    flag_dist = train.groupby("Segment")["초고한도_flag"].mean().rename("비율")
    print(flag_dist.sort_index().to_string())

    print("\n" + "=" * 60)
    print("Segment별 vip_score 중앙값")
    print("=" * 60)
    vip_med = train.groupby("Segment")["vip_score"].median().rename("중앙값")
    print(vip_med.sort_index().to_string())


# %% [저장] train_features_v2.parquet, test_features_v2.parquet 저장
TRAIN_OUT = FEATURES_DIR / "train_features_v2.parquet"
TEST_OUT  = FEATURES_DIR / "test_features_v2.parquet"

print(f"\n[저장] {TRAIN_OUT} ...")
train.to_parquet(TRAIN_OUT)

print(f"[저장] {TEST_OUT} ...")
test.to_parquet(TEST_OUT)

print("\n저장 완료:")
print(f"  TRAIN → {TRAIN_OUT}  shape={train.shape}")
print(f"  TEST  → {TEST_OUT}   shape={test.shape}")

print("\n추가된 파생 피처 목록:")
for i, col in enumerate(DERIVED_COLS, 1):
    print(f"  {i}. {col}")
