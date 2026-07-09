# %% [셋업]
# preprocess_v3.py — modeling_plan.md 2절 전처리 파이프라인
# 8개 테이블 병합(월 단위 행 유지) → 상수 컬럼 제거 → 다운캐스팅 → Label Encoding → parquet 저장
# 이후 모든 모델링 스크립트(model_v3_*.py)는 여기서 만든 parquet에서 시작한다.
import gc
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR     = Path(__file__).resolve().parents[1]   # "segment analysis" 루트 (데탑 이식 시 수정 불필요)
FEATURES_DIR = BASE_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)

MONTHS = [201807, 201808, 201809, 201810, 201811, 201812]
TABLES = [
    ("1.회원정보",   "회원정보"),
    ("2.신용정보",   "신용정보"),
    ("3.승인매출정보", "승인매출정보"),
    ("4.청구입금정보", "청구정보"),   # 폴더명과 달리 파일명은 '청구정보'
    ("5.잔액정보",   "잔액정보"),
    ("6.채널정보",   "채널정보"),
    ("7.마케팅정보",  "마케팅정보"),
    ("8.성과정보",   "성과정보"),
]
KEY_COLS = ["ID", "기준년월", "Segment"]  # 인코딩/제거 대상에서 제외

# 디버그용: 정수를 넣으면 ID를 그 수만큼 샘플링해서 빠르게 검증 (예: 5000). None이면 전체.
DEBUG_N_IDS = None

print(f"BASE_DIR : {BASE_DIR}")


# %% [유틸]
def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """int64→int32, float64→float32 (2·3등 방식). 메모리 절반."""
    for col in df.columns:
        dt = df[col].dtype
        if dt == np.int64:
            df[col] = df[col].astype(np.int32)
        elif dt == np.float64:
            df[col] = df[col].astype(np.float32)
    return df


def load_split(split: str) -> pd.DataFrame:
    """월별로 8개 테이블을 ID 기준 병합 후 6개월 세로 결합 → (ID×월) 단위 행."""
    month_frames = []
    for ym in MONTHS:
        merged = None
        for folder, stem in TABLES:
            path = BASE_DIR / split / folder / f"{ym}_{split}_{stem}.parquet"
            df = pd.read_parquet(path)
            df = downcast(df)
            if merged is None:
                merged = df
            else:
                # 기준년월은 첫 테이블 것만 유지
                df = df.drop(columns=[c for c in ["기준년월"] if c in df.columns])
                merged = merged.merge(df, on="ID", how="left")
            del df
        if DEBUG_N_IDS is not None:
            keep_ids = set(merged["ID"].drop_duplicates().sort_values().iloc[:DEBUG_N_IDS])
            if "Segment" in merged.columns:
                # 디버그 샘플에도 A/B 고객은 전원 포함 (VIP 모델 검증용)
                keep_ids |= set(merged.loc[merged["Segment"].isin(["A", "B"]), "ID"])
            merged = merged[merged["ID"].isin(keep_ids)]
        month_frames.append(merged)
        print(f"  [{split}] {ym} 병합 완료 shape={merged.shape}")
        gc.collect()
    out = pd.concat(month_frames, axis=0, ignore_index=True)
    del month_frames
    gc.collect()
    print(f"[{split}] 전체 결합 shape={out.shape}")
    return out


# %% [Step 1] 테이블 병합 (월 단위 행 유지 — train 240만 / test 60만)
train = load_split("train")
test  = load_split("test")


# %% [Step 2] 단일값(상수) 컬럼 제거 — 3등 방식, train 기준으로 판정
const_cols = []
for col in train.columns:
    if col in KEY_COLS:
        continue
    if train[col].nunique(dropna=False) <= 1:
        const_cols.append(col)

print(f"상수 컬럼 {len(const_cols)}개 제거")
train = train.drop(columns=const_cols)
test  = test.drop(columns=[c for c in const_cols if c in test.columns])
print(f"  train {train.shape} / test {test.shape}")

# 준한이 EDA 상수 목록과 교차 확인용으로 저장
pd.Series(const_cols, name="const_col").to_csv(FEATURES_DIR / "const_cols_v3.csv", index=False)


# %% [Step 3] Label Encoding — 문자열 컬럼, train+test 범주 합집합으로 인코딩 (3등 방식)
# pandas 버전에 따라 문자열 dtype 표기가 object/string/str로 달라서 select_dtypes로 감지
str_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
            if c not in KEY_COLS]
print(f"문자열 컬럼 {len(str_cols)}개 인코딩: {str_cols}")

for col in str_cols:
    cats = pd.Index(
        pd.concat([train[col], test[col]], ignore_index=True).dropna().unique()
    ).sort_values()
    mapping = {v: i for i, v in enumerate(cats)}
    train[col] = train[col].map(mapping).fillna(-1).astype(np.int32)
    test[col]  = test[col].map(mapping).fillna(-1).astype(np.int32)

# sentinel 결측(예: 날짜 10101)은 값 그대로 유지 — 트리 모델이 분기로 처리 (일괄 drop 하지 않음)


# %% [Step 4] 최종 확인 및 저장
assert train["Segment"].notna().all(), "train Segment에 결측 존재"
left_str = [c for c in train.select_dtypes(include=["object", "string"]).columns
            if c not in ("ID", "Segment")]
assert not left_str, f"인코딩 안 된 문자열 컬럼 잔존: {left_str}"

mem_gb = train.memory_usage(deep=True).sum() / 1e9
print(f"train {train.shape}, 메모리 {mem_gb:.2f} GB")
print("Segment 분포 (201812 스냅샷, ID 단위):")
print(train[train["기준년월"] == 201812]["Segment"].value_counts().sort_index().to_string())

suffix = "" if DEBUG_N_IDS is None else f"_debug{DEBUG_N_IDS}"
train.to_parquet(FEATURES_DIR / f"train_v3{suffix}.parquet", index=False)
test.to_parquet(FEATURES_DIR / f"test_v3{suffix}.parquet", index=False)
print(f"저장 완료: features/train_v3{suffix}.parquet, features/test_v3{suffix}.parquet")
