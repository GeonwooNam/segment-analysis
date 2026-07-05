# %% [셋업]
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

BASE_DIR     = Path("/Users/namgeon-u/Desktop/claude/segment analysis")
TRAIN_DIR    = BASE_DIR / "train"
TEST_DIR     = BASE_DIR / "test"
FEATURES_DIR = BASE_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)

MONTHS     = [201807, 201808, 201809, 201810, 201811, 201812]
LAST_MONTH = 201812

print(f"BASE_DIR     : {BASE_DIR}")
print(f"FEATURES_DIR : {FEATURES_DIR}")


# %% [유틸] 날짜 sentinel 처리 / recency 계산 함수

def load_months_table(split: str, folder_name: str, file_stem: str,
                      columns=None) -> pd.DataFrame:
    """
    6개월 parquet → 하나의 DataFrame.
    columns 지정 시 해당 컬럼만 로드 (ID·기준년월은 자동 포함).
    존재하지 않는 컬럼은 조용히 건너뜀.
    """
    base_dir = TRAIN_DIR if split == "train" else TEST_DIR
    dfs = []
    for ym in MONTHS:
        path = base_dir / folder_name / f"{ym}_{split}_{file_stem}.parquet"
        if columns is not None:
            load_cols = list(dict.fromkeys(["ID", "기준년월"] + list(columns)))
            actual    = pq.read_schema(str(path)).names
            load_cols = [c for c in load_cols if c in actual]
            df = pd.read_parquet(path, columns=load_cols)
        else:
            df = pd.read_parquet(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def process_date_sentinel(series: pd.Series, base_ym: int = 201812,
                          sentinel: int = 10101):
    """
    날짜 컬럼 sentinel 처리 → (이용없음 플래그, 경과월) 반환.
    sentinel(10101) 또는 NaN → 플래그=1, 경과월=999.
    입력: yyyymmdd 정수 컬럼.
    """
    is_none  = (series == sentinel) | series.isna()
    recency  = series.where(~is_none).astype("float64")
    recv_ym  = (recency // 100).astype("float64")   # yyyymmdd → yyyymm
    elapsed  = (base_ym - recv_ym).clip(lower=0)
    elapsed[is_none] = 999.0
    return is_none.astype(int), elapsed


def make_missing_flag(df: pd.DataFrame, col: str) -> pd.Series:
    """MNAR 결측 플래그 (1=결측, 0=존재)"""
    return df[col].isna().astype(int).rename(col + "_결측")


def safe_cols(df: pd.DataFrame, wanted: list) -> list:
    """wanted 중 df에 실제 존재하는 컬럼만 반환"""
    return [c for c in wanted if c in df.columns]


def agg_mean_std(df: pd.DataFrame, cols: list,
                 suffix: str = "6m") -> pd.DataFrame:
    """groupby ID → mean + std, 컬럼명 = {col}_mean_{suffix} / _std_{suffix}"""
    valid = safe_cols(df, cols)
    if not valid:
        return pd.DataFrame()
    agg = df.groupby("ID")[valid].agg(["mean", "std"])
    agg.columns = [f"{col}_{fn}_{suffix}" for col, fn in agg.columns]
    return agg


def last_month_snap(df: pd.DataFrame, cols: list,
                    suffix: str = "last") -> pd.DataFrame:
    """201812 단일월 스냅샷 (index=ID). suffix가 있으면 컬럼명에 붙임."""
    valid = safe_cols(df, cols)
    last  = df[df["기준년월"] == LAST_MONTH].set_index("ID")[valid].copy()
    if suffix:
        last.columns = [f"{c}_{suffix}" for c in last.columns]
    return last


# %% [빌더] split별 피처 생성 함수

def build_features(split: str) -> pd.DataFrame:
    """split = 'train' | 'test' → 고객 단위 피처 DataFrame (index=ID)"""
    print(f"\n{'='*55}")
    print(f"  Building features : {split.upper()}")
    print(f"{'='*55}")

    feat_parts: list[pd.DataFrame] = []
    target: pd.Series | None = None

    # ------------------------------------------------------------------ #
    # %% [로드-회원] 6개월 회원정보 → 고객 단위 집계
    # ------------------------------------------------------------------ #
    print("[1/8] 회원정보 ...")
    df_raw = load_months_table(split, "1.회원정보", "회원정보")

    # 스냅샷 컬럼 (201812 last)
    snap_wanted = [
        "남녀구분코드", "연령", "입회경과개월수_신용",
        "소지카드수_유효_신용", "소지카드수_이용가능_신용",
        "_1순위신용체크구분", "_2순위신용체크구분",
        "가입통신회사코드", "직장시도명",
        "Life_Stage", "탈회횟수_누적", "최종카드발급경과월",
        "마케팅동의여부",
        "이용금액_R3M_신용",     # 회원 테이블 내 R3M 이용금액
        "이용카드수_신용",        # 실제 사용 카드 수
    ]
    snap = last_month_snap(df_raw, snap_wanted, suffix="last")

    # train: Segment 추출 (201812 기준; 6개월 불변)
    if split == "train" and "Segment" in df_raw.columns:
        target = (
            df_raw[df_raw["기준년월"] == LAST_MONTH]
            .set_index("ID")["Segment"]
            .copy()
        )

    # 결측 플래그 (MNAR 컬럼)
    snap_201812 = df_raw[df_raw["기준년월"] == LAST_MONTH].set_index("ID")
    for col in ["_2순위신용체크구분", "가입통신회사코드", "직장시도명"]:
        if col in snap_201812.columns:
            snap[col + "_결측"] = snap_201812[col].isna().astype(int)

    # 이진 ever 피처 (6개월 max: 한 번이라도 연체=1)
    ever_wanted = ["회원여부_연체", "이용거절여부_카드론"]
    ever_valid  = safe_cols(df_raw, ever_wanted)
    if ever_valid:
        agg_ever = df_raw.groupby("ID")[ever_valid].max()
        agg_ever.columns = [c + "_ever" for c in agg_ever.columns]
        snap = snap.join(agg_ever, how="left")

    feat_parts.append(snap)
    del df_raw, snap_201812
    gc.collect()
    print(f"       → {snap.shape[1]}개 피처")

    # ------------------------------------------------------------------ #
    # %% [로드-신용] 6개월 신용정보 → 고객 단위 집계
    # ------------------------------------------------------------------ #
    print("[2/8] 신용정보 ...")
    df_raw = load_months_table(split, "2.신용정보", "신용정보")

    snap_wanted = [
        "카드이용한도금액", "최초한도금액", "CA한도금액",
        "CA이자율_할인전", "CL이자율_할인전",
        "RV약정청구율", "RV최소결제비율",
        "시장단기연체여부_R6M", "시장단기연체여부_R3M",
        "한도증액횟수_R12M", "자발한도감액횟수_R12M", "강제한도감액횟수_R12M",
        "상향가능한도금액", "상향가능CA한도금액",
    ]
    snap = last_month_snap(df_raw, snap_wanted, suffix="last")

    # 6개월 mean/std (한도·이자율 변동성 포착)
    mean_wanted = ["카드이용한도금액", "CA이자율_할인전", "CL이자율_할인전"]
    agg = agg_mean_std(df_raw, mean_wanted)
    if not agg.empty:
        snap = snap.join(agg, how="left")

    feat_parts.append(snap)
    del df_raw
    gc.collect()
    print(f"       → {snap.shape[1]}개 피처")

    # ------------------------------------------------------------------ #
    # %% [로드-승인매출] 6개월 승인매출정보 → 고객 단위 집계
    # 406컬럼 테이블 → 필요 컬럼만 columns= 로 로드
    # ------------------------------------------------------------------ #
    print("[3/8] 승인매출정보 (선택 컬럼) ...")
    승인_cols = [
        # B0M 금액/건수 → 6m mean/std
        "이용금액_일시불_B0M", "이용금액_할부_B0M",
        "이용금액_CA_B0M",     "이용금액_카드론_B0M",
        "이용건수_신용_B0M",   "이용건수_신판_B0M",
        "이용금액_온라인_B0M", "이용금액_오프라인_B0M",
        "이용건수_온라인_B0M", "이용건수_오프라인_B0M",
        "RP건수_B0M", "RP금액_B0M",
        # R3M / R6M → 201812 스냅샷
        "이용금액_일시불_R3M", "이용금액_할부_R3M", "이용금액_CA_R3M",
        "이용금액_일시불_R6M", "이용금액_할부_R6M", "이용금액_CA_R6M",
        "이용개월수_신용_R6M",
        # 이미 계산된 recency → 201812 스냅샷
        "이용후경과월_신용", "이용후경과월_신판",
        "이용후경과월_일시불", "이용후경과월_CA",
        # 날짜 sentinel → 이용없음 플래그 생성용
        "최종이용일자_기본", "최종이용일자_신판", "최종이용일자_할부",
        # 연속 무실적/유실적 → 스냅샷
        "연속무실적개월수_기본_24M_카드", "연속유실적개월수_기본_24M_카드",
    ]
    df_raw = load_months_table(
        split, "3.승인매출정보", "승인매출정보", columns=승인_cols
    )

    # 파생: 이용금액_신용합계_B0M
    amt_base = ["이용금액_일시불_B0M", "이용금액_할부_B0M",
                "이용금액_CA_B0M",     "이용금액_카드론_B0M"]
    amt_valid = safe_cols(df_raw, amt_base)
    if amt_valid:
        df_raw["이용금액_신용합계_B0M"] = df_raw[amt_valid].fillna(0).sum(axis=1)

    # 날짜 sentinel → 이용없음 플래그 (201812 기준)
    last_raw   = df_raw[df_raw["기준년월"] == LAST_MONTH].set_index("ID")
    date_feats = pd.DataFrame(index=last_raw.index)
    for dcol, fcol in [
        ("최종이용일자_기본", "이용없음_기본"),
        ("최종이용일자_신판", "이용없음_신판"),
        ("최종이용일자_할부", "이용없음_할부"),
    ]:
        if dcol in last_raw.columns:
            flag, _ = process_date_sentinel(last_raw[dcol])
            date_feats[fcol] = flag

    # 6개월 mean/std (B0M 컬럼)
    b0m_agg_cols = safe_cols(df_raw, [
        "이용금액_일시불_B0M", "이용금액_할부_B0M",
        "이용금액_CA_B0M",     "이용금액_카드론_B0M",
        "이용금액_신용합계_B0M",
        "이용건수_신용_B0M",   "이용건수_신판_B0M",
        "이용금액_온라인_B0M", "이용금액_오프라인_B0M",
        "RP건수_B0M", "RP금액_B0M",
    ])
    agg_b0m = agg_mean_std(df_raw, b0m_agg_cols)

    # 스냅샷: R6M/R3M + recency + 연속무실적
    snap_wanted = [
        "이용금액_일시불_R3M", "이용금액_할부_R3M", "이용금액_CA_R3M",
        "이용금액_일시불_R6M", "이용금액_할부_R6M", "이용금액_CA_R6M",
        "이용개월수_신용_R6M",
        "이용후경과월_신용",   "이용후경과월_신판",
        "이용후경과월_일시불", "이용후경과월_CA",
        "연속무실적개월수_기본_24M_카드", "연속유실적개월수_기본_24M_카드",
    ]
    snap = last_month_snap(df_raw, snap_wanted, suffix="last")
    snap = snap.join(date_feats, how="left")
    if not agg_b0m.empty:
        snap = snap.join(agg_b0m, how="left")

    feat_parts.append(snap)
    del df_raw, last_raw, date_feats, agg_b0m
    gc.collect()
    print(f"       → {snap.shape[1]}개 피처")

    # ------------------------------------------------------------------ #
    # %% [로드-청구] 6개월 청구입금정보 → 고객 단위 집계
    # 주의: 실제 파일명은 '청구정보' (폴더명은 '4.청구입금정보')
    # ------------------------------------------------------------------ #
    print("[4/8] 청구입금정보 ...")
    df_raw = load_months_table(split, "4.청구입금정보", "청구정보")

    # 스냅샷 (201812)
    snap_wanted = [
        "대표결제방법코드", "대표결제일",
        "포인트_마일리지_건별_B0M",
        "청구금액_R3M", "청구금액_R6M",
        "할인금액_R3M", "할인건수_R3M",
        "포인트_잔여포인트_B0M", "마일_잔여포인트_B0M",
        "혜택수혜금액_R3M", "혜택수혜금액",
        "연체건수_R6M", "연체건수_R3M",
        "선결제건수_R6M",
    ]
    snap = last_month_snap(df_raw, snap_wanted, suffix="last")

    # 6개월 mean/std (당월 청구금액: 컬럼명 '청구금액_B0')
    mean_wanted = ["청구금액_B0", "포인트_마일리지_건별_B0M", "할인금액_B0M"]
    agg = agg_mean_std(df_raw, mean_wanted)
    if not agg.empty:
        snap = snap.join(agg, how="left")

    feat_parts.append(snap)
    del df_raw
    gc.collect()
    print(f"       → {snap.shape[1]}개 피처")

    # ------------------------------------------------------------------ #
    # %% [로드-잔액] 6개월 잔액정보 → 고객 단위 집계
    # ------------------------------------------------------------------ #
    print("[5/8] 잔액정보 ...")
    df_raw = load_months_table(split, "5.잔액정보", "잔액정보")

    # 파생: 총잔액_B0M
    bal_base  = ["잔액_일시불_B0M", "잔액_할부_B0M",
                 "잔액_현금서비스_B0M", "잔액_카드론_B0M"]
    bal_valid = safe_cols(df_raw, bal_base)
    if bal_valid:
        df_raw["총잔액_B0M"] = df_raw[bal_valid].fillna(0).sum(axis=1)

    # 파생: 연체여부_B0M (이진)
    if "연체잔액_B0M" in df_raw.columns:
        df_raw["연체여부_B0M"] = (df_raw["연체잔액_B0M"].fillna(0) > 0).astype(int)

    # 스냅샷 (201812)
    snap_wanted = [
        "잔액_일시불_B0M", "잔액_할부_B0M",
        "잔액_현금서비스_B0M", "잔액_카드론_B0M",
        "잔액_리볼빙일시불이월_B0M",
        "연체잔액_B0M", "월중평잔_일시불_B0M",
        "총잔액_B0M",
        "RV_평균잔액_R6M", "RV_최대잔액_R6M",
        "평잔_6M", "평잔_카드론_6M",
    ]
    snap = last_month_snap(df_raw, snap_wanted, suffix="last")

    # 6개월 mean/std
    mean_wanted = ["총잔액_B0M", "월중평잔_일시불_B0M", "연체잔액_B0M"]
    agg = agg_mean_std(df_raw, mean_wanted)
    if not agg.empty:
        snap = snap.join(agg, how="left")

    # 연체여부 ever (6개월 max)
    if "연체여부_B0M" in df_raw.columns:
        ever = df_raw.groupby("ID")["연체여부_B0M"].max().rename("연체여부_ever")
        snap = snap.join(ever, how="left")

    feat_parts.append(snap)
    del df_raw
    gc.collect()
    print(f"       → {snap.shape[1]}개 피처")

    # ------------------------------------------------------------------ #
    # %% [로드-채널] 6개월 채널정보 → 고객 단위 집계
    # 이미 R6M 집계된 컬럼 위주 → 201812 스냅샷 사용
    # ------------------------------------------------------------------ #
    print("[6/8] 채널정보 ...")
    df_raw = load_months_table(split, "6.채널정보", "채널정보")

    # 주의: 일부 R6M 횟수 컬럼은 str (버킷화된 범주형, 예: "10회 이상")
    #       → LightGBM 범주형 피처로 사용
    #       채널활동성_합산은 int64인 일수/월수 컬럼으로 계산
    snap_wanted = [
        # str (범주형) - LightGBM 직접 처리
        "인입횟수_ARS_R6M",          # str: "X회 이상"
        "방문횟수_PC_R6M",           # str: "X회 이상"
        "방문횟수_앱_R6M",           # str: "X회 이상"
        "OS구분코드",                 # str: OS 구분
        # int64 - 수치형
        "인입일수_ARS_R6M", "인입후경과월_ARS",
        "방문일수_PC_R6M",  "방문월수_PC_R6M",  "방문후경과월_PC_R6M",
        "방문일수_앱_R6M",  "방문월수_앱_R6M",  "방문후경과월_앱_R6M",
        "방문횟수_모바일웹_R6M", "방문일수_모바일웹_R6M",
        "방문월수_모바일웹_R6M", "방문후경과월_모바일웹_R6M",
        "인입횟수_IB_R6M",  "이용메뉴건수_IB_R6M", "인입후경과월_IB_R6M",
        "상담건수_R6M",
        "불만제기건수_R12M", "불만제기후경과월_R12M",
        "당사PAY_방문횟수_R6M", "당사멤버쉽_방문횟수_R6M",
        "홈페이지_금융건수_R6M", "홈페이지_선결제건수_R6M",
    ]
    snap = last_month_snap(df_raw, snap_wanted, suffix="last")

    # 파생: 채널활동성_합산 (int64 컬럼만 사용: 일수/월수/횟수)
    chan_sum_cols = safe_cols(snap, [
        "인입일수_ARS_R6M_last",       # ARS 인입 일수
        "방문월수_PC_R6M_last",        # PC 방문 월수
        "방문월수_앱_R6M_last",        # 앱 방문 월수
        "방문횟수_모바일웹_R6M_last",  # 모바일웹 방문 횟수 (int64)
    ])
    if chan_sum_cols:
        snap["채널활동성_합산_last"] = snap[chan_sum_cols].fillna(0).sum(axis=1)

    feat_parts.append(snap)
    del df_raw
    gc.collect()
    print(f"       → {snap.shape[1]}개 피처")

    # ------------------------------------------------------------------ #
    # %% [로드-마케팅] 6개월 마케팅정보 → 고객 단위 집계
    # B0M은 당월 컨택 건수 → 6m mean 적용
    # R6M 컬럼은 201812 스냅샷
    # ------------------------------------------------------------------ #
    print("[7/8] 마케팅정보 ...")
    df_raw = load_months_table(split, "7.마케팅정보", "마케팅정보")

    # 파생: TM / LMS 컨택 합산 B0M
    tm_base  = ["컨택건수_카드론_TM_B0M",  "컨택건수_이용유도_TM_B0M",
                "컨택건수_리볼빙_TM_B0M",  "컨택건수_CA_TM_B0M"]
    lms_base = ["컨택건수_카드론_LMS_B0M", "컨택건수_이용유도_LMS_B0M",
                "컨택건수_리볼빙_LMS_B0M", "컨택건수_CA_LMS_B0M"]
    tm_valid  = safe_cols(df_raw, tm_base)
    lms_valid = safe_cols(df_raw, lms_base)
    if tm_valid:
        df_raw["TM컨택합산_B0M"]  = df_raw[tm_valid].fillna(0).sum(axis=1)
    if lms_valid:
        df_raw["LMS컨택합산_B0M"] = df_raw[lms_valid].fillna(0).sum(axis=1)

    # 스냅샷 (201812): R6M 컬럼
    snap_wanted = [
        "컨택건수_카드론_TM_R6M",  "컨택건수_이용유도_TM_R6M",
        "컨택건수_리볼빙_TM_R6M",
        "컨택건수_카드론_LMS_R6M", "컨택건수_이용유도_LMS_R6M",
        "컨택건수_리볼빙_LMS_R6M",
        "캠페인접촉건수_R12M", "캠페인접촉일수_R12M",
        "컨택건수_채권_R6M",   "컨택건수_FDS_R6M",
    ]
    snap = last_month_snap(df_raw, snap_wanted, suffix="last")

    # 6개월 mean/std (B0M 파생합산)
    mean_wanted = safe_cols(df_raw, ["TM컨택합산_B0M", "LMS컨택합산_B0M"])
    agg = agg_mean_std(df_raw, mean_wanted)
    if not agg.empty:
        snap = snap.join(agg, how="left")

    feat_parts.append(snap)
    del df_raw
    gc.collect()
    print(f"       → {snap.shape[1]}개 피처")

    # ------------------------------------------------------------------ #
    # %% [로드-성과] 6개월 성과정보 → 고객 단위 집계
    # ------------------------------------------------------------------ #
    print("[8/8] 성과정보 ...")
    df_raw = load_months_table(split, "8.성과정보", "성과정보")

    snap_wanted = [
        "잔액_신판평균한도소진율_r6m", "잔액_신판최대한도소진율_r6m",
        "잔액_신판ca평균한도소진율_r6m", "잔액_신판ca최대한도소진율_r6m",
        "증감율_이용금액_신용_전월",   "증감율_이용건수_신용_전월",
        "증감율_이용금액_신용_분기",   "증감율_이용건수_신용_분기",
        "혜택수혜율_R3M", "혜택수혜율_B0M",
        "변동률_잔액_B1M",
    ]
    snap = last_month_snap(df_raw, snap_wanted, suffix="last")

    # 6개월 mean/std (월별 변동률/소진율의 평균·변동성)
    mean_wanted = [
        "잔액_신판평균한도소진율_r6m",
        "증감율_이용금액_신용_전월",
        "혜택수혜율_R3M",
    ]
    agg = agg_mean_std(df_raw, mean_wanted)
    if not agg.empty:
        snap = snap.join(agg, how="left")

    feat_parts.append(snap)
    del df_raw
    gc.collect()
    print(f"       → {snap.shape[1]}개 피처")

    # ------------------------------------------------------------------ #
    # %% [병합] 고객 단위 마스터 피처 테이블 생성
    # ------------------------------------------------------------------ #
    print("\n[병합] 피처 테이블 통합 ...")
    master = feat_parts[0]
    for part in feat_parts[1:]:
        master = master.join(part, how="left")

    # train: Segment 맨 앞에 삽입
    if split == "train" and target is not None:
        master.insert(0, "Segment", target)

    print(f"  최종 shape : {master.shape}")
    return master


# %% [실행] train / test 피처 빌드
feat_train = build_features("train")
feat_test  = build_features("test")


# %% [검증] shape, 결측 확인, Segment 분포 확인
print("\n" + "=" * 60)
print("TRAIN 피처 검증")
print("=" * 60)
print(f"shape     : {feat_train.shape}")
print(f"컬럼 수   : {feat_train.shape[1]}")

na_ratio = feat_train.isnull().mean()
high_na  = na_ratio[na_ratio > 0.5]
if not high_na.empty:
    print(f"\n결측률 > 50% 컬럼 ({len(high_na)}개):")
    print(high_na.sort_values(ascending=False).to_string())
else:
    print("결측률 > 50% 컬럼 : 없음")

if "Segment" in feat_train.columns:
    print("\nSegment 분포 (value_counts) :")
    print(feat_train["Segment"].value_counts().sort_index().to_string())

print("\n" + "=" * 60)
print("TEST 피처 검증")
print("=" * 60)
print(f"shape     : {feat_test.shape}")
na_ratio_t = feat_test.isnull().mean()
high_na_t  = na_ratio_t[na_ratio_t > 0.5]
if not high_na_t.empty:
    print(f"\n결측률 > 50% 컬럼 ({len(high_na_t)}개):")
    print(high_na_t.sort_values(ascending=False).to_string())
else:
    print("결측률 > 50% 컬럼 : 없음")


# %% [저장] features/ 폴더에 parquet 저장
TRAIN_OUT = FEATURES_DIR / "train_features.parquet"
TEST_OUT  = FEATURES_DIR / "test_features.parquet"

feat_train.to_parquet(TRAIN_OUT)
feat_test.to_parquet(TEST_OUT)

print(f"\n저장 완료:")
print(f"  TRAIN → {TRAIN_OUT}  shape={feat_train.shape}")
print(f"  TEST  → {TEST_OUT}   shape={feat_test.shape}")


# %% [컬럼 목록 출력] 생성된 피처 전체 리스트
print("\n" + "=" * 60)
print("최종 피처 컬럼 목록 (train 기준)")
print("=" * 60)
for i, col in enumerate(feat_train.columns, 1):
    print(f"  {i:3d}. {col}")
