# -*- coding: utf-8 -*-
"""
신용카드 고객 세그먼트 분류 - 간단 EDA
====================================================
PyCharm Community 사용법:
  - 아래 각 블록은 `# %%` 로 구분된 "셀" 입니다.
  - 실행할 줄들을 드래그 선택 후  Alt + Shift + E  (Execute Selection in Python Console)
  - 한 셀 전체를 실행하려면 셀 안쪽을 트리플클릭/드래그해서 선택 → Alt+Shift+E
  - 변수는 콘솔에 계속 살아있어서 위→아래 순서로 주피터처럼 작업하면 됩니다.
  - 그래프는 plt.show() 실행 시 별도 창으로 뜹니다.
"""

# %% [셋업] 라이브러리 / 한글폰트 / 경로  -------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)   # 컬럼 많아서 잘리지 않게
pd.set_option("display.width", 200)

# Mac 한글 폰트 (한글 컬럼명/라벨 깨짐 방지)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False   # 마이너스 기호 깨짐 방지

# 프로젝트 경로 (환경 다르면 이 줄만 수정)
from pathlib import Path
BASE_DIR = Path("/Users/namgeon-u/Desktop/claude/segment analysis")

print("셋업 완료. BASE_DIR =", BASE_DIR)


# %% [로드] 회원정보 한 달치(201807) 로드  -----------------------------------
# 타깃(Segment)이 회원정보에 있으므로 EDA는 회원정보 중심.
# 우선 한 달치(40만 행)만 봐도 분포 파악에 충분.
member_path = BASE_DIR / "train" / "1.회원정보" / "201807_train_회원정보.parquet"
df = pd.read_parquet(member_path)

print("shape:", df.shape)          # (행, 열)
df.head()


# %% [개요] 기본 정보  ----------------------------------------------------------
print("=== 행/열 ===", df.shape)
print("\n=== dtype 분포 ===")
print(df.dtypes.value_counts())
print("\n=== 수치형 기초통계 (상위 8개 컬럼) ===")
df.describe().iloc[:, :8]


# %% [타깃] Segment 분포 (불균형 확인)  ----------------------------------------
seg_counts = df["Segment"].value_counts().sort_index()
seg_ratio = (df["Segment"].value_counts(normalize=True).sort_index() * 100).round(2)

print("=== Segment 건수 ===")
print(seg_counts)
print("\n=== Segment 비율(%) ===")
print(seg_ratio)
# 참고: E가 80%, A/B는 수십~수백건 수준의 극심한 클래스 불균형


# %% [시각화] Segment 분포 막대그래프  -----------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

seg_counts.plot(kind="bar", ax=ax[0], color="#4C72B0")
ax[0].set_title("Segment 건수")
ax[0].set_xlabel("Segment")
ax[0].set_ylabel("건수")

# 로그 스케일로 보면 소수 클래스(A,B,C)도 보임
seg_counts.plot(kind="bar", ax=ax[1], color="#DD8452", logy=True)
ax[1].set_title("Segment 건수 (로그 스케일)")
ax[1].set_xlabel("Segment")
ax[1].set_ylabel("건수 (log)")

plt.tight_layout()
plt.show()


# %% [결측] 결측치 점검  -------------------------------------------------------
na = df.isna().sum()
na = na[na > 0].sort_values(ascending=False)
na_ratio = (na / len(df) * 100).round(2)

miss = pd.DataFrame({"결측수": na, "결측비율(%)": na_ratio})
print(f"결측 있는 컬럼: {len(miss)} / 전체 {df.shape[1]}")
miss.head(20)


# %% [인구통계] 연령 / 성별 분포  ----------------------------------------------
# 연령: '20대'~'70대이상' 범주형 / 남녀구분코드: 1, 2
age_order = ["20대", "30대", "40대", "50대", "60대", "70대이상"]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

df["연령"].value_counts().reindex(age_order).plot(kind="bar", ax=ax[0], color="#55A868")
ax[0].set_title("연령대 분포")
ax[0].set_ylabel("건수")

df["남녀구분코드"].map({1: "남", 2: "여"}).value_counts().plot(
    kind="bar", ax=ax[1], color="#C44E52"
)
ax[1].set_title("성별 분포 (1=남, 2=여)")
ax[1].set_ylabel("건수")

plt.tight_layout()
plt.show()


# %% [교차분석] 연령대 × Segment  ---------------------------------------------
# 연령대별로 Segment 구성비가 어떻게 다른지 (행 기준 100% 정규화)
ct = pd.crosstab(df["연령"], df["Segment"], normalize="index").reindex(age_order)
print("=== 연령대별 Segment 비율 ===")
print((ct * 100).round(1))

ct.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="viridis")
plt.title("연령대별 Segment 구성비")
plt.ylabel("비율")
plt.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# %% [조인 예시] 신용정보 합쳐서 한도 비교  ------------------------------------
# 다른 카테고리는 ID + 기준년월로 조인. 예: 신용정보의 카드이용한도금액
credit_path = BASE_DIR / "train" / "2.신용정보" / "201807_train_신용정보.parquet"
credit = pd.read_parquet(credit_path, columns=["ID", "기준년월", "카드이용한도금액"])

merged = df[["ID", "기준년월", "Segment"]].merge(credit, on=["ID", "기준년월"], how="left")
limit_by_seg = merged.groupby("Segment")["카드이용한도금액"].agg(["mean", "median", "count"])
print("=== Segment별 카드이용한도금액 ===")
print(limit_by_seg.round(0))

merged.boxplot(column="카드이용한도금액", by="Segment", figsize=(9, 5), showfliers=False)
plt.title("Segment별 카드이용한도금액 분포")
plt.suptitle("")          # 자동 생성되는 상단 제목 제거
plt.ylabel("카드이용한도금액")
plt.tight_layout()
plt.show()


# %% [전체월 로드 함수] 필요할 때 6개월 합치기  --------------------------------
def load_all_months(category_dir: str, columns=None) -> pd.DataFrame:
    """카테고리 폴더(예: '1.회원정보') 안의 월별 parquet를 모두 읽어 세로로 합침."""
    folder = BASE_DIR / "train" / category_dir
    files = sorted(folder.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=columns) for f in files]
    out = pd.concat(parts, ignore_index=True)
    print(f"{category_dir}: {len(files)}개월 → shape {out.shape}")
    return out

# 사용 예 (필요할 때 주석 풀고 실행):
# member_all = load_all_months("1.회원정보", columns=["ID", "기준년월", "연령", "Segment"])
# print(member_all.groupby("기준년월")["Segment"].value_counts().unstack())


# %% [P1-1] 마스터키 테이블 (201812 단면)  ------------------------------------
# 201812 단면(40만 행) 회원정보 로드 → 이후 모든 EDA의 기준
# Segment 6개월 불변이므로 201812 하나가 전체 고객을 대표
master = BASE_DIR / "train" / "1.회원정보" / "201812_train_회원정보.parquet"
df_master = pd.read_parquet(master)

# 마스터키 (이후 모든 테이블 조인 기준)
key = df_master[["ID", "기준년월", "Segment"]].copy()

# Segment별 색상 팔레트 (이후 공통 사용)
SEG_COLORS = {"A": "#C0392B", "B": "#E67E22", "C": "#F1C40F", "D": "#2ECC71", "E": "#95A5A6"}
SEG_ORDER  = ["A", "B", "C", "D", "E"]

print("마스터키 shape:", key.shape)
print(key["Segment"].value_counts().sort_index())


# %% [P1-2] 2.신용정보 EDA  ---------------------------------------------------
# 주요 컬럼만 선택 로드 (메모리 절약)
# ※ Segment A(162명), B(24명) 샘플 희소 → boxplot 이상치로 보일 수 있음
CREDIT_COLS = [
    "ID", "기준년월",
    "카드이용한도금액", "최초한도금액", "CA한도금액",
    "CA이자율_할인전", "CL이자율_할인전",
    "RV약정청구율", "RV최소결제비율",
]
credit_path_12 = BASE_DIR / "train" / "2.신용정보" / "201812_train_신용정보.parquet"
df_credit = pd.read_parquet(credit_path_12, columns=CREDIT_COLS)
df_credit = key.merge(df_credit, on=["ID", "기준년월"], how="left")

print("신용정보 shape:", df_credit.shape)
print("결측 수:\n", df_credit.isna().sum())

# --- Segment별 수치형 컬럼 boxplot (한 figure, subplot) ---
credit_num_cols = [
    "카드이용한도금액", "최초한도금액", "CA한도금액",
    "CA이자율_할인전", "CL이자율_할인전", "RV약정청구율", "RV최소결제비율",
]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(credit_num_cols):
    data = [df_credit.loc[df_credit["Segment"] == s, col].dropna() for s in SEG_ORDER]
    axes[i].boxplot(data, labels=SEG_ORDER, showfliers=False)
    axes[i].set_title(col)
    axes[i].set_xlabel("Segment")
axes[-1].set_visible(False)
fig.suptitle("Segment별 신용정보 주요 수치형 분포\n(Segment A=162, B=24 샘플 희소 — 통계 불안정)", y=1.01)
plt.tight_layout()
plt.show()

# --- Segment별 이자율/한도 groupby median ---
print("\n=== Segment별 신용정보 median ===")
print(df_credit.groupby("Segment")[credit_num_cols].median().round(2))

# --- _1순위/_2순위신용체크구분 결측 패턴 (회원정보에 존재) ---
# 참고: _1순위/_2순위신용체크구분은 신용정보가 아닌 회원정보(df_master)에 존재
print("\n=== Segment별 _1/_2순위신용체크구분 결측비율(%) ===")
df_check = df_master[["Segment", "_1순위신용체크구분", "_2순위신용체크구분"]].copy()
miss_rate = (
    df_check.groupby("Segment")[["_1순위신용체크구분", "_2순위신용체크구분"]]
    .apply(lambda g: g.isna().mean() * 100)
    .round(2)
)
print(miss_rate)

ct_check = (
    pd.crosstab(df_master["Segment"], df_master["_1순위신용체크구분"], normalize="index") * 100
).round(2)
print("\n=== _1순위신용체크구분 × Segment 비율(%) ===")
print(ct_check)


# %% [P1-3] 5.잔액정보 EDA  ---------------------------------------------------
# 실측 컬럼: 잔액_리볼빙일시불이월_B0M 존재, 잔액_현금서비스_B0M 존재 (명세와 일치)
BAL_COLS = [
    "ID", "기준년월",
    "잔액_일시불_B0M", "잔액_할부_B0M", "잔액_현금서비스_B0M", "잔액_카드론_B0M",
    "잔액_리볼빙일시불이월_B0M", "연체잔액_B0M", "월중평잔_일시불_B0M",
]
bal_path = BASE_DIR / "train" / "5.잔액정보" / "201812_train_잔액정보.parquet"
df_bal = pd.read_parquet(bal_path, columns=BAL_COLS)
df_bal = key.merge(df_bal, on=["ID", "기준년월"], how="left")

# 총잔액 파생: 일시불 + 할부 + 현금서비스 + 카드론
df_bal["총잔액"] = (
    df_bal["잔액_일시불_B0M"].fillna(0)
    + df_bal["잔액_할부_B0M"].fillna(0)
    + df_bal["잔액_현금서비스_B0M"].fillna(0)
    + df_bal["잔액_카드론_B0M"].fillna(0)
)

print("잔액정보 shape:", df_bal.shape)

# --- Segment별 총잔액 boxplot (log1p 스케일) ---
# ※ Segment A(162명), B(24명) 샘플 희소 — 이상치로 보일 수 있음
fig, ax = plt.subplots(figsize=(9, 5))
data = [np.log1p(df_bal.loc[df_bal["Segment"] == s, "총잔액"].dropna()) for s in SEG_ORDER]
ax.boxplot(data, labels=SEG_ORDER, showfliers=False)
ax.set_title("Segment별 총잔액(일시불+할부+CA+카드론) 분포 (log1p 스케일)\n※ Segment A=162, B=24 샘플 희소")
ax.set_xlabel("Segment")
ax.set_ylabel("log1p(총잔액)")
plt.tight_layout()
plt.show()

print("\n=== Segment별 총잔액 median ===")
print(df_bal.groupby("Segment")["총잔액"].median().round(0))

# --- 연체잔액 > 0 비율 (Segment별) ---
df_bal["연체여부"] = (df_bal["연체잔액_B0M"] > 0).astype(int)
delinq_rate = df_bal.groupby("Segment")["연체여부"].mean() * 100
print("\n=== Segment별 연체잔액>0 비율(%) ===")
print(delinq_rate.round(2))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(SEG_ORDER, [delinq_rate.get(s, 0) for s in SEG_ORDER],
       color=[SEG_COLORS[s] for s in SEG_ORDER])
ax.set_title("Segment별 연체 보유 비율(%)\n※ Segment A=162, B=24 샘플 희소")
ax.set_xlabel("Segment")
ax.set_ylabel("연체잔액>0 비율(%)")
plt.tight_layout()
plt.show()


# %% [P1-4] 3.승인매출정보 EDA  -----------------------------------------------
# 406컬럼 중 주요 컬럼만 선택 (columns= 파라미터로 메모리 절약)
# ※ 실측 결과: 이용금액_신용_B0M / 이용금액_신판_B0M 컬럼은 실제 존재하지 않음
#   → 이용금액_일시불_B0M + 이용금액_할부_B0M + 이용금액_CA_B0M 합산으로 대체
SALES_COLS = [
    "ID", "기준년월",
    "이용금액_일시불_B0M", "이용금액_할부_B0M",
    "이용금액_CA_B0M", "이용금액_카드론_B0M",
    "이용건수_신용_B0M",
    "이용금액_일시불_R3M", "이용금액_일시불_R6M",
    "최종이용일자_기본", "최종이용일자_신판", "최종이용일자_할부",
    "이용후경과월_신용",
]
sales_path = BASE_DIR / "train" / "3.승인매출정보" / "201812_train_승인매출정보.parquet"
df_sales = pd.read_parquet(sales_path, columns=SALES_COLS)
df_sales = key.merge(df_sales, on=["ID", "기준년월"], how="left")

print("승인매출정보 shape:", df_sales.shape)

# --- sentinel 처리: 10101 = 미이용 (= 0001-01-01, YYYYMMDD 정수형) ---
SENTINEL = 10101
for date_col, flag_col in [
    ("최종이용일자_기본", "이용없음_기본"),
    ("최종이용일자_신판", "이용없음_신판"),
    ("최종이용일자_할부", "이용없음_할부"),
]:
    df_sales[flag_col] = (df_sales[date_col] == SENTINEL).astype(int)

print("\n=== 최종이용일자 미이용(sentinel=10101) 비율(%) ===")
flag_cols = ["이용없음_기본", "이용없음_신판", "이용없음_할부"]
print((df_sales[flag_cols].mean() * 100).round(2))

# --- 신용합계 파생 (일시불 + 할부 + CA) ---
df_sales["이용금액_신용합계_B0M"] = (
    df_sales["이용금액_일시불_B0M"].fillna(0)
    + df_sales["이용금액_할부_B0M"].fillna(0)
    + df_sales["이용금액_CA_B0M"].fillna(0)
)

# --- Segment별 이용금액_신용합계_B0M boxplot (log1p) ---
# ※ Segment A(162명), B(24명) 샘플 희소
fig, ax = plt.subplots(figsize=(9, 5))
data = [
    np.log1p(df_sales.loc[df_sales["Segment"] == s, "이용금액_신용합계_B0M"].dropna())
    for s in SEG_ORDER
]
ax.boxplot(data, labels=SEG_ORDER, showfliers=False)
ax.set_title("Segment별 이용금액(일시불+할부+CA) 합산 분포 (log1p)\n※ Segment A=162, B=24 샘플 희소")
ax.set_xlabel("Segment")
ax.set_ylabel("log1p(이용금액_신용합계_B0M)")
plt.tight_layout()
plt.show()

print("\n=== Segment별 이용금액_신용합계_B0M 통계 ===")
print(df_sales.groupby("Segment")["이용금액_신용합계_B0M"].agg(["mean", "median", "std"]).round(0))

# --- Segment별 미이용(기본) 비율 bar chart ---
no_use_rate = df_sales.groupby("Segment")["이용없음_기본"].mean() * 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(SEG_ORDER, [no_use_rate.get(s, 0) for s in SEG_ORDER],
       color=[SEG_COLORS[s] for s in SEG_ORDER])
ax.set_title("Segment별 최종이용일자_기본 미이용(10101) 비율(%)\n※ Segment A=162, B=24 샘플 희소")
ax.set_xlabel("Segment")
ax.set_ylabel("미이용 비율(%)")
plt.tight_layout()
plt.show()


# %% [P1-5] 8.성과정보 EDA  ---------------------------------------------------
# 핵심 6개 컬럼 선택 (성과정보는 결측 매우 적음 — 실측 확인)
PERF_COLS = [
    "ID", "기준년월",
    "증감율_이용건수_신용_전월",
    "증감율_이용금액_신용_전월",
    "증감율_이용건수_신용_분기",
    "증감율_이용금액_신용_분기",
    "잔액_신판평균한도소진율_r6m",
    "혜택수혜율_R3M",
]
perf_path = BASE_DIR / "train" / "8.성과정보" / "201812_train_성과정보.parquet"
df_perf = pd.read_parquet(perf_path, columns=PERF_COLS)
df_perf = key.merge(df_perf, on=["ID", "기준년월"], how="left")

print("성과정보 shape:", df_perf.shape)

# --- NaN 비율 확인 ---
na_perf = df_perf.isna().sum()
na_perf_pos = na_perf[na_perf > 0]
print("\n=== 결측 있는 컬럼 ===")
print(na_perf_pos if len(na_perf_pos) > 0 else "결측 없음")

# --- Segment별 주요 증감율 boxplot ---
# ※ Segment A(162명), B(24명) 샘플 희소 — 분산이 매우 크게 나타날 수 있음
perf_metric_cols = [
    "증감율_이용건수_신용_전월", "증감율_이용금액_신용_전월",
    "증감율_이용건수_신용_분기", "증감율_이용금액_신용_분기",
    "잔액_신판평균한도소진율_r6m", "혜택수혜율_R3M",
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(perf_metric_cols):
    data = [df_perf.loc[df_perf["Segment"] == s, col].dropna() for s in SEG_ORDER]
    axes[i].boxplot(data, labels=SEG_ORDER, showfliers=False)
    axes[i].set_title(col)
    axes[i].set_xlabel("Segment")
fig.suptitle("Segment별 성과정보 주요 증감율/소진율 분포\n※ Segment A=162, B=24 샘플 희소 — 통계 불안정", y=1.01)
plt.tight_layout()
plt.show()

print("\n=== Segment별 성과정보 median ===")
print(df_perf.groupby("Segment")[perf_metric_cols].median().round(4))


# %% [P1-6] 1.회원정보 추가 EDA  ----------------------------------------------
# df_master 재사용 (P1-1에서 이미 로드됨)
# 상위 세그먼트 고객이 입회 기간이 길 것이라는 가설 확인
mem_extra_cols = [
    "입회경과개월수_신용", "소지카드수_유효_신용",
    "소지카드수_이용가능_신용", "회원여부_연체", "이용거절여부_카드론",
]

print("=== 회원정보 추가 컬럼 기초통계 ===")
print(df_master[mem_extra_cols].describe().round(2))

# --- Segment별 입회경과개월수_신용 분포 (boxplot) ---
# ※ Segment A(162명), B(24명) 샘플 희소
fig, ax = plt.subplots(figsize=(9, 5))
data = [df_master.loc[df_master["Segment"] == s, "입회경과개월수_신용"].dropna()
        for s in SEG_ORDER]
ax.boxplot(data, labels=SEG_ORDER, showfliers=False)
ax.set_title("Segment별 입회경과개월수_신용 분포\n※ Segment A=162, B=24 샘플 희소")
ax.set_xlabel("Segment")
ax.set_ylabel("입회경과개월수")
plt.tight_layout()
plt.show()

print("\n=== Segment별 입회경과개월수_신용 median ===")
print(df_master.groupby("Segment")["입회경과개월수_신용"].median().round(0))

# --- 회원여부_연체 비율 Segment별 bar ---
delinq_mem = df_master.groupby("Segment")["회원여부_연체"].mean() * 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(SEG_ORDER, [delinq_mem.get(s, 0) for s in SEG_ORDER],
       color=[SEG_COLORS[s] for s in SEG_ORDER])
ax.set_title("Segment별 회원여부_연체 비율(%)\n※ Segment A=162, B=24 샘플 희소")
ax.set_xlabel("Segment")
ax.set_ylabel("연체 비율(%)")
plt.tight_layout()
plt.show()

print("\n=== Segment별 이용거절여부_카드론 비율(%) ===")
rej_rate = df_master.groupby("Segment")["이용거절여부_카드론"].mean() * 100
print(rej_rate.round(2))


# %% [P1-7] 4.청구입금정보 EDA  -----------------------------------------------
# 실측: 46컬럼, 결측 없음(대부분) — 그대로 사용
BILL_COLS = [
    "ID", "기준년월",
    "청구금액_B0", "청구금액_R3M", "청구금액_R6M",
    "대표결제방법코드", "포인트_마일리지_건별_B0M",
]
bill_path = BASE_DIR / "train" / "4.청구입금정보" / "201812_train_청구입금정보.parquet"
df_bill = pd.read_parquet(bill_path, columns=BILL_COLS)
df_bill = key.merge(df_bill, on=["ID", "기준년월"], how="left")

print("청구입금정보 shape:", df_bill.shape)
na_bill = df_bill.isna().sum()
na_bill_pos = na_bill[na_bill > 0]
print("\n=== 결측 있는 컬럼 ===")
print(na_bill_pos if len(na_bill_pos) > 0 else "결측 없음")

# --- Segment별 청구금액_R6M boxplot (log1p) ---
# ※ Segment A(162명), B(24명) 샘플 희소
fig, ax = plt.subplots(figsize=(9, 5))
data = [
    np.log1p(df_bill.loc[df_bill["Segment"] == s, "청구금액_R6M"].clip(lower=0).dropna())
    for s in SEG_ORDER
]
ax.boxplot(data, labels=SEG_ORDER, showfliers=False)
ax.set_title("Segment별 청구금액_R6M 분포 (log1p 스케일)\n※ Segment A=162, B=24 샘플 희소")
ax.set_xlabel("Segment")
ax.set_ylabel("log1p(청구금액_R6M)")
plt.tight_layout()
plt.show()

print("\n=== Segment별 청구금액_R6M 통계 ===")
print(df_bill.groupby("Segment")["청구금액_R6M"].agg(["mean", "median"]).round(0))

# --- 대표결제방법코드 × Segment 비율 crosstab ---
ct_pay = (
    pd.crosstab(df_bill["Segment"], df_bill["대표결제방법코드"], normalize="index") * 100
).round(2)
print("\n=== 대표결제방법코드 × Segment 비율(%) ===")
print(ct_pay)


# %% [P1-8] 6.채널정보 EDA  ---------------------------------------------------
chan_path = BASE_DIR / "train" / "6.채널정보" / "201812_train_채널정보.parquet"

# 실제 컬럼 목록 먼저 출력하여 ARS/PC/앱/모바일 컬럼 파악
chan_all_cols = pd.read_parquet(chan_path).columns.tolist()
print("=== 채널정보 전체 컬럼 목록 ===")
print(chan_all_cols)

# 주요 채널 컬럼 선택 (ARS / PC / 앱 / 모바일웹 / IB)
CHAN_COLS = [
    "ID", "기준년월",
    "인입횟수_ARS_R6M",
    "방문횟수_PC_R6M", "방문일수_PC_R6M",
    "방문횟수_앱_R6M",  "방문일수_앱_R6M",
    "방문횟수_모바일웹_R6M", "방문일수_모바일웹_R6M",
    "인입횟수_IB_R6M",
]
df_chan = pd.read_parquet(chan_path, columns=CHAN_COLS)
df_chan = key.merge(df_chan, on=["ID", "기준년월"], how="left")

print("\n채널정보 shape:", df_chan.shape)

# --- 채널 합산 활동성 지표 파생 ---
df_chan["채널활동성_합산"] = (
    df_chan["인입횟수_ARS_R6M"].fillna(0)
    + df_chan["방문횟수_PC_R6M"].fillna(0)
    + df_chan["방문횟수_앱_R6M"].fillna(0)
    + df_chan["방문횟수_모바일웹_R6M"].fillna(0)
)

# --- Segment별 채널 활동성 groupby mean ---
chan_metric_cols = [
    "인입횟수_ARS_R6M", "방문횟수_PC_R6M",
    "방문횟수_앱_R6M", "방문횟수_모바일웹_R6M",
    "인입횟수_IB_R6M", "채널활동성_합산",
]
print("\n=== Segment별 채널 활동성 mean ===")
print(df_chan.groupby("Segment")[chan_metric_cols].mean().round(2))

# --- Segment별 채널활동성 합산 boxplot (log1p) ---
# ※ Segment A(162명), B(24명) 샘플 희소
fig, ax = plt.subplots(figsize=(9, 5))
data = [
    np.log1p(df_chan.loc[df_chan["Segment"] == s, "채널활동성_합산"].dropna())
    for s in SEG_ORDER
]
ax.boxplot(data, labels=SEG_ORDER, showfliers=False)
ax.set_title("Segment별 채널 활동성(ARS+PC+앱+모바일웹) 합산 분포 (log1p)\n※ Segment A=162, B=24 샘플 희소")
ax.set_xlabel("Segment")
ax.set_ylabel("log1p(채널활동성_합산)")
plt.tight_layout()
plt.show()


# %% [P1-9] 7.마케팅정보 EDA  -------------------------------------------------
MKT_COLS = [
    "ID", "기준년월",
    "컨택건수_카드론_TM_B0M", "컨택건수_리볼빙_TM_B0M",
    "컨택건수_CA_TM_B0M",     "컨택건수_이용유도_TM_B0M",
    "컨택건수_카드론_LMS_B0M", "컨택건수_이용유도_LMS_B0M",
    "컨택건수_카드론_TM_R6M",  "컨택건수_이용유도_TM_R6M",
    "캠페인접촉건수_R12M",     "캠페인접촉일수_R12M",
]
mkt_path = BASE_DIR / "train" / "7.마케팅정보" / "201812_train_마케팅정보.parquet"
df_mkt = pd.read_parquet(mkt_path, columns=MKT_COLS)
df_mkt = key.merge(df_mkt, on=["ID", "기준년월"], how="left")

print("마케팅정보 shape:", df_mkt.shape)

# --- TM/LMS 총합 파생 ---
tm_cols_b0m  = ["컨택건수_카드론_TM_B0M", "컨택건수_리볼빙_TM_B0M",
                "컨택건수_CA_TM_B0M",     "컨택건수_이용유도_TM_B0M"]
lms_cols_b0m = ["컨택건수_카드론_LMS_B0M", "컨택건수_이용유도_LMS_B0M"]

df_mkt["마케팅_TM합산_B0M"]    = df_mkt[tm_cols_b0m].fillna(0).sum(axis=1)
df_mkt["마케팅_LMS합산_B0M"]   = df_mkt[lms_cols_b0m].fillna(0).sum(axis=1)
df_mkt["마케팅_노출합산_B0M"]  = df_mkt["마케팅_TM합산_B0M"] + df_mkt["마케팅_LMS합산_B0M"]

# --- Segment별 마케팅 노출 groupby mean ---
mkt_summary_cols = [
    "마케팅_TM합산_B0M", "마케팅_LMS합산_B0M", "마케팅_노출합산_B0M",
    "캠페인접촉건수_R12M", "캠페인접촉일수_R12M",
]
print("\n=== Segment별 마케팅 노출 mean ===")
print(df_mkt.groupby("Segment")[mkt_summary_cols].mean().round(3))

# --- 시각화: Segment별 마케팅_노출합산_B0M 평균 bar ---
mkt_mean = df_mkt.groupby("Segment")["마케팅_노출합산_B0M"].mean()
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(SEG_ORDER, [mkt_mean.get(s, 0) for s in SEG_ORDER],
       color=[SEG_COLORS[s] for s in SEG_ORDER])
ax.set_title("Segment별 마케팅 노출 건수 평균(TM+LMS 합산, B0M)\n※ Segment A=162, B=24 샘플 희소")
ax.set_xlabel("Segment")
ax.set_ylabel("마케팅 노출 건수 평균")
plt.tight_layout()
plt.show()


# %% [P1-10] 종합 정리  -------------------------------------------------------
# 테이블별 Segment 분리력 상위 피처 정리 (코드 아닌 주석으로)
# ─────────────────────────────────────────────────────────────
# [2.신용정보] 분리력 높은 피처
#   - 카드이용한도금액     : A,B >> C,D,E (우수 고객 한도 높음)
#   - CA이자율_할인전      : E,D > C > A,B (하위 세그먼트 이자율 높음)
#   - RV약정청구율         : E 높음 (리볼빙 부채 성향)
#   - RV최소결제비율       : 하위 세그먼트에서 최소결제 비율 높음
#
# [5.잔액정보] 분리력 높은 피처
#   - 총잔액(일시불+할부+CA+카드론): A,B >> C,D,E
#   - 연체잔액_B0M        : E,D에서 높을 가능성 (연체 성향)
#   - 잔액_카드론_B0M     : A,B 고객 카드론 잔액 높음
#
# [3.승인매출정보] 분리력 높은 피처
#   - 이용금액_신용합계_B0M (일시불+할부+CA 파생): A,B >> E
#   - 이용건수_신용_B0M   : 상위 세그먼트 이용 빈도 높음
#   - 이용없음_기본 (미이용비율): E 고객 미이용 비율 높음
#   - 이용후경과월_신용   : E > D > C > B > A (미이용 경과월)
#
# [8.성과정보] 분리력 높은 피처
#   - 잔액_신판평균한도소진율_r6m: 소진율 높을수록 상위 세그먼트 경향
#   - 증감율_이용금액_신용_전월  : 성장 트렌드 반영
#   - 혜택수혜율_R3M             : 상위 세그먼트 혜택 활용 높음
#
# [1.회원정보] 분리력 높은 피처
#   - 입회경과개월수_신용  : 상위 세그먼트 오래된 고객 多
#   - 회원여부_연체        : 연체=1이면 하위 세그먼트 집중
#   - 이용거절여부_카드론  : 하위 세그먼트에서 거절 높음
#   - _1순위신용체크구분   : 신용 vs 체크 사용 패턴
#
# [4.청구입금정보] 분리력 높은 피처
#   - 청구금액_R6M         : A,B >> C,D,E
#   - 대표결제방법코드     : 결제 방법으로 세그먼트 구분 가능
#   - 포인트_마일리지_건별_B0M: 상위 세그먼트 포인트 적립 활발
#
# [6.채널정보] 분리력 높은 피처
#   - 방문횟수_앱_R6M     : 디지털 활용도 - 상위 세그먼트 높음
#   - 인입횟수_ARS_R6M    : 하위 세그먼트(연체·문의) 높을 가능성
#   - 방문횟수_PC_R6M     : 디지털 채널 활용도 차이
#   - 채널활동성_합산 (파생): 전반적 디지털 참여도
#
# [7.마케팅정보] 분리력 높은 피처
#   - 캠페인접촉건수_R12M        : 마케팅 타겟팅 집중 세그먼트 식별
#   - 컨택건수_카드론_TM_B0M     : 카드론 TM = 하위 세그먼트 집중 타겟
#   - 컨택건수_이용유도_TM_B0M   : 저활동 세그먼트(E) 이용 유도 집중
# ─────────────────────────────────────────────────────────────
#
# 모델링 피처 후보 목록 (우선순위 순)
# ─────────────────────────────────────────────────────────────
# [Tier 1 - 핵심 피처] Segment 분리력 매우 높음
#   - 카드이용한도금액             (2.신용정보)
#   - 청구금액_R6M                 (4.청구입금정보)
#   - 이용금액_신용합계_B0M        (3.승인매출정보 - 파생)
#   - 총잔액                       (5.잔액정보 - 파생)
#   - 입회경과개월수_신용           (1.회원정보)
#   - 회원여부_연체                 (1.회원정보)
#   - 이용후경과월_신용             (3.승인매출정보)
#   - 잔액_신판평균한도소진율_r6m   (8.성과정보)
#
# [Tier 2 - 보조 피처] 분리력 중간
#   - CA이자율_할인전               (2.신용정보)
#   - 연체잔액_B0M                  (5.잔액정보)
#   - 방문횟수_앱_R6M               (6.채널정보)
#   - 이용건수_신용_B0M             (3.승인매출정보)
#   - 대표결제방법코드               (4.청구입금정보)
#   - 포인트_마일리지_건별_B0M      (4.청구입금정보)
#   - 이용거절여부_카드론            (1.회원정보)
#   - 컨택건수_카드론_TM_R6M        (7.마케팅정보)
#   - 캠페인접촉건수_R12M           (7.마케팅정보)
#
# [Tier 3 - 참고 피처] 분리력 낮거나 결측 多
#   - _1순위신용체크구분             (1.회원정보, 결측 있음)
#   - _2순위신용체크구분             (1.회원정보, 결측 39%)
#   - 소지카드수_유효_신용           (1.회원정보)
#   - 인입횟수_ARS_R6M              (6.채널정보)
#   - 혜택수혜율_R3M                 (8.성과정보)
#
# [모델링 주의사항]
#   - Segment A(162명), B(24명) 극희소 → SMOTE 또는 클래스 가중치 조정 필수
#   - Segment는 6개월 내내 불변 → 피처는 201812 단일월 또는 월별 평균 사용
#   - MNAR 결측(미이용=0 처리 or is_null 플래그) 반드시 고려
#   - 날짜 sentinel 10101(=0001-01-01) → 별도 이진 플래그로 변환
# ─────────────────────────────────────────────────────────────

print("P1-10 종합 정리 완료. EDA 셀 P1-1 ~ P1-10 추가 완료.")