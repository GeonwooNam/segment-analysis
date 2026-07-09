# %% [셋업]
# model_v3_vip.py — modeling_plan.md 3절 Track 2: VIP 모델 (v3.1)
# ① AB vs 나머지 이진 → ID 단위 후보 "랭킹" 생성 (컷은 combine의 VIP_CAP이 유일)
# ② 랭킹 상위권에서 A vs B 정밀 분류
#
# v3.1 변경 (docs/score_log_v3.md 반영):
#  - threshold 탐색 + TARGET_RECALL assert 제거 — combine이 상위 VIP_CAP명만 쓰므로
#    제출물에 반영되지 않는 죽은 코드였음. VIP는 랭킹만 제공하고 컷은 combine이 담당.
#  - 5-class OOF 평가(eval_v3_oof.py)용으로 train 상위 후보 풀도 저장.
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR     = Path(__file__).resolve().parents[1]
FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_FOLDS     = 5
SEED        = 42
USE_GPU     = True
TOP_K_TEST  = 300    # test 후보 export 수 (VIP_CAP=60의 5배 — cap 스윕 여지 확보)
TOP_K_TRAIN = 1200   # train 후보 export 수 (test의 4배 스케일)


# %% [로드]
train = pd.read_parquet(FEATURES_DIR / "train_v3.parquet")
test  = pd.read_parquet(FEATURES_DIR / "test_v3.parquet")
feat_cols = [c for c in train.columns if c not in ["ID", "기준년월", "Segment"]]

y_ab   = train["Segment"].isin(["A", "B"]).astype(int).to_numpy()
groups = train["ID"].to_numpy()
n_ab_ids = train.loc[y_ab == 1, "ID"].nunique()
print(f"train {train.shape} / AB 고객 {n_ab_ids}명 ({y_ab.sum()}행)")


# %% [유틸]
def make_model(binary_params: dict):
    params = dict(random_seed=SEED, verbose=0, **binary_params)
    if USE_GPU:
        params.update(task_type="GPU", devices="0")
    return CatBoostClassifier(**params)


def fit_with_fallback(model, X_tr, y_tr, **kw):
    try:
        model.fit(X_tr, y_tr, **kw)
    except Exception as e:
        print(f"[경고] GPU 학습 실패({e}) → CPU 폴백")
        model = CatBoostClassifier(**{**model.get_params(), "task_type": "CPU"})
        model.fit(X_tr, y_tr, **kw)
    return model


def id_max_prob(ids: np.ndarray, prob: np.ndarray) -> pd.Series:
    """ID별 6개월 중 최대 확률 — 한 달이라도 AB 신호가 강하면 후보로."""
    return pd.Series(prob, index=ids).groupby(level=0).max()


# %% [Stage ①] AB vs 나머지 — 5-fold OOF로 ID 랭킹 생성
params_s1 = dict(
    iterations=1000, depth=6, learning_rate=0.05,
    loss_function="Logloss",
    scale_pos_weight=(y_ab == 0).sum() / max(y_ab.sum(), 1),
    early_stopping_rounds=100,
)

X = train[feat_cols]
oof_s1  = np.zeros(len(train), dtype=np.float32)
test_s1 = np.zeros(len(test), dtype=np.float32)

skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y_ab, groups)):
    print(f"Stage① Fold {fold + 1}/{N_FOLDS} ...")
    model = fit_with_fallback(make_model(params_s1), X.iloc[idx_tr], y_ab[idx_tr],
                              eval_set=(X.iloc[idx_va], y_ab[idx_va]))
    oof_s1[idx_va] = model.predict_proba(X.iloc[idx_va])[:, 1]
    test_s1 += model.predict_proba(test[feat_cols])[:, 1] / N_FOLDS

# 랭킹 품질 진단 — cap 후보 구간의 정밀도/구성 (컷 결정은 combine의 VIP_CAP)
id_prob_tr = id_max_prob(groups, oof_s1)
id_true    = pd.Series(y_ab, index=groups).groupby(level=0).max()
id_seg     = train.groupby("ID")["Segment"].first()

print("\n[Stage① 랭킹 진단] 상위 K명 구성 (train, ~K/4 == test cap):")
for k in [120, 240, 480, 1200]:
    top = id_prob_tr.nlargest(k).index
    comp = id_seg[top].value_counts().to_dict()
    print(f"  top {k:>5}: precision {id_true[top].mean():.3f} | {comp}")


# %% [Stage ②] A vs B — 실제 AB 고객으로 학습 (1등: B 확장 불가 → 풀 내 정밀 분류에 집중)
ab = train[train["Segment"].isin(["A", "B"])].reset_index(drop=True)
y_b     = (ab["Segment"] == "B").astype(int).to_numpy()
g_ab    = ab["ID"].to_numpy()
n_b_ids = ab.loc[y_b == 1, "ID"].nunique()
print(f"\nStage② 학습: AB {ab['ID'].nunique()}명({len(ab)}행), B {n_b_ids}명")

params_s2 = dict(
    iterations=500, depth=4, learning_rate=0.05,
    loss_function="Logloss",
    scale_pos_weight=(ab["Segment"] == "A").sum() / max(y_b.sum(), 1),
    min_data_in_leaf=1,   # B fold당 ~5명 → 작은 leaf 허용
)

oof_s2 = np.zeros(len(ab), dtype=np.float32)
skf2 = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
s2_models = []
for fold, (idx_tr, idx_va) in enumerate(skf2.split(ab[feat_cols], y_b, g_ab)):
    model = fit_with_fallback(make_model(params_s2), ab[feat_cols].iloc[idx_tr], y_b[idx_tr])
    oof_s2[idx_va] = model.predict_proba(ab[feat_cols].iloc[idx_va])[:, 1]
    s2_models.append(model)

# ID 단위(평균 확률) threshold: A/B F1 평균 최대화
id_prob_b = pd.Series(oof_s2, index=g_ab).groupby(level=0).mean()
id_true_b = pd.Series(y_b, index=g_ab).groupby(level=0).max()

best_thr_b, best_score = 0.5, -1
for thr in np.arange(0.05, 0.95, 0.05):
    pred = (id_prob_b >= thr).astype(int)
    score = (f1_score(id_true_b, pred) + f1_score(1 - id_true_b, 1 - pred)) / 2
    if score > best_score:
        best_thr_b, best_score = thr, score

pred_b = (id_prob_b >= best_thr_b).astype(int)
print(f"Stage② OOF (ID 단위, 실제 AB 내부): A F1={f1_score(1 - id_true_b, 1 - pred_b):.3f} / "
      f"B F1={f1_score(id_true_b, pred_b):.3f} / threshold={best_thr_b:.2f}")
print(f"  B {int(id_true_b.sum())}명 중 {int((pred_b & id_true_b).sum())}명 적중")


# %% [유틸] 후보 풀에 Stage② 확률 부여
def stage2_prob_for_ids(pool_ids: pd.Index, df: pd.DataFrame) -> pd.Series:
    """풀 ID들의 B 확률 (ID 단위 평균). 실제 AB ID는 OOF 사용(누수 방지), 나머지는 앙상블 예측."""
    out = pd.Series(np.nan, index=pool_ids, dtype=np.float64)
    known = pool_ids.intersection(id_prob_b.index)
    out[known] = id_prob_b[known]
    todo = out[out.isna()].index
    if len(todo) > 0:
        rows = df[df["ID"].isin(todo)]
        p = np.zeros(len(rows), dtype=np.float32)
        for m in s2_models:
            p += m.predict_proba(rows[feat_cols])[:, 1] / len(s2_models)
        out[out.isna()] = pd.Series(p, index=rows["ID"].to_numpy()).groupby(level=0).mean()
    return out


# %% [Export ①] train 후보 풀 → eval_v3_oof.py 입력
pool_tr_ids = id_prob_tr.nlargest(TOP_K_TRAIN).index
prob_b_tr   = stage2_prob_for_ids(pool_tr_ids, train)
oof_pool = pd.DataFrame({
    "ID": pool_tr_ids,
    "prob_ab_stage1": id_prob_tr[pool_tr_ids].values,
    "prob_b_stage2": prob_b_tr.values,
    "Segment": np.where(prob_b_tr.values >= best_thr_b, "B", "A"),
    "y_true": id_seg[pool_tr_ids].values,
}).sort_values("prob_ab_stage1", ascending=False)
oof_pool.to_csv(RESULTS_DIR / "vip_v3_oof_pool.csv", index=False)


# %% [Export ②] test 후보 랭킹 → combine 입력
id_prob_test  = id_max_prob(test["ID"].to_numpy(), test_s1)
pool_test_ids = id_prob_test.nlargest(TOP_K_TEST).index
prob_b_test   = stage2_prob_for_ids(pool_test_ids, test)
vip_out = pd.DataFrame({
    "ID": pool_test_ids,
    "prob_ab_stage1": id_prob_test[pool_test_ids].values,
    "prob_b_stage2": prob_b_test.values,
    "Segment": np.where(prob_b_test.values >= best_thr_b, "B", "A"),
}).sort_values("prob_ab_stage1", ascending=False)
vip_out.to_csv(RESULTS_DIR / "vip_v3_test_pred.csv", index=False)

oof_save = train[["ID", "기준년월"]].copy()
oof_save["prob_ab_stage1"] = oof_s1
oof_save.to_parquet(RESULTS_DIR / "vip_v3_oof_stage1.parquet", index=False)
print(f"\n저장 완료: vip_v3_test_pred.csv (상위 {TOP_K_TEST}명), "
      f"vip_v3_oof_pool.csv (상위 {TOP_K_TRAIN}명), vip_v3_oof_stage1.parquet")
