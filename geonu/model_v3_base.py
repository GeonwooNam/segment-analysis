# %% [셋업]
# model_v3_base.py — modeling_plan.md 3절 Track 1: BASE 모델
# train에서 A/B 제거 → C/D/E 3분류 (1등 방식). class_weight C=2.
# 행 단위: 월 1행 (240만) + 최종 ID 집계는 combine 단계에서 수행.
# CV: StratifiedGroupKFold(group=ID) — 같은 고객의 6개월이 fold를 넘나들지 않게 (누수 방지)
# v3.1: A/B 고객도 fold 모델 평균으로 예측해 OOF에 포함 (5-class 평가기 eval_v3_oof.py 입력)
import gc
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

N_FOLDS  = 5
SEED     = 42
USE_GPU  = True          # 데탑(4060 Ti)에서 True, 실패 시 자동 CPU 폴백
CLASSES  = ["C", "D", "E"]
CLASS_WEIGHTS = [2.0, 1.0, 1.0]   # 1등 방식: C↔D 혼동 보정


# %% [로드]
train = pd.read_parquet(FEATURES_DIR / "train_v3.parquet")
test  = pd.read_parquet(FEATURES_DIR / "test_v3.parquet")
print(f"train {train.shape} / test {test.shape}")

feat_cols = [c for c in train.columns if c not in ["ID", "기준년월", "Segment"]]
assert feat_cols == [c for c in test.columns if c not in ["ID", "기준년월"]], "train/test 피처 불일치"

# BASE는 A/B를 아예 모른 채 학습 (1등 방식 — 186명 제거는 전체의 0.05%)
mask_cde = train["Segment"].isin(CLASSES)
tr = train[mask_cde].reset_index(drop=True)
# A/B 고객 행 — 학습에서는 제외하되, VIP가 덮어쓰지 못한 경우 base가 뭐라고 예측하는지가
# 5-class OOF 평가에 필요하므로 fold 모델 평균으로 예측해 둔다 (train fold에 전혀 안 들어가므로 누수 없음)
ab = train[~mask_cde].reset_index(drop=True)
print(f"A/B 제거 후 학습 행: {len(tr):,} (A/B {len(ab):,}행은 평가용 예측만)")

X       = tr[feat_cols]
y       = tr["Segment"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
groups  = tr["ID"].to_numpy()
X_test  = test[feat_cols]

del train   # 메모리 ~7GB 확보 (4060 Ti 데탑에서 working set 22.6GB까지 갔던 원인)
gc.collect()


# %% [학습] 5-fold OOF + test 확률 평균
def make_model():
    params = dict(
        iterations=1500, depth=8, learning_rate=0.05,
        loss_function="MultiClass", class_weights=CLASS_WEIGHTS,
        random_seed=SEED, early_stopping_rounds=100, verbose=200,
    )
    if USE_GPU:
        params.update(task_type="GPU", devices="0")
    return CatBoostClassifier(**params)


oof_prob  = np.zeros((len(tr), len(CLASSES)), dtype=np.float32)
test_prob = np.zeros((len(test), len(CLASSES)), dtype=np.float32)
models    = []

skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y, groups)):
    print(f"\n===== Fold {fold + 1}/{N_FOLDS} =====")
    model = make_model()
    try:
        model.fit(X.iloc[idx_tr], y[idx_tr],
                  eval_set=(X.iloc[idx_va], y[idx_va]))
    except Exception as e:
        if USE_GPU:
            print(f"[경고] GPU 학습 실패({e}) → CPU 폴백")
            model = CatBoostClassifier(**{**make_model().get_params(), "task_type": "CPU"})
            model.fit(X.iloc[idx_tr], y[idx_tr], eval_set=(X.iloc[idx_va], y[idx_va]))
        else:
            raise
    oof_prob[idx_va] = model.predict_proba(X.iloc[idx_va])
    test_prob += model.predict_proba(X_test) / N_FOLDS
    models.append(model)

    f1_fold = f1_score(y[idx_va], oof_prob[idx_va].argmax(1), average="macro")
    print(f"Fold {fold + 1} macro-F1 (행 단위): {f1_fold:.4f}")

# A/B 고객 예측 (fold 모델 평균 — 어떤 fold에도 학습되지 않은 고객들)
ab_prob = np.zeros((len(ab), len(CLASSES)), dtype=np.float32)
for model in models:
    ab_prob += model.predict_proba(ab[feat_cols]) / len(models)


# %% [평가] 행 단위 / ID 단위(확률 평균 = soft 다수결, 3등 방식) 모두 기록
f1_row = f1_score(y, oof_prob.argmax(1), average="macro")

oof_df = pd.DataFrame(oof_prob, columns=CLASSES)
oof_df["ID"], oof_df["y"] = tr["ID"].values, y
id_prob = oof_df.groupby("ID")[CLASSES].mean()
id_y    = oof_df.groupby("ID")["y"].first()
f1_id   = f1_score(id_y, id_prob.to_numpy().argmax(1), average="macro")

# ⚠ 아래는 C/D/E 3-class macro-F1이다. 리더보드(5-class)와 직접 비교 금지 —
#   5-class OOF는 vip까지 돌린 뒤 eval_v3_oof.py로 계산한다. (docs/score_log_v3.md 참고)
print(f"\nOOF macro-F1 [C/D/E 3-class] — 행 단위: {f1_row:.4f} / ID 단위(확률평균): {f1_id:.4f}")
for i, c in enumerate(CLASSES):
    print(f"  {c} F1 (ID 단위): {f1_score(id_y == i, id_prob.to_numpy().argmax(1) == i):.4f}")


# %% [저장] combine 단계 입력물 (ID 정렬 보존을 위해 parquet)
oof_out = tr[["ID", "기준년월"]].copy()
for i, c in enumerate(CLASSES):
    oof_out[f"prob_{c}"] = oof_prob[:, i]
oof_out["y_true"] = tr["Segment"].values

ab_out = ab[["ID", "기준년월"]].copy()
for i, c in enumerate(CLASSES):
    ab_out[f"prob_{c}"] = ab_prob[:, i]
ab_out["y_true"] = ab["Segment"].values

oof_out = pd.concat([oof_out, ab_out], ignore_index=True)  # 전체 40만 고객 커버
oof_out.to_parquet(RESULTS_DIR / "base_v3_oof.parquet", index=False)

test_out = test[["ID", "기준년월"]].copy()
for i, c in enumerate(CLASSES):
    test_out[f"prob_{c}"] = test_prob[:, i]
test_out.to_parquet(RESULTS_DIR / "base_v3_test_prob.parquet", index=False)
print("저장 완료: results/base_v3_oof.parquet, results/base_v3_test_prob.parquet")
