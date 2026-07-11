# %% model_cascade_v2.py — OOF 라우팅 기반 cascade (데이터 누수 수정)
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# %% 데이터 로드
tr = pd.read_parquet(f'{BASE}/features/train_features_v2.parquet')
te = pd.read_parquet(f'{BASE}/features/test_features_v2.parquet')

tr.replace([np.inf, -np.inf], np.nan, inplace=True)
te.replace([np.inf, -np.inf], np.nan, inplace=True)

cat_cols = [c for c in tr.columns
            if (tr[c].dtype == object or pd.api.types.is_string_dtype(tr[c]))
            and c not in ['ID', 'Segment']]
for c in cat_cols:
    tr[c] = pd.Categorical(tr[c].astype(str))
    te[c] = pd.Categorical(te[c].astype(str))

feat_cols = [c for c in tr.columns if c not in ['ID', '기준년월', 'Segment']]
X = tr[feat_cols].reset_index(drop=True)
y_seg = tr['Segment'].reset_index(drop=True)
X_test = te[feat_cols].reset_index(drop=True)

# %% Stage 1 — E vs 非E (5-fold OOF)
y1 = (y_seg != 'E').astype(int)

params1 = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'num_leaves': 63,
    'min_child_samples': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': (y1 == 0).sum() / (y1 == 1).sum(),
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_s1_prob = np.zeros(len(tr))
test_s1_prob = np.zeros(len(te))

for fold, (tr_idx, val_idx) in enumerate(skf1.split(X, y1)):
    m = lgb.LGBMClassifier(**params1)
    m.fit(X.iloc[tr_idx], y1.iloc[tr_idx],
          eval_set=[(X.iloc[val_idx], y1.iloc[val_idx])],
          categorical_feature=cat_cols,
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof_s1_prob[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
    test_s1_prob += m.predict_proba(X_test)[:, 1] / 5
    print(f"Stage1 fold {fold+1} done")

# threshold: Non-E recall >= 0.95 보장 (0.99는 달성 불가)
best_thr1, best_f1_s1 = 0.5, 0
for thr in np.arange(0.05, 0.60, 0.01):
    pred = (oof_s1_prob > thr).astype(int)
    rec = recall_score(y1, pred, pos_label=1, zero_division=0)
    mac = f1_score(y1, pred, average='macro', zero_division=0)
    if rec >= 0.95 and mac > best_f1_s1:
        best_f1_s1, best_thr1 = mac, thr

# recall >= 0.95 달성 못하면 최대 Non-E recall threshold 사용
if best_f1_s1 == 0:
    for thr in np.arange(0.05, 0.60, 0.01):
        pred = (oof_s1_prob > thr).astype(int)
        rec = recall_score(y1, pred, pos_label=1, zero_division=0)
        if rec > best_f1_s1:
            best_f1_s1, best_thr1 = rec, thr
    print(f"[Warning] recall>=0.95 달성 불가, 최대 recall threshold={best_thr1:.2f}")

oof_s1 = (oof_s1_prob > best_thr1).astype(int)
test_s1 = (test_s1_prob > best_thr1).astype(int)

s1_rec = recall_score(y1, oof_s1, pos_label=1, zero_division=0)
print(f"\nStage1: threshold={best_thr1:.2f}, Non-E recall={s1_rec:.4f}")
print(f"OOF: {oof_s1.sum()} 예측 Non-E / 실제 {y1.sum()} Non-E")

# %% Stage 2 — AB vs C vs D (OOF Stage 1 예측 기반 필터)
# 핵심 수정: ground truth 대신 oof_s1 사용
s2_mask = (oof_s1 == 1)
X_s2 = X[s2_mask].reset_index(drop=True)
y_seg_s2 = y_seg[s2_mask].reset_index(drop=True)

# E가 섞임 (Stage 1 오류) → D class (저활동)로 처리
y_s2 = y_seg_s2.map({'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 2})

ab_in_s2 = y_seg_s2.isin(['A', 'B']).sum()
print(f"\nStage2 훈련셋: {len(X_s2)}명 (실제 AB {ab_in_s2}명 포함)")

counts2 = y_s2.value_counts()
sw2 = y_s2.map({c: len(y_s2) / (3 * counts2[c]) for c in counts2.index})

params2 = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'num_leaves': 63,
    'min_child_samples': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_s2_prob = np.zeros((len(X_s2), 3))
X_test_s1 = X_test[test_s1 == 1].reset_index(drop=True)
test_s2_prob = np.zeros((len(X_test_s1), 3))

for fold, (tr_idx, val_idx) in enumerate(skf2.split(X_s2, y_s2)):
    m = lgb.LGBMClassifier(**params2)
    m.fit(X_s2.iloc[tr_idx], y_s2.iloc[tr_idx],
          sample_weight=sw2.iloc[tr_idx],
          eval_set=[(X_s2.iloc[val_idx], y_s2.iloc[val_idx])],
          categorical_feature=cat_cols,
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof_s2_prob[val_idx] = m.predict_proba(X_s2.iloc[val_idx])
    if len(X_test_s1) > 0:
        test_s2_prob += m.predict_proba(X_test_s1) / 5
    print(f"Stage2 fold {fold+1} done")

oof_s2 = np.argmax(oof_s2_prob, axis=1)

ab_true_s2 = y_seg_s2.isin(['A', 'B']).values
ab_pred_s2 = (oof_s2 == 0)
print(f"\nStage2 AB recall: {recall_score(ab_true_s2, ab_pred_s2, zero_division=0):.4f}")
print(f"AB 예측 {ab_pred_s2.sum()}명, 실제 AB {ab_true_s2.sum()}명")

# %% Stage 3 — A vs B (OOF Stage 2 예측 기반 필터)
# 핵심 수정: ground truth 대신 oof_s2==0 사용
s3_mask_in_s2 = (oof_s2 == 0)
X_s3 = X_s2[s3_mask_in_s2].reset_index(drop=True)
y_seg_s3 = y_seg_s2[s3_mask_in_s2].reset_index(drop=True)
y_s3 = (y_seg_s3 == 'B').astype(int)

b_in_s3 = y_s3.sum()
print(f"\nStage3 훈련셋: {len(X_s3)}명 (실제 B {b_in_s3}명 포함)")

# test_s2_AB_mask는 skip_s3 분기 양쪽에서 필요하므로 미리 계산
test_s2_AB_mask = (np.argmax(test_s2_prob, axis=1) == 0)
X_test_s2_AB = X_test_s1[test_s2_AB_mask].reset_index(drop=True)

if b_in_s3 < 3:
    print("[Warning] Stage3 B가 3명 미만 → Stage3 생략, AB → 전부 A 처리")
    oof_s3 = np.zeros(len(X_s3), dtype=int)
    test_s3_prob_arr = np.zeros(len(X_test_s2_AB))
    skip_s3 = True
else:
    skip_s3 = False
    sp3 = max(1.0, (len(y_s3) - b_in_s3) / b_in_s3)
    params3 = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'num_leaves': 15,
        'min_child_samples': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': sp3,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    skf3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_s3_prob = np.zeros(len(X_s3))
    test_s3_prob_arr = np.zeros(len(X_test_s2_AB))

    for fold, (tr_idx, val_idx) in enumerate(skf3.split(X_s3, y_s3)):
        m = lgb.LGBMClassifier(**params3)
        m.fit(X_s3.iloc[tr_idx], y_s3.iloc[tr_idx],
              eval_set=[(X_s3.iloc[val_idx], y_s3.iloc[val_idx])],
              categorical_feature=cat_cols,
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
        oof_s3_prob[val_idx] = m.predict_proba(X_s3.iloc[val_idx])[:, 1]
        if len(X_test_s2_AB) > 0:
            test_s3_prob_arr += m.predict_proba(X_test_s2_AB)[:, 1] / 5
        print(f"Stage3 fold {fold+1} done")

    # Stage3 threshold: B F1 최대화
    best_thr3, best_b_f1 = 0.3, 0
    for thr in np.arange(0.05, 0.95, 0.01):
        pred3 = (oof_s3_prob > thr).astype(int)
        if pred3.sum() == 0:
            continue
        mf1 = f1_score(y_s3, pred3, average='macro', zero_division=0)
        if mf1 > best_b_f1:
            best_b_f1, best_thr3 = mf1, thr

    oof_s3 = (oof_s3_prob > best_thr3).astype(int)
    b_rec = recall_score(y_s3, oof_s3, pos_label=1, zero_division=0)
    print(f"\nStage3: threshold={best_thr3:.2f}, B recall={b_rec:.4f}")
    print(f"B 예측 {oof_s3.sum()}명, 실제 B {b_in_s3}명")

# %% 최종 OOF 조합 (OOF 예측 인덱스 기반 — 핵심 수정)
oof_final = pd.Series(['E'] * len(tr))

# Stage 1 predicted Non-E → Stage 2 predictions
s2_idx = np.where(s2_mask)[0]
oof_final.iloc[s2_idx[oof_s2 == 1]] = 'C'
oof_final.iloc[s2_idx[oof_s2 == 2]] = 'D'

# Stage 2 predicted AB → Stage 3 predictions
s3_idx_in_s2 = np.where(s3_mask_in_s2)[0]
s3_global_idx = s2_idx[s3_idx_in_s2]
oof_final.iloc[s3_global_idx[oof_s3 == 1]] = 'B'
oof_final.iloc[s3_global_idx[oof_s3 == 0]] = 'A'

# 최종 Macro-F1
macro_f1 = f1_score(y_seg, oof_final, average='macro', zero_division=0)
per_class = f1_score(y_seg, oof_final, average=None,
                     labels=['A','B','C','D','E'], zero_division=0)
print(f"\n★ Cascade v2 OOF Macro-F1: {macro_f1:.4f}")
print(f"A:{per_class[0]:.3f} B:{per_class[1]:.3f} C:{per_class[2]:.3f} D:{per_class[3]:.3f} E:{per_class[4]:.3f}")

from sklearn.metrics import confusion_matrix
print("\n혼동행렬:")
print(pd.DataFrame(
    confusion_matrix(y_seg, oof_final, labels=['A','B','C','D','E']),
    index=['실제_'+s for s in list('ABCDE')],
    columns=['예측_'+s for s in list('ABCDE')]
))
print("\nOOF 예측 분포:"); print(oof_final.value_counts().sort_index())

# %% Test 예측 조합
test_final = pd.Series(['E'] * len(te))

test_s1_idx = np.where(test_s1 == 1)[0]
test_s2_pred = np.argmax(test_s2_prob, axis=1)
test_final.iloc[test_s1_idx[test_s2_pred == 1]] = 'C'
test_final.iloc[test_s1_idx[test_s2_pred == 2]] = 'D'

if not skip_s3:
    test_s3_AB_idx = test_s1_idx[test_s2_AB_mask]
    test_s3_pred = (test_s3_prob_arr > best_thr3).astype(int)
    test_final.iloc[test_s3_AB_idx[test_s3_pred == 1]] = 'B'
    test_final.iloc[test_s3_AB_idx[test_s3_pred == 0]] = 'A'
else:
    test_s3_AB_idx = test_s1_idx[test_s2_AB_mask]
    test_final.iloc[test_s3_AB_idx] = 'A'

print("\nTest 예측 분포:"); print(test_final.value_counts().sort_index())

# %% 저장
os.makedirs(f'{BASE}/results', exist_ok=True)
os.makedirs(f'{BASE}/submissions', exist_ok=True)
np.save(f'{BASE}/results/cascade_v2_oof.npy', oof_final.values)
np.save(f'{BASE}/results/cascade_v2_test_pred.npy', test_final.values)

sub = pd.read_csv(f'{BASE}/sample_submission.csv')
sub['Segment'] = test_final.values
sub.to_csv(f'{BASE}/submissions/submission_cascade_v2.csv', index=False)
print("submission_cascade_v2.csv 저장 완료:", sub.shape)
