# %% [0] Imports & Data Load
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import os

BASE_DIR = '/Users/namgeon-u/Desktop/claude/segment analysis'

tr = pd.read_parquet(f'{BASE_DIR}/features/train_features_v2.parquet')
te = pd.read_parquet(f'{BASE_DIR}/features/test_features_v2.parquet')

print(f"Train shape: {tr.shape}, Test shape: {te.shape}")
print(f"Segment 분포:\n{tr['Segment'].value_counts().sort_index()}")

# %% [1] 전처리
# inf → NaN
tr.replace([np.inf, -np.inf], np.nan, inplace=True)
te.replace([np.inf, -np.inf], np.nan, inplace=True)

# 범주형: object 또는 pandas StringDtype 컬럼 감지
cat_cols = [
    c for c in tr.columns
    if (tr[c].dtype == object or pd.api.types.is_string_dtype(tr[c]))
    and c not in ['ID', 'Segment']
]
# LightGBM은 StringDtype 미지원 → label encoding (integer code)
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    tr_vals = tr[c].astype(str).fillna('__NAN__')
    te_vals = te[c].astype(str).fillna('__NAN__')
    combined = pd.concat([tr_vals, te_vals], ignore_index=True)
    le.fit(combined)
    tr[c] = le.transform(tr_vals).astype('int32')
    te[c] = le.transform(te_vals).astype('int32')
    label_encoders[c] = le

print(f"cat_cols ({len(cat_cols)}개): {cat_cols}")

# 피처 컬럼 (ID, 기준년월, Segment 제외)
feature_cols = [c for c in tr.columns if c not in ['ID', '기준년월', 'Segment']]
X = tr[feature_cols]
y_seg = tr['Segment']   # A/B/C/D/E 문자열
X_test = te[feature_cols]

print(f"feature_cols 수: {len(feature_cols)}")

# %% [2] Stage 1 — E vs 非E (이진 분류)
print("\n" + "="*60)
print("Stage 1: E vs 非E")
print("="*60)

# 타깃: E=0, 非E=1
y1 = (y_seg != 'E').astype(int)
print(f"E={y1.eq(0).sum()}, 非E={y1.eq(1).sum()}")

params1 = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'num_leaves': 63,
    'min_child_samples': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 320342 / 79658,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_s1_prob = np.zeros(len(tr))    # 非E 확률
test_s1_prob = np.zeros(len(te))

for fold, (tr_idx, val_idx) in enumerate(skf1.split(X, y1)):
    print(f"  Fold {fold+1}/5 ...", flush=True)
    m = lgb.LGBMClassifier(**params1)
    m.fit(X.iloc[tr_idx], y1.iloc[tr_idx],
          eval_set=[(X.iloc[val_idx], y1.iloc[val_idx])],
          categorical_feature=cat_cols,
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof_s1_prob[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
    test_s1_prob += m.predict_proba(X_test)[:, 1] / 5

# threshold: Non-E recall >= 0.99 보장 목표
best_thr, best_f1_s1 = 0.5, 0
for thr in np.arange(0.1, 0.6, 0.01):
    pred = (oof_s1_prob > thr).astype(int)
    rec_nonE = recall_score(y1, pred, pos_label=1)
    mac = f1_score(y1, pred, average='macro')
    if rec_nonE >= 0.99 and mac > best_f1_s1:
        best_f1_s1, best_thr = mac, thr

s1_nonE_recall = recall_score(y1, (oof_s1_prob > best_thr).astype(int), pos_label=1)
print(f"Stage1 threshold={best_thr:.2f}, Non-E recall={s1_nonE_recall:.4f}, Macro-F1={best_f1_s1:.4f}")

oof_s1 = (oof_s1_prob > best_thr).astype(int)    # 1=非E
test_s1 = (test_s1_prob > best_thr).astype(int)
print(f"OOF 非E 예측: {oof_s1.sum()}, Test 非E 예측: {test_s1.sum()}")

# %% [3] Stage 2 — AB vs C vs D (非E만)
print("\n" + "="*60)
print("Stage 2: AB vs C vs D (非E만)")
print("="*60)

# 非E 실제 레이블로만 학습 (Stage 1 오류 전파 방지)
mask_nonE_tr = (y_seg != 'E')
X_nonE = X[mask_nonE_tr]
y2_map = {'A': 0, 'B': 0, 'C': 1, 'D': 2}    # AB=0, C=1, D=2
y2 = y_seg[mask_nonE_tr].map(y2_map)

print(f"非E 학습 샘플: {len(X_nonE)}")
print(f"AB={y2.eq(0).sum()}, C={y2.eq(1).sum()}, D={y2.eq(2).sum()}")

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

# AB(186명) 가중치 극단적으로 높임
counts2 = y2.value_counts()
sw2 = y2.map({0: len(y2) / (3 * counts2[0]),
              1: len(y2) / (3 * counts2[1]),
              2: len(y2) / (3 * counts2[2])})

skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_s2_prob = np.zeros((len(X_nonE), 3))
test_s2_prob = np.zeros((len(te), 3))

# test 중 非E로 예측된 것만 대상 (test_s1==1)
X_test_nonE = X_test[test_s1 == 1]

for fold, (tr_idx, val_idx) in enumerate(skf2.split(X_nonE, y2)):
    print(f"  Fold {fold+1}/5 ...", flush=True)
    m = lgb.LGBMClassifier(**params2)
    m.fit(X_nonE.iloc[tr_idx], y2.iloc[tr_idx],
          sample_weight=sw2.iloc[tr_idx],
          eval_set=[(X_nonE.iloc[val_idx], y2.iloc[val_idx])],
          categorical_feature=cat_cols,
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof_s2_prob[val_idx] = m.predict_proba(X_nonE.iloc[val_idx])
    if len(X_test_nonE) > 0:
        test_s2_prob[test_s1 == 1] += m.predict_proba(X_test_nonE) / 5

oof_s2 = np.argmax(oof_s2_prob, axis=1)    # 0=AB, 1=C, 2=D

# AB recall 확인
ab_true = (y2 == 0).astype(int)
ab_pred = (oof_s2 == 0).astype(int)
s2_ab_recall = recall_score(ab_true, ab_pred)
print(f"Stage2 AB recall: {s2_ab_recall:.4f}")
print(f"Stage2 AB precision: {ab_pred.sum()} 예측, {ab_true.sum()} 실제")

# %% [4] Stage 3 — A vs B (AB만)
print("\n" + "="*60)
print("Stage 3: A vs B (AB만)")
print("="*60)

# AB 실제 레이블로만 학습
mask_AB_tr = (y_seg == 'A') | (y_seg == 'B')
X_AB = X[mask_AB_tr]
y3 = (y_seg[mask_AB_tr] == 'B').astype(int)    # B=1, A=0

print(f"AB 샘플: {len(X_AB)} (A={y3.eq(0).sum()}, B={y3.eq(1).sum()})")

params3 = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 500,
    'learning_rate': 0.01,
    'num_leaves': 15,        # 작게 (과적합 방지, 186명뿐)
    'min_child_samples': 1,  # leaf 1명도 허용
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 162 / 24,   # A:B = 162:24
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# Stage 3는 186명뿐 → StratifiedKFold 5-fold (fold당 B ≈ 5명)
skf3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_s3_prob = np.zeros(len(X_AB))

# test 중 Stage2에서 AB로 예측된 것
test_AB_mask = (test_s1 == 1) & (np.argmax(test_s2_prob, axis=1) == 0)
X_test_AB = X_test[test_AB_mask]
test_s3_prob = np.zeros(len(X_test_AB))

for fold, (tr_idx, val_idx) in enumerate(skf3.split(X_AB, y3)):
    print(f"  Fold {fold+1}/5 ...", flush=True)
    m = lgb.LGBMClassifier(**params3)
    m.fit(X_AB.iloc[tr_idx], y3.iloc[tr_idx],
          eval_set=[(X_AB.iloc[val_idx], y3.iloc[val_idx])],
          categorical_feature=cat_cols,
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    oof_s3_prob[val_idx] = m.predict_proba(X_AB.iloc[val_idx])[:, 1]
    if len(X_test_AB) > 0:
        test_s3_prob += m.predict_proba(X_test_AB)[:, 1] / 5

# Stage 3 threshold (OOF로 macro-F1 최적화)
best_thr3, best_b_f1 = 0.3, 0
for thr in np.arange(0.1, 0.9, 0.02):
    pred3 = (oof_s3_prob > thr).astype(int)
    bf1 = f1_score(y3, pred3, pos_label=1, zero_division=0)
    mf1 = f1_score(y3, pred3, average='macro', zero_division=0)
    if mf1 > best_b_f1:
        best_b_f1, best_thr3 = mf1, thr

oof_s3_final = (oof_s3_prob > best_thr3).astype(int)
s3_b_recall = recall_score(y3, oof_s3_final, pos_label=1, zero_division=0)
print(f"Stage3 threshold={best_thr3:.2f}, B recall={s3_b_recall:.4f}, macro-F1={best_b_f1:.4f}")
oof_s3 = oof_s3_final

# %% [5] 최종 OOF 조합 및 Macro-F1
print("\n" + "="*60)
print("최종 OOF 조합 & Macro-F1")
print("="*60)

# OOF 최종 예측 조합
oof_final = pd.Series(['E'] * len(tr), index=tr.index)

# 非E 영역 (oof_s1==1) → Stage 2 결과 반영
s2_pred_series = pd.Series(oof_s2, index=X_nonE.index)

oof_final[s2_pred_series[s2_pred_series == 1].index] = 'C'
oof_final[s2_pred_series[s2_pred_series == 2].index] = 'D'

# AB 영역 → Stage 3 결과 반영
s3_pred_series = pd.Series(oof_s3, index=X_AB.index)
oof_final[s3_pred_series[s3_pred_series == 1].index] = 'B'
oof_final[s3_pred_series[s3_pred_series == 0].index] = 'A'

# Macro-F1 계산
macro_f1 = f1_score(y_seg, oof_final, average='macro')
per_class = f1_score(y_seg, oof_final, average=None, labels=['A', 'B', 'C', 'D', 'E'])
print(f"\n★ Cascade OOF Macro-F1: {macro_f1:.4f}")
print(f"A:{per_class[0]:.3f} B:{per_class[1]:.3f} C:{per_class[2]:.3f} D:{per_class[3]:.3f} E:{per_class[4]:.3f}")

# 혼동행렬
print("\n혼동행렬:")
cm = confusion_matrix(y_seg, oof_final, labels=['A', 'B', 'C', 'D', 'E'])
print(pd.DataFrame(
    cm,
    index=['실제_' + s for s in list('ABCDE')],
    columns=['예측_' + s for s in list('ABCDE')]
))

# OOF 분포
print("\nOOF 예측 분포:")
print(oof_final.value_counts().sort_index())

# %% [6] Test 예측 & 저장
print("\n" + "="*60)
print("Test 예측 & 저장")
print("="*60)

# Test 최종 조합
test_final = pd.Series(['E'] * len(te), index=te.index)
s2_test = np.argmax(test_s2_prob, axis=1)

# 非E 예측된 곳에 C/D 할당
nonE_te_idx = te.index[test_s1 == 1]
s2_test_nonE = s2_test[test_s1 == 1]
test_final[nonE_te_idx[s2_test_nonE == 1]] = 'C'
test_final[nonE_te_idx[s2_test_nonE == 2]] = 'D'

# AB 예측 → Stage 3 적용
s3_test = (test_s3_prob > best_thr3).astype(int)
test_final[te.index[test_AB_mask][s3_test == 1]] = 'B'
test_final[te.index[test_AB_mask][s3_test == 0]] = 'A'

print("Test 예측 분포:")
print(test_final.value_counts().sort_index())

# 저장
os.makedirs(f'{BASE_DIR}/results', exist_ok=True)
np.save(f'{BASE_DIR}/results/cascade_oof.npy', oof_final.values)
np.save(f'{BASE_DIR}/results/cascade_test_pred.npy', test_final.values)

# sample_submission 형식으로 제출 파일 생성
sub = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')
sub['Segment'] = test_final.values
sub.to_csv(f'{BASE_DIR}/results/submission_cascade_v1.csv', index=False)
print(f"submission_cascade_v1.csv 저장 완료: {sub.shape}")

print("\n모든 작업 완료!")
