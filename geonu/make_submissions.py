# %% 제출 파일 생성 (baseline + cascade v1)
import numpy as np
import pandas as pd
import lightgbm as lgb
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tr = pd.read_parquet(f'{BASE}/features/train_features.parquet')
te = pd.read_parquet(f'{BASE}/features/test_features.parquet')

cat_cols = [c for c in tr.columns
            if (tr[c].dtype == object or pd.api.types.is_string_dtype(tr[c]))
            and c not in ['ID', 'Segment']]
for c in cat_cols:
    tr[c] = pd.Categorical(tr[c].astype(str))
    te[c] = pd.Categorical(te[c].astype(str))

feat_cols = [c for c in tr.columns if c not in ['ID', '기준년월', 'Segment']]
X = tr[feat_cols]
y = tr['Segment']
X_test = te[feat_cols]

# class_weight 빈도역수 (baseline과 동일)
counts = y.value_counts()
sw = y.map({s: len(y) / (len(counts) * counts[s]) for s in counts.index})

# %% baseline — 전체 데이터 학습 후 test 예측
print("=== Baseline 학습 중 ===")
params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}
model = lgb.LGBMClassifier(**params)
model.fit(X, y, sample_weight=sw, categorical_feature=cat_cols)

test_pred = model.predict(X_test)
sub = pd.read_csv(f'{BASE}/sample_submission.csv')
sub['Segment'] = test_pred
os.makedirs(f'{BASE}/submissions', exist_ok=True)
sub.to_csv(f'{BASE}/submissions/submission_baseline.csv', index=False)
print("submission_baseline.csv 저장 완료")
print(pd.Series(test_pred).value_counts().sort_index())

# %% cascade v1 — 이미 존재하면 확인만
cascade_path = f'{BASE}/results/submission_cascade_v1.csv'
cascade_dest  = f'{BASE}/submissions/submission_cascade_v1.csv'
if os.path.exists(cascade_path):
    sub2 = pd.read_csv(cascade_path)
    sub2.to_csv(cascade_dest, index=False)
    print("\nsubmission_cascade_v1.csv 확인 완료")
    print(sub2['Segment'].value_counts().sort_index())
else:
    print(f"\n[경고] {cascade_path} 없음 — cascade 먼저 실행 필요")
