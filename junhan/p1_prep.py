# -*- coding: utf-8 -*-
"""P1 데이터 준비 (1회 실행) — 8개 카테고리 201812 조인 → 전처리 → 디스크 저장.
산출: X_train.parquet, X_test.parquet, y.npy, meta.npz (modeling/artifacts/)."""
import os, glob, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

MONTH = '201812'
SEG_ORDER = ['A', 'B', 'C', 'D', 'E']; SEG2I = {s: i for i, s in enumerate(SEG_ORDER)}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ADIR = os.path.join(ROOT, 'modeling', 'artifacts'); os.makedirs(ADIR, exist_ok=True)


def load_split(split, month=MONTH):
    frames, seg = None, None
    for n in range(1, 9):
        f = glob.glob(os.path.join(ROOT, split, f'{n}.*', f'{month}_*_*.parquet'))[0]
        d = pd.read_parquet(f)
        if n == 1 and split == 'train':
            seg = d[['ID', 'Segment']].copy()
        d = d.drop(columns=[c for c in ('기준년월', 'Segment') if c in d.columns])
        feat = [c for c in d.columns if c != 'ID']
        d = d.rename(columns={c: f'c{n}_{c}' for c in feat})
        frames = d if frames is None else frames.merge(d, on='ID', how='left')
    assert frames['ID'].is_unique
    return frames, seg


t0 = time.time()
train_df, seg = load_split('train')
test_df, _ = load_split('test')
print(f'로드 train={train_df.shape} test={test_df.shape} ({time.time()-t0:.0f}s)', flush=True)

test_ids = test_df['ID'].values
y = seg.set_index('ID').loc[train_df['ID']]['Segment'].map(SEG2I).values.astype('int8')

tr = train_df.drop(columns=['ID']); te = test_df.drop(columns=['ID'])
del train_df, test_df

# 상수 제거 (train nunique<=1)
nun = tr.nunique(dropna=True)
const = nun[nun <= 1].index.tolist()
tr = tr.drop(columns=const); te = te.drop(columns=const)

# 범주형(비수치) 합집합 카테고리 정렬
cat_cols = [c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]
for c in cat_cols:
    cats = pd.api.types.union_categoricals(
        [pd.Categorical(tr[c].astype('object')), pd.Categorical(te[c].astype('object'))]).categories
    tr[c] = pd.Categorical(tr[c].astype('object'), categories=cats)
    te[c] = pd.Categorical(te[c].astype('object'), categories=cats)

# 수치형 float32 다운캐스트
num_cols = [c for c in tr.columns if c not in cat_cols]
for c in num_cols:
    tr[c] = tr[c].astype('float32'); te[c] = te[c].astype('float32')

print(f'전처리 완료: {tr.shape[1]}개 특징 (상수제거 {len(const)}, 범주형 {len(cat_cols)})', flush=True)

tr.to_parquet(os.path.join(ADIR, 'X_train.parquet'))
te.to_parquet(os.path.join(ADIR, 'X_test.parquet'))
np.save(os.path.join(ADIR, 'y.npy'), y)
np.savez(os.path.join(ADIR, 'meta.npz'), test_ids=test_ids, cat_cols=np.array(cat_cols, dtype=object))
print(f'저장 완료 -> {ADIR} ({time.time()-t0:.0f}s)', flush=True)
print('클래스 분포:', {SEG_ORDER[i]: int((y == i).sum()) for i in range(5)}, flush=True)
