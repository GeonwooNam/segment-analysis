# -*- coding: utf-8 -*-
"""P1 폴드 단위 학습 (재시작/체크포인트). 사용: python p1_train_fold.py <fold 0-4>
없거나 'auto'면 아직 안 끝난 다음 폴드 1개를 학습."""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

RNG = 42; N_FOLDS = 5; SEG_ORDER = ['A', 'B', 'C', 'D', 'E']; E_I = 4; E_CAP = 40000
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ADIR = os.path.join(ROOT, 'modeling', 'artifacts')

X = pd.read_parquet(os.path.join(ADIR, 'X_train.parquet'))
Xte = pd.read_parquet(os.path.join(ADIR, 'X_test.parquet'))
y = np.load(os.path.join(ADIR, 'y.npy'))
oof_p = os.path.join(ADIR, 'oof.npy'); tp_p = os.path.join(ADIR, 'testproba.npy'); dn_p = os.path.join(ADIR, 'folddone.npy')
oof = np.load(oof_p) if os.path.exists(oof_p) else np.zeros((len(X), 5), dtype='float32')
test_proba = np.load(tp_p) if os.path.exists(tp_p) else np.zeros((len(Xte), 5), dtype='float32')
done = set(np.load(dn_p).tolist()) if os.path.exists(dn_p) else set()

arg = sys.argv[1] if len(sys.argv) > 1 else 'auto'
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)
splits = list(skf.split(X, y))
if arg == 'auto':
    todo = [i for i in range(N_FOLDS) if i not in done]
    if not todo:
        print('모든 폴드 완료:', sorted(done)); sys.exit(0)
    fold = todo[0]
else:
    fold = int(arg)

tr_i, va_i = splits[fold]
params = dict(objective='multiclass', num_class=5, learning_rate=0.05, n_estimators=700,
              num_leaves=63, max_bin=127, min_child_samples=60, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.5, reg_lambda=5.0, class_weight='balanced',
              random_state=RNG, n_jobs=-1, verbosity=-1)

t1 = time.time()
rs = np.random.RandomState(RNG + fold)
e_mask = y[tr_i] == E_I
keep_e = rs.choice(tr_i[e_mask], size=min(E_CAP, int(e_mask.sum())), replace=False)
tr_use = np.sort(np.concatenate([tr_i[~e_mask], keep_e]))
m = lgb.LGBMClassifier(**params)
m.fit(X.iloc[tr_use], y[tr_use], eval_set=[(X.iloc[va_i], y[va_i])], eval_metric='multi_logloss',
      callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
oof[va_i] = m.predict_proba(X.iloc[va_i]).astype('float32')
test_proba += (m.predict_proba(Xte) / N_FOLDS).astype('float32')
done.add(fold)
np.save(oof_p, oof); np.save(tp_p, test_proba); np.save(dn_p, np.array(sorted(done)))
f_fold = f1_score(y[va_i], oof[va_i].argmax(1), average='macro')
print(f'[fold {fold}] n_tr={len(tr_use)} best_iter={m.best_iteration_} '
      f'macroF1(argmax)={f_fold:.4f} ({time.time()-t1:.0f}s) done={sorted(done)}', flush=True)
