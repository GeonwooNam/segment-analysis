# -*- coding: utf-8 -*-
"""P1 최종화 — OOF 평가 + 클래스가중 튜닝 + 제출 생성."""
import os, numpy as np, pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix

SEG_ORDER = ['A', 'B', 'C', 'D', 'E']
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ADIR = os.path.join(ROOT, 'modeling', 'artifacts')

oof = np.load(os.path.join(ADIR, 'oof.npy'))
test_proba = np.load(os.path.join(ADIR, 'testproba.npy'))
y = np.load(os.path.join(ADIR, 'y.npy'))
meta = np.load(os.path.join(ADIR, 'meta.npz'), allow_pickle=True)
test_ids = meta['test_ids']


def macro_f1(yt, p): return f1_score(yt, p, average='macro')


def tune_class_weights(y, proba, n_restart=8, n_iter=600, seed=42):
    rs = np.random.RandomState(seed); K = proba.shape[1]
    best_w = np.ones(K); best_f1 = macro_f1(y, proba.argmax(1))
    grid = np.array([0.2, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    for r in range(n_restart):
        w = np.ones(K) if r == 0 else np.exp(rs.uniform(-1.8, 1.8, K))
        f = macro_f1(y, (proba * w).argmax(1))
        for it in range(n_iter):
            k = rs.randint(K); cand = w.copy(); improved = False
            for g in grid:
                cand[k] = g; fc = macro_f1(y, (proba * cand).argmax(1))
                if fc > f: f = fc; w = cand.copy(); improved = True
            if not improved and it > K * 4: break
        if f > best_f1: best_f1 = f; best_w = w.copy()
    return best_w, best_f1


f_argmax = macro_f1(y, oof.argmax(1))
w, f_tuned = tune_class_weights(y, oof)
print('=== OOF 결과 (전체 400k, 정직한 검증) ===')
print(f'Macro F1 (argmax)      : {f_argmax:.4f}')
print(f'Macro F1 (weight-tuned): {f_tuned:.4f}   w={np.round(w, 2)}\n')
print('[argmax] per-class:')
print(classification_report(y, oof.argmax(1), target_names=SEG_ORDER, digits=4, zero_division=0))
print('[weight-tuned] per-class:')
print(classification_report(y, (oof * w).argmax(1), target_names=SEG_ORDER, digits=4, zero_division=0))
print('[weight-tuned] confusion matrix (행=실제 A..E):')
print(confusion_matrix(y, (oof * w).argmax(1)))

pred_i = (test_proba * w).argmax(1)
sub = pd.DataFrame({'ID': test_ids, 'Segment': [SEG_ORDER[i] for i in pred_i]})
out = os.path.join(ROOT, 'modeling', 'submission_p1.csv')
sub.to_csv(out, index=False, encoding='utf-8-sig')
print('\n제출 저장:', out)
print('test 예측 분포:', sub['Segment'].value_counts().reindex(SEG_ORDER).to_dict())
print('train 실제 분포:', {SEG_ORDER[i]: int((y == i).sum()) for i in range(5)})
