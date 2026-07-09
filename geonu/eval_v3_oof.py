# %% [셋업]
# eval_v3_oof.py — 5-class OOF 평가기 (docs/score_log_v3.md, 결과_브리핑_v3.md 6절 1순위)
# combine 로직(BASE ID 확률평균 + VIP 상위 cap 덮어쓰기)을 OOF에 그대로 적용해
# 리더보드와 같은 지표(A~E 5-class, ID 단위 macro-F1)를 제출 없이 계산한다.
#
# 입력 (v3.1 스크립트를 먼저 실행해야 생성됨):
#   results/base_v3_oof.parquet — 전체 40만 고객의 C/D/E 확률 (A/B 고객 포함, base v3.1)
#   results/vip_v3_oof_pool.csv — Stage① 상위 후보 풀 + Stage② A/B 판정 (vip v3.1)
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

BASE_DIR    = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

CLASSES      = ["A", "B", "C", "D", "E"]
BASE_CLASSES = ["C", "D", "E"]
VIP_CAP_TEST = 60          # model_v3_combine.py의 VIP_CAP과 맞출 것
TRAIN_TEST_RATIO = 4       # train 40만 ID / test 10만 ID
CAP_SWEEP = [30, 46, 60, 80, 100, 150, 225]   # test 기준 cap 후보


# %% [로드]
base = pd.read_parquet(RESULTS_DIR / "base_v3_oof.parquet")
pool = pd.read_csv(RESULTS_DIR / "vip_v3_oof_pool.csv")

y_true_id = base.groupby("ID")["y_true"].first()
assert set(y_true_id.unique()) == set(CLASSES), (
    "base_v3_oof.parquet에 A/B 고객이 없음 — model_v3_base.py를 v3.1로 재실행 필요"
)
print(f"고객 {len(y_true_id):,}명 / VIP 후보 풀 {len(pool):,}명")

# BASE: ID 단위 soft 다수결
prob_cols = [f"prob_{c}" for c in BASE_CLASSES]
id_prob = base.groupby("ID")[prob_cols].mean()
base_pred = pd.Series(
    np.array(BASE_CLASSES)[id_prob.to_numpy().argmax(1)],
    index=id_prob.index, name="pred",
)


# %% [평가 함수] combine 로직 재현
def evaluate(cap_test: int, verbose: bool = False) -> float:
    cap_train = cap_test * TRAIN_TEST_RATIO
    pred = base_pred.copy()
    vip_top = pool.nlargest(cap_train, "prob_ab_stage1")
    pred.loc[vip_top["ID"].values] = vip_top["Segment"].values

    f1 = f1_score(y_true_id, pred[y_true_id.index], labels=CLASSES, average="macro")
    if verbose:
        print(f"\n===== cap_test={cap_test} (cap_train={cap_train}) =====")
        print(f"5-class OOF macro-F1 (ID 단위): {f1:.4f}")
        per = f1_score(y_true_id, pred[y_true_id.index], labels=CLASSES, average=None)
        for c, v in zip(CLASSES, per):
            print(f"  {c} F1: {v:.4f}")
        cm = confusion_matrix(y_true_id, pred[y_true_id.index], labels=CLASSES)
        print("혼동행렬 (행=실제, 열=예측, A~E):")
        print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    return f1


# %% [실행] 현재 설정 상세 + cap 스윕
evaluate(VIP_CAP_TEST, verbose=True)

print("\n[cap 스윕] — score_log_v3.md: 곡선 평평, 이득 ~0.002 확인용")
print("cap_test | 5-class macro-F1")
for cap in CAP_SWEEP:
    print(f"  {cap:>6} | {evaluate(cap):.4f}")

print("\n제출 전 이 값을 docs/score_log_v3.md 표의 'OOF (5-class)' 칸에 기록할 것.")
