# %% [셋업]
# model_v3_combine.py — modeling_plan.md 3절 최종 결합
# BASE 예측(C/D/E, ID 단위 확률 평균) 위에 VIP 예측(A/B)을 덮어쓰기 → submission 생성
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
SUB_DIR     = BASE_DIR / "submissions"
SUB_DIR.mkdir(exist_ok=True)

CLASSES = ["C", "D", "E"]

# VIP 덮어쓰기 인원 = Stage① 확률 상위 N명. 이 컷이 VIP 인원을 결정하는 유일한 지점
# (vip 스크립트는 랭킹만 제공). OOF cap 스윕 결과 곡선이 평평해 튜닝 이득 ~0.002 —
# 건드리지 말 것 (docs/score_log_v3.md). 변경 시 eval_v3_oof.py의 VIP_CAP_TEST도 맞출 것.
VIP_CAP = 60


# %% [로드]
base = pd.read_parquet(RESULTS_DIR / "base_v3_test_prob.parquet")
vip  = pd.read_csv(RESULTS_DIR / "vip_v3_test_pred.csv")
print(f"base {base.shape} / vip 후보 {len(vip)}명")

# BASE: ID 단위 soft 다수결 (6개월 확률 평균 후 argmax — 3등 다수결의 soft 버전)
prob_cols = [f"prob_{c}" for c in CLASSES]
id_prob = base.groupby("ID")[prob_cols].mean()
final = pd.Series(
    np.array(CLASSES)[id_prob.to_numpy().argmax(1)],
    index=id_prob.index, name="Segment",
)
print("BASE 예측 분포:")
print(final.value_counts().sort_index().to_string())


# %% [VIP 덮어쓰기] Stage① 확률 상위 VIP_CAP명만
vip_top = vip.sort_values("prob_ab_stage1", ascending=False).head(VIP_CAP)
final.loc[vip_top["ID"].values] = vip_top["Segment"].values
print(f"\nVIP 덮어쓰기 {len(vip_top)}명 (cap={VIP_CAP}):")
print(vip_top["Segment"].value_counts().to_string())


# %% [제출 파일 생성]
sub = pd.read_csv(BASE_DIR / "sample_submission.csv")
assert sub["ID"].is_unique and len(sub) == final.index.nunique(), "제출 ID 불일치"
sub["Segment"] = sub["ID"].map(final)
assert sub["Segment"].notna().all(), "매핑 안 된 ID 존재"

out_path = SUB_DIR / "submission_v3.csv"
sub.to_csv(out_path, index=False)
print(f"\n최종 분포:")
print(sub["Segment"].value_counts().sort_index().to_string())
print(f"저장 완료: {out_path}")
# 제출 전: eval_v3_oof.py로 5-class OOF를 먼저 확인하고, 제출 후 Public과 함께
# docs/score_log_v3.md에 페어로 기록할 것 (base 출력 0.83대는 3-class라 비교 대상 아님)
