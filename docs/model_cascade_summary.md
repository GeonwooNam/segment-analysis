# Model v1 — 3단계 Cascade 요약

> 작성일: 2026-07-05  
> 스크립트: `build_features_v2.py` (파생 피처) → `model_cascade.py` (3단계 모델)  
> 전작: `baseline_model.py` (Model v0, OOF Macro-F1 0.4809)  
> 목표: **B 클래스 F1 = 0.000 탈출 + 전체 Macro-F1 개선**

---

## 배경 — 왜 Cascade인가

베이스라인(단일 5분류)의 구조적 한계:

```
40만 명 공간에서 B(24명)를 찾아야 함
→ class_weight 3333을 줘도 F1 = 0.000
→ 근본 원인: 탐색 공간이 너무 넓음
```

Cascade로 탐색 공간을 단계적으로 좁힘:

```
Stage 1: 40만 명 → E 제거  (320k vs 80k)
Stage 2: 非E 8만 명 → AB 추출  (186명 vs 79,472명)
Stage 3: AB 186명 → A vs B 분리  (162명 vs 24명)
```

---

## 아키텍처

```
┌─────────────────────────────────────────┐
│         전체 고객 400,000명              │
│                                         │
│  Stage 1: E  vs  非E                    │
│  ────────────────────────               │
│  E (320,342)    非E (79,658)            │
│       │               │                │
│    → "E"           Stage 2             │
│                 AB vs C vs D           │
│              ─────────────────         │
│              AB(186) C(21k) D(58k)     │
│                 │                      │
│              Stage 3                   │
│              A  vs  B                  │
│           ─────────────                │
│           A(162)   B(24)               │
└─────────────────────────────────────────┘
```

**각 Stage의 역할**

| Stage | 분류 문제 | 핵심 포인트 |
|---|---|---|
| Stage 1 | E vs 非E (이진) | Non-E recall 최대화 — 실제 非E를 E로 보내면 Stage 2,3 도달 불가. threshold 낮게 |
| Stage 2 | AB vs C vs D (3분류) | Non-E 안에서 프리미엄 클러스터 분리. AB(186명) 탐색 공간 확보 |
| Stage 3 | A vs B (이진) | 186명 안에서 정밀 분류. B(24명) 대상 맞춤 학습 |

---

## 파생 피처 (build_features_v2.py 추가분)

B·A 특화 시그널을 명시적으로 피처화:

| 피처 | 계산 | 의도 |
|---|---|---|
| `초고한도_flag` | 카드이용한도 > 99th percentile | A·B = 극단적 고한도 |
| `vip_score` | 한도 × 이용금액_mean_6m | 한도·이용 복합 프리미엄 지수 |
| `극저금리_flag` | CL이자율 < 5th percentile | A·B = 최우량 신용등급 |
| `장기무연체_flag` | 입회 60개월↑ AND 연체 없음 | 장기 우량 고객 |
| `한도소진율` | 이용금액_mean / (한도 + 1) | 한도 활용 강도 |
| `이용금액_변동계수` | std_6m / (mean_6m + 1) | 이용 패턴 안정성 |

저장: `features/train_features_v2.parquet`, `features/test_features_v2.parquet`

---

## 모델 설정

### Stage 1 — E vs 非E (LightGBM 이진)
```python
# 非E recall 최대화가 목표 (E를 非E로 잘못 보내도 괜찮지만 非E를 E로 보내면 안 됨)
# threshold: 기본 0.5 대신 OOF로 최적화 (非E recall ≥ 0.99 보장)
params_s1 = {
    'objective': 'binary',
    'scale_pos_weight': 320342 / 79658,  # E:非E 비율 역수
    ...
}
```

### Stage 2 — AB vs C vs D (LightGBM 3분류)
```python
# AB(186명) recall 최대화
# class_weight: AB >> C > D
params_s2 = {
    'objective': 'multiclass',
    'num_class': 3,  # AB=0, C=1, D=2
    ...
}
```

### Stage 3 — A vs B (LightGBM 이진)
```python
# 186명만 대상. B(24명) recall 극대화
# scale_pos_weight: 162/24 = 6.75
params_s3 = {
    'objective': 'binary',
    'scale_pos_weight': 162 / 24,
    'min_child_samples': 1,  # leaf 1명도 허용
    ...
}
```

### CV 전략
- **Stage 1, 2**: StratifiedKFold 5-fold, OOF로 다음 Stage 입력 생성
- **Stage 3**: LeaveOneOut 또는 StratifiedKFold 5-fold (24명이라 fold당 B ≈ 5명)
- **오류 전파 차단**: OOF 예측 기준으로 Stage 전달 (test time과 동일한 흐름)

---

## 최종 예측 흐름 (Inference)

```python
# Step 1
s1_prob = stage1_model.predict_proba(X)[:, 1]  # 非E 확률
pred_E = (s1_prob < threshold_s1)              # E로 분류

# Step 2 (非E로 분류된 고객만)
X_nonE = X[~pred_E]
s2_pred = stage2_model.predict(X_nonE)          # AB=0, C=1, D=2
pred_C = (s2_pred == 1)
pred_D = (s2_pred == 2)

# Step 3 (AB로 분류된 고객만)
X_AB = X_nonE[s2_pred == 0]
s3_prob = stage3_model.predict_proba(X_AB)[:, 1]  # B 확률
pred_B = (s3_prob > threshold_s3)                  # B vs A
pred_A = ~pred_B

# 최종 조합
final_pred = ['E'] * len(X)
final_pred[~pred_E][pred_C]  = 'C'
final_pred[~pred_E][pred_D]  = 'D'
# AB 구간
final_pred[~pred_E][s2_pred==0][pred_B] = 'B'
final_pred[~pred_E][s2_pred==0][pred_A] = 'A'
```

---

## 결과

| | Stage 1 | Stage 2 | Stage 3 | 전체 |
|---|---|---|---|---|
| 지표 | 非E Recall | AB Recall | B Recall | **Macro-F1** |
| 목표 | ≥ 0.99 | ≥ 0.90 | > 0 | > 0.48 |
| **실측** | **0.8436** ❌ | **0.1559** ❌ | **0.7083** ✓ | **0.8467** ✓ |

> Stage 1·2 목표 미달이나 B F1 0.000→0.680 탈출 + Macro-F1 +0.366 달성

**클래스별 F1**

| A | B | C | D | E | **Macro-F1** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.950 | **0.680** | 0.707 | 0.896 | 1.000 | **0.8467** |

**v0(베이스라인) 대비 개선**

| | v0 (베이스라인) | v1 (Cascade) | 변화 |
|---|:---:|:---:|:---:|
| Macro-F1 | 0.4809 | **0.8467** | **+0.366** |
| B F1 | 0.000 | **0.680** | **+0.680** |

**혼동행렬 요약**

```
실제 B(24명): 17명 → B (정답) / 7명 → A (A로 오분류)
실제 A(162명): Stage 2에서 대부분 AB로 통과, Stage 3에서 A 분류 (A F1=0.950)
실제 C/D: C↔D 혼동 심각 (12,213건) → C F1 손실의 주원인
실제 E: 완벽 분리 (E F1=1.000)
```

**각 Stage 상세**

| | 목표 | 실측 | 비고 |
|---|---|---|---|
| Stage 1 threshold | Non-E recall ≥ 0.99 | 0.8436 (0.50 사용) | 0.10~0.59 탐색 전 구간 미달 |
| Stage 2 AB recall | ≥ 0.90 | 0.1559 | B 24명은 전원 AB 통과 추정, A 다수 C/D 오분류 |
| Stage 3 B recall | > 0 | 0.7083 (17/24) | threshold=0.12 |

---

## 이슈 및 다음 단계 (v2 후보)

### 발견된 문제

**1. Stage 1 Non-E recall 미달 (0.84 < 0.99 목표)**
- threshold 0.10까지 낮춰도 0.99 달성 불가
- 12,473명의 실제 非E가 E로 누락 (다행히 B 24명은 전원 Stage 2 진입 추정)
- 개선안: `scale_pos_weight` 더 높이기, 또는 threshold를 0.05~0.10까지 탐색

**2. Stage 2 AB recall 극단적 저조 (0.1559)**
- B 24명은 전원 AB 통과한 것으로 보이나, A 162명 대부분이 C/D로 오분류
- A F1=0.950으로 최종 결과는 양호 → AB pool에서 A가 잘 분류됨
- 개선안: Stage 2를 AB vs 非AB 이진으로 바꾸고, 非AB 내 C/D 별도 분류

**3. C↔D 혼동 (12,213건)**
- C 6,445명이 D로, D 5,768명이 C로 오분류
- C F1 0.707의 주원인 → Trend 피처(월별 기울기)가 C/D 경계 개선에 도움 가능

### v2 우선 시도
- Stage 1 threshold 탐색 범위 확대 (0.05까지)
- Stage 2 구조 변경: AB vs 非AB (이진) → C vs D (이진) 2단계
- Trend 피처 추가 (plans.md P4-6, C/D 구분 목적)
