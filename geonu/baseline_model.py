# %% [셋업]
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

PROJ = "/Users/namgeon-u/Desktop/claude/segment analysis"
os.makedirs(os.path.join(PROJ, "results"), exist_ok=True)

print("LightGBM version:", lgb.__version__)

# %% [로드] train_features 로드 + 기본 확인
df = pd.read_parquet(os.path.join(PROJ, "features/train_features.parquet"))
print(f"Train shape: {df.shape}")
print("Segment distribution:")
print(df['Segment'].value_counts())

# %% [전처리] 타깃 인코딩 / 범주형 컬럼 처리
SEG_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
SEG_INV = {v: k for k, v in SEG_MAP.items()}
y = df['Segment'].map(SEG_MAP)

EXCLUDE = ['Segment', 'ID', '기준년월']
feature_all = [c for c in df.columns if c not in EXCLUDE]

# object(string) 컬럼 → pd.Categorical 변환
obj_cols = [c for c in df.select_dtypes(include='object').columns if c not in EXCLUDE]
print(f"\nObject columns to categorize ({len(obj_cols)}): {obj_cols}")

X = df[feature_all].copy()
for col in obj_cols:
    X[col] = pd.Categorical(X[col])

cat_features_all = obj_cols  # LightGBM categorical_feature 인자로 사용

print(f"\nFeature matrix shape: {X.shape}")

# %% [Tier1 피처] Tier 1 + 핵심 Tier 2 컬럼 목록 정의
all_cols = set(X.columns)

def pick(keywords):
    """키워드를 포함하는 컬럼 중 실제 존재하는 것만 반환"""
    result = []
    for kw in keywords:
        matched = [c for c in all_cols if kw in c]
        result.extend(matched)
    return list(dict.fromkeys(result))  # 순서 유지 + 중복 제거

# Tier 1: 핵심 신용/한도/이용 피처
tier1_explicit = [
    '카드이용한도금액_last',
    '청구금액_R6M_last',
    '총잔액_B0M_last',
    '입회경과개월수_신용_last',
    '회원여부_연체_ever',
    '이용후경과월_신용_last',
    '잔액_신판평균한도소진율_r6m_last',
    '이용없음_기본',
    '이용없음_신판',
    '이용없음_할부',
]
tier1_mean6m = pick(['이용금액_일시불_B0M_mean_6m', '이용금액_할부_B0M_mean_6m',
                     '이용금액_CA_B0M_mean_6m', '이용금액_카드론_B0M_mean_6m',
                     '이용금액_신용합계_B0M_mean_6m', '이용금액_온라인_B0M_mean_6m',
                     '이용금액_오프라인_B0M_mean_6m', '카드이용한도금액_mean_6m'])

tier1 = [c for c in tier1_explicit if c in all_cols] + [c for c in tier1_mean6m if c in all_cols]

# Tier 2: 보조 피처
tier2_explicit = [
    'CA이자율_할인전_last',
    'CL이자율_할인전_last',
    '연체잔액_B0M_last',
    '포인트_마일리지_건별_B0M_last',
    '이용거절여부_카드론_ever',
    'RV약정청구율_last',
    'RV최소결제비율_last',
    '연속무실적개월수_기본_24M_카드_last',
]
tier2_count_mean = pick(['이용건수_신용_B0M_mean_6m', '이용건수_신판_B0M_mean_6m'])
tier2_missing = [c for c in all_cols if c.endswith('_결측')]

tier2 = [c for c in tier2_explicit if c in all_cols] + \
        [c for c in tier2_count_mean if c in all_cols] + \
        [c for c in tier2_missing if c in all_cols]

tier12 = list(dict.fromkeys(tier1 + tier2))  # 중복 제거
print(f"\nTier1 features ({len(tier1)}): {tier1}")
print(f"\nTier2 features ({len(tier2)}): {tier2}")
print(f"\nTier1+2 total: {len(tier12)}")

# cat_features for Tier1+2 subset
cat_features_t12 = [c for c in cat_features_all if c in tier12]
print(f"Categorical in Tier1+2: {cat_features_t12}")

# %% [class_weight] 빈도역수 가중치 계산
counts = Counter(y.tolist())
total = len(y)
n_classes = len(counts)
class_weight = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}
print("\nClass weights (frequency inverse):")
for cls, w in sorted(class_weight.items()):
    print(f"  Class {cls} ({SEG_INV[cls]}): count={counts[cls]:,}  weight={w:.4f}")

sample_weight = y.map(class_weight)

# %% [CV 함수] StratifiedKFold OOF 평가 함수
params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

def run_cv(X, y, sample_weight, feature_cols, cat_features, params, tag):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=int)
    importances = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X[feature_cols], y)):
        X_tr = X[feature_cols].iloc[tr_idx]
        X_val = X[feature_cols].iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]
        sw_tr = sample_weight.iloc[tr_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_features if cat_features else 'auto',
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)]
        )
        oof[val_idx] = model.predict(X_val)
        importances.append(model.feature_importances_)

        fold_f1 = f1_score(y_val, oof[val_idx], average='macro')
        per_class = f1_score(y_val, oof[val_idx], average=None, labels=[0, 1, 2, 3, 4])
        print(f"[{tag}] Fold {fold+1} Macro-F1: {fold_f1:.4f} | "
              f"A:{per_class[0]:.3f} B:{per_class[1]:.3f} C:{per_class[2]:.3f} "
              f"D:{per_class[3]:.3f} E:{per_class[4]:.3f}")

    oof_f1 = f1_score(y, oof, average='macro')
    per_class_oof = f1_score(y, oof, average=None, labels=[0, 1, 2, 3, 4])
    print(f"\n[{tag}] ★ OOF Macro-F1: {oof_f1:.4f}")
    print(f"[{tag}] OOF 클래스별 F1: "
          f"A:{per_class_oof[0]:.3f} B:{per_class_oof[1]:.3f} C:{per_class_oof[2]:.3f} "
          f"D:{per_class_oof[3]:.3f} E:{per_class_oof[4]:.3f}")
    return oof, np.mean(importances, axis=0), model  # 마지막 fold 모델 반환

# %% [모델A] Tier 1+2 핵심 피처만으로 LightGBM 학습
print("\n" + "="*60)
print("모델A: Tier1+2 피처")
print("="*60)
oof_t12, imp_t12, _ = run_cv(X, y, sample_weight, tier12, cat_features_t12, params, "ModelA")

# %% [모델B] 전체 피처로 LightGBM 학습
print("\n" + "="*60)
print("모델B: 전체 피처")
print("="*60)
oof_all, imp_all, _ = run_cv(X, y, sample_weight, feature_all, cat_features_all, params, "ModelB")

# %% [비교] 두 모델 OOF Macro-F1 + 클래스별 F1 출력
print("\n" + "="*60)
print("최종 비교")
print("="*60)
for tag, oof in [("ModelA Tier1+2", oof_t12), ("ModelB All   ", oof_all)]:
    macro = f1_score(y, oof, average='macro')
    per = f1_score(y, oof, average=None, labels=[0, 1, 2, 3, 4])
    print(f"[{tag}] Macro-F1={macro:.4f} | A:{per[0]:.3f} B:{per[1]:.3f} C:{per[2]:.3f} D:{per[3]:.3f} E:{per[4]:.3f}")

# %% [중요도] 피처 중요도 시각화 (모델B 기준 상위 30개)
imp_df = pd.DataFrame({'feature': feature_all, 'importance': imp_all})
imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

top30 = imp_df.head(30)
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'][::-1], top30['importance'][::-1])
ax.set_title('ModelB Feature Importance (Top 30, gain avg over folds)')
ax.set_xlabel('Importance')
plt.tight_layout()
fig.savefig(os.path.join(PROJ, "results/feat_importance_top30.png"), dpi=100)
plt.close(fig)
print("\n피처 중요도 상위 10개 (모델B):")
print(imp_df.head(10).to_string(index=False))

# %% [저장] OOF 예측, 피처 중요도 CSV 저장
np.save(os.path.join(PROJ, "results/oof_tier1.npy"), oof_t12)
np.save(os.path.join(PROJ, "results/oof_all.npy"), oof_all)
imp_df.to_csv(os.path.join(PROJ, "results/feat_importance.csv"), index=False)
print(f"\n저장 완료:")
print(f"  results/oof_tier1.npy  shape={oof_t12.shape}")
print(f"  results/oof_all.npy    shape={oof_all.shape}")
print(f"  results/feat_importance.csv  rows={len(imp_df)}")
print(f"  results/feat_importance_top30.png")
