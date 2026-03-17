# 💳 신용카드 사기 검출 — 실전 불균형 분류 파이프라인

> Kaggle 신용카드 사기 검출 데이터셋을 활용하여  
> **클래스 불균형 대응 → 다중 모델 학습 → 앙상블 → 임계값 최적화**까지의  
> 실전 수준 불균형 이진 분류 파이프라인을 구현하는 프로젝트입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/MainQuest4.ipynb)

---

## 📁 파일 구조

```
etc/
└── MainQuest4.ipynb     # 전체 파이프라인 노트북 (Google Colab / Kaggle)

데이터 (Kaggle에서 별도 다운로드 필요)
├── train.csv            # 학습 데이터 (Class 라벨 포함)
├── test.csv             # 테스트 데이터 (Class 라벨 없음)
└── submission.csv       # 최종 예측 제출 파일 (생성됨)
```

---

## 📊 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/competitions/playground-series-s3e4) |
| 피처 | `V1` ~ `V28` (PCA 변환된 익명 피처) + `Amount` |
| 제거 컬럼 | `Time` (예측에 불필요), `id` |
| 타겟 | `Class` (0: 정상, 1: 사기) |
| 특징 | **극심한 클래스 불균형** — 사기 비율 매우 낮음 |

---

## 🔄 전체 파이프라인

```
train.csv / test.csv
        │
        ▼
① EDA (클래스 분포, 상관관계 히트맵, PCA 2D)
        │
        ▼
② 전처리 (결측치 중앙값 대체 + StandardScaler)
        │
        ▼
③ Train / Val 분할 (80:20, Stratified)
        │
        ▼
④ 모델 학습 + 불균형 대응 + RandomizedSearchCV
   ├── Logistic Regression (L1/L2) + SMOTE
   ├── XGBoost + scale_pos_weight
   └── LightGBM + SMOTE
        │
        ▼
⑤ 피처 중요도 분석 (XGBoost, LightGBM)
        │
        ▼
⑥ 앙상블
   ├── Soft Voting
   └── Stacking (meta: Logistic Regression)
        │
        ▼
⑦ 임계값 탐색 (PR-curve 기반, F1 최대화)
        │
        ▼
⑧ 최종 예측 및 submission.csv 생성
```

---

## 🔍 STEP 1 — EDA

### 클래스 분포 시각화

```python
plot_class_distribution(train_df['Class'], title='Train Class Distribution')
```

사기(Class=1)와 정상(Class=0)의 극심한 불균형을 확인합니다.

### 상관관계 히트맵

```python
sns.heatmap(train_df_imputed.corr(), cmap='RdBu_r', center=0)
```

PCA 변환된 V1~V28 피처 간의 상관 구조를 파악합니다.

### PCA 2D 시각화

```python
plot_pca_2d(X, y, title='PCA 2D - Train')
```

2개 주성분으로 정상·사기 샘플의 분포 분리 가능성을 시각적으로 탐색합니다.

---

## 🔧 STEP 2 — 전처리

### 결측치 처리

수치형 컬럼의 결측치를 **중앙값**으로 대체합니다. (train fit → test transform 방식으로 누출 방지)

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
imputer.fit(train_df.drop(['Class', 'id'], axis=1, errors='ignore'))

X_train_imputed = pd.DataFrame(imputer.transform(...), columns=...)
X_test_imputed  = pd.DataFrame(imputer.transform(...), columns=...)
```

### Train / Val 분할

클래스 비율을 유지하는 **Stratified 분할**로 불균형 데이터의 대표성을 보장합니다.

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## ⚖️ STEP 3 — 클래스 불균형 대응 전략

불균형 문제에 대해 모델별로 다른 전략을 적용하고 비교합니다.

| 전략 | 설명 | 적용 모델 |
|------|------|-----------|
| **SMOTE** | 소수 클래스 합성 오버샘플링 | Logistic Regression, LightGBM |
| **scale_pos_weight** | 양성 클래스에 가중치 부여 (`neg/pos` 비율) | XGBoost |
| **class_weight='balanced'** | 클래스별 손실 가중치 자동 조정 | Logistic Regression |
| RandomOverSampler | 소수 클래스 단순 복사 | (옵션) |
| RandomUnderSampler | 다수 클래스 다운샘플링 | (옵션) |

```python
# XGBoost scale_pos_weight 계산
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
```

---

## 🤖 STEP 4 — 모델 학습 및 하이퍼파라미터 튜닝

### 공통 구조: `tune_and_fit()` 함수

샘플러 유무에 따라 `sklearn Pipeline` 또는 `imblearn Pipeline`을 자동 선택하고,
`RandomizedSearchCV`로 하이퍼파라미터를 탐색합니다.

```python
def tune_and_fit(model, param_dist, sampler=None, n_iter=20):
    if sampler is None:
        pipe = Pipeline([('scaler', scaler), ('clf', model)])
    else:
        pipe = ImbPipeline([('scaler', scaler), ('sampler', sampler), ('clf', model)])

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        pipe, param_distributions=param_dist,
        n_iter=n_iter,
        scoring='average_precision',   # ← PR-AUC를 최적화 기준으로 사용
        cv=skf, n_jobs=-1, refit=True
    )
    rs.fit(X_train, y_train)
    return rs
```

> 클래스 불균형 환경에서는 **PR-AUC(average_precision)** 가 ROC-AUC보다 신뢰도 높은 평가 지표입니다.

### 모델 A — Logistic Regression + SMOTE

```python
# L1(Lasso) / L2(Ridge) 비교, saga solver
lr_param_dist = {
    'clf__penalty': ['l1', 'l2'],
    'clf__C': np.logspace(-3, 2, 20)
}
results['lr_smote'] = tune_and_fit(lr, lr_param_dist, sampler=SMOTE(random_state=42))
```

### 모델 B — XGBoost + scale_pos_weight

```python
xgb_param_dist = {
    'clf__n_estimators'     : [50, 100, 200],
    'clf__max_depth'        : [3, 5, 7],
    'clf__learning_rate'    : [0.01, 0.05, 0.1],
    'clf__subsample'        : [0.6, 0.8, 1.0],
    'clf__colsample_bytree' : [0.6, 0.8, 1.0],
    'clf__scale_pos_weight' : [scale_pos_weight]   # 불균형 보정
}
results['xgb_spw'] = tune_and_fit(xgb_clf, xgb_param_dist, sampler=None)
```

### 모델 C — LightGBM + SMOTE

```python
lgb_param_dist = {
    'clf__n_estimators' : [50, 100, 200],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__num_leaves'   : [15, 31, 63],
    'clf__max_depth'    : [-1, 5, 7]
}
results['lgb_smote'] = tune_and_fit(lgb_clf, lgb_param_dist, sampler=SMOTE(random_state=42))
```

---

## 📊 STEP 5 — 피처 중요도 분석

XGBoost와 LightGBM의 `feature_importances_`를 추출하여 상위 10개 피처를 시각화합니다.

```python
fi = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=fi.head(10).values, y=fi.head(10).index)
```

> **핵심 피처 (LGBM 기준):** `V26`, `V4`, `V12`, `V14`  
> → PCA 기반 익명 변수이지만, 사기 여부 구분에 결정적 역할을 하며 신호 대 잡음비가 높은 피처입니다.

---

## 🏆 STEP 6 — 앙상블

3개 모델의 최적 파이프라인을 조합하여 앙상블을 구성합니다.

### Soft Voting

각 모델의 예측 확률을 평균하여 최종 예측을 결정합니다.

```python
voting = VotingClassifier(
    estimators=[('lr', ...), ('xgb', ...), ('lgb', ...)],
    voting='soft', n_jobs=-1
)
voting.fit(X_train, y_train)
```

### Stacking

3개 모델을 베이스 레이어로, **Logistic Regression**을 메타 모델로 사용합니다.

```python
stack = StackingClassifier(
    estimators=[('lr', ...), ('xgb', ...), ('lgb', ...)],
    final_estimator=LogisticRegression(max_iter=2000),
    n_jobs=-1
)
stack.fit(X_train, y_train)
```

### 앙상블 효과 비교

| 구분 | 특징 | 적합한 목적 |
|------|------|-------------|
| **Voting** | 재현율(Recall) 높음 | 사기 탐지율 극대화 |
| **Stacking** | 정밀도(Precision) 높음 | 오탐 최소화 (운영 효율) |
| 공통 PR-AUC | **0.9487** | 매우 우수한 탐지 성능 |

---

## 🎚️ STEP 7 — 임계값 탐색

기본 임계값 0.5 대신 **PR-curve 기반으로 F1을 최대화하는 최적 임계값**을 탐색합니다.

```python
ts = threshold_search(y_val, voting_proba, metric='f1')
```

| 결과 | 값 | 의미 |
|------|-----|------|
| **Best F1 threshold** | **0.719** | F1 최대화 임계값 |
| Precision @ best F1 | 0.952 | 낮은 오탐율 |
| Recall @ best F1 | 0.952 | 높은 탐지율 |
| Threshold (Precision ≥ 0.90) | 0.588 | 정밀도 보장 최소 임계값 |

> 금융권에서는 오탐 비용이 크므로 **정밀도 ≥ 0.90** 조건의 임계값(0.588)을 실무에 적용할 수 있습니다.

---

## 📋 STEP 8 — 모델 성능 비교 (Val Set)

5개 모델을 PR-AUC 기준으로 정렬하여 비교합니다.

```python
summary_df = pd.DataFrame(summary).sort_values(by='pr_auc', ascending=False)
```

| 모델 | 주요 지표 |
|------|-----------|
| `stack` | PR-AUC 최상위, 정밀도 우선 |
| `voting` | PR-AUC 최상위, 재현율 우선 |
| `xgb_spw` | 단일 모델 최고 성능 |
| `lgb_smote` | 안정적 성능 |
| `lr_smote` | 기준 모델(Baseline) |

> PR-AUC ≥ 0.94 이상 → 사기 케이스 탐지 성능이 **매우 우수한 수준**

---

## 🔢 유틸리티 함수 목록

| 함수 | 기능 |
|------|------|
| `print_metrics()` | Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC 출력 |
| `plot_class_distribution()` | 클래스 분포 막대 그래프 |
| `plot_pca_2d()` | PCA 2D 산점도 (클래스별 색상) |
| `threshold_search()` | PR-curve 기반 최적 임계값 탐색 (F1 최대화 / Precision ≥ 0.90) |
| `tune_and_fit()` | 샘플러 포함 파이프라인 + RandomizedSearchCV 자동 구성 |

---

## 🛠️ 기술 스택

```
Python
├── pandas, numpy                       # 데이터 처리
├── matplotlib, seaborn                 # 시각화
├── scikit-learn
│   ├── LogisticRegression (L1/L2)      # Lasso/Ridge 회귀 분류
│   ├── RandomizedSearchCV              # 하이퍼파라미터 탐색 (PR-AUC 기준)
│   ├── StratifiedKFold                 # 불균형 데이터 교차 검증
│   ├── VotingClassifier                # Soft Voting 앙상블
│   ├── StackingClassifier              # Stacking 앙상블
│   ├── PCA                             # 2D 시각화
│   └── StandardScaler, SimpleImputer  # 전처리
├── imbalanced-learn
│   ├── SMOTE                           # 합성 오버샘플링
│   ├── RandomOverSampler               # 단순 오버샘플링
│   └── RandomUnderSampler             # 언더샘플링
├── xgboost                             # XGBoost 분류 + scale_pos_weight
└── lightgbm                            # LightGBM 분류
```

환경: **Google Colab / Kaggle Notebook**

---

## 🚀 실행 방법

1. Google Colab 또는 Kaggle Notebook에서 `MainQuest4.ipynb`를 엽니다.
2. Kaggle에서 `train.csv`와 `test.csv`를 다운로드하여 현재 작업 디렉토리에 넣습니다.
3. 첫 번째 셀에서 필요한 패키지를 설치합니다:
   ```
   !pip install -q xgboost lightgbm imbalanced-learn
   ```
4. 셀을 위에서부터 순서대로 실행합니다.
5. 실행 완료 후 `submission.csv`가 생성됩니다.

---

## 📎 참고

- 데이터셋: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/competitions/playground-series-s3e4)
- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/MainQuest4.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/etc/MainQuest4.ipynb)
