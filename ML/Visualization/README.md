# 🚲 바이크 수요 예측 모델 — 하이퍼파라미터 튜닝과 모델 해석 (2일차)

> Kaggle 자전거 공유 수요 데이터셋을 활용하여  
> **교차 검증 → GridSearchCV 하이퍼파라미터 튜닝 → XAI(피처 중요도 · PDP)**까지의  
> 머신러닝 성능 극대화 및 모델 해석 파이프라인을 구현하는 프로젝트입니다.  
> *(1일차: EDA + 모델 비교 → 2일차: 튜닝 + 해석)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/ML/Visualization/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_2%EC%9D%BC%EC%B0%A8_%EC%84%B1%EB%8A%A5_%EA%B7%B9%EB%8C%80%ED%99%94%EB%A5%BC_%EC%9C%84%ED%95%9C_%ED%8A%9C%EB%8B%9D.ipynb)

---

## 📁 파일 구조

```
ML/Visualization/
└── 머신러닝_2일차_성능_극대화를_위한_튜닝.ipynb    # 전체 파이프라인 노트북

데이터 (Kaggle에서 별도 다운로드 필요)
└── train.csv    # 자전거 공유 수요 데이터
```

> 데이터 출처: [Kaggle — Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv)

---

## 🎯 프로젝트 목표

| 항목 | 내용 |
|------|------|
| 예측 대상 (y) | `count` — 시간당 자전거 대여 수요 |
| 평가 지표 | **RMSLE** (Root Mean Squared Log Error) |
| 핵심 모델 | Gradient Boosting Regressor |

---

## 📊 1일차 Recap — 주요 발견

| 발견 | 내용 |
|------|------|
| 데이터 특성 | 시간·계절·온도 등이 수요와 **비선형 관계** |
| 모델 성능 차이 | 선형 모델 < 트리 기반 앙상블 모델 (RF, GBM) |
| 원인 | 트리 모델의 **귀납적 편향(Inductive Bias)** 이 데이터 패턴과 잘 부합 |

---

## 📅 2일차 목표

```
① 더 신뢰성 있는 평가  →  K-Fold 교차 검증
② 성능 극대화          →  GridSearchCV 하이퍼파라미터 튜닝
③ 모델 해석 (XAI)      →  Feature Importance + PDP
```

---

## 🔄 전체 파이프라인

```
train.csv
    │
    ▼
① 데이터 로드 및 전처리
   ├── datetime → year / month / day / hour 분리
   └── 불필요 컬럼 제거 (atemp, humidity, casual, registered)
    │
    ▼
② 평가 지표 정의 — RMSLE
    │
    ▼
③ K-Fold 교차 검증 (5-Fold)
   └── 기본 GBR 성능 기준선 설정
    │
    ▼
④ GridSearchCV 하이퍼파라미터 튜닝
   └── 최적 GBR 파라미터 탐색 (3-Fold)
    │
    ▼
⑤ 모델 해석 (XAI)
   ├── Feature Importance (MDI 기반)
   └── Partial Dependence Display (PDP)
```

---

## 🔧 STEP 1 — 데이터 로드 및 전처리

### datetime 피처 분리

```python
train_df['datetime'] = pd.to_datetime(train_df['datetime'])

train_df['year']  = train_df['datetime'].dt.year
train_df['month'] = train_df['datetime'].dt.month
train_df['day']   = train_df['datetime'].dt.day
train_df['hour']  = train_df['datetime'].dt.hour
```

### 불필요 컬럼 제거

```python
X = train_df.drop(['datetime', 'count', 'casual', 'registered', 'humidity'], axis=1)
y = train_df['count']
```

| 제거 컬럼 | 제거 이유 |
|-----------|-----------|
| `atemp` | `temp`와 역할이 거의 동일 (1일차 확인) |
| `humidity` | 수요와 낮은 상관성 |
| `casual` / `registered` | `count`의 구성 요소 → 타겟 누출(Data Leakage) |
| `datetime` | 년·월·일·시로 분해 완료 |

---

## 📏 STEP 2 — 평가 지표: RMSLE

```python
def rmsle(y_true, y_pred):
    log_true = np.log1p(np.maximum(y_true, 0))
    log_pred = np.log1p(np.maximum(y_pred, 0))
    return np.sqrt(np.mean((log_true - log_pred) ** 2))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
```

### RMSLE를 사용하는 이유

| 특성 | 설명 |
|------|------|
| 우편향(Skewed) 분포 대응 | 로그 변환으로 큰 값의 오차가 과도하게 반영되는 것을 방지 |
| 과소 예측 페널티 | 수요를 너무 낮게 예측하는 것에 더 큰 패널티 부여 |
| 수요 예측 적합성 | 자전거 재고·운영 계획 수립에 과소 예측이 더 치명적이기 때문 |

> 값이 작을수록 좋은 지표입니다 (RMSLE ↓)

---

## 🔁 STEP 3 — K-Fold 교차 검증

단일 `train_test_split`의 한계(우연에 의한 성능 왜곡)를 극복하기 위해 **5-Fold CV**를 적용합니다.

```python
gbr = GradientBoostingRegressor(random_state=42)

def k_fold_cross_validation(model, X, y, n_splits=5):
    scores = cross_val_score(model, X, y, cv=n_splits, scoring=rmsle_scorer)
    mean_score = -np.mean(scores)
    print(f"폴드별 RMSLE: {-scores}")
    print(f"평균 RMSLE (기본 모델): {mean_score:.4f}")

k_fold_cross_validation(gbr, X, y)
```

### 단일 분할 vs K-Fold 비교

| 방식 | 특징 | 단점 |
|------|------|------|
| train_test_split | 빠름 | 특정 분할에 따라 성능 왜곡 가능 |
| **K-Fold (5-Fold)** | 신뢰도 높은 평균 성능 | 학습 시간 K배 증가 |

---

## ⚙️ STEP 4 — GridSearchCV 하이퍼파라미터 튜닝

최적 파라미터 조합을 자동 탐색하여 성능을 극대화합니다.

```python
param_grid = {
    'n_estimators' : [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth'    : [4, 5],
    'subsample'    : [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    scoring=rmsle_scorer,
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X, y)
best_model = grid_search.best_estimator_
```

### 탐색 파라미터 설명

| 파라미터 | 탐색 범위 | 의미 |
|----------|-----------|------|
| `n_estimators` | 100 · 200 · 300 | 앙상블에 사용할 트리 수 |
| `learning_rate` | 0.05 · 0.1 | 각 트리의 기여도 조정 (작을수록 보수적) |
| `max_depth` | 4 · 5 | 개별 트리의 최대 깊이 (클수록 복잡) |
| `subsample` | 0.8 · 1.0 | 각 트리 학습에 사용할 샘플 비율 |

> 총 탐색 조합 수: 3 × 2 × 2 × 2 × 3(CV) = **72가지**

---

## 🔬 STEP 5 — 모델 해석 (XAI)

### 5-1. 특성 중요도 (Feature Importance)

GBM의 각 피처별 **MDI(Mean Decrease in Impurity)** 기반 중요도를 시각화합니다.

```python
feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
importance_df = importance_df.sort_values('importance', ascending=False)

sns.barplot(x='importance', y='feature', data=importance_df)
```

#### 계산 원리

```
단일 트리에서 피처 X의 중요도
  = 해당 트리 내에서 X가 사용된 모든 분기의 불순도 감소량 합산

GBM 전체 피처 중요도
  = 모든 트리에 걸친 단일 트리 중요도의 평균 → 정규화 (합계 = 1)
```

#### 피처 중요도 결과

| 순위 | 피처 | 해석 |
|------|------|------|
| 🥇 1위 | **`hour`** | 출퇴근 시간대 패턴 — 압도적 핵심 변수 |
| 🥈 2위 | **`workingday`** | 근무일 여부 — 전역적 수요 구분 |
| 🥉 3위 | **`year`** | 연도별 수요 증가 추세 |
| 4위 | **`temp`** | 온도에 따른 수요 변화 |

---

### 5-2. 부분 의존성 플롯 (Partial Dependence Display, PDP)

다른 모든 변수의 효과를 평균내어 제거했을 때, **특정 변수 하나**가 예측값에 미치는 순수 영향을 시각화합니다.

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(14, 6))
PartialDependenceDisplay.from_estimator(
    best_model, X,
    features=['hour', 'workingday', 'year', 'temp'],
    ax=ax
)
```

#### PDP 결과 해석

| 피처 | PDP 패턴 | 인사이트 |
|------|----------|----------|
| **`hour`** | 오전 8시 · 오후 5~6시 **두 개의 뚜렷한 피크** | 출퇴근 시간대 수요 급증 — 강한 비선형 패턴 |
| **`temp`** | 20~30°C 구간에서 최고, 양 끝에서 감소하는 **역 U자 곡선** | 적정 온도 범위에서 수요 최대 |
| `workingday` | 상대적으로 완만한 변화 | 다른 변수와의 강한 상호작용으로 단독 효과는 작음 |
| `year` | 완만한 우상향 | 전역적 효과로 조기 학습 → PDP 변화량 작음 |

> 💡 **`workingday`와 `year`의 Feature Importance는 높지만 PDP 변화량이 작은 이유:**
> - 부스팅 초기 단계에서 전역적 분산을 크게 줄이는 데 기여 → 이후 단계에서 단독 효과는 작아짐
> - `workingday`는 `hour` 등 다른 변수와 강한 상호작용을 가져 단독 효과가 희석됨

---

## 📋 최종 결론

```
비선형 패턴 검증
  hour: 출퇴근 두 피크 (오전 8시 / 오후 5~6시)
  temp: 역 U자 곡선 (20~30°C 최적)
  → 선형 모델 실패 원인 명확히 규명

모델 성능
  기본 GBR (5-Fold CV) → GridSearchCV 튜닝 후 RMSLE 개선

핵심 피처: hour > workingday > year > temp
```

> **"단순히 이 모델이 좋다"를 넘어 "왜 좋은지"를 설명할 수 있게 됨** — 현업 의사결정 및 모델 신뢰성 확보의 핵심 역량

---

## 🛠️ 기술 스택

```
Python
├── pandas                              # 데이터 처리 및 datetime 분해
├── matplotlib, seaborn                 # 피처 중요도 시각화
└── scikit-learn
    ├── GradientBoostingRegressor       # 핵심 예측 모델
    ├── cross_val_score                 # K-Fold 교차 검증
    ├── GridSearchCV                    # 하이퍼파라미터 탐색
    ├── make_scorer                     # 커스텀 RMSLE scorer 등록
    └── PartialDependenceDisplay        # PDP 시각화 (XAI)
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 노트북을 열거나 위 배지를 클릭합니다.
2. [Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv)에서 `train.csv`를 다운로드하여 현재 작업 디렉토리에 넣습니다.
3. 셀을 위에서부터 순서대로 실행합니다.

> ⚠️ GridSearchCV는 총 72가지 조합을 탐색하므로 **실행에 수 분이 소요**됩니다.

---

## 📎 참고

- 데이터셋: [Kaggle — Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv)
- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/ML/Visualization/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_2%EC%9D%BC%EC%B0%A8_%EC%84%B1%EB%8A%A5_%EA%B7%B9%EB%8C%80%ED%99%94%EB%A5%BC_%EC%9C%84%ED%95%9C_%ED%8A%9C%EB%8B%9D.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository)
