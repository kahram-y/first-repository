# 🏠 주택 가격 예측 — 고급 회귀 기법 (Kaggle)

> Ames 주택 데이터셋을 활용하여  
> **EDA → 전처리 → 다중 회귀 모델 비교 → 앙상블 → 하이퍼파라미터 튜닝**까지의  
> 전체 회귀 파이프라인을 구현하는 Kaggle 프로젝트입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/MainQuest5-house%20price%20prediction-kahram.ipynb)

---

## 📁 파일 구조

```
etc/
└── MainQuest5-house price prediction-kahram.ipynb    # 전체 파이프라인 노트북

데이터 (Kaggle에서 별도 다운로드 필요)
├── train.csv        # 학습 데이터 (SalePrice 라벨 포함)
├── test.csv         # 테스트 데이터 (SalePrice 없음)
└── submission.csv   # 최종 예측 제출 파일 (생성됨)
```

---

## 🎯 프로젝트 목표

| 항목 | 내용 |
|------|------|
| 예측 대상 (y) | `SalePrice` — 주택 판매 가격 ($) |
| 평가 지표 | **RMSE** (Root Mean Squared Error) |
| 데이터 | Ames Housing Dataset (80개 이상의 피처) |

---

## 📊 데이터셋

Ames, Iowa 지역의 주택 거래 데이터로, 건물·부지·내외부 품질 등 다양한 범주형·수치형 피처로 구성됩니다.

### 주요 피처 (SalePrice와 상관성 상위)

| 피처명 | 설명 |
|--------|------|
| `OverallQual` | 자재 및 마감 품질 종합 평가 |
| `GrLivArea` | 지상 거주 면적 (평방 피트) |
| `GarageCars` | 차고 수용 차량 수 |
| `TotalBsmtSF` | 지하실 전체 면적 |
| `1stFlrSF` | 1층 면적 |
| `YearBuilt` | 최초 건축 연도 |
| `Neighborhood` | 지역 위치 |

---

## 🔄 전체 파이프라인

```
train.csv / test.csv
        │
        ▼
① EDA
   ├── SalePrice 분포 확인 (원본 vs 로그 변환)
   ├── 상위 상관 변수 탐색 (Top 10)
   ├── 주요 수치형 변수 산점도
   └── 범주형 변수 박스플롯 (OverallQual, Neighborhood)
        │
        ▼
② 전처리
   ├── SalePrice 로그 변환 (log1p)
   ├── 결측치 처리 (수치형: 중앙값 / 범주형: 최빈값)
   ├── 이상치 제거 (GrLivArea ≥ 4000)
   ├── One-Hot Encoding (drop_first=True)
   ├── 컬럼 정렬 (train/test 동기화)
   └── StandardScaler 스케일링
        │
        ▼
③ 모델 비교 (5종)
   ├── LinearRegression
   ├── Ridge
   ├── Lasso ← 최우수
   ├── RandomForest
   └── GradientBoosting
        │
        ▼
④ 앙상블 (VotingRegressor)
   └── Ridge + RandomForest + GradientBoosting
        │
        ▼
⑤ 하이퍼파라미터 튜닝 (GridSearchCV)
   └── Lasso alpha 탐색 → 최적 alpha 선택
        │
        ▼
⑥ 최종 예측 및 submission.csv 생성
   └── 예측값 역변환 (expm1)
```

---

## 🔍 STEP 1 — EDA

### SalePrice 분포 분석

```python
# 원본 vs 로그 변환 비교
sns.histplot(train_df['SalePrice'], kde=True)           # 우편향 분포
sns.histplot(np.log1p(train_df['SalePrice']), kde=True) # 정규분포에 가까워짐
```

> SalePrice는 **오른쪽 꼬리가 긴 우편향 분포** → `log1p` 변환으로 정규성 확보

### 상위 상관 변수 탐색

```python
corr = train_df.corr(numeric_only=True)
top_corr_features = corr['SalePrice'].sort_values(ascending=False).head(10)
```

상위 5개 변수 Pairplot과 상위 6개 변수 산점도로 SalePrice와의 선형 관계를 확인합니다.

### 범주형 변수 분석

```python
sns.boxplot(x='OverallQual', y='SalePrice', data=train_df)
sns.boxplot(x='Neighborhood', y='SalePrice', data=train_df)
```

`OverallQual`이 높을수록, `Neighborhood`에 따라 SalePrice 분포가 크게 달라지는 것을 확인합니다.

---

## 🔧 STEP 2 — 전처리

### 타겟 로그 변환

```python
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
```

RMSE가 고가 주택에 과도하게 치우치지 않도록 분포를 안정화합니다.

### 결측치 처리

```python
# 수치형: 중앙값
train_df = train_df.fillna(train_df.median(numeric_only=True))

# 범주형: 최빈값
for col in train_df.select_dtypes(include='object'):
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
```

### 이상치 제거

```python
# 거주 면적 4000 평방피트 이상 극단값 제거
train_df = train_df[train_df['GrLivArea'] < 4000]
```

### 인코딩 및 컬럼 정렬

```python
train_df = pd.get_dummies(train_df, drop_first=True)
test_df  = pd.get_dummies(test_df,  drop_first=True)

# train 기준으로 test 컬럼 정렬 (없는 컬럼은 0으로 채움)
test_df = test_df.reindex(columns=X.columns, fill_value=0)
```

### StandardScaler 스케일링

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 🤖 STEP 3 — 모델 비교

5가지 회귀 모델을 동일한 조건(train 80% / val 20%)으로 학습하고 RMSE를 비교합니다.

```python
models = {
    "LinearRegression" : LinearRegression(),
    "Ridge"            : Ridge(alpha=1.0),
    "Lasso"            : Lasso(alpha=0.001),
    "RandomForest"     : RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting" : GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
```

| 모델 | 특징 |
|------|------|
| LinearRegression | 기준 모델(Baseline) |
| Ridge (L2) | 다중공선성 완화 |
| **Lasso (L1)** | **최우수 RMSE — 불필요한 피처 계수를 0으로 축소** |
| RandomForest | 비선형 관계 포착 |
| GradientBoosting | 순차적 앙상블 |

> ✅ **Lasso**가 가장 낮은 RMSE → 80개 이상의 피처 중 중요한 피처만 자동 선택하는 정규화 효과

---

## 🏆 STEP 4 — 앙상블 (VotingRegressor)

Ridge, RandomForest, GradientBoosting의 예측값 **평균**으로 편향·분산을 동시에 완화합니다.

```python
from sklearn.ensemble import VotingRegressor

best_models = [
    ('ridge', Ridge(alpha=1.0)),
    ('rf',    RandomForestRegressor(n_estimators=200, random_state=42)),
    ('gb',    GradientBoostingRegressor(random_state=42))
]

ensemble = VotingRegressor(best_models)
ensemble.fit(X_train, y_train)
ensemble_rmse = mean_squared_error(y_val, ensemble.predict(X_val), squared=False)
```

> VotingRegressor는 단일 모델 대비 RMSE를 소폭 개선하는 효과를 보입니다.

---

## ⚙️ STEP 5 — 하이퍼파라미터 튜닝 (GridSearchCV)

최우수 모델인 **Lasso**의 정규화 강도(`alpha`)를 5-Fold CV로 탐색합니다.

```python
from sklearn.model_selection import GridSearchCV

lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
lasso_grid = GridSearchCV(
    Lasso(max_iter=10000, random_state=42),
    lasso_params,
    cv=5,
    scoring='neg_root_mean_squared_error'
)
lasso_grid.fit(X_scaled, y)

print("Best alpha:", lasso_grid.best_params_)
print("Best RMSE:", -lasso_grid.best_score_)
```

---

## 📤 STEP 6 — 최종 예측 및 제출

최적 alpha로 전체 학습 데이터를 학습한 후, **`expm1`** 으로 로그 역변환하여 실제 가격을 복원합니다.

```python
best_alpha = lasso_grid.best_params_['alpha']
final_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
final_model.fit(X_scaled, y)

test_scaled = scaler.transform(test_df)

# log1p 역변환으로 실제 SalePrice 복원
test_preds = np.expm1(final_model.predict(test_scaled))

submission = pd.DataFrame({
    "Id": test_df_original["Id"],
    "SalePrice": test_preds
})
submission.to_csv("submission.csv", index=False)
```

---

## 📋 핵심 설계 결정 요약

| 항목 | 선택 | 이유 |
|------|------|------|
| 타겟 변환 | `log1p` | 우편향 분포 정규화, 고가 주택 오차 안정화 |
| 결측치 전략 | 수치형 중앙값 / 범주형 최빈값 | 이상치에 강건한 중앙값 선택 |
| 이상치 제거 | GrLivArea ≥ 4000 제거 | 극단값이 회귀선을 왜곡 |
| 인코딩 | get_dummies (drop_first=True) | 다중공선성 방지 |
| 최종 모델 | Lasso (최적 alpha) | 피처 자동 선택 + 최저 RMSE |
| 역변환 | `expm1` | log1p의 역함수로 실제 가격 복원 |

---

## 🛠️ 기술 스택

```
Python
├── pandas, numpy                        # 데이터 처리
├── matplotlib, seaborn                  # EDA 시각화
└── scikit-learn
    ├── LinearRegression                 # 선형 회귀 (Baseline)
    ├── Ridge                            # L2 정규화 회귀
    ├── Lasso                            # L1 정규화 회귀 (최종 모델)
    ├── RandomForestRegressor            # 랜덤 포레스트
    ├── GradientBoostingRegressor        # 그래디언트 부스팅
    ├── VotingRegressor                  # 평균 앙상블
    ├── GridSearchCV                     # 하이퍼파라미터 탐색
    ├── StandardScaler                   # 피처 스케일링
    └── mean_squared_error               # RMSE 평가
```

환경: **Kaggle Notebook**

---

## 🚀 실행 방법

1. [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s3e4)에서 `train.csv`와 `test.csv`를 다운로드합니다.
2. Kaggle Notebook 또는 Google Colab에서 노트북을 열고 데이터 경로를 수정합니다.
3. 셀을 위에서부터 순서대로 실행합니다.
4. 실행 완료 후 `submission.csv`가 생성됩니다.

---

## 📎 참고

- 데이터셋: [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/MainQuest5-house%20price%20prediction-kahram.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/etc/MainQuest5-house%20price%20prediction-kahram.ipynb)
