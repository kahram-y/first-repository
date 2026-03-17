# 📱 스마트폰 가격대 분류하기 — 다중 분류 파이프라인

> 스마트폰의 하드웨어 스펙 데이터를 활용하여  
> **EDA → 다중공선성 제거 → 6가지 분류 모델 비교 → 하이퍼파라미터 튜닝 → 모델 해석**까지의  
> 전체 분류 파이프라인을 구현하는 프로젝트입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/ML3.ipynb)

---

## 📁 파일 구조

```
etc/
└── ML3.ipynb       # 전체 파이프라인 노트북 (Google Colab)

데이터 (별도 준비 필요)
└── train.csv       # 스마트폰 스펙 + 가격대 데이터
```

> 데이터 출처: [Kaggle — Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

---

## 🎯 프로젝트 목표

| 항목 | 내용 |
|------|------|
| 예측 대상 (y) | `price_range` — 가격대 (0~3) |
| 문제 유형 | **4클래스 다중 분류** |
| 평가 지표 | Accuracy, Classification Report |

### 타겟 클래스

| 값 | 가격대 |
|----|--------|
| 0 | Very Low (저가) |
| 1 | Low (중저가) |
| 2 | High (고가) |
| 3 | Very High (초고가) |

---

## 📊 데이터셋

| 항목 | 내용 |
|------|------|
| 데이터 크기 | 2,000개 |
| 피처 수 | 20개 + 타겟 1개 |
| 결측치 | 없음 |
| 자료형 | 대부분 `int64` (2개 `float64`: `clock_speed`, `m_dep`) |

### 주요 피처

| 피처명 | 설명 |
|--------|------|
| `battery_power` | 배터리 용량 (mAh) |
| `ram` | 메모리 용량 (MB) ← **핵심 피처** |
| `px_height` / `px_width` | 픽셀 해상도 (높이/너비) |
| `fc` / `pc` | 전면/후면 카메라 화소 (MP) |
| `int_memory` | 내장 메모리 (GB) |
| `clock_speed` | 프로세서 속도 |
| `n_cores` | CPU 코어 수 |
| `blue` / `wifi` / `four_g` 등 | 기능 지원 여부 (이진 변수) |

---

## 🔄 전체 파이프라인

```
train.csv
    │
    ▼
① EDA
   ├── 타겟 분포 (bar plot)
   ├── 수치형 피처 분포 (histplot)
   ├── 이진형 피처 분포 (countplot)
   ├── 상관관계 히트맵
   └── RAM vs price_range (violin plot)
    │
    ▼
② 전처리 및 피처 엔지니어링
   ├── 파생 변수 생성 (camera_sum, fc_pc_ratio, pixel_area)
   ├── IQR 기반 이상치 제거 (ram)
   ├── 다중공선성 확인 (VIF) → fc, pc, px_height, px_width 제거
   ├── Stratified Train/Test 분할 (80:20)
   └── StandardScaler 스케일링
    │
    ▼
③ 6가지 분류 모델 비교
   ├── Logistic Regression ← 최우수
   ├── KNN
   ├── Decision Tree
   ├── Random Forest
   ├── Naive Bayes
   └── SVM
    │
    ▼
④ 5-Fold 교차 검증 (Cross Validation)
    │
    ▼
⑤ SVM 하이퍼파라미터 튜닝 (GridSearchCV)
    │
    ▼
⑥ 모델 해석
   ├── Permutation Importance
   ├── 결정 경계 시각화 (ram + battery_power)
   └── 학습 곡선 (Learning Curve)
```

---

## 🔍 STEP 1 — EDA

### 타겟 분포

```python
df['price_range'].value_counts().plot(kind='bar')
```

4개 클래스(0~3)가 각 500개씩 **균등하게 분포** — 클래스 불균형 없음을 확인합니다.

### 수치형 / 이진형 피처 시각화

```python
# 수치형 8개: 히스토그램
continuous_features = ['battery_power','clock_speed','int_memory',
                       'mobile_wt','ram','talk_time','px_height','px_width']
sns.histplot(df[col], kde=True)

# 이진형 6개: 카운트플롯
binary_features = ['blue','dual_sim','four_g','three_g','touch_screen','wifi']
sns.countplot(x=df[col])
```

### 상관관계 분석

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm")

# price_range와의 상관계수 순위
df.corr()['price_range'].sort_values(ascending=False)
```

### EDA 주요 인사이트

| 인사이트 | 내용 |
|----------|------|
| ⭐ 핵심 발견 | `price_range` ↔ `ram` **매우 강한 양의 상관관계** |
| 피처 간 공선성 | `pc` ↔ `fc`, `px_height` ↔ `px_width` 간 강한 상관관계 |
| 기타 피처 | `price_range`와 나머지 피처들은 약한 상관관계만 존재 |

---

## 🔧 STEP 2 — 전처리 및 피처 엔지니어링

### 파생 변수 생성

```python
df['camera_sum']  = df['fc'] + df['pc']            # 전체 카메라 화소 합
df['fc_pc_ratio'] = np.where(df['pc']==0, 0,
                             df['fc'] / df['pc'])  # 전면/후면 카메라 비율
df['pixel_area']  = df['px_height'] * df['px_width']  # 픽셀 해상도 면적
```

### IQR 기반 이상치 제거

```python
Q1, Q3 = df['ram'].quantile(0.25), df['ram'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['ram'] >= Q1 - 1.5*IQR) & (df['ram'] <= Q3 + 1.5*IQR)]
```

### 다중공선성(VIF) 분석 및 피처 제거

`variance_inflation_factor`로 VIF를 계산하고, 값이 무한대(∞)에 가까운 피처를 제거합니다.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

| 단계 | 제거 피처 | 대체 피처 |
|------|-----------|-----------|
| 제거 전 | `fc`, `pc`, `px_height`, `px_width` (VIF ∞) | - |
| 제거 후 | - | `camera_sum`, `pixel_area` (VIF 안정화) |

---

## 🤖 STEP 3 — 6가지 분류 모델 비교

동일 조건(Stratified 80:20 분할)에서 6가지 모델을 학습하고 정확도를 비교합니다.

```python
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000),
    "KNN"                 : KNeighborsClassifier(),
    "Decision Tree"       : DecisionTreeClassifier(),
    "Random Forest"       : RandomForestClassifier(),
    "Naive Bayes"         : GaussianNB(),
    "SVM"                 : SVC()
}
```

결과는 수평 막대 그래프로 시각화합니다.

---

## 📊 STEP 4 — 5-Fold 교차 검증

단일 홀드아웃 평가의 한계를 보완하기 위해 5-Fold CV로 **평균 성능 ± 표준편차**를 비교합니다.

```python
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
# 평균 ± 표준편차 시각화
plt.barh(cv_df["Model"], cv_df["CV_Mean"], xerr=cv_df["CV_Std"])
```

### 최종 모델 비교 결과

| 모델 | CV 성능 | 특징 |
|------|---------|------|
| **Logistic Regression** | **최우수** | train-CV gap 작음, 일반화 성능 우수 |
| SVM | 높은 train acc, 낮은 CV acc | 과적합 경향 |
| Random Forest | 높은 train acc, 낮은 CV acc | 과적합 경향 |
| Decision Tree | 분산 큼 | 과적합에 민감 |

> 💡 **결론**: 이 데이터의 선형 구조(RAM → price_range)에는 복잡한 모델보다 **단순 선형 모델(Logistic Regression)** 이 더 적합

---

## ⚙️ STEP 5 — SVM 하이퍼파라미터 튜닝 (GridSearchCV)

SVM이 Logistic Regression을 넘어설 수 있는지 성능 역전을 시도합니다.

```python
param_grid = {
    'C'      : [0.1, 1, 10],
    'gamma'  : ['scale', 0.1, 0.01, 0.001],
    'kernel' : ['linear', 'rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)
```

---

## 🔬 STEP 6 — 모델 해석

### Permutation Importance (순열 중요도)

특정 피처를 무작위로 섞었을 때 정확도 하락 폭을 측정하여 피처 중요도를 계산합니다.

```python
perm = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=15, random_state=42, scoring='accuracy'
)
```

| 순위 | 피처 | 해석 |
|------|------|------|
| 1위 | **`ram`** | 가격대 예측의 압도적 핵심 변수 |
| 2위 | **`battery_power`** | 두 번째로 중요한 변수 |

### 결정 경계 시각화

`ram`과 `battery_power` 두 핵심 피처만으로 3개 모델의 결정 경계를 비교합니다.

```python
models2d = {
    "LogReg" : LogisticRegression(max_iter=1000),
    "SVC"    : SVC(kernel="rbf", gamma=0.1, C=1),
    "RF"     : RandomForestClassifier(n_estimators=200)
}
```

| 모델 | 결정 경계 특성 |
|------|---------------|
| LogReg | 선형 경계 → 깔끔한 구분 |
| SVC (RBF) | 비선형 경계 → 곡선 형태 |
| Random Forest | 일부 구간에서 **과적합 흔적** 관찰 |

### 학습 곡선 (Learning Curve)

훈련 샘플 크기를 10%~100%로 변화시키며 Train vs CV 정확도를 비교합니다.

```python
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)
```

> SVM과 RF는 초기부터 빠르게 훈련 데이터를 학습 → **전형적인 과적합(overfitting)** 패턴  
> Logistic Regression은 train-CV gap이 작고 안정적

---

## 📋 최종 결론

```
EDA → RAM이 price_range를 거의 선형적으로 결정
전처리 → fc/pc/px 원본 제거, 파생 변수(camera_sum, pixel_area) 도입
모델 비교 → Logistic Regression 최우수 (CV 정확도 & 일반화)
SVM 튜닝 → GridSearchCV로 최적 파라미터 탐색
모델 해석 → Permutation Importance로 ram > battery_power 순서 확인
학습 곡선 → RF/SVM 과적합, LR 일반화 성능 우수 시각적 확인

→ 단순 선형 모델이 이 데이터 구조에 더 적합
```

---

## 🛠️ 기술 스택

```
Python
├── pandas, numpy                          # 데이터 처리
├── matplotlib, seaborn                    # EDA·결과 시각화
├── statsmodels
│   └── variance_inflation_factor          # 다중공선성(VIF) 분석
└── scikit-learn
    ├── LogisticRegression                 # 최우수 분류 모델
    ├── KNeighborsClassifier               # KNN
    ├── DecisionTreeClassifier             # 결정 트리
    ├── RandomForestClassifier             # 랜덤 포레스트
    ├── GaussianNB                         # 나이브 베이즈
    ├── SVC                               # 서포트 벡터 머신
    ├── GridSearchCV                       # SVM 하이퍼파라미터 탐색
    ├── cross_val_score                    # 5-Fold 교차 검증
    ├── learning_curve                     # 학습 곡선
    ├── permutation_importance             # 순열 중요도
    └── StandardScaler                     # 피처 스케일링
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `ML3.ipynb`를 열거나 위 배지를 클릭합니다.
2. [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)에서 `train.csv`를 다운로드합니다.
3. 셀을 위에서부터 순서대로 실행합니다.

---

## 📎 참고

- 데이터셋: [Kaggle — Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/ML3.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/etc/ML3.ipynb)
