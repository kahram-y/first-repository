# 📡 시계열 데이터 분류 — tsfresh + 머신러닝

> 로봇 실행 실패 감지(Robot Execution Failures) 시계열 데이터를 대상으로  
> **tsfresh 자동 피처 추출 → 피처 선택 → 분류 모델 학습 및 평가**까지의 전체 파이프라인을 구현하는 프로젝트입니다.  
> 항공기 승객 데이터를 활용한 **정상성 확보** 실습도 함께 포함되어 있습니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/ds4_timeseries_4_5.ipynb)

---

## 📁 파일 구조

```
etc/
└── ds4_timeseries_4_5.ipynb    # 전체 파이프라인 노트북 (Google Colab)
```

> 모든 데이터는 코드 실행 시 자동으로 다운로드됩니다. 별도 파일 준비 불필요.

---

## 📋 노트북 구성

| 파트 | 주제 | 데이터 |
|------|------|--------|
| **Part 4** | 시계열 정상성 확보 | 항공기 승객 수 (`airline-passengers`) |
| **Part 5** | 시계열 분류 미니 프로젝트 | 로봇 실행 실패 (`robot_execution_failures`) |

---

## 📊 데이터셋

### 1. Robot Execution Failures (`tsfresh` 내장)

로봇 팔이 작업을 수행하는 동안 측정된 힘(Force)과 토크(Torque) 6채널 시계열 데이터입니다.

| 컬럼 | 설명 |
|------|------|
| `id` | 시계열 샘플 ID |
| `time` | 시간 스텝 |
| `F_x`, `F_y`, `F_z` | X·Y·Z축 힘 (Force) |
| `T_x`, `T_y`, `T_z` | X·Y·Z축 토크 (Torque) |
| `y` | 실행 실패 여부 (True / False) |

| 항목 | 값 |
|------|-----|
| 전체 샘플 수 | 88개 |
| 실패 샘플 (True) | 21개 |
| 정상 샘플 (False) | 67개 |

### 2. Airline Passengers (정상성 실습용)

1949~1960년 월별 국제선 항공기 승객 수 데이터입니다.

---

## 🔄 전체 파이프라인

```
Robot Execution Failures 시계열 데이터
              │
              ▼
① 커스텀 Train/Test 분할 (클래스 비율 유지)
              │
              ▼
② tsfresh 피처 추출
   ├── Part 4: MinimalFCParameters (경량)
   └── Part 5: EfficientFCParameters (고성능)
              │
              ▼
③ 결측치 처리 (impute)
              │
              ▼
④ 피처 선택 (select_features) — Part 4
              │
              ▼
⑤ 분류 모델 학습 및 평가
   ├── Logistic Regression (Part 4)
   ├── Random Forest       (Part 5)
   └── XGBoost             (Part 5)
              │
              ▼
⑥ 피처 중요도 분석 (XGBoost plot_importance)
              │
              ▼
⑦ Classification Report
```

---

## 🔧 PART 4 — tsfresh 기초 실습

### 커스텀 Train/Test 분할

클래스 불균형을 고려하여 True/False 비율을 유지하는 커스텀 분할 함수를 구현합니다.

```python
def custom_classification_split(x, y, test_size=0.3):
    num_true  = int(y.sum() * test_size)           # True 샘플 수: int(21 * 0.3) = 6
    num_false = int((len(y) - y.sum()) * test_size) # False 샘플 수: int(67 * 0.3) = 20

    id_list = (y[y==False].head(num_false).index.to_list() +
               y[y==True].head(num_true).index.to_list())

    X_train = x[~x['id'].isin(id_list)]
    X_test  = x[ x['id'].isin(id_list)]
    y_train = y.drop(id_list).sort_index()
    y_test  = y.loc[id_list].sort_index()

    return X_train, y_train, X_test, y_test
```

### 피처 추출 — MinimalFCParameters

계산 효율을 위해 최소한의 통계적 피처만 추출합니다.

```python
from tsfresh.feature_extraction import MinimalFCParameters

settings = MinimalFCParameters()
minimal_features_train = extract_features(
    X_train, column_id="id", column_sort="time",
    default_fc_parameters=settings
)
```

추출된 피처 예시: `F_x__sum_values`, `T_z__maximum` 등

### 피처 선택 — select_features

통계적으로 유의미한 피처만 남깁니다.

```python
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)                        # NaN 대체
features_filtered = select_features(extracted_features, y)
```

### 분류 모델 — Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
logistic.fit(minimal_features_train, y_train)
logistic.score(minimal_features_test, y_test)
```

---

## 🔬 PART 4 (부록) — 시계열 정상성 확보

항공기 승객 데이터를 활용하여 비정상 시계열을 정상화하는 과정을 실습합니다.

### 1. 로그 변환 — 분산 안정화

```python
log_transformed = np.log(ap)
```

분산이 일정해지는 효과를 시각화로 확인합니다.

### 2. 1차 차분 — 추세 제거

```python
diffed = log_transformed.diff()
```

증가하는 추세와 커지는 분산을 제거합니다.

### 3. 계절 차분 (주기=12) — 계절성 제거

```python
seasonally_diffed = diffed.diff(12)
seasonally_diffed.dropna(inplace=True)
```

### 4. ADF 검정 — 정상성 확인

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(x):
    stat, p_value, *_ = adfuller(x)
    print('ADF statistics:', stat)
    print('P-value:', p_value)

adf_test(seasonally_diffed)
```

> p-value < 0.05이면 정상성 확보 완료

| 변환 단계 | 목적 |
|-----------|------|
| 로그 변환 | 분산 안정화 (이분산성 제거) |
| 1차 차분 | 추세 제거 |
| 계절 차분 (lag=12) | 계절성 제거 |
| ADF 검정 | 정상성 통계적 검증 |

---

## 🏆 PART 5 — 시계열 분류 미니 프로젝트

### 데이터 분할

test_size를 **0.25**로 설정합니다.

```python
X_train, y_train, X_test, y_test = custom_classification_split(
    timeseries, y, test_size=0.25
)
```

### 피처 추출 — EfficientFCParameters

Part 4의 Minimal보다 더 많은 피처를 추출하는 효율적 설정을 사용합니다.

```python
from tsfresh.feature_extraction import EfficientFCParameters

settings = EfficientFCParameters()

comprehensive_features_train = extract_features(
    X_train, column_id="id", column_sort="time",
    default_fc_parameters=settings
)
comprehensive_features_test = extract_features(
    X_test, column_id="id", column_sort="time",
    default_fc_parameters=settings
)

impute(comprehensive_features_train)
impute(comprehensive_features_test)
```

### 모델 1 — Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=10, max_depth=3)
rf_clf.fit(comprehensive_features_train, y_train)
rf_clf.score(comprehensive_features_test, y_test)
# → 약 0.66 (불만족스러운 결과)
```

### 모델 2 — XGBoost

```python
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(n_estimators=10, max_depth=3)
xgb_clf.fit(comprehensive_features_train, y_train)
xgb_clf.score(comprehensive_features_test, y_test)
```

### 피처 중요도 분석

`gain` 기준 피처 중요도를 시각화합니다.

```python
xgb.plot_importance(xgb_clf, importance_type='gain')
plt.show()

# 유의미한 피처 수 확인
sum(xgb_clf.feature_importances_ != 0)
# → F_x의 abs_energy 피처가 핵심 피처로 파악됨
```

### 최종 성능 평가 — Classification Report

```python
from sklearn.metrics import classification_report

classification_report(
    y_test,
    xgb_clf.predict(comprehensive_features_test),
    target_names=['true', 'false'],
    output_dict=True
)
```

---

## ⚙️ 피처 추출 설정 비교

| 설정 | 클래스 | 추출 피처 수 | 용도 |
|------|--------|-------------|------|
| `MinimalFCParameters` | 최소 | 소량 | 빠른 프로토타이핑 |
| `EfficientFCParameters` | 효율 | 중간 | 실용적 균형 |
| `ComprehensiveFCParameters` | 전체 | 대량 | 최대 성능 탐색 |

---

## 🛠️ 기술 스택

```
Python
├── tsfresh
│   ├── extract_features          # 시계열 피처 자동 추출
│   ├── select_features           # 통계적 피처 선택
│   ├── MinimalFCParameters       # 경량 피처 추출 설정
│   ├── EfficientFCParameters     # 효율 피처 추출 설정
│   └── impute                    # 결측치 대체
├── scikit-learn
│   ├── LogisticRegression        # 로지스틱 회귀 분류
│   ├── RandomForestClassifier    # 랜덤 포레스트 분류
│   └── classification_report     # 분류 성능 평가
├── xgboost
│   ├── XGBClassifier             # XGBoost 분류
│   └── plot_importance           # 피처 중요도 시각화
├── statsmodels
│   └── adfuller                  # ADF 정상성 검정
├── pandas / numpy                # 데이터 처리
└── matplotlib                    # 시각화
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `ds4_timeseries_4_5.ipynb`를 열거나 위 배지를 클릭합니다.
2. 데이터는 코드 실행 시 자동으로 다운로드됩니다.
3. 셀을 위에서부터 순서대로 실행합니다.

> ⚠️ `EfficientFCParameters`를 사용하는 피처 추출 단계는 피처 수가 많아 **수 분의 실행 시간**이 소요됩니다.

---

## 📎 참고

- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/ds4_timeseries_4_5.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/etc/ds4_timeseries_4_5.ipynb)
- tsfresh 공식 문서: [tsfresh.readthedocs.io](https://tsfresh.readthedocs.io/)
