# 📈 ARIMA · ARCH 모델을 활용한 시계열 분석

> **항공기 승객 수** 데이터로 `AutoARIMA`를 통한 **시계열 예측**을,  
> **S&P 500 수익률** 데이터로 `ARCH/GARCH`를 통한 **변동성(Volatility) 모델링**을 실습하는 프로젝트입니다.  
> ACF·PACF 분석 → 로그 변환 → 차분 → 모델 자동 선택 → 예측 시각화까지의 전체 파이프라인을 다룹니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/DS_ts_chp8.ipynb)

---

## 📁 파일 구조

```
etc/
└── DS_ts_chp8.ipynb    # 전체 파이프라인 노트북 (Google Colab)
```

> 모든 데이터는 코드 실행 시 자동으로 로드됩니다. 별도 파일 준비 불필요.

---

## 📋 노트북 구성

| 파트 | 모델 | 목적 | 데이터 |
|------|------|------|--------|
| **Part 1** | AutoARIMA | 평균(수준) 예측 | 항공기 승객 수 (`airline-passengers`) |
| **Part 2** | ARCH / GARCH | 변동성 예측 | S&P 500 수익률 (`arch.data.sp500`) |

---

## 📊 데이터셋

### 1. Airline Passengers

1949~1960년 월별 국제선 항공기 승객 수 데이터입니다.

| 항목 | 내용 |
|------|------|
| 출처 | [jbrownlee/Datasets (GitHub)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv) |
| 기간 | 1949년 1월 ~ 1960년 12월 |
| 주기 | 월별 (월 12회 계절성) |
| 특징 | 증가하는 추세 + 증가하는 분산 (이분산성) + 계절성 |

### 2. S&P 500

S&P 500 지수 조정 종가 및 일별 수익률 데이터입니다.

| 항목 | 내용 |
|------|------|
| 출처 | `arch.data.sp500` 내장 데이터셋 |
| 활용 컬럼 | `Adj Close` (조정 종가) |
| 변환 | `100 × pct_change()` → 일별 수익률 (%) |
| 특징 | 변동성 군집(Volatility Clustering) 현상 |

---

## 🔄 전체 파이프라인

```
PART 1 — AutoARIMA (항공기 승객 예측)
  원본 시계열
      │
      ▼
  로그 변환 → ACF/PACF 분석
      │
      ▼
  1차 차분 → ACF/PACF 재분석
      │
      ▼
  Train/Test 분할 (8:2)
      │
      ▼
  AutoARIMA 모델 선택 (seasonal, m=12)
      │
      ▼
  예측 + 신뢰 구간 시각화


PART 2 — ARCH/GARCH (S&P 500 변동성 모델링)
  S&P 500 일별 수익률
      │
      ▼
  수익률 시각화 (변동성 군집 확인)
      │
      ▼
  ARCH 모델 적합
      │
      ▼
  모델 요약 (p-value, AIC, BIC, GARCH 계수)
      │
      ▼
  변동성 결과 시각화
```

---

## 📉 PART 1 — AutoARIMA 시계열 예측

### Step 1. 데이터 로드 및 시각화

```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
ap = pd.read_csv(url)
ap.drop('Month', axis=1, inplace=True)

plt.plot(ap)
plt.show()
```

원본 데이터에서 **증가하는 추세**와 **커지는 분산(이분산성)** 을 시각적으로 확인합니다.

---

### Step 2. 로그 변환 — 분산 안정화

```python
ap_transformed = np.log(ap)
```

이분산성을 제거하여 분산을 일정하게 만듭니다.

---

### Step 3. ACF / PACF 분석

```python
plot_acf(ap_transformed)   # 점차 감소하는 형태 → 자기상관 존재
plot_pacf(ap_transformed)  # lag 2까지 유의미 → AR(2) 가능성 암시
```

| 분석 | 결과 | 해석 |
|------|------|------|
| ACF (로그 변환 후) | 점차 감소 | 자기상관 존재, 차분 필요 |
| PACF (로그 변환 후) | lag 2까지 유의 | AR 항 존재 |
| ACF (1차 차분 후) | lag 1, lag 12에서 유의 | MA 항 + 계절성 존재 |
| PACF (1차 차분 후) | lag 1, lag 8~12에서 유의 | AR 항 + 계절성 존재 |

---

### Step 4. 1차 차분 — 추세 제거

```python
ap_diff = ap_transformed.diff().dropna()
```

차분 후 ACF/PACF를 재분석하여 계절 주기(lag 12)를 확인합니다.

---

### Step 5. Train / Test 분할

```python
train_size = int(len(ap_transformed) * 0.8)     # 80% 학습

ap_transformed_train = ap_transformed[:train_size]   # 115개
ap_transformed_test  = ap_transformed[train_size:]   #  29개
```

---

### Step 6. AutoARIMA 모델 자동 선택

계절성이 있는 시계열에 최적의 ARIMA 파라미터를 자동으로 탐색합니다.

```python
import pmdarima as pm

model = pm.AutoARIMA(
    seasonal=True,
    suppress_warnings=True,
    trace=True,        # 탐색 과정 출력
    max_D=12,          # 최대 계절 차분 차수
    m=12               # 계절 주기 = 12 (월별)
)
res = model.fit(ap_transformed_train)
```

---

### Step 7. 예측 + 신뢰 구간 시각화

```python
# 예측값 + 95% 신뢰 구간
preds, conf_int = res.predict(
    n_periods=ap_transformed_test.shape[0],
    return_conf_int=True
)

# 시각화: 학습 데이터, 예측선, 실제값(산점도), 신뢰 구간
x_axis = np.arange(ap_transformed_train.shape[0] + preds.shape[0])

plt.plot(x_axis[:ap_transformed_train.shape[0]], ap_transformed_train, alpha=0.75)  # 학습 데이터
plt.plot(x_axis[ap_transformed_train.shape[0]:], preds, alpha=0.75)                 # 예측선
plt.scatter(x_axis[ap_transformed_train.shape[0]:], ap_transformed_test,
            alpha=0.4, marker='o')                                                    # 실제 테스트 데이터
plt.fill_between(x_axis[-preds.shape[0]:],
                 conf_int[:, 0], conf_int[:, 1],
                 alpha=0.1, color='b')                                                # 신뢰 구간 음영
plt.title("Log Transformed Air Passengers Forecast")
plt.show()
```

---

## 📊 PART 2 — ARCH/GARCH 변동성 모델링

### Step 1. S&P 500 수익률 데이터 로드

```python
import arch.data.sp500

data    = arch.data.sp500.load()
market  = data["Adj Close"]
returns = 100 * market.pct_change().dropna()   # 일별 수익률 (%)

ax   = returns.plot()
xlim = ax.set_xlim(returns.index.min(), returns.index.max())
plt.show()
```

수익률 그래프에서 **변동성 군집(Volatility Clustering)** — 큰 변동이 연속적으로 나타나는 현상 — 을 시각적으로 확인합니다.

---

### Step 2. ARCH 모델 적합

```python
from arch import arch_model

am  = arch_model(returns)   # 기본 ARCH 모델
res = am.fit(update_freq=5) # 5 반복마다 수렴 과정 출력
```

---

### Step 3. 모델 요약 및 해석

```python
print(res.summary())
```

주요 확인 항목:

| 지표 | 의미 |
|------|------|
| **p-value** | 각 계수의 통계적 유의성 (< 0.05이면 유의) |
| **AIC** | 모델 적합도 비교 기준 (낮을수록 좋음) |
| **BIC** | 모델 복잡도 패널티 포함 비교 기준 |
| **alpha\[1\]** | ARCH 효과 계수 (전 시점 잔차 제곱 영향) |
| **beta\[1\]** | GARCH 효과 계수 (전 시점 분산 영향) |

> alpha\[1\]과 beta\[1\]은 GARCH(1,1)을 의미하며,  
> p-value < 0.05로 신뢰도 95%에서 두 계수 모두 **통계적으로 유의**함을 확인합니다.

---

### Step 4. 변동성 결과 시각화

```python
res.plot()
```

표준화 잔차, 조건부 변동성 등을 한눈에 시각화하여 모델 적합 결과를 확인합니다.

---

## 📌 ARIMA vs ARCH/GARCH 핵심 차이

| 구분 | ARIMA | ARCH / GARCH |
|------|-------|--------------|
| 모델링 대상 | 시계열의 **평균(수준)** | 시계열의 **분산(변동성)** |
| 주 활용 분야 | 수요 예측, 재고 예측 | 금융 수익률 변동성 예측 |
| 가정 | 등분산성 | 이분산성 (변동성 군집) |
| 데이터 예시 | 항공기 승객 수 | 주식 수익률 |

---

## 🛠️ 기술 스택

```
Python
├── pmdarima
│   └── AutoARIMA              # ARIMA 파라미터 자동 선택 (AIC 기준)
├── statsmodels
│   ├── ARIMA                  # ARIMA 모델
│   ├── plot_acf               # 자기상관 함수 시각화
│   └── plot_pacf              # 편자기상관 함수 시각화
├── arch
│   ├── arch_model             # ARCH/GARCH 변동성 모델
│   └── arch.data.sp500        # S&P 500 내장 데이터셋
├── pandas / numpy             # 데이터 처리 및 수치 연산
└── matplotlib                 # 시각화
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `DS_ts_chp8.ipynb`를 열거나 위 배지를 클릭합니다.
2. 첫 번째 셀에서 필요한 패키지를 설치합니다 (`pmdarima`, `arch`).
3. 셀을 위에서부터 순서대로 실행합니다.
   - 데이터는 URL에서 자동으로 로드됩니다.

> ⚠️ `AutoARIMA(trace=True)` 실행 시 파라미터 탐색 과정이 상세하게 출력됩니다.

---

## 📎 참고

- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/DS_ts_chp8.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/etc/DS_ts_chp8.ipynb)
- pmdarima 공식 문서: [alkaline-ml.com/pmdarima](https://alkaline-ml.com/pmdarima/)
- arch 공식 문서: [arch.readthedocs.io](https://arch.readthedocs.io/)
