# 📈 수익률 예측을 활용한 최적 포트폴리오 찾기

> 국내 보험사 대장주 5종목의 주가 데이터를 바탕으로 **ARIMA · Prophet · LSTM** 세 가지 모델로 수익률을 예측하고,  
> **몬테카를로 시뮬레이션 + 샤프 지수(Sharpe Ratio)** 를 활용해 위험 대비 수익이 최적인 포트폴리오를 찾는 프로젝트입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/WrapUp/pred_portfolio.ipynb)

---

## 📁 파일 구조

```
WrapUp/
└── pred_portfolio.ipynb    # 전체 파이프라인 노트북 (Google Colab)
```

> 데이터는 `FinanceDataReader` 라이브러리로 자동 수집됩니다. 별도 파일 준비 불필요.

---

## 🗂️ 분석 대상 종목

국내 보험업종 대형주 5종목을 선정했습니다.

| 종목명 | 종목 코드 |
|--------|-----------|
| 삼성생명 | 032830 |
| 삼성화재 | 000810 |
| DB손해보험 | 005830 |
| 현대해상 | 001450 |
| 한화생명 | 088350 |

---

## 📅 데이터 분할

| 구분 | 기간 | 용도 |
|------|------|------|
| Train | 2015-01-01 ~ 2023-12-31 (9년) | 모델 학습, 공분산 행렬 산출 |
| Test | 2024-01-01 ~ 2024-12-31 (1년) | 예측 성능 평가 |

---

## 🔄 전체 파이프라인

```
주가 데이터 수집 (FinanceDataReader)
          │
          ▼
    로그 수익률 계산
          │
    ┌─────┼──────┐
    ▼     ▼      ▼
 ARIMA  LSTM  Prophet   ← 수익률 예측 (3가지 모델)
    └─────┼──────┘
          │
          ▼
  예측 연간 수익률 산출
          │
          ▼
  몬테카를로 시뮬레이션 (10,000회)
          │
          ▼
  최적 포트폴리오 선택
  ├── 최대 샤프 지수 (Max Sharpe Ratio)
  └── 최소 리스크 (Min Risk)
```

---

## 📊 STEP 1 — 과거 데이터 기반 포트폴리오 분석 (베이스라인)

예측 모델을 사용하기 전, 과거 평균 수익률로 기본 포트폴리오 성능을 먼저 확인합니다.

```python
# 일간 & 연간 수익률
daily_ret = data.pct_change()
annual_ret = daily_ret.mean() * 252

# 연간 공분산 행렬 (리스크 측정용)
annual_cov = daily_ret.cov() * 252
```

---

## 🎲 몬테카를로 시뮬레이션

10,000번의 무작위 비중 조합을 생성하여 각 포트폴리오의 기대 수익률, 리스크, 샤프 지수를 계산합니다.

```python
for _ in range(10000):
    w = np.random.random(len(stocks))
    w /= w.sum()                                         # 비중 합계 = 1

    ret    = np.dot(w, forecast_ret)                     # 기대 수익률
    risk   = np.sqrt(np.dot(w.T, np.dot(annual_cov, w)))  # 표준편차(리스크)
    sharpe = ret / risk                                  # 샤프 지수
```

**최적 포트폴리오 선택 기준:**
- ⭐ **Max Sharpe** — 위험 대비 수익이 가장 높은 포트폴리오 (별표 표시)
- ✖ **Min Risk** — 리스크가 가장 낮은 포트폴리오 (X 표시)

---

## 📉 STEP 2 — ARIMA (통계 기반 시계열 모델)

자기회귀이동평균(ARIMA) 모델로 테스트 구간의 로그 수익률을 예측합니다.

```python
model = ARIMA(train_series, order=(1, 0, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test_series))
```

| 파라미터 | 값 |
|----------|----|
| order (p, d, q) | (1, 0, 1) |
| 입력 | 로그 수익률 |
| 예측 기간 | 2024년 전체 (Test 구간) |

예측된 일별 수익률의 평균 × 252 = **연간 예측 수익률**로 변환 후 몬테카를로 시뮬레이션에 투입합니다.

---

## 🧠 STEP 3 — LSTM (딥러닝 기반 시계열 모델)

과거 20일 창(window)을 입력으로 다음 날 수익률을 예측합니다.

```python
model = Sequential([
    LSTM(50, input_shape=(window, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

| 파라미터 | 값 |
|----------|----|
| window 크기 | 20일 |
| LSTM 유닛 수 | 50 |
| Epochs | 10 |
| Batch Size | 32 |
| 스케일링 | MinMaxScaler |

---

## 🔮 STEP 4 — Prophet (구조적 시계열 모델)

추세(Trend)와 연간 계절성(Yearly Seasonality)을 분리하여 학습하는 Facebook Prophet 모델을 적용합니다.

```python
model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=True
)
model.fit(df_train)
forecast = model.predict(future)
```

> Prophet은 통계 모델과 머신러닝의 중간 성격인 **구조적(Structural) 시계열 모델**로, 추세·계절성을 기반으로 안정적인 예측이 가능합니다.

---

## 🏆 모델 성능 비교

3가지 모델의 예측 정확도를 **MAE** · **RMSE** 기준으로 비교합니다.

```python
model_summary = pd.DataFrame({
    'MAE':  [arima_scores['MAE'].mean(),
             prophet_scores['MAE'].mean(),
             lstm_scores['MAE'].mean()],
    'RMSE': [arima_scores['RMSE'].mean(),
             prophet_scores['RMSE'].mean(),
             lstm_scores['RMSE'].mean()]
}, index=['ARIMA', 'Prophet', 'LSTM'])
```

| 모델 | MAE | RMSE | 순위 |
|------|-----|------|------|
| **ARIMA** | **가장 낮음** | **가장 낮음** | 🥇 1위 |
| Prophet | 중간 | 중간 | 🥈 2위 |
| LSTM | 가장 높음 | 가장 높음 | 🥉 3위 |

**결론:** 변동성이 크지 않은 대형 보험주의 경우 **선형 패턴을 잘 포착하는 ARIMA**가 LSTM보다 예측 안정성이 높았습니다.

---

## 📊 포트폴리오 시각화

각 모델의 몬테카를로 시뮬레이션 결과를 **효율적 프론티어(Efficient Frontier)** 형태로 시각화합니다.

- X축: 리스크(표준편차)
- Y축: 기대 수익률
- 색상: 샤프 지수 (밝을수록 높음)
- ⭐ 빨간 별: 최대 샤프 포트폴리오
- ✖ 파란 X: 최소 리스크 포트폴리오

추가로 **KOSPI 지수**와 비교하여 시장 전체 대비 포트폴리오 성능을 확인합니다.

---

## 🛠️ 기술 스택

```
Python
├── FinanceDataReader        # 주가 데이터 수집
├── pandas, numpy            # 데이터 처리 및 수치 연산
├── matplotlib               # 효율적 프론티어 시각화
├── statsmodels (ARIMA)      # 통계 기반 시계열 예측
├── prophet                  # 구조적 시계열 예측
├── tensorflow / keras       # LSTM 딥러닝 예측
│   └── LSTM, Dense, Sequential
└── scikit-learn
    ├── MinMaxScaler         # LSTM 입력 정규화
    └── mean_absolute_error, mean_squared_error  # 성능 평가
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `pred_portfolio.ipynb`를 열거나 위 배지를 클릭합니다.
2. 첫 번째 셀에서 필요한 패키지를 설치합니다 (`finance-datareader`, `prophet` 등).
3. 셀을 위에서부터 순서대로 실행합니다.
   - 데이터는 `FinanceDataReader`로 자동 수집됩니다.
   - LSTM 학습은 종목 5개 × 10 epoch 단위로 진행됩니다.
4. 마지막 셀에서 3가지 모델의 성능 비교 테이블을 확인합니다.

---

## 📎 참고

- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/WrapUp/pred_portfolio.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/WrapUp/pred_portfolio.ipynb)
- 데이터 출처: [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader)