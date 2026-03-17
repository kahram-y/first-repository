# 📈 시계열과 선형 회귀 — Time-step & Lag 피처 엔지니어링

> Kaggle의 [Store Sales - Time Series Forecasting](https://www.kaggle.com/c/29781) 강의 실습 노트북입니다.  
> **타임 더미(Time-step)** 와 **래그(Lag)** 피처를 직접 설계하여 선형 회귀 모델로 시계열을 예측하고,  
> 계열 의존성·시간 의존성의 개념을 시각화와 함께 이해하는 것을 목표로 합니다.

---

## 📁 파일 구조

```
etc/
└── exercise-linear-reg-with-time-series-kor.ipynb    # 실습 노트북 (한국어 번역본)

데이터 (Kaggle에서 자동 다운로드)
├── book_sales.csv       # 양장본 도서 30일 판매 데이터 (개념 설명용)
├── tunnel.csv           # 스위스 Baregg 터널 일별 차량 수 (예제 실습용)
├── ar.csv               # AR 모델 시뮬레이션 데이터 (계열 의존성 실습용)
└── train.csv            # Store Sales 대회 데이터 (연습 문제용)
```

---

## 🎯 학습 목표

이 과정을 마치면 다음을 할 수 있습니다.

| 목표 | 내용 |
|------|------|
| 피처 설계 | 추세·계절성·주기를 모델링하는 시계열 피처 설계 |
| 시각화 | 시간 플롯, 래그 플롯 등 시계열 전용 그래프 활용 |
| 하이브리드 | 서로 보완적인 모델을 결합한 예측 앙상블 구성 |
| 응용 | 다양한 예측 과제에 머신러닝 기법 적용 |

---

## 📋 노트북 구성

| 파트 | 주제 | 데이터 |
|------|------|--------|
| **개념 설명** | 시간 단계 피처 · 래그 피처 이론 | `book_sales.csv` |
| **예제 실습** | 터널 교통량 선형 회귀 | `tunnel.csv` |
| **연습 문제 1** | 회귀 계수 해석 | `book_sales.csv` |
| **연습 문제 2** | 계열 의존성 패턴 구분 | `ar.csv` |
| **연습 문제 3** | Store Sales — 타임 더미 모델 | `train.csv` |
| **연습 문제 4** | Store Sales — 래그 피처 모델 | `train.csv` |

---

## 📊 데이터셋

### 1. Book Sales (`book_sales.csv`)
30일간 양장본(`Hardcover`) 도서 일별 판매량. 시간 의존성·회귀 계수 해석 실습에 사용됩니다.

### 2. Tunnel Traffic (`tunnel.csv`)
스위스 Baregg Tunnel 일별 통과 차량 수 (2003.11 ~ 2005.11). 타임 더미와 래그 피처 모두 적용하는 핵심 예제입니다.

### 3. AR Simulated (`ar.csv`)
`weight = +0.95` / `weight = -0.95` 두 가지 AR 계수를 가진 시뮬레이션 시계열. 계열 의존성 방향을 시각적으로 구분하는 실습에 사용됩니다.

### 4. Store Sales (`train.csv`)
에콰도르 식료품 체인 *Corporación Favorita* 의 2013~2017년 매출 데이터.
실습에서는 전체 약 1,800개 시계열의 **하루 평균 매출**(`average_sales`)만 활용합니다.

---

## 🔄 전체 흐름

```
개념 이해
  ├── 시계열 = 시간 인덱스 + 관측값 열
  ├── 타임 더미 → 시간 의존성 모델링
  └── 래그 피처 → 계열 의존성 모델링
        │
        ▼
예제 실습 (Tunnel Traffic)
  ├── 타임 더미 선형 회귀 → 추세선 적합
  └── 래그-1 선형 회귀 → 자기상관 적합
        │
        ▼
연습 문제 (4문제)
  ├── Q1. 회귀 계수 해석 (기울기 × 시간 = 예측 변화량)
  ├── Q2. AR 계열 의존성 방향 구분 (양·음 weight)
  ├── Q3. Store Sales — 타임 더미 모델 구현
  └── Q4. Store Sales — 래그-1 모델 구현
```

---

## 🔑 핵심 개념

### 시간 단계 피처 (Time-step Feature)

```python
df['Time'] = np.arange(len(df.index))
```

- 시계열 시작부터 끝까지 순서를 매기는 **타임 더미** 생성
- 모델: `target = weight × Time + bias`
- 그래프: 시간 플롯(Time Plot)에서 **추세선** 적합
- 모델링 대상: **시간 의존성** (발생 시점만으로 예측 가능한 패턴)

```python
# Tunnel Traffic 모델 결과 예시
# Vehicles ≈ 22.5 × Time + 98176
```

---

### 래그 피처 (Lag Feature)

```python
df['Lag_1'] = df['target'].shift(1)   # 1스텝 래그
```

- 관측값을 한 스텝 뒤로 미뤄 **이전 시점 값**을 피처로 활용
- 모델: `target = weight × Lag_1 + bias`
- 그래프: 래그 플롯(Lag Plot)에서 **자기상관** 적합
- 모델링 대상: **계열 의존성** (이전 관측으로 현재 값 예측 가능한 패턴)
- 결측치 처리: `dropna()` + `y.align(X, join='inner')`로 인덱스 동기화

---

### AR 계수 해석

| weight 값 | 의미 | 시계열 패턴 |
|-----------|------|-------------|
| **+0.95** (1에 가까움) | 다음 값이 이전 값과 **같은 방향** | 완만하게 오르내리는 부드러운 곡선 |
| **−0.95** (−1에 가까움) | 다음 값이 이전 값과 **반대 방향** | 위아래로 빠르게 진동하는 패턴 |

---

## 🧪 연습 문제 풀이 요약

### Q1. 타임 더미 회귀 계수 해석

```
Hardcover ≈ 3.33 × Time + 150.5

→ 6일 동안 예상 변화량 = 3.33 × 6 ≈ 20권
```

### Q2. AR 계열 의존성 구분

```
Series 1 → target = +0.95 × lag_1 + error  (양의 계열 의존성 — 완만한 곡선)
Series 2 → target = −0.95 × lag_1 + error  (음의 계열 의존성 — 빠른 진동)
```

### Q3. Store Sales — 타임 더미 모델

```python
df['time'] = np.arange(len(df))
X = df[['time']]
y = df['sales']

model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)
```

### Q4. Store Sales — 래그 피처 모델

```python
df['lag_1'] = df['sales'].shift(1)

X = df[['lag_1']].dropna()
y = df['sales']
y, X = y.align(X, join='inner')

model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)
```

---

## 📊 주요 시각화

| 시각화 | 목적 |
|--------|------|
| **시간 플롯 (Time Plot)** | 타임 더미 피처의 추세선 적합 결과 확인 |
| **래그 플롯 (Lag Plot)** | 현재값 vs 이전값 산점도 — 계열 의존성 강도 파악 |
| **AR 시계열 비교** | 양·음 weight에 따른 시각적 패턴 차이 확인 |
| **예측값 오버레이** | 실제 시계열 위에 예측선을 겹쳐 적합도 시각화 |

---

## 🛠️ 기술 스택

```
Python
├── pandas                   # 시계열 인덱싱, shift(), align(), PeriodIndex
├── numpy                    # 타임 더미 생성 (np.arange)
├── matplotlib, seaborn      # 시간 플롯, 래그 플롯 시각화
├── scikit-learn
│   └── LinearRegression     # 시계열 회귀 모델
└── kagglehub                # Kaggle 데이터셋 자동 다운로드
    └── learntools           # 연습 문제 자동 채점 시스템
```

환경: **Google Colab / Kaggle Notebook**

---

## 🚀 실행 방법

1. Google Colab에서 노트북을 열고 **Kaggle API 인증**(`kagglehub.login()`)을 완료합니다.
2. 두 번째 셀을 실행하면 대회 데이터와 강의 데이터가 자동 다운로드됩니다.
3. 세 번째 셀에서 심볼릭 링크를 생성합니다.
4. 셀을 위에서부터 순서대로 실행합니다.

> ⚠️ Kaggle 계정 및 대회 참여(Rules Acceptance)가 사전에 필요합니다.

---

## 📎 참고

- 강의: [Kaggle Learn — Time Series](https://www.kaggle.com/learn/time-series)
- 대회: [Store Sales - Time Series Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting)
- 데이터: [ts-course-data](https://www.kaggle.com/datasets/ryanholbrook/ts-course-data)
