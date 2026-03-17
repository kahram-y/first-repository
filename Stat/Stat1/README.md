# 📊 기초 통계 실습 — 통계량 · 분포 시각화 · 신뢰구간

> 통신사 고객 이탈(Churn) 데이터의 **월 청구금액(`MonthlyCharges`)** 을 활용하여  
> **기초 통계량 계산 → 분포 시각화 → 신뢰구간 추정**까지의  
> 기초 통계 핵심 개념을 실습하는 노트북입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/stat1.ipynb)

---

## 📁 파일 구조

```
etc/
└── stat1.ipynb    # 기초 통계 실습 노트북 (Google Colab)

데이터 (별도 업로드 필요)
└── WA_Fn-UseC_-Telco-Customer-Churn.csv    # IBM 통신사 고객 이탈 데이터셋
```

> 데이터 출처: [IBM Sample Data — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🎯 분석 대상

| 항목 | 내용 |
|------|------|
| 데이터셋 | IBM Telco Customer Churn |
| 분석 변수 | `MonthlyCharges` — 고객 월 청구금액 ($) |
| 핵심 목표 | 평균·중앙값·최빈값 비교, 분포 형태 파악, 표본 크기별 신뢰구간 변화 이해 |

---

## 🔄 전체 구성

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
          │
          ▼
① 기초 통계량 계산 및 시각화
   ├── 평균 (Mean)
   ├── 중앙값 (Median)
   └── 최빈값 (Mode) — 구간 이산화 후 중간값 추출
          │
          ▼
② 분포 시각화 (3종 비교)
   ├── 상자수염그림 (Box Plot)
   ├── 바이올린 플롯 (Violin Plot)
   └── 스웜 플롯 (Swarm Plot)
          │
          ▼
③ 신뢰구간 추정
   └── 표본 크기 n = 100 / 300 / 1000 별 95% CI 비교
```

---

## 📐 STEP 1 — 기초 통계량 계산 및 시각화

`MonthlyCharges`의 평균·중앙값·최빈값을 계산하고 히스토그램 위에 세 값을 함께 표시합니다.

```python
mean_val   = churn['MonthlyCharges'].mean()
median_val = churn['MonthlyCharges'].median()

# 연속형 데이터의 최빈값: 30개 구간으로 이산화 후 최빈 구간의 중간값 추출
binned_total   = pd.cut(churn['MonthlyCharges'], bins=30, right=False)
modal_interval = binned_total.mode()[0]
mode_val       = modal_interval.mid
```

### 시각화

```python
sns.histplot(churn['MonthlyCharges'], bins=30)
plt.axvline(mean_val,   color='red',   linestyle='--', label=f'Mean:   ${mean_val:.2f}')
plt.axvline(median_val, color='green', linestyle='--', label=f'Median: ${median_val:.2f}')
plt.axvline(mode_val,   color='blue',  linestyle='--', label=f'Mode:   ${mode_val:.2f}')
```

| 통계량 | 특징 | 분포 해석 |
|--------|------|-----------|
| **평균 (Mean)** | 모든 값을 합산하여 나눔 | 이상치에 민감 |
| **중앙값 (Median)** | 정렬 후 가운데 값 | 이상치에 강건 |
| **최빈값 (Mode)** | 가장 자주 등장하는 구간 | 분포의 봉우리 위치 |

> 세 값의 위치 차이로 `MonthlyCharges` 분포의 **비대칭성(Skewness)** 을 직관적으로 확인합니다.

---

## 📊 STEP 2 — 분포 시각화 3종 비교

200개 표본을 추출하여 동일 데이터를 3가지 그래프로 비교합니다.

```python
churn_sample = churn['MonthlyCharges'].sample(n=200, random_state=42)

sns.boxplot(churn_sample)     # 상자수염그림
sns.violinplot(churn_sample)  # 바이올린 플롯
sns.swarmplot(churn_sample)   # 스웜 플롯
```

### 3종 시각화 비교

| 그래프 | 보여주는 것 | 특징 |
|--------|-------------|------|
| **Box Plot** | 중앙값, IQR, 이상치 | 요약 통계 한눈에 파악 |
| **Violin Plot** | 분포의 형태(밀도) + Box Plot | 데이터 분포 모양까지 확인 |
| **Swarm Plot** | 개별 데이터 포인트 위치 | 데이터 군집·겹침 시각화 |

---

## 🎯 STEP 3 — 표본 크기별 95% 신뢰구간 추정

모집단(전체 `MonthlyCharges`)의 표준편차를 알고 있다는 가정 하에, 표본 크기를 달리하며 **모평균의 95% 신뢰구간**을 계산합니다.

```python
population_mean = np.mean(population)
population_std  = np.std(population, ddof=0)

z_crit = norm.ppf((1 + 0.95) / 2)   # z 임계값 ≈ 1.96

for n in [100, 300, 1000]:
    sample    = np.random.choice(population, size=n)
    sample_mean = np.mean(sample)
    se        = population_std / np.sqrt(n)   # 표준오차
    margin    = z_crit * se                   # 오차 범위

    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin
```

### 표준오차 및 신뢰구간 공식

```
표준오차 (SE) = σ / √n

신뢰구간 = x̄ ± z_(α/2) × SE
         = 표본평균 ± 1.96 × (모표준편차 / √n)
```

### 표본 크기와 신뢰구간의 관계

| 표본 크기 (n) | 표준오차 변화 | 신뢰구간 너비 | 해석 |
|---------------|---------------|---------------|------|
| 100 | 크다 | **넓다** | 불확실성 높음 |
| 300 | 중간 | 중간 | - |
| 1000 | 작다 | **좁다** | 정밀도 높음 |

> 표본 크기가 커질수록 표준오차가 감소하여 신뢰구간이 좁아지고, 모평균 추정이 더 정밀해짐을 시각화(`plt.errorbar`)로 확인합니다.

---

## 🛠️ 기술 스택

```
Python
├── pandas          # 데이터 로드 및 기초 통계량 계산
├── numpy           # 표본 추출, 표준오차, 신뢰구간 계산
├── matplotlib      # 히스토그램, 신뢰구간 에러바 시각화
├── seaborn         # 분포 시각화 (boxplot / violinplot / swarmplot)
└── scipy.stats
    └── norm.ppf    # 정규분포 z 임계값 산출
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `stat1.ipynb`를 열거나 위 배지를 클릭합니다.
2. [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)에서 `WA_Fn-UseC_-Telco-Customer-Churn.csv`를 다운로드하여 업로드합니다.
3. 셀을 위에서부터 순서대로 실행합니다.

---

## 📎 참고

- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/stat1.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/etc/stat1.ipynb)
- 데이터: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
