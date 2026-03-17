# 🧪 가설 검정 실습 — t검정 · ANOVA · 카이제곱 검정

> 통신사 고객 이탈(Churn) 데이터를 활용하여  
> **이표본 t검정 → ANOVA 분산분석 → 카이제곱 독립성 검정**의  
> 3가지 핵심 가설 검정을 단계적으로 실습하는 노트북입니다.  
> 각 검정에서 **정규성 가정 확인 → 모수/비모수 검정 선택**의 올바른 흐름을 따릅니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/stat2.ipynb)

---

## 📁 파일 구조

```
etc/
└── stat2.ipynb    # 가설 검정 실습 노트북 (Google Colab)

데이터 (별도 업로드 필요)
└── WA_Fn-UseC_-Telco-Customer-Churn.csv    # IBM 통신사 고객 이탈 데이터셋
```

> 데이터 출처: [IBM Sample Data — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🎯 분석 목표

| 검정 | 질문 | 변수 |
|------|------|------|
| 이표본 t검정 | 이탈 여부에 따라 월 청구금액 평균이 다른가? | `Churn` × `MonthlyCharges` |
| ANOVA | 계약 유형·인터넷 서비스 등 그룹별 월 청구금액 평균이 다른가? | `MultipleLines` / `InternetService` / `Contract` × `MonthlyCharges` |
| 카이제곱 | 파트너 유무와 고객 이탈은 관련이 있는가? | `Partner` × `Churn` |

---

## 🔄 전체 구성

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
          │
          ▼
① 이표본 t검정
   ├── 정규성 확인 (K-S 검정)
   └── 비모수 대안 → Mann–Whitney U Test
          │
          ▼
② ANOVA 분산분석
   ├── 3개 범주형 변수의 그룹별 정규성 확인 (K-S 검정)
   └── 비모수 대안 → Kruskal-Wallis Test
          │
          ▼
③ 카이제곱 독립성 검정
   ├── 분할표 (Contingency Table) 생성
   └── Chi2 검정 → 기대 빈도 확인 및 해석
```

---

## 🔬 STEP 1 — 이표본 t검정

### 가설 설정

| 가설 | 내용 |
|------|------|
| **귀무가설 (H₀)** | Churn 여부에 따른 MonthlyCharges 평균은 차이가 없다 (μ이탈 = μ유지) |
| **대립가설 (H₁)** | Churn 여부에 따른 MonthlyCharges 평균은 차이가 있다 (μ이탈 ≠ μ유지) |

### 사전 가정 확인 — KS 정규성 검정

t검정 수행 전, 두 그룹이 정규분포를 따르는지 **Kolmogorov–Smirnov 검정**으로 확인합니다.

```python
# Z-score 표준화 후 정규분포와 비교
churn_yes_z = (churn_yes - np.mean(churn_yes)) / np.std(churn_yes, ddof=1)
churn_no_z  = (churn_no  - np.mean(churn_no))  / np.std(churn_no,  ddof=1)

ks_yes = stats.kstest(churn_yes_z, 'norm')
ks_no  = stats.kstest(churn_no_z,  'norm')
```

### 비모수 대안 — Mann–Whitney U Test

정규성 가정이 충족되지 않을 경우 **평균 대신 중앙값**을 비교하는 비모수 검정을 수행합니다.

```python
u_stat, p_value = stats.mannwhitneyu(churn_yes, churn_no, alternative='two-sided')
```

| p-value | 해석 |
|---------|------|
| < 0.05 | 두 그룹 간 MonthlyCharges 중앙값에 **유의한 차이 있음** → H₀ 기각 |
| ≥ 0.05 | 두 그룹 간 차이가 통계적으로 유의하지 않음 → H₀ 채택 |

---

## 📊 STEP 2 — ANOVA 분산분석

### 가설 설정

| 가설 | 내용 |
|------|------|
| **귀무가설 (H₀)** | 그룹별 MonthlyCharges 평균은 모두 같다 |
| **대립가설 (H₁)** | 적어도 한 그룹의 평균이 다르다 |

### 분석 대상 변수

```python
cat_vars = ['MultipleLines', 'InternetService', 'Contract']
```

| 변수 | 설명 |
|------|------|
| `MultipleLines` | 다중 회선 가입 여부 |
| `InternetService` | 인터넷 서비스 종류 (DSL / Fiber optic / No) |
| `Contract` | 계약 유형 (Month-to-month / One year / Two year) |

### 사전 가정 확인 — 그룹별 KS 정규성 검정

각 범주형 변수의 그룹별로 `MonthlyCharges`가 정규분포를 따르는지 확인합니다.

```python
for var in cat_vars:
    for name, group in data.groupby(var):
        charges_z = (group['MonthlyCharges'] - np.mean(group['MonthlyCharges'])) \
                    / np.std(group['MonthlyCharges'], ddof=1)
        ks_stat, p_value = stats.kstest(charges_z, 'norm')
```

### 비모수 대안 — Kruskal-Wallis Test

정규성 가정 불충족 시 3개 이상 그룹의 중앙값을 비교하는 **비모수 ANOVA**를 수행합니다.

```python
for var in cat_vars:
    groups = [group['MonthlyCharges'].values for name, group in data.groupby(var)]
    h_stat, p_value = stats.kruskal(*groups)
```

| p-value | 해석 |
|---------|------|
| < 0.05 | 그룹별 MonthlyCharges 중앙값에 **유의한 차이 있음** → H₀ 기각 |
| ≥ 0.05 | 차이가 통계적으로 유의하지 않음 → H₀ 채택 |

---

## 🔲 STEP 3 — 카이제곱 독립성 검정

### 가설 설정

| 가설 | 내용 |
|------|------|
| **귀무가설 (H₀)** | Partner 여부와 Churn은 서로 **독립**이다 (관련 없음) |
| **대립가설 (H₁)** | Partner 여부와 Churn은 서로 **독립이 아니다** (관련 있음) |

### 분할표 및 카이제곱 검정

```python
# 분할표 생성
contingency_table = pd.crosstab(data['Partner'], data['Churn'])

# 카이제곱 검정
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
```

### 출력 결과 구성

| 항목 | 설명 |
|------|------|
| **Chi2 통계량** | 관측 빈도와 기대 빈도의 차이 크기 |
| **p-value** | 귀무가설이 참일 때 이 통계량이 나올 확률 |
| **자유도 (dof)** | (행 범주 수 - 1) × (열 범주 수 - 1) |
| **기대 빈도표** | 독립 가정 하에서 예상되는 각 셀의 빈도 |

| p-value | 해석 |
|---------|------|
| < 0.05 | Partner 여부는 Churn과 **관련 있음** → H₀ 기각 |
| ≥ 0.05 | Partner 여부는 Churn과 관련 없다고 볼 수 있음 → H₀ 채택 |

---

## 🗺️ 검정 선택 흐름도

```
분석 목적
    │
    ├── 두 그룹의 평균 비교
    │       │
    │       ├── 정규성 O  →  이표본 t검정 (Independent t-test)
    │       └── 정규성 X  →  Mann–Whitney U Test (비모수)
    │
    ├── 세 그룹 이상의 평균 비교
    │       │
    │       ├── 정규성 O  →  ANOVA (일원분산분석)
    │       └── 정규성 X  →  Kruskal-Wallis Test (비모수)
    │
    └── 두 범주형 변수의 관련성 검정
            └── 카이제곱 독립성 검정 (Chi-square Test)
```

---

## 🔑 핵심 판단 기준 요약

| 항목 | 기준 | 의미 |
|------|------|------|
| 유의수준 (α) | **0.05** | 5% 오류를 허용하는 판단 기준 |
| 정규성 검정 | KS p-value < 0.05 | 정규분포를 따르지 않음 → 비모수 검정 전환 |
| 가설 기각 | 검정 p-value < α | 귀무가설 기각, 대립가설 채택 |
| 가설 채택 | 검정 p-value ≥ α | 귀무가설 기각 실패 (차이 없음) |

---

## 🛠️ 기술 스택

```
Python
├── pandas                           # 데이터 로드, crosstab, groupby
├── numpy                            # z-score 표준화
├── matplotlib, seaborn              # (시각화 라이브러리 로드)
└── scipy.stats
    ├── norm.ppf                     # 정규분포 참조
    ├── kstest                       # Kolmogorov–Smirnov 정규성 검정
    ├── mannwhitneyu                 # Mann–Whitney U Test (비모수 이표본)
    ├── kruskal                      # Kruskal-Wallis Test (비모수 다집단)
    └── chi2_contingency             # 카이제곱 독립성 검정
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `stat2.ipynb`를 열거나 위 배지를 클릭합니다.
2. [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)에서 `WA_Fn-UseC_-Telco-Customer-Churn.csv`를 다운로드하여 업로드합니다.
3. 셀을 위에서부터 순서대로 실행합니다.

---

## 📎 참고

- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/stat2.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/etc/stat2.ipynb)
- 데이터: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
