# 💳 신용카드 이상거래 탐지 — Feature Engineering

> 신용카드 거래 데이터에서 사기 거래(Fraud)를 탐지하기 위한 피처 엔지니어링 파이프라인입니다.
> 카드 사용자별 소비 패턴을 분석하여 이상 거래를 수치화하는 파생 변수를 생성합니다.

---

## 📁 파일 구조

```
DataCleansing/
└── fe.ipynb    # 전체 Feature Engineering 코드 (Google Colab)
```

---

## 📊 데이터셋

| 항목 | 내용 |
|------|------|
| 파일명 | `fraud.csv` |
| 데이터 규모 | 약 49만 건 (491,134행 × 22열) |
| 타겟 변수 | `is_fraud` (0: 정상, 1: 사기) — 클래스 불균형 심각 (사기 약 0.25%) |
| 기간 | 2019년 1월 ~ 2020년 12월 |

### 주요 원본 변수

| 변수명 | 설명 |
|--------|------|
| `trans_date_trans_time` | 거래 일시 |
| `cc_num` | 신용카드 번호 (카드 식별자) |
| `merchant` | 구매 상점 이름 |
| `category` | 상점 분류 (14종) |
| `amt` | 결제 금액 |
| `gender` | 성별 |
| `lat`, `long` | 고객 위치 (위경도) |
| `merch_lat`, `merch_long` | 상점 위치 (위경도) |
| `city_pop` | 도시 인구 수 |
| `dob` | 고객 생년월일 |
| `is_fraud` | 사기 여부 (타겟 변수) |

---

## 🗑️ 불필요한 컬럼 제거

사기 거래 탐지와 직접적인 관련이 낮거나 고유값이 너무 많아 모델에 노이즈가 될 수 있는 컬럼들을 제거합니다.

| 제거된 컬럼 | 제거 이유 |
|-------------|-----------|
| `merchant` | 고유값 693개 — 너무 많아 직접 사용 불가 |
| `first`, `last` | 개인 식별 정보, 사기 탐지에 불필요 |
| `street`, `city`, `state`, `zip` | 위경도 정보로 대체 가능 |
| `job` | 고유값 110개로 카드 수(124개)와 유사 — 의미 없음 |
| `trans_num` | 거래 ID, 분석에 불필요 |
| `unix_time` | `trans_date_trans_time`과 중복 |

---

## ⚙️ Feature Engineering

### 1. 결제 금액 Z-score (`amt_z`)

카드별 평균/표준편차를 기준으로 결제 금액의 표준 점수를 계산합니다.

```python
amt_info = cc_df.groupby('cc_num')['amt'].agg(['mean', 'std']).reset_index()
cc_df = cc_df.merge(amt_info, on='cc_num', how='left')
cc_df['amt_z'] = (cc_df['amt'] - cc_df['mean']) / cc_df['std']
```

**효과:** 사용자별 평소 소비 패턴을 벗어난 거래 금액을 탐지합니다. 중간 저장 파일: `amt_info.pkl`

---

### 2. 카테고리별 결제 금액 Z-score (`cat_amt_z`)

카드번호 + 상점 분류 조합을 기준으로 결제 금액의 표준 점수를 계산합니다.

```python
cat_info = cc_df.groupby(['cc_num', 'category'])['amt'].agg(['mean', 'std']).reset_index()
cc_df = cc_df.merge(cat_info, on=['cc_num', 'category'], how='left')
cc_df['cat_amt_z'] = (cc_df['amt'] - cc_df['mean']) / cc_df['std']
```

**효과:** 개인별, 상점 분류별 이상 거래 금액을 탐지합니다. 상황을 고려한 맥락적 이상 탐지가 가능합니다. 중간 저장 파일: `cat_info.pkl`

---

### 3. 시간대별 결제 비율 (`hour_perc`)

거래 시각을 4개 시간대로 구분하고, 카드별 시간대별 결제 비율을 계산합니다.

| 시간대 | 범위 |
|--------|------|
| morning | 06:00 ~ 11:59 |
| afternoon | 12:00 ~ 17:59 |
| night | 18:00 ~ 22:59 |
| evening | 23:00 ~ 05:59 |

```python
cc_df['hour'] = pd.to_datetime(cc_df['trans_date_trans_time']).dt.hour
cc_df['hour_cat'] = cc_df['hour'].apply(hour_func)

hour_cnt['hour_perc'] = hour_cnt['hour_cnt'] / hour_cnt['total_cnt']
cc_df = cc_df.merge(hour_cnt[['cc_num','hour_cat','hour_perc']], on=['cc_num', 'hour_cat'], how='left')
```

**효과:** 카드 사용자의 평소 시간대별 소비 패턴에서 벗어난 거래를 감지합니다. 중간 저장 파일: `hour_cnt.pkl`

---

### 4. 고객-상점 간 거리 Z-score (`dist_z`)

`geopy` 라이브러리로 고객 위치와 상점 위치 간의 실제 거리(km)를 계산하고, 카드별 평균 거래 거리 기준으로 표준 점수를 산출합니다.

```python
from geopy.distance import distance

cc_df['distance'] = cc_df.apply(
    lambda x: distance((x['lat'], x['long']), (x['merch_lat'], x['merch_long'])).km,
    axis=1
)

dist_info = cc_df.groupby('cc_num')['distance'].agg(['mean', 'std']).reset_index()
cc_df = cc_df.merge(dist_info, on='cc_num', how='left')
cc_df['dist_z'] = (cc_df['distance'] - cc_df['mean']) / cc_df['std']
```

**효과:** 카드 사용자의 평소 거래 활동 반경을 벗어난 비정상 거래를 탐지합니다. 중간 저장 파일: `dist_info.pkl`

---

### 5. 생년 추출 (`dob`)

생년월일에서 연도만 추출하여 출생연도 변수로 변환합니다.

```python
cc_df['dob'] = pd.to_datetime(cc_df['dob']).dt.year
```

---

### 6. 범주형 변수 One-Hot Encoding

`category` (14종)와 `gender` (2종) 컬럼에 원-핫 인코딩을 적용합니다. `drop_first=True`로 다중공선성을 방지합니다.

```python
cc_df = pd.get_dummies(cc_df, drop_first=True)
```

---

## 📋 최종 피처 목록

| 피처 | 설명 |
|------|------|
| `amt` | 결제 금액 |
| `city_pop` | 도시 인구 |
| `dob` | 출생 연도 |
| `amt_z` | 카드별 결제 금액 Z-score |
| `cat_amt_z` | 카드 + 카테고리별 결제 금액 Z-score |
| `hour_perc` | 카드별 해당 시간대 결제 비율 |
| `distance` | 고객-상점 간 거리 (km) |
| `dist_z` | 카드별 거리 Z-score |
| `category_*` | 상점 분류 더미 변수 (13개) |
| `gender_M` | 성별 더미 변수 |
| `is_fraud` | 타겟 변수 (사기 여부) |

> 최종 데이터셋: **491,134행 × 24열**

---

## 🛠️ 기술 스택

```
Python
├── pandas      # 데이터 로드 및 조작
├── numpy       # 수치 연산
├── seaborn     # 데이터 시각화
└── geopy       # 위경도 기반 실거리 계산
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `fe.ipynb`를 엽니다.
2. `fraud.csv` 파일을 업로드합니다.
3. 셀을 위에서부터 순서대로 실행합니다.
4. 중간 결과물(`amt_info.pkl`, `cat_info.pkl`, `hour_cnt.pkl`, `dist_info.pkl`)이 자동 저장됩니다.

---

## 📎 참고

- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/DataCleansing/fe.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/DataCleansing/fe.ipynb)
