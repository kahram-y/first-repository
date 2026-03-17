# 🚕 택시 요금 데이터 다루기 — Data Cleansing & Feature Engineering

> NYC 택시 운행 데이터(`trip.csv`)를 대상으로 **중복 제거 · 결측치 처리 · 이상치 탐지 및 제거 · 범주형 전처리 · 파생 변수 생성**까지의 전체 데이터 클렌징 파이프라인을 실습하는 프로젝트입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/DataCleansing/trip.ipynb)

---

## 📁 파일 구조

```
DataCleansing/
└── trip.ipynb     # 전체 데이터 클렌징 노트북 (Google Colab)

데이터 (별도 준비 필요)
└── trip.csv       # NYC 택시 운행 원본 데이터
```

---

## 📊 데이터셋

NYC 택시 운행 기록 데이터로, 주요 컬럼은 아래와 같습니다.

| 컬럼명 | 설명 |
|--------|------|
| `passenger_name` | 승객 이름 |
| `passenger_count` | 탑승 승객 수 |
| `tpep_pickup_datetime` | 승차 일시 |
| `tpep_dropoff_datetime` | 하차 일시 |
| `trip_distance` | 운행 거리 (miles) |
| `fare_amount` | 기본 요금 ($) |
| `tip_amount` | 팁 금액 ($) |
| `tolls_amount` | 통행료 ($) |
| `payment_method` | 결제 수단 |

---

## 🔄 전체 파이프라인

```
원본 데이터 (trip.csv)
        │
        ▼
① 데이터 탐색 (head / info / describe)
        │
        ▼
② 중복 데이터 제거
        │
        ▼
③ 결측치 처리
        │
        ▼
④ 이상치 탐지 및 제거
   ├── passenger_count
   ├── trip_distance  (IQR)
   ├── fare_amount    (캡핑)
   └── tip_amount     (IQR)
        │
        ▼
⑤ 범주형 데이터 전처리
   ├── payment_method 통합
   └── passenger_name 분리
        │
        ▼
⑥ 파생 변수 생성
   ├── travel_time / travel_time_seconds
   └── total_amount
```

---

## 🔍 STEP 1 — 데이터 탐색

```python
data.head()       # 상위 5행 미리보기
data.info()       # 컬럼명, 자료형, 결측치 수, 데이터 크기 확인
data.describe()   # 컬럼별 기초 통계량 (평균·표준편차·사분위수 등)
```

---

## 🗑️ STEP 2 — 중복 데이터 제거

`duplicated()`로 중복 행을 확인하고 `drop_duplicates()`로 제거합니다.
`Sarah Gross`, `Lisa Bullock` 등 특정 승객명으로 중복 여부를 교차 검증했습니다.

```python
data[data.duplicated()]       # 중복 행 확인
data = data.drop_duplicates() # 중복 제거
```

---

## 🔧 STEP 3 — 결측치 처리

```python
data.isna().sum()    # 컬럼별 결측치 개수
data.isna().mean()   # 컬럼별 결측치 비율
data = data.dropna() # 결측치 포함 행 전체 제거
```

---

## 📐 STEP 4 — 이상치 탐지 및 제거

### passenger_count

승객 수가 비현실적인 값(0명, 6명 초과)을 산점도로 시각화한 후 필터링합니다.

```python
# 6 초과 제거 (일반 택시 최대 탑승 인원 기준)
data = data[data['passenger_count'] <= 6]

# 0명 제거 (탑승자 없는 운행 기록)
data = data[data['passenger_count'] != 0]
```

---

### trip_distance — IQR 기반 이상치 제거

IQR(사분위 범위)를 활용해 운행 거리의 상·하한을 정의하고 이상치를 제거합니다.
히스토그램으로 제거 전·후 분포 변화를 비교합니다.

```python
Q1 = data['trip_distance'].quantile(0.25)
Q3 = data['trip_distance'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data = data[(data['trip_distance'] >= lower_bound) &
            (data['trip_distance'] <= upper_bound)]
```

> 제거 전: 분포의 꼬리가 길게 늘어진 형태 → 제거 후: 중앙에 데이터가 모인 정상 분포

---

### fare_amount — 조건 필터링 + 캡핑(Capping)

0 이하 요금을 제거하고, 150 초과 값은 150으로 캡핑합니다.
`trip_distance` vs `fare_amount` 산점도로 양의 상관관계를 확인했습니다.

```python
# 0 이하 제거
data = data[data['fare_amount'] > 0]

# 150 초과 값 캡핑 (이상치를 제거가 아닌 상한값으로 대체)
data['fare_amount'] = data['fare_amount'].apply(
    lambda x: 150 if x > 150 else x
)
```

---

### tip_amount — IQR 기반 이상치 제거

`trip_distance` vs `tip_amount` 산점도로 이상치를 시각적으로 먼저 확인한 후 IQR 기준으로 제거합니다.

```python
Q1 = data['tip_amount'].quantile(0.25)
Q3 = data['tip_amount'].quantile(0.75)
IQR = Q3 - Q1

data = data[(data['tip_amount'] >= Q1 - 1.5 * IQR) &
            (data['tip_amount'] <= Q3 + 1.5 * IQR)]
```

---

### tolls_amount — 시각화 확인

`tolls_amount` vs `trip_distance`, `tolls_amount` vs `fare_amount` 산점도로 통행료 패턴을 시각적으로 검토합니다.

---

## 🏷️ STEP 5 — 범주형 데이터 전처리

### payment_method 통합

`Debit Card`와 `Credit Card`를 `Card`로 통합하여 카테고리 수를 줄입니다.

```python
data['payment_method'].unique()        # 고유값 종류 확인
data['payment_method'].value_counts()  # 각 값의 빈도 확인

data['payment_method'] = data['payment_method'].replace(
    ['Debit Card', 'Credit Card'], 'Card'
)
```

### passenger_name 분리

승객 이름에서 성(last name)만 추출하여 `passenger_first_name` 컬럼으로 저장합니다.

```python
# 공백 기준으로 분리 후 두 번째 요소(성) 추출
data['passenger_first_name'] = data['passenger_name'].str.split().str[1]
```

---

## ⚙️ STEP 6 — 파생 변수 생성 (Feature Engineering)

### 날짜시간 자료형 변환

```python
data['tpep_pickup_datetime']  = pd.to_datetime(data['tpep_pickup_datetime'])
data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])
```

### travel_time (운행 시간)

하차 시각과 승차 시각의 차이를 분 단위와 초 단위로 계산합니다.

```python
# 분 단위
data['travel_time'] = (
    data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']
).dt.total_seconds() / 60

# 초 단위
data['travel_time_seconds'] = data['travel_time'] * 60
```

### total_amount (총 결제 금액)

기본 요금 + 팁 + 통행료를 합산합니다.

```python
data['total_amount'] = data['fare_amount'] + data['tip_amount'] + data['tolls_amount']
```

---

## 📊 주요 시각화

| 시각화 | 목적 |
|--------|------|
| `passenger_count` 산점도 | 이상치(0명, 6 초과) 확인 |
| `trip_distance` 히스토그램 (전·후) | IQR 이상치 제거 효과 비교 |
| `fare_amount` vs `trip_distance` 산점도 | 요금-거리 양의 상관관계 확인 |
| `tip_amount` vs `trip_distance` 산점도 | 팁 이상치 시각적 탐지 |
| `tolls_amount` vs `trip_distance` 산점도 | 통행료 패턴 확인 |
| `fare_amount` vs `travel_time` 산점도 | 요금-시간 관계 확인 |
| `trip_distance` vs `travel_time` 산점도 | 거리-시간 관계 확인 |

---

## 📋 최종 컬럼 목록

| 컬럼 | 설명 | 구분 |
|------|------|------|
| `passenger_name` | 승객 이름 | 원본 |
| `passenger_first_name` | 승객 성(last name) | 파생 |
| `passenger_count` | 탑승 승객 수 | 원본 |
| `tpep_pickup_datetime` | 승차 일시 (datetime) | 변환 |
| `tpep_dropoff_datetime` | 하차 일시 (datetime) | 변환 |
| `trip_distance` | 운행 거리 (miles) | 원본 |
| `fare_amount` | 기본 요금 ($, 캡핑 150) | 원본+캡핑 |
| `tip_amount` | 팁 금액 ($) | 원본 |
| `tolls_amount` | 통행료 ($) | 원본 |
| `payment_method` | 결제 수단 (Card 통합) | 변환 |
| `travel_time` | 운행 시간 (분) | 파생 |
| `travel_time_seconds` | 운행 시간 (초) | 파생 |
| `total_amount` | 총 결제 금액 ($) | 파생 |

---

## 🛠️ 기술 스택

```
Python
├── pandas      # 데이터 로드 및 전처리
├── numpy       # 수치 연산
├── matplotlib  # 시각화
└── seaborn     # 산점도·히스토그램 시각화
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `trip.ipynb`를 열거나 위 배지를 클릭합니다.
2. `trip.csv` 파일을 업로드합니다.
3. 셀을 위에서부터 순서대로 실행합니다.

---

## 📎 참고

- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/DataCleansing/trip.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/DataCleansing/trip.ipynb)
