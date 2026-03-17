# 🚗 영국 중고 자동차 가격 데이터 다루기 — Data Cleansing & Transformation

> 영국 중고차 시장 데이터(`cars.csv`, `brand.csv`)를 대상으로 **데이터 병합 → 자료형 변환 → 결측치 처리 → 이상치 제거 → 인코딩 → 스케일링 → PCA 차원 축소**까지의 전체 전처리 파이프라인을 실습하는 프로젝트입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/DataCleansing/datatrans.ipynb)

---

## 📁 파일 구조

```
DataCleansing/
└── datatrans.ipynb    # 전체 파이프라인 노트북 (Google Colab)

데이터 (별도 준비 필요)
├── cars.csv           # 중고차 매물 원본 데이터
└── brand.csv          # 브랜드-국가 매핑 데이터
```

---

## 📊 데이터셋

두 개의 CSV 파일을 병합하여 사용합니다.

### cars.csv 주요 컬럼

| 컬럼명 | 설명 |
|--------|------|
| `title` | 차량 제목 (브랜드 포함) |
| `Price` | 판매 가격 (£) |
| `Mileage(miles)` | 주행 거리 (마일) |
| `Registration_Year` | 등록 연도 |
| `Previous Owners` | 이전 소유자 수 |
| `Engine` | 엔진 배기량 (예: `2.0L`) |
| `Fuel type` | 연료 유형 |
| `Body type` | 차체 유형 |
| `Gearbox` | 변속기 유형 |
| `Doors` | 문 개수 |
| `Seats` | 좌석 수 |
| `Emission Class` | 배출가스 등급 (예: `Euro 6`) |
| `Service history` | 서비스 이력 |

### brand.csv

| 컬럼명 | 설명 |
|--------|------|
| `title` | 브랜드명 |
| `country` | 브랜드 원산지 국가 |

---

## 🔄 전체 파이프라인

```
cars.csv + brand.csv
        │
        ▼
① 데이터 병합 (Left Merge)
        │
        ▼
② 자료형 변환
   ├── Engine: "2.0L" → 2.0 (float)
   └── Emission Class: "Euro 6" → 6 (int)
        │
        ▼
③ 결측치 처리
   ├── Service history → 'Unknown' 대체
   ├── 결측치 4개 이상 행 제거
   └── 나머지 수치 컬럼 → 중앙값 대체
        │
        ▼
④ 이상치 제거
   ├── Mileage(miles) ≤ 1000 제거
   ├── Registration_Year ≥ 2025 제거
   └── 브랜드별 Price 분포 분석
        │
        ▼
⑤ 인코딩 & 스케일링
   ├── One-Hot Encoding (범주형 전체)
   └── Robust Scaler
        │
        ▼
⑥ PCA 차원 축소 (7개 주성분)
```

---

## 🔗 STEP 1 — 데이터 병합

`title` 컬럼의 첫 번째 단어로 `brand`를 추출하고, `brand_df`를 대문자로 표준화한 뒤 Left Merge로 `country` 컬럼을 추가합니다.

```python
# title에서 브랜드명 추출
car_df['brand'] = car_df['title'].str.split().str[0]

# brand_df 대문자 표준화
brand_df['title'] = brand_df['title'].str.upper()

# Left Merge로 country 컬럼 추가
car_df = pd.merge(car_df, brand_df, how='left',
                  left_on='brand', right_on='title')

# 병합 후 중복 컬럼 정리
car_df = car_df.drop(columns=['title_y'])
car_df = car_df.rename(columns={'title_x': 'title'})
```

---

## 🔢 STEP 2 — 자료형 변환

문자열로 저장된 수치 컬럼에서 단위 문자를 제거하고 숫자형으로 변환합니다.

```python
# "2.0L" → "2.0" → float
car_df['Engine'] = car_df['Engine'].str.replace('L', '', regex=False)

# "Euro 6" → "6" → int
car_df['Emission Class'] = car_df['Emission Class'].str.split().str[-1]

# 숫자형으로 변환 (변환 불가 값은 NaN 처리)
car_df['Engine']         = pd.to_numeric(car_df['Engine'],         errors='coerce')
car_df['Emission Class'] = pd.to_numeric(car_df['Emission Class'], errors='coerce')
```

---

## 🔧 STEP 3 — 결측치 처리

### Service history — 'Unknown' 대체

서비스 이력이 없는 경우도 의미 있는 정보이므로 제거 대신 `'Unknown'`으로 대체합니다.

```python
# 그룹별 가격 평균 확인 후 처리 전략 결정
car_df.groupby('Service history')['Price'].mean()

car_df['Service history'] = car_df['Service history'].fillna('Unknown')
```

### 결측치 과다 행 제거

행별 결측치 수를 계산하여 결측치가 4개 이상인 행을 제거합니다.

```python
car_df['na_values'] = car_df.isna().sum(axis=1)
car_df = car_df[car_df['na_values'] < 4]
car_df = car_df.drop(columns=['na_values'])
```

### 나머지 수치 컬럼 — 중앙값 대체

히스토그램으로 분포를 확인한 후, 왜도가 있는 컬럼은 평균 대신 **중앙값**으로 결측치를 채웁니다.

```python
columns_to_fill = ['Previous Owners', 'Engine', 'Doors', 'Seats', 'Emission Class']

for col in columns_to_fill:
    car_df[col] = car_df[col].fillna(car_df[col].median())
```

| 컬럼 | 대체 기준 | 이유 |
|------|-----------|------|
| `Previous Owners` | 중앙값 | 왜도 존재 |
| `Engine` | 중앙값 | 평균과 중앙값 차이 확인 후 선택 |
| `Doors` | 중앙값 | 이산형 분포 |
| `Seats` | 중앙값 | 이산형 분포 |
| `Emission Class` | 중앙값 | 왜도 존재 |

---

## 📐 STEP 4 — 이상치 제거 및 분석

### Mileage(miles)

1,000마일 이하의 비현실적으로 낮은 주행 거리 데이터를 제거합니다.

```python
car_df = car_df[car_df['Mileage(miles)'] > 1000]
```

### Registration_Year

2025년 이상 미래 연도를 오류값으로 판단하여 제거합니다.

```python
car_df = car_df[car_df['Registration_Year'] < 2025]
```

### 브랜드별 Price 분포 분석

브랜드별 평균·표준편차를 계산하고, 표준편차가 NaN인 브랜드(데이터 1개)를 파악합니다.

```python
car_df.groupby('brand')['Price'].agg(['mean', 'std'])

# 데이터 1개 브랜드 확인
car_df[car_df['brand'].isin(
    ['DAEWOO', 'DODGE', 'ISUZU', 'LAGONDA', 'MARCOS']
)]['brand'].value_counts()
```

피벗 테이블로 브랜드 × 연료 유형별 평균 가격도 확인합니다.

```python
car_df.pivot_table(index='brand', columns='Fuel type',
                   values='Price', aggfunc='mean')
```

### 가격 로그 변환 시각화

고가 차량으로 인한 Price의 오른쪽 꼬리 분포를 로그 변환으로 확인합니다.

```python
# 변환 전
sns.scatterplot(x=car_df['Registration_Year'], y=car_df['Price'])

# 로그 변환 후 — 정규분포에 가까운 형태로 변환
sns.scatterplot(x=car_df['Registration_Year'], y=np.log(car_df['Price']))
```

---

## 🏷️ STEP 5 — 인코딩 & 스케일링

### One-Hot Encoding

범주형 컬럼 전체에 원-핫 인코딩을 적용합니다. `drop_first=True`로 다중공선성을 방지합니다.

```python
# 고유값 수 확인 후 적용 대상 결정
car_df[['Fuel type', 'Body type', 'Gearbox',
        'Emission Class', 'Service history',
        'brand', 'country']].nunique()

# title 컬럼 제거 (고유값 과다)
car_df.drop('title', axis=1, inplace=True)

# 원-핫 인코딩 적용
car_df = pd.get_dummies(car_df, drop_first=True)
```

### Robust Scaler

이상치에 강건한 **RobustScaler**를 사용하여 중앙값과 IQR 기준으로 스케일링합니다.

```python
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
car_df = pd.DataFrame(rs.fit_transform(car_df), columns=car_df.columns)
```

> MinMaxScaler·StandardScaler 대신 RobustScaler를 선택한 이유: 가격 데이터 특성상 잔존하는 이상치에 덜 민감하게 스케일링할 수 있습니다.

---

## 🔬 STEP 6 — PCA 차원 축소

주성분 개수를 2~10개로 변화시키며 설명 분산 비율(Explained Variance Ratio)을 비교하고 최적 주성분 수를 결정합니다.

```python
from sklearn.decomposition import PCA

# 주성분 수별 설명 분산 비율 탐색
for i in range(2, 11):
    pca = PCA(i)
    pca.fit(car_df)
    print(i, round(pca.explained_variance_ratio_.sum(), 2))
```

| 주성분 수 | 설명 분산 비율 (누적) |
|-----------|----------------------|
| 2 | 확인 |
| 3 | 확인 |
| ... | ... |
| **7** | **최적으로 선택** |
| 10 | 확인 |

**최종 선택: 7개 주성분 (PC1 ~ PC7)**

```python
pca = PCA(7)
pca_result = pd.DataFrame(
    pca.fit_transform(car_df),
    columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7']
)
```

---

## 🎁 보너스 분석 (`bonus_df`)

원본 데이터 복사본으로 추가 분석을 수행합니다.

```python
bonus_df = car_df.copy()

# 국가별 브랜드 수
bonus_df.groupby('country')['brand'].nunique()

# 수치형 컬럼 간 상관 계수 행렬
bonus_df.select_dtypes(include='number').corr()
```

---

## 📊 주요 시각화

| 시각화 | 목적 |
|--------|------|
| `Previous Owners` 히스토그램 | 분포 확인 → 중앙값 대체 근거 |
| `Engine` 히스토그램 | 평균 vs 중앙값 비교 |
| `Doors` / `Seats` 히스토그램 | 이산형 분포 확인 |
| `Emission Class` 히스토그램 | 왜도 확인 |
| `Previous Owners` vs `Price` 산점도 | 소유 이력-가격 관계 |
| `Registration_Year` vs `Price` 산점도 | 연식-가격 관계 (로그 변환 전·후 비교) |
| 브랜드 × 연료 유형 피벗 테이블 | 그룹별 평균 가격 분석 |

---

## 🛠️ 기술 스택

```
Python
├── pandas                       # 데이터 로드, 병합, 전처리
├── numpy                        # 수치 연산, 로그 변환
├── matplotlib / seaborn         # 히스토그램, 산점도 시각화
└── scikit-learn
    ├── RobustScaler             # 이상치 강건 스케일링
    └── PCA                      # 주성분 분석 (차원 축소)
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `datatrans.ipynb`를 열거나 위 배지를 클릭합니다.
2. `cars.csv`와 `brand.csv` 두 파일을 업로드합니다.
3. 셀을 위에서부터 순서대로 실행합니다.

---

## 📎 참고

- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/DataCleansing/datatrans.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/DataCleansing/datatrans.ipynb)
