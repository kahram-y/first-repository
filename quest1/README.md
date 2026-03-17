# 🛍️ 고객을 세그먼테이션하자 — RFM 분석 with BigQuery SQL

> 온라인 리테일 거래 데이터를 **Google BigQuery**에서 SQL로 전처리하고,  
> **RFM(Recency · Frequency · Monetary)** 지표와 추가 피처를 추출하여  
> 고객별 행동 패턴 데이터셋(`user_data`)을 구축하는 프로젝트입니다.

---

## 📁 파일 구조

```
quest1/
└── MainQuest1_kahramyoon.pdf    # 전체 프로젝트 결과 보고서 (윤가람)
```

---

## 📊 데이터셋

| 항목 | 내용 |
|------|------|
| 플랫폼 | Google BigQuery |
| 테이블 | `stone-nuance-228223.modulabs_project.data` |
| 원본 행 수 | **541,909행** |
| 전처리 후 행 수 | **401,604행** (결측치·중복·오류 제거 후) |

### 원본 컬럼

| 컬럼명 | 설명 |
|--------|------|
| `InvoiceNo` | 거래 번호 (`C`로 시작하면 취소 거래) |
| `StockCode` | 상품 코드 |
| `Description` | 상품 설명 |
| `Quantity` | 구매 수량 |
| `InvoiceDate` | 거래 일시 |
| `UnitPrice` | 단가 |
| `CustomerID` | 고객 ID |
| `Country` | 국가 |

---

## 🔄 전체 파이프라인

```
원본 데이터 (541,909행)
         │
         ▼
① 데이터 전처리 (결측치·중복·오류 제거)
         │
         ▼
② RFM 스코어 계산
   ├── Recency  → user_r 테이블
   ├── Frequency
   └── Monetary → user_rfm 테이블
         │
         ▼
③ 추가 피처 추출
   ├── 구매 제품 다양성
   ├── 평균 구매 주기
   └── 구매 취소 경향성 → user_data 테이블
         │
         ▼
④ 최종 고객 행동 프로파일 완성
```

---

## 🧹 STEP 1 — 데이터 전처리

### 11-4. 결측치 제거

컬럼별 누락 비율을 `UNION ALL`로 한눈에 계산합니다.

```sql
SELECT 'CustomerID' AS column_name,
  ROUND(SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2)
  AS missing_percentage
FROM `...data`
UNION ALL
SELECT 'Description' AS column_name, ...
```

| 컬럼 | 누락 비율 |
|------|-----------|
| `CustomerID` | **24.93%** |
| `Description` | 0.27% |

`CustomerID` 또는 `Description`이 NULL인 행을 `DELETE`로 제거합니다. → **135,080행 삭제**

---

### 11-5. 중복값 제거

8개 컬럼 전체를 기준으로 `GROUP BY + HAVING COUNT(*) > 1`로 중복 확인 후,
`CREATE OR REPLACE TABLE ... SELECT DISTINCT *`로 제거합니다.

> 중복 제거 후 행 수: **401,604행**

---

### 11-6. 오류값 처리

**InvoiceNo 오류 처리**

`C`로 시작하는 취소 거래 비율을 확인합니다 → **2.2%**

```sql
SELECT ROUND(
  SUM(CASE WHEN InvoiceNo LIKE 'C%' THEN 1 ELSE 0 END) / COUNT(*) * 100, 1
) AS canceled_percentage
FROM `...data`;
```

**StockCode 오류 처리**

숫자가 0~1개인 비정형 코드(`POST`, `M`, `BANK CHARGES` 등 서비스·관리용 코드)를 정규식으로 탐지하여 제거합니다.

```sql
-- 숫자 자릿수 계산
LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, r'[0-9]', '')) AS number_count

-- 비정형 코드 삭제 (전체의 0.48% 해당)
DELETE FROM `...data`
WHERE StockCode IN (... WHERE number_count = 0 OR number_count = 1);
```

> **1,915행 삭제**

**Description 오류 처리**

- 서비스 관련 행(`Next Image`, `High Resolution Image`) 삭제 → **3행 삭제**
- 대소문자 혼재 데이터를 `UPPER()` 함수로 대문자 표준화

**UnitPrice 오류 처리**

- 단가 통계: min `0.0` / max `649.5` / avg `2.91`
- 단가 0원 거래(33건)를 `UnitPrice <> 0` 조건으로 제거

---

## 📐 STEP 2 — RFM 스코어 계산

### Recency (최신성)

데이터 전체에서 가장 최근 구매일과 고객별 마지막 구매일 간의 일수 차이를 계산합니다.

```sql
-- user_r 테이블 생성
CREATE OR REPLACE TABLE ...user_r AS
SELECT
  CustomerID,
  EXTRACT(DAY FROM MAX(InvoiceDay) OVER () - InvoiceDay) AS recency
FROM (
  SELECT CustomerID, MAX(DATE(InvoiceDate)) AS InvoiceDay
  FROM ...data
  GROUP BY CustomerID
);
```

### Frequency (구매 빈도)

고객별 고유 `InvoiceNo` 수(거래 건수)와 총 구매 아이템 수를 계산합니다.

```sql
COUNT(DISTINCT InvoiceNo) AS purchase_cnt   -- 거래 건수
SUM(Quantity)             AS item_cnt       -- 총 구매 수량
```

### Monetary (구매 금액)

고객별 총 지출액과 거래당 평균 금액을 계산하고 `user_rfm` 테이블에 통합합니다.

```sql
-- user_rfm 테이블 생성
CREATE OR REPLACE TABLE ...user_rfm AS
SELECT
  rf.CustomerID,
  rf.purchase_cnt,
  rf.item_cnt,
  rf.recency,
  ut.user_total,
  ROUND(ut.user_total / rf.purchase_cnt, 1) AS user_average
FROM ...user_rf rf
LEFT JOIN (
  SELECT CustomerID, SUM(UnitPrice * Quantity) AS user_total
  FROM ...data GROUP BY CustomerID
) ut ON rf.CustomerID = ut.CustomerID;
```

### 중간 테이블 정리

| 테이블명 | 내용 |
|----------|------|
| `user_r` | CustomerID + recency |
| `user_rf` | user_r + purchase_cnt + item_cnt |
| `user_rfm` | user_rf + user_total + user_average |

---

## ➕ STEP 3 — 추가 피처 추출

### 1. 구매 제품 다양성 (`unique_products`)

고객별로 구매한 고유 상품 종류 수를 계산합니다.

```sql
COUNT(DISTINCT StockCode) AS unique_products
```

### 2. 평균 구매 주기 (`average_interval`)

`LAG()` 윈도우 함수로 직전 구매일과의 날짜 차이를 계산하고, 고객별 평균을 구합니다.
구매가 1회인 고객은 NULL → 0으로 처리합니다.

```sql
DATE_DIFF(
  InvoiceDate,
  LAG(InvoiceDate) OVER (PARTITION BY CustomerID ORDER BY InvoiceDate),
  DAY
) AS interval_

-- 고객별 평균 (NULL → 0 처리)
CASE WHEN ROUND(AVG(interval_), 2) IS NULL
     THEN 0
     ELSE ROUND(AVG(interval_), 2) END AS average_interval
```

### 3. 구매 취소 경향성 (`cancel_frequency`, `cancel_rate`)

`InvoiceNo`가 `C`로 시작하는 취소 거래를 집계하고 취소 비율을 산출합니다.

```sql
WITH TransactionInfo AS (
  SELECT
    CustomerID,
    COUNT(*) AS total_transactions,
    COUNTIF(STARTS_WITH(InvoiceNo, 'C')) AS cancel_frequency
  FROM ...data
  GROUP BY CustomerID
)
SELECT
  ...,
  ROUND(COALESCE(cancel_frequency, 0) / total_transactions, 2) AS cancel_rate
FROM ...user_data
LEFT JOIN TransactionInfo ON ...CustomerID;
```

---

## 📋 최종 피처 목록 (`user_data` 테이블)

| 피처 | 설명 | 분류 |
|------|------|------|
| `purchase_cnt` | 총 거래 건수 | Frequency |
| `item_cnt` | 총 구매 아이템 수 | Frequency |
| `recency` | 마지막 구매 후 경과 일수 | Recency |
| `user_total` | 총 지출액 | Monetary |
| `user_average` | 거래당 평균 금액 | Monetary |
| `unique_products` | 구매 제품 다양성 | 추가 피처 |
| `average_interval` | 평균 구매 주기 (일) | 추가 피처 |
| `total_transactions` | 전체 거래 수 | 추가 피처 |
| `cancel_frequency` | 취소 거래 횟수 | 추가 피처 |
| `cancel_rate` | 취소 비율 | 추가 피처 |

---

## 🛠️ 기술 스택

```
Google BigQuery (SQL)
├── DML          : SELECT, DELETE, CREATE OR REPLACE TABLE
├── 집계 함수    : COUNT, SUM, AVG, MIN, MAX, ROUND, COUNTIF
├── 윈도우 함수  : OVER(), LAG(), MAX() OVER(), PARTITION BY
├── 문자열 함수  : LIKE, REGEXP_REPLACE, UPPER, LENGTH, STARTS_WITH
├── 날짜 함수    : DATE(), DATE_DIFF(), EXTRACT(DAY FROM ...)
└── 기타         : WITH (CTE), UNION ALL, LEFT JOIN, HAVING, DISTINCT
```

---

## 💡 회고

**Keep**
`user_data`를 활용하면 구매 최신성·빈도·금액·다양성·주기·취소율 등 다차원으로 고객을 분류하고, 그룹별 맞춤 판매 전략 수립이 가능합니다.

**Problem**
다양한 속성 정보를 기반으로 유저를 세분화하는 데 RFM 분석 프레임워크만으로는 한계가 있습니다.

**Try**
유저 그룹의 심층 분석을 위해 **K-Means 등 클러스터링 알고리즘**을 추가로 활용해볼 예정입니다.

---

## 📎 참고

- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/quest1/MainQuest1_kahramyoon.pdf)
- 플랫폼: [Google BigQuery](https://cloud.google.com/bigquery)