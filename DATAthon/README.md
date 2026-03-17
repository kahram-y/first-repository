# 🚭 노담케어 포인트 (NoDam Care Point)
### 분류 모델을 기반으로 한 생체 신호 기반 흡연 예측 · 금연 동기부여 시스템

> **"담배는 NO, 건강은 CARE, 혜택은 POINT UP!"**  
> 생체 신호 AI 예측으로 흡연 여부를 감지하고, 금연을 포인트로 동기부여하는 공익 헬스케어 시스템

---

## 👥 팀원

| 역할 | 이름 |
|------|------|
| 모델링 마스터 | 김효진 |
| 사이버 수사대 | 서주연 |
| 묵묵한 공헌자 | 윤가람 |
| 로드맵 매니저 | 이미현 |
| 만능 치트키 | 조은별 |

> 데이터사이언스 6기 데이터톤 — M2팀 (2025.12.01)

---

## 📌 프로젝트 개요

건강검진 문진표의 '비흡연' 체크가 항상 사실일까요?  
생체 신호 데이터는 이미 흡연 위험을 **80% 확률**로 예측할 수 있습니다.

이 프로젝트는 세 가지 문제의식에서 출발합니다:

1. **숨겨진 흡연자** — 문진표 자기 보고의 한계를 생체 신호 데이터로 보완
2. **간접 흡연 위험** — 흡연자 주변 고위험군까지 포괄하는 공익적 개입 필요
3. **금연 동기 부재** — 눈에 보이지 않는 건강 개선을 포인트로 적립해 지속적 동기부여

---

## 📊 데이터

| 항목 | 내용 |
|------|------|
| 출처 | [Kaggle — Binary Prediction of Smoker Status using Bio-Signals](https://www.kaggle.com/competitions/playground-series-s3e24) |
| 규모 | 약 15만 명 (익명) |
| 타겟 변수 | `smoking` (0: 비흡연, 1: 흡연) — 클래스 비율 56.3% vs 43.7% |

### 주요 변수

**신체 기본 정보**
- `age`, `height(cm)`, `weight(kg)`, `waist(cm)`
- `eyesight(left/right)`, `hearing(left/right)`, `dental caries`

**혈액/소변 건강 지표**
- `systolic`, `relaxation` — 최고/최저 혈압
- `fasting blood sugar` — 공복 혈당
- `cholesterol`, `HDL`, `LDL`, `triglyceride` — 콜레스테롤 관련
- `hemoglobin` — 헤모글로빈 (흡연 시 산소 운반 능력 저하)
- `GTP`, `AST`, `ALT` — 간 기능
- `urine protein`, `serum creatinine` — 신장 기능

---

## 🔬 데이터 분석 및 전처리

### STEP 1 · 초기 데이터 확인
- 타겟 변수 클래스 불균형 여부 파악
- 데이터 크기/타입/결측치 확인
- 변수 설명서 작성 및 도메인 지식 서칭

### STEP 2 · 타겟 변수와의 상관관계 확인
- Correlation Heatmap 분석
- 흡연과 연관성 높은 지표 도출: **헤모글로빈, 중성지방(GTP), 키/몸무게**

### STEP 3 · 독립변수 별 분포 파악
- 왜도가 심한 변수 식별: `GTP`, `Hemoglobin`, `HDL/LDL`, `Triglyceride`
- 구간별 성향 파악: `eyesight`, `hearing`, `dental caries`, `AST`, `ALT`, `Creatinine`

### STEP 4 · 데이터 문제점 파악
- VIF(분산팽창지수)로 다중공선성 검토
- 타겟 변수와의 상관관계 및 도메인 지식 기반으로 제거 변수 확정
- 이상치 확인 및 처리 방법 논의

---

## ⚙️ Feature Engineering

의학적으로 검증된 지표를 기반으로 12개의 파생변수 생성:

| 카테고리 | 파생변수 |
|----------|----------|
| 연령 · 체질량 | `Age_group`, `BMI` (비만도 표준 지표) |
| 상호작용 지표 | `Weight_waist_interaction`, `BMI_AST_interaction`, `BMI_ALT_interaction`, `BMI_GTP_interaction` |
| 심혈관 · 지질 | `HDL_risk`, `TG_risk`, `Average_blood_pressure`, `Pulse_pressure` |
| 감각기관 비대칭 | `Eyesight_diff`, `Hearing_diff` |

**공통 전처리:** 왜도가 큰 5개 지표 로그 변환 / 낮은 상관관계 변수 4개 제거

---

## 🧪 모델링 전략

세 가지 전처리 전략을 비교 실험:

| 전략 | 설명 |
|------|------|
| **전략 A** | 파생변수 생성(12개) + IQR 기반 이상치 제거 |
| **전략 B** | 파생변수 생성(12개) + 이상치 제거 없음 |
| **전략 C** | 파생변수 생성 없음 + 이상치 제거 없음 (베이스라인) |

### 모델별 AUC Score 비교

| 모델 | 전략 A | 전략 B | 전략 C |
|------|--------|--------|--------|
| Logistic Regression | 0.8277 | 0.8363 | 0.8498 |
| Random Forest | 0.8570 | 0.8508 | 0.8535 |
| LightGBM | 0.8539 | 0.8626 | 0.8635 |
| **XGBoost** | **0.8635** | **0.9014** | **0.8644** |

> **XGBoost**가 모든 전략에서 최상위 성능 기록.  
> 단, 전략 B에서 과적합 위험이 발견되어 5-Fold CV로 재점검 → **이상치는 노이즈가 아닌 건강 위험 신호**로 판단, 전략 B 최적화 결정 (CV AUC: **0.8667**)

---

## 🏆 최종 모델 성능

### 1단계 · 과적합 방지 튜닝 (RandomizedSearchCV)
- 50 조합 × 5-Fold = 250 fits
- Train AUC: 0.8915 / CV AUC: 0.8669 (gap 0.0246 — 일반화 안정)
- 주요 파라미터: `learning_rate=0.05`, `n_estimators=470`, `max_depth=6`, `lambda=1.55`

### 2단계 · 성능 극대화 튜닝 (GridSearchCV)
- 108 조합 × 5-Fold = 540 fits
- AUC: **0.8677** (+0.0008 개선)
- 조정 파라미터: `n_estimators=550↑`, `max_depth=7↑`, `lambda=1.0↓`

### 3단계 · 스태킹 앙상블
- 베이스 모델: Logistic Regression + LightGBM + XGBoost
- 메타 모델: Logistic Regression
- **최종 ROC-AUC: 0.9052** (+0.0375 개선, 단일 모델 대비 3.75% 향상)
- 5-Fold CV로 일반화 보장

---

## ⚠️ 데이터 한계점

**성별 데이터 부재**  
헤모글로빈, 크레아티닌 등 핵심 생체 지표는 성별에 따라 정상 범위가 다르지만, 데이터셋에 Gender 변수가 없어 정확한 기준치 설정이 불가능했습니다. 키/몸무게를 통한 성별 예측은 노이즈 발생 우려로 사용하지 않았습니다.

**이상치 제거의 딜레마**  
의료 데이터 특성상 극단값은 기계 오류일 수도 있지만 중증 흡연자 신호일 가능성도 존재합니다. 제거 시 일반화 성능이 향상되나 고위험군 신호를 손실하고, 유지 시 고위험군 패턴 학습이 가능하나 모델 안정성이 저하되는 trade-off 관계입니다.

**상관관계의 한계**  
단면 연구(cross-sectional) 기반이므로 흡연과 생체 신호 간의 인과 관계를 명확히 증명할 수 없습니다. 향후 시계열 데이터를 통한 금연 후 수치 변화 추적이 필요합니다.

---

## 💡 비즈니스 아이디어 — 노담케어 포인트

XGBoost 예측 결과를 바탕으로 사용자를 **3개 그룹**으로 분류하여 선제적으로 관리합니다.

| 그룹 | 기준 | 관리 방법 |
|------|------|-----------|
| 🔴 **Code RED** | 흡연 확률 80% 이상 (헤모글로빈, GTP 위험 수치) | 집중 관리 알림 발송, 3개월 집중 케어 |
| 🟡 **Code YELLOW** | 금연 유지 중 헤모글로빈/콜레스테롤 50% 악화 | 수치 재악화 경고, 금연 의지 강화 지원 |
| 🟢 **Code GREEN** | AI 감지 생체 신호 30% 이하 · 정상 범위 유지 | 특별 리워드 제공, 지역 화폐 전환 |

**인센티브 생태계**
- 건강검진 데이터 연동 → 포인트 적립 → 지역 화폐/공과금 납부
- 3개월 주기 재검사로 지속적인 건강 추적
- 보험료 최대 15% 할인 서비스

**기대 효과**
- 개인: 금연 성공률 증가, 경제적 이득
- 사회: 지역 경제 활성화, 건보료 재정 건전성 확보
- 민간기업: Loss Ratio 감소, 우량 고객 확보

---

## 🛠️ 기술 스택

```
Python
├── pandas, numpy          # 데이터 전처리
├── matplotlib, seaborn    # 시각화
├── scikit-learn           # 모델링 (Logistic Regression, Random Forest, Stacking)
├── xgboost                # XGBoost
└── lightgbm               # LightGBM
```

---

## 📁 파일 구조

```
DATAthon/
└── byteam.ipynb    # 전체 분석 및 모델링 코드
```

---

## 📎 참고

- 데이터: [Kaggle — Binary Prediction of Smoker Status using Bio-Signals](https://www.kaggle.com/competitions/playground-series-s3e24)
- 발표일: 2025년 12월 1일
- 소속: 데이터사이언스 6기 데이터톤 M2팀