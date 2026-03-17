# 📈 금융 시계열 기반 머신러닝 트레이딩 시그널 분류

> 업비트 ETH(이더리움) 분봉 데이터를 활용하여 **추세 라벨링 → 피처 엔지니어링 → 피처 선택 → 분류 모델 학습 → 성능 평가**까지의 전체 파이프라인을 구현하는 프로젝트입니다.
> "Advances in Financial Machine Learning" (Lopez de Prado)의 기법들을 실제 암호화폐 데이터에 적용합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/mproject10.ipynb)

---

## 📁 파일 구조

```
etc/
└── mproject10.ipynb                       # 전체 파이프라인 노트북 (Google Colab)

데이터 (LMS에서 별도 다운로드 필요)
├── sub_upbit_eth_min_tick.csv             # 업비트 ETH 분봉 OHLCV 원본 데이터
├── sub_upbit_eth_min_tick_label.pkl       # 추세 라벨 (t_value: 0/1/2)
└── sub_upbit_eth_min_feature_labels.pkl   # 피처 + 라벨 병합 최종 데이터
```

> ⚠️ 데이터는 LMS에서 별도로 다운로드하여 사용해야 합니다.

---

## 📊 데이터셋

| 항목 | 내용 |
|------|------|
| 거래소 | 업비트 (Upbit) |
| 종목 | ETH/KRW (이더리움) |
| 시간 단위 | 분봉 (1분) |
| 기간 | 2017년 11월 ~ 2019년 |
| 주요 컬럼 | `open`, `high`, `low`, `close`, `volume` |

---

## 🔄 전체 파이프라인

```
원본 데이터
    │
    ▼
① 추세 라벨링 (Triple Barrier / t-값 기반)
    │
    ▼
② 기술적 지표 피처 생성 (TA-Lib)
    │
    ▼
③ 피처 중요도 분석 (MDI / MDA / SHAP)
    │
    ▼
④ 피처 선택 (RFE-CV / SFS)
    │
    ▼
⑤ 분류 모델 학습 (BaggingClassifier + RandomForest + Purged K-Fold CV)
    │
    ▼
⑥ 성능 평가 (Confusion Matrix, ROC-AUC)
```

---

## 🏷️ PART 1 — 추세 라벨링

### 1-1. 모멘텀 시그널

과거 `window`개 봉 전 종가 대비 현재 종가의 방향(상승/하락)으로 이진 시그널을 생성합니다.

```python
# 방법 1: 이전 종가 대비
momentum_signal = np.sign(np.sign(close - close.shift(window)) + 1)

# 방법 2: 이동평균 대비
momentum_signal = np.sign(np.sign(close - close.rolling(window).mean()) + 1)
```

결과를 산점도로 시각화하여 상승(빨강) / 하락(파랑) 구간을 확인합니다.

---

### 1-2. 극소/극대 포인트 기반 라벨링

`get_local_min_max()` 함수로 가격의 극소(local min)와 극대(local max) 포인트를 추출하고, 각 구간의 방향(상승=1 / 하락=0)으로 라벨을 부여합니다.

```python
mins, maxes = get_local_min_max(close, wait=3)

extrema_df = pd.concat([
    mins.rename(columns={'min_time': 'time', 'local_min': 'value'}).assign(type='min'),
    maxes.rename(columns={'max_time': 'time', 'local_max': 'value'}).assign(type='max')
]).sort_values('time')

# 다음 극점이 현재보다 높으면 1(상승), 낮으면 0(하락)
extrema_df['label'] = [1 if next_val > cur_val else 0 for ...]
```

---

### 1-3. t-값 기반 추세 라벨링 (Financial ML 핵심 기법)

`t_val_lin_r()` 함수로 선형 회귀의 t 통계량을 계산하여, 가격 시계열이 유의미한 상승/하락 추세를 보이는지를 검정하고 라벨을 부여합니다.

```python
def t_val_lin_r(close):
    # OLS 회귀의 시간 계수(기울기)에 대한 t-값 반환
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sml.OLS(close, x).fit()
    return ols.tvalues[1]
```

각 시점에서 `look_forward_window=60`봉 앞을 탐색하며, t-값의 절댓값이 최대가 되는 시점까지의 추세를 라벨로 저장합니다.

| 파라미터 | 값 |
|----------|----|
| `look_forward_window` | 60 |
| `min_sample_length` | 5 |
| `step` | 1 |

---

## 🔧 PART 2 — 피처 엔지니어링

`ta` 라이브러리로 다양한 기술적 지표(Technical Indicators)를 생성합니다.

### 거래량 지표 (Volume Indicators)

| 피처명 | 지표 | 설명 |
|--------|------|------|
| `volume_cmf` | Chaikin Money Flow | 자금 유입/유출 추세 |
| `volume_fi` | Force Index | 가격 × 거래량 변화 강도 |
| `volume_mfi` | Money Flow Index | 거래량 기반 RSI |

### 모멘텀 / 수익률 지표

| 피처명 | 설명 |
|--------|------|
| `vol_change_{5,10,20}` | 거래량 변화율 (5/10/20봉) |
| `ret_{5,10,20}` | 종가 수익률 (5/10/20봉) |

### 변동성 지표

| 피처명 | 설명 |
|--------|------|
| `std_30` | 종가 30봉 표준편차 |
| `vol_std_30` | 거래량 30봉 표준편차 |

> ATR(Average True Range) 등 추가 변동성 지표도 포함됩니다.

---

## 🎯 PART 3 — 피처 중요도 분석

### MDI (Mean Decrease Impurity)

랜덤 포레스트의 트리 분기 시 불순도 감소량으로 피처 중요도를 계산합니다 (in-sample).

```python
rfc = RandomForestClassifier(class_weight='balanced')
rfc.fit(X_sc, y)
feat_imp = mean_decrease_impurity(rfc, X.columns)
```

### MDA (Mean Decrease Accuracy)

교차 검증에서 각 피처를 제거했을 때 정확도 하락 폭으로 중요도를 계산합니다 (out-of-sample).

```python
svc_rbf = SVC(kernel='rbf', probability=True)
cv = KFold(n_splits=5)
feat_imp_mda = mean_decrease_accuracy(svc_rbf, X_sc, y, cv_gen=cv)
```

### SHAP (SHapley Additive exPlanations)

게임 이론 기반으로 각 피처가 개별 예측에 미치는 기여도를 계산합니다. 클래스별 Summary Plot을 통해 시각화합니다.

```python
explainer = shap.TreeExplainer(rfc, model_output="raw")
shap_value = explainer.shap_values(X_sc)

# 클래스별 Summary Plot (0, 1, 2)
shap.summary_plot(shap_value[:, :, 0], X_sc)
```

---

## 🔍 PART 4 — 피처 선택

### RFE-CV (Recursive Feature Elimination with Cross Validation)

선형 SVC를 기반으로 교차 검증을 반복하며 최적 피처 조합을 선택합니다.

```python
svc_rbf = SVC(kernel='linear', probability=True)
rfe_cv = RFECV(svc_rbf, cv=cv)
rfe_fitted = rfe_cv.fit(X_sc, y)
```

### SFS (Sequential Feature Selector)

순방향(forward) 탐색으로 상위 n개의 최적 피처를 선택합니다.

```python
sfs_forward = SequentialFeatureSelector(svc_rbf, n_features_to_select=2, direction='forward')
sfs_fitted = sfs_forward.fit(X_sc, y)
```

---

## 🤖 PART 5 — 분류 모델 학습

### Purged K-Fold CV

금융 시계열의 데이터 누출(look-ahead bias)을 방지하기 위해 학습/검증 구간 사이의 겹치는 샘플을 제거(purge)하는 교차 검증을 사용합니다.

```python
t1 = pd.Series(train_y.index.values, index=train_y.index)
cv = PKFold(n_cv=4, t1=t1, pct_embargo=0)
```

### BaggingClassifier + RandomForestClassifier + GridSearchCV

```python
rfc = RandomForestClassifier(class_weight='balanced')
bag_rfc = BaggingClassifier(rfc)

bc_params = {
    'n_estimators': [5, 10, 20],
    'max_features': [0.5, 0.7],
    'estimator__max_depth': [3, 5, 10, 20],
    'estimator__max_features': [None, 'auto'],
    'estimator__min_samples_leaf': [3, 5, 10],
    'bootstrap_features': [False, True]
}

gs_rfc = GridSearchCV(bag_rfc, bc_params, cv=cv, n_jobs=-1)
gs_rfc.fit(train_x, train_y)
```

| 학습/테스트 분할 | 비율 |
|----------------|------|
| Train | 70% |
| Test | 20% |

---

## 📊 PART 6 — 성능 평가

혼동 행렬, 정확도, 정밀도, 재현율, ROC-AUC로 모델 성능을 종합 평가합니다.

```python
# 예측
pred_y = gs_rfc_best.predict(test_x)
prob_y = gs_rfc_best.predict_proba(test_x)

# 평가 지표
confusion = confusion_matrix(test_y, pred_y)
accuracy  = accuracy_score(test_y, pred_y)
precision = precision_score(test_y, pred_y)
recall    = recall_score(test_y, pred_y)

# ROC Curve
fpr, tpr, thresholds = roc_curve(test_y, prob_y[:, 1])
auc = roc_auc_score(test_y, prob_y[:, 1])
```

---

## 🛠️ 기술 스택

```
Python
├── pandas, numpy                        # 데이터 처리
├── matplotlib                           # 시각화
├── ta==0.9.0                            # 기술적 지표 생성
├── statsmodels                          # OLS 선형 회귀 (t-값 계산)
├── scikit-learn
│   ├── RandomForestClassifier           # 기본 분류기
│   ├── BaggingClassifier                # 앙상블
│   ├── SVC                             # 피처 중요도 / 선택용
│   ├── GridSearchCV                     # 하이퍼파라미터 탐색
│   ├── RFECV, SequentialFeatureSelector # 피처 선택
│   └── StandardScaler                  # 피처 스케일링
└── shap                                 # 모델 해석 (SHAP)
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `mproject10.ipynb`를 열거나 배지를 클릭합니다.
2. LMS에서 데이터 파일 2개를 다운로드합니다:
   - `sub_upbit_eth_min_tick.csv`
   - `sub_upbit_eth_min_tick_label.pkl`
3. 셀을 위에서부터 순서대로 실행합니다.

> ⚠️ t-값 라벨링 단계(PART 1-3)는 실행에 약 **20~30분**이 소요됩니다.
> ⚠️ GridSearchCV 단계(PART 5)는 실행에 약 **최대 20분**이 소요됩니다.

---

## 📎 참고

- 참고 도서: *Advances in Financial Machine Learning* — Marcos Lopez de Prado
- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/mproject10.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/etc/mproject10.ipynb)