# 💹 금융 시계열 데이터 활용하기 — 피처 엔지니어링 & 머신러닝 분류

> 업비트 **ETH(이더리움) 분봉 데이터**를 바탕으로  
> 추세 라벨링 → 기술적 지표 피처 생성 → 피처 중요도 분석 → 피처 선택 → 분류 모델 학습 및 평가까지의  
> **금융 머신러닝(Financial ML) 전체 파이프라인**을 구현하는 프로젝트입니다.  
> *"Advances in Financial Machine Learning"* (Lopez de Prado)의 핵심 기법을 실제 암호화폐 데이터에 적용합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/mproject10.ipynb)

---

## 📁 파일 구조

```
etc/
└── mproject10.ipynb                         # 전체 파이프라인 노트북 (Google Colab)

데이터 (LMS에서 별도 다운로드 필요)
├── sub_upbit_eth_min_tick.csv               # ETH 분봉 OHLCV 원본 데이터
├── sub_upbit_eth_min_tick_label.pkl         # 추세 라벨 데이터
└── sub_upbit_eth_min_feature_labels.pkl     # 피처 + 라벨 병합 최종 데이터 (생성됨)
```

> ⚠️ 데이터는 LMS에서 별도로 다운로드하여 업로드해야 합니다.

---

## 📊 데이터셋

| 항목 | 내용 |
|------|------|
| 거래소 | 업비트 (Upbit) |
| 종목 | ETH/KRW (이더리움) |
| 시간 단위 | 분봉 (1분) |
| 기간 | 2017년 11월 ~ 2019년 |
| 컬럼 | `open`, `high`, `low`, `close`, `volume` |

---

## 🔄 전체 파이프라인

```
ETH 분봉 OHLCV 데이터
        │
        ▼
① 추세 라벨링 (3가지 방법)
   ├── 모멘텀 시그널 (shift / rolling mean)
   ├── 극소·극대 포인트 기반 라벨링
   └── t-값 기반 추세 라벨링 ← 핵심 기법
        │
        ▼
② 기술적 지표 피처 생성 (ta 라이브러리)
   ├── 거래량 지표 (CMF, FI, MFI, EMV, VPT)
   ├── 변동성 지표 (ATR, UI)
   ├── 모멘텀/수익률 지표
   └── 표준편차 지표
        │
        ▼
③ 피처 중요도 분석
   ├── MDI (Mean Decrease Impurity)
   ├── MDA (Mean Decrease Accuracy)
   └── SHAP (SHapley Additive exPlanations)
        │
        ▼
④ 피처 선택
   ├── RFE-CV (Recursive Feature Elimination)
   └── SFS (Sequential Feature Selector)
        │
        ▼
⑤ 분류 모델 학습
   └── BaggingClassifier + RandomForest
       + Purged K-Fold CV + GridSearchCV
        │
        ▼
⑥ 성능 평가
   └── Confusion Matrix / Accuracy /
       Precision / Recall / ROC-AUC
```

---

## 🏷️ STEP 1 — 추세 라벨링

### 방법 1 · 모멘텀 시그널

과거 `window`봉 전 종가 또는 이동평균 대비 현재 종가 방향으로 이진 시그널을 생성하고, 산점도에 상승(빨강)·하락(파랑)으로 시각화합니다.

```python
# 이전 종가 대비 방향
momentum_signal = np.sign(
    np.sign(close - close.shift(window)) + 1
)

# 이동평균 대비 방향
momentum_signal = np.sign(
    np.sign(close - close.rolling(window).mean()) + 1
)
```

---

### 방법 2 · 극소/극대 포인트 기반 라벨링

`get_local_min_max()` 함수로 가격의 극소(local min)·극대(local max)를 추출하고,
인접한 두 극점 간의 방향(상승=1 / 하락=0)으로 라벨을 부여합니다.

```python
mins, maxes = get_local_min_max(close, wait=3)

# 극점 합치기 → 시간순 정렬 → 방향 라벨링
extrema_df['label'] = [
    1 if next_val > cur_val else 0
    for cur_val, next_val in zip(values[:-1], values[1:])
]
```

---

### 방법 3 · t-값 기반 추세 라벨링 ⭐ (핵심)

각 시점에서 앞으로 `look_forward_window`봉을 탐색하며
**OLS 선형 회귀의 기울기 t-통계량**이 가장 큰 시점을 라벨로 결정합니다.

```python
def t_val_lin_r(close):
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sml.OLS(close, x).fit()
    return ols.tvalues[1]    # 기울기의 t-값 반환
```

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `look_forward_window` | 60 | 전방 탐색 구간 |
| `min_sample_length` | 5 | 최소 회귀 샘플 수 |
| `label['bin']` | -1 / 0 / 1 | 하락 / 중립 / 상승 |

> ⚠️ 이 단계는 실행에 약 **20~30분**이 소요됩니다.

---

## 🔧 STEP 2 — 기술적 지표 피처 생성

`ta==0.9.0` 라이브러리로 거래량·변동성·추세·모멘텀 지표를 생성합니다.

### 거래량 지표 (Volume)

| 피처명 | 지표 | 설명 |
|--------|------|------|
| `volume_cmf` | Chaikin Money Flow (window=20) | 자금 유입/유출 추세 |
| `volume_fi` | Force Index (window=15) | 가격 × 거래량 변화 강도 |
| `volume_mfi` | Money Flow Index (window=15) | 거래량 기반 RSI |
| `volume_sma_em` | Ease of Movement (window=15) | 가격 이동 용이성 |
| `volume_vpt` | Volume Price Trend | 가격·거래량 누적 추세 |

### 변동성 지표 (Volatility)

| 피처명 | 지표 | 설명 |
|--------|------|------|
| `volatility_atr` | Average True Range (window=10) | 시장 변동 진폭 |
| `volatility_ui` | Ulcer Index (window=15) | 하락 위험도 |

### 모멘텀·수익률 지표

| 피처명 | 설명 |
|--------|------|
| `vol_change_{5,10,20}` | 거래량 변화율 (5/10/20봉) |
| `ret_{5,10,20}` | 종가 수익률 (5/10/20봉) |
| `std_30` | 종가 30봉 표준편차 |
| `vol_std_30` | 거래량 30봉 표준편차 |

---

## 🎯 STEP 3 — 피처 중요도 분석

### MDI (Mean Decrease Impurity) — In-Sample

랜덤 포레스트 트리 분기 시 불순도 감소량으로 피처 중요도를 측정합니다.

```python
rfc = RandomForestClassifier(class_weight='balanced')
rfc.fit(X_sc, y)
feat_imp = mean_decrease_impurity(rfc, X.columns)
plot_feature_importance(feat_imp)  # 평균 ± 표준편차 수평 막대 그래프
```

> 핵심 피처: **`volatility_atr`** (Average True Range — 시장 변동성 측정 지표)

### MDA (Mean Decrease Accuracy) — Out-of-Sample

교차 검증 중 피처를 무작위로 섞었을 때 정확도 하락 폭으로 중요도를 측정합니다.

```python
svc_rbf = SVC(kernel='rbf', probability=True)
cv = KFold(n_splits=5)
feat_imp_mda = mean_decrease_accuracy(svc_rbf, X_sc, y, cv_gen=cv)
plot_feature_importance(feat_imp_mda)
```

### SHAP (SHapley Additive exPlanations)

게임 이론 기반으로 개별 예측에 대한 각 피처의 기여도를 계산합니다.
클래스별(0·1·2) Summary Plot을 통해 방향성까지 시각화합니다.

```python
explainer = shap.TreeExplainer(rfc, model_output="raw")
shap_value = explainer.shap_values(X_sc)  # shape: (샘플, 피처, 클래스)

# 클래스별 Summary Plot
shap.summary_plot(shap_value[:, :, 0], X_sc, class_names='class 0')
shap.summary_plot(shap_value[:, :, 1], X_sc, class_names='class 1')
shap.summary_plot(shap_value[:, :, 2], X_sc, class_names='class 2')

# 전체 피처 중요도 요약 (클래스 평균 절댓값)
mean_shap = np.mean(np.abs(shap_value), axis=(0, 2))
```

---

## 🔍 STEP 4 — 피처 선택

### RFE-CV (Recursive Feature Elimination with Cross Validation)

선형 SVC 기반으로 교차 검증을 반복하며 최적 피처 조합을 선택합니다.

```python
svc_rbf = SVC(kernel='linear', probability=True)
rfe_cv = RFECV(svc_rbf, cv=cv)
rfe_fitted = rfe_cv.fit(X_sc, y)

# 최적 피처 및 랭킹 확인
rfe_df[rfe_df["Optimal_Features"] == True]
```

### SFS (Sequential Feature Selector)

순방향(forward) 탐색으로 성능 기여도가 높은 상위 **2개** 피처를 선택합니다.

```python
sfs_forward = SequentialFeatureSelector(
    svc_rbf, n_features_to_select=2, direction='forward'
)
sfs_fitted = sfs_forward.fit(X_sc, y)
```

---

## 🤖 STEP 5 — 분류 모델 학습

### 데이터 분할

t-값 라벨의 이진 변환 후 시간 순서를 유지하며 분할합니다.

```python
# 이진 변환: t_value == 1이면 1, 나머지는 0
df_data['t_value'] = df_data['t_value'].apply(lambda x: x if x == 1 else 0)

# 시간 순 분할 (70% Train / 20% Test)
train_ratio, test_ratio = 0.7, 0.2
```

### Purged K-Fold Cross Validation

금융 시계열의 **데이터 누출(look-ahead bias)** 을 방지하기 위해 학습·검증 구간 사이의 겹치는 샘플을 제거(purge)하는 교차 검증을 사용합니다.

```python
t1 = pd.Series(train_y.index.values, index=train_y.index)
cv = PKFold(n_cv=4, t1=t1, pct_embargo=0)
```

### BaggingClassifier + RandomForest + GridSearchCV

```python
bc_params = {
    'n_estimators'              : [5, 10, 20],
    'max_features'              : [0.5, 0.7],
    'estimator__max_depth'      : [3, 5, 10, 20],
    'estimator__max_features'   : [None, 'auto'],
    'estimator__min_samples_leaf': [3, 5, 10],
    'bootstrap_features'        : [False, True]
}

rfc = RandomForestClassifier(class_weight='balanced')
bag_rfc = BaggingClassifier(rfc)
gs_rfc = GridSearchCV(bag_rfc, bc_params, cv=cv, n_jobs=-1, verbose=1)
gs_rfc.fit(train_x, train_y)
```

> ⚠️ GridSearchCV 실행에 약 **최대 20분**이 소요됩니다.

---

## 📊 STEP 6 — 성능 평가

```python
pred_y = gs_rfc_best.predict(test_x)
prob_y = gs_rfc_best.predict_proba(test_x)

# 혼동 행렬 + 정확도 / 정밀도 / 재현율
print(confusion_matrix(test_y, pred_y))
print(f'정확도:{accuracy_score(test_y, pred_y)}, '
      f'정밀도:{precision_score(test_y, pred_y)}, '
      f'재현율:{recall_score(test_y, pred_y)}')

# ROC-AUC 곡선
fpr, tpr, _ = roc_curve(test_y, prob_y[:, 1])
auc = roc_auc_score(test_y, prob_y[:, 1])
plt.plot(fpr, tpr)
print(f'auc:{auc}')
```

---

## 🛠️ 기술 스택

```
Python
├── pandas, numpy                  # 데이터 처리 및 수치 연산
├── matplotlib                     # 시각화
├── tqdm                           # 진행률 표시
├── ta==0.9.0                      # 기술적 지표 생성
├── statsmodels                    # OLS 선형 회귀 (t-값 라벨링)
├── shap                           # 모델 해석 (SHAP)
├── scikit-learn
│   ├── RandomForestClassifier     # 기본 분류기
│   ├── BaggingClassifier          # 앙상블
│   ├── SVC                        # 피처 중요도·선택용
│   ├── GridSearchCV               # 하이퍼파라미터 탐색
│   ├── RFECV                      # 재귀적 피처 제거
│   ├── SequentialFeatureSelector  # 순방향 피처 선택
│   └── StandardScaler             # 피처 스케일링
└── 커스텀 구현
    ├── mean_decrease_impurity()   # MDI 피처 중요도
    ├── mean_decrease_accuracy()   # MDA 피처 중요도
    ├── plot_feature_importance()  # 중요도 시각화
    ├── get_local_min_max()        # 극소·극대 추출
    └── PKFold                     # Purged K-Fold CV
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 노트북을 열거나 위 배지를 클릭합니다.
2. LMS에서 아래 두 파일을 다운로드하여 업로드합니다:
   - `sub_upbit_eth_min_tick.csv`
   - `sub_upbit_eth_min_tick_label.pkl`
3. 셀을 위에서부터 순서대로 실행합니다.

### ⏱️ 실행 시간 안내

| 단계 | 소요 시간 |
|------|-----------|
| t-값 라벨링 (STEP 1) | 약 **20~30분** |
| GridSearchCV (STEP 5) | 약 **최대 20분** |

---

## 📎 참고

- 참고 도서: *Advances in Financial Machine Learning* — Marcos Lopez de Prado
- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/etc/mproject10.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/etc/mproject10.ipynb)
