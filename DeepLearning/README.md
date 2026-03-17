# 🎵 LSTM 기반 가사(Lyrics) 생성 AI

> 영어 가사 텍스트 데이터로 LSTM 언어 모델을 학습시키고,  
> **Sampling** 및 **Beam Search** 두 가지 디코딩 전략으로 새로운 가사를 생성하는 프로젝트입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/DeepLearning/lyricsgnrt.ipynb)

---

## 📁 파일 구조

```
DeepLearning/
└── lyricsgnrt.ipynb       # 전체 파이프라인 노트북 (Google Colab)

데이터 (별도 준비 필요)
└── lyrics.zip             # 가사 .txt 파일 묶음
    └── lyrics/*.txt       # 가사 원본 텍스트 파일들
```

---

## 🔄 전체 파이프라인

```
가사 데이터 (.txt)
       │
       ▼
① 데이터 로드 및 전처리
       │
       ▼
② 토크나이징 & 패딩 (maxlen=30)
       │
       ▼
③ LSTM 언어 모델 학습
       │
       ▼
④ 텍스트 생성
   ├── Sampling (Temperature + Top-k)
   └── Beam Search
```

---

## 📊 데이터 전처리

여러 `.txt` 가사 파일을 읽어 하나의 코퍼스로 통합한 뒤, 아래 전처리를 적용합니다.

```python
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)  # 구두점 앞뒤 공백
    sentence = re.sub(r'[" "]+', " ", sentence)          # 중복 공백 제거
    sentence = "<start> " + sentence + " <end>"          # 시작/종료 토큰 추가
    return sentence
```

전처리 시 특수문자/숫자/구두점을 과도하게 제거하지 않고 **의미를 최대한 보존**하는 방향으로 설계했습니다.

---

## 🔢 토크나이징 & 데이터셋 구성

| 설정 | 값 |
|------|----|
| 어휘 크기 (`num_words`) | 12,000 |
| 최대 시퀀스 길이 (`maxlen`) | 30 |
| 패딩 방식 | post |
| OOV 토큰 | `<unk>` |
| 배치 크기 | 64 |
| 검증 데이터 비율 | 20% |

언어 모델의 특성에 맞게 입력(`enc_inputs`)과 타겟(`dec_targets`)을 한 토큰씩 shift하여 구성합니다.

```python
enc_inputs = tensor[:, :-1]   # 입력: <start> w1 w2 ... wN-1
dec_targets = tensor[:, 1:]   # 타겟: w1 w2 ... wN <end>
```

---

## 🧠 모델 구조

`Embedding → LSTM × 2 → Dense` 구조의 언어 모델입니다.

```python
class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        self.embedding = Embedding(vocab_size, embedding_size)
        self.rnn_1     = LSTM(hidden_size, return_sequences=True)
        self.rnn_2     = LSTM(hidden_size, return_sequences=True)
        self.linear    = Dense(vocab_size)
```

| 하이퍼파라미터 | 값 | 설명 |
|--------------|-----|------|
| `embedding_size` | 256 | 단어 벡터 차원 수 |
| `hidden_size` | 1024 | LSTM 히든 유닛 수 |
| `vocab_size` | 12,001 | 어휘 크기 + 패드 토큰 |

---

## 🏋️ 모델 학습

```python
optimizer = Adam()
loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

model.compile(loss=loss, optimizer=optimizer)
history = model.fit(dataset, validation_data=val_dataset, epochs=5)
```

| 설정 | 값 |
|------|----|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 5 |
| 학습 소요 시간 | 약 30분 (Colab 기준) |
| 최종 val_loss | **1.2769** |

---

## ✍️ 텍스트 생성 전략

### 1. Sampling (Temperature + Top-k)

확률 분포에서 무작위로 단어를 샘플링합니다. Temperature와 Top-k를 조합하여 다양하고 자연스러운 문장을 생성합니다.

```python
def generate_text_sampling(model, tokenizer,
                            init_sentence="<start>",
                            max_len=30,
                            temperature=1.0,
                            top_k=20):
    ...
    # 확률 상위 top_k 단어 중에서 temperature에 따라 샘플링
    top_k_indices = prob.argsort()[-top_k:]
    next_id = np.random.choice(top_k_indices, p=top_k_probs)
```

| 파라미터 | 권장값 | 설명 |
|----------|--------|------|
| `temperature` | 0.7 ~ 0.9 | 낮을수록 보수적, 높을수록 다양 |
| `top_k` | 20 | 후보 단어 수 |

**특징:** 랜덤성이 있어 실행마다 다른 문장 생성, 창의적인 가사 표현에 적합

---

### 2. Beam Search

매 스텝에서 확률 상위 `beam_width`개의 후보 시퀀스를 유지하며 가장 높은 누적 확률의 경로를 선택합니다.

```python
def generate_text_beam(model, tokenizer,
                       init_sentence="<start>",
                       max_len=30,
                       beam_width=3):
    ...
    # beam_width개 후보 유지, 누적 로그 확률로 정렬
    sequences = sorted(candidates, key=lambda x: x[1])[:beam_width]
```

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `beam_width` | 3 | 유지할 후보 경로 수 |

**특징:** Sampling보다 랜덤성이 낮고, 문법적으로 더 논리적인 문장 생성에 적합

---

### 생성 예시

```python
# Sampling
generate_text_sampling(model, tokenizer, init_sentence="<start> it", temperature=0.8)

# Beam Search
generate_text_beam(model, tokenizer, init_sentence="<start> it", beam_width=3)
```

두 방법을 모두 제공하여 상황에 따라 선택할 수 있도록 구성했습니다.

---

## 🛠️ 개선 포인트

기존 베이스라인 대비 아래 개선 사항을 적용해 생성 품질을 향상시켰습니다.

| 항목 | 기존 | 개선 |
|------|------|------|
| 전처리 | 특수문자/구두점 모두 제거 | 최소한만 제거, 의미 보존 |
| 시퀀스 토큰 | 없음 | `<start>` / `<end>` 추가 |
| 최대 시퀀스 길이 | 15 | **30** |
| 디코딩 방식 | Argmax (greedy) | **Temperature + Top-k Sampling** & **Beam Search** |

> **향후 개선 가능 사항:**  
> Attention 메커니즘, Bidirectional LSTM, Layer Normalization + Dropout 적용

---

## 🛠️ 기술 스택

```
Python
├── TensorFlow / Keras
│   ├── Embedding
│   ├── LSTM × 2
│   └── Dense
├── NumPy
├── scikit-learn    # train_test_split
└── re, glob, os   # 전처리 및 파일 처리
```

환경: **Google Colab**

---

## 🚀 실행 방법

1. Google Colab에서 `lyricsgnrt.ipynb`를 열거나 위 배지를 클릭합니다.
2. `lyrics.zip` 파일을 업로드합니다.
3. 셀을 위에서부터 순서대로 실행합니다.
4. 학습 완료 후 원하는 시작 문구로 가사를 생성합니다:

```python
# Sampling 방식
generate_text_sampling(model, tokenizer, init_sentence="<start> love", temperature=0.8)

# Beam Search 방식
generate_text_beam(model, tokenizer, init_sentence="<start> love", beam_width=3)
```

> ⚠️ 5 epoch 기준 학습에 약 **30분**이 소요됩니다 (Colab GPU 기준).

---

## 📎 참고

- 노트북: [Google Colab에서 열기](https://colab.research.google.com/github/kahram-y/first-repository/blob/master/DeepLearning/lyricsgnrt.ipynb)
- 저장소: [GitHub](https://github.com/kahram-y/first-repository/blob/master/DeepLearning/lyricsgnrt.ipynb)