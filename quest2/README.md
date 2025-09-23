# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 윤가람
- 리뷰어 : 강호신


# PRT(Peer Review Template)
- [Y]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
import re
UNK_ID = -1  # 사전에 없는 단어는 -1로 처리
def preprocess(text):
    """입력 문장을 전처리 → 토큰 리스트 반환"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()
def encoder(text_line, vocab):
    """문장을 정수 시퀀스로 변환"""
    tokens = preprocess(text_line)
    return [vocab.get(tok, UNK_ID) for tok in tokens]
# 사용 예시
sample = "I am a boy"
print(sample, "->", encoder(sample, vocab))
    
- [Y]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
items = sorted(word_count.items(), key=lambda kv: (-kv[1], kv[0]))
vocab = {word: idx for idx, (word, _) in enumerate(items)}
빈도 순서로 내림차순을 수행한 후 인덱스를 부여하고, 동률이면 단어 별로 오름 차순으로 정령함
        
- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        
- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        
- [Y]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

이정도 코드를 분석 할 만큼 이해도가 높지 않아 되었다, 그렇지 않다고 판단하기 여려움.
그러나 문제를 순차적으로 풀어나가기 위한 코딩이 좋았습니다.
clean_lines = [
    ''.join(
      ch if (ch.islower() or ch.isdigit() or ch.isspace()) else ' '
      for ch in line
    )
    for line in data_lower
]


# 회고(참고 링크 및 코드 개선)
```
# 리뷰어의 회고를 작성합니다.
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
