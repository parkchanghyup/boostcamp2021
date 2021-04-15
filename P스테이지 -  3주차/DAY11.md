## 수업 복습

## 1. 인공지능의 탄생과 자연어처리

### 1.1 자연어처리 소개

- `ELIZA` : 최초의 대화형 챗봇

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1c4f8016-3f5c-43a8-a7ed-3f2b6c833c32/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1c4f8016-3f5c-43a8-a7ed-3f2b6c833c32/Untitled.png)

특허 문서 분류, 오타 교정, 정보 추출, 검색

### 1.2 자연어처리 응용분야

인간의 자연어 처리 : 화자는 객체를 자연어로(청자가 이해할 수 있는 방향으로 예쁘게) 인코딩, 청자는 자연어를 객체로(본인 지식을 바탕으로) 디코딩 함 

- 컴퓨터는 벡터 형태로 인코딩, 인코딩된 정보를 자연어로 디코딩 함

### 1.3 자연어 단어 임베딩

좌표평면 위에 데이터가 예쁘게 표현될 수 있다면, 어떤 task도 가능하다😜

- 자연어(문장, 문단 단위의)의 특징을 추출하는 건 컴퓨터보다 인간에게 더 어려울 수 있음
- `word2Vec` : 원핫인코딩 후 중심단어에서 주변단어 예측을 학습 ⇒ 단어 연산 가능, OOV 문제
- `FastText` : 단어의 subword를 고려하여 OOV 문제 해결, 알고리즘 자체는 word2vec과 비슷

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4da1945c-1427-4c4c-92fe-a0bc55887b30/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4da1945c-1427-4c4c-92fe-a0bc55887b30/Untitled.png)

    앞 뒤 꺽쇄 표시!(subword로 분리하면 어디가 앞이고 뒤인지 몰라서,,)

    ⇒ `FastText`의 경우 오탈자, OOV, 등장 회수가 적은 단어에서 임베딩 성능이 높았음

`단어 임베딩 방식` : 전체의 문맥을 담고 있지 않고, 동형어, 다의어에 약함

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/70197b7e-eceb-484e-a9c8-1126b402226c/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/70197b7e-eceb-484e-a9c8-1126b402226c/Untitled.png)

## 2. 딥러닝 기반의 자연어처리와 언어모델

### 2.1 언어모델

주어진 단어들로 다음에 등장할 단어의 확률을 계산하는 방식

- 다음에 등장한 단어를 잘 예측하는 모델은, 그 모델 내부에 언어적 특성이 잘 반영된 것이며, 문맥을 잘 계산할 수 있는 좋은 언어 모델이라 할 수 있음
- `Markov Chain Model` : 가장 전통적인 방식의 언어모델

    ⇒ 현재 state에서 다음 단어에 나올 단어들의 확률을 최대화하는 방향으로 언어모델 학습

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ee9a1f8f-9615-41b4-bc11-3a7fe3dd1c6a/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ee9a1f8f-9615-41b4-bc11-3a7fe3dd1c6a/Untitled.png)

- `RNN(Recurrent Neural Netwowrk)` 기반의 언어모델

    ⇒ 이전 state 정보(hidden)가 다음 state 예측에 사용 ⇒ 시계열 데이터 처리에 특화

    ⇒ `Context vector` : 마지막 출력은 앞선 단어들의 문맥을 모두 고려해서 만들어진 최종 출력, 이를 활용하여 다양한 task를 수행할 수 있음

### 2.2 Seq2Seq

RNN 기반, Encoder와 Decoder로 구성됨

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/db31260d-489f-4e89-b67e-089af0e2abbd/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/db31260d-489f-4e89-b67e-089af0e2abbd/Untitled.png)

### 2.3 Attention

RNN의 문제점 : 시퀀스가 긴 경우 앞단의 정보가 희석됨 ⇒ 중요한 정보에 주목하자!

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/66d7619a-ca33-4965-bd2f-98a69c57a925/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/66d7619a-ca33-4965-bd2f-98a69c57a925/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/20884487-ef6c-4072-96b6-e4f36809213d/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/20884487-ef6c-4072-96b6-e4f36809213d/Untitled.png)

(마지막 뿐 아니라) 각 노드의 hidden state 정보 활용

- Attention 역시 순차적 연산 때문에 속도가 느렸음 ⇒ **연결 구조를 없애보자!!!**

### 2.4 Self-attention

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2fd7f8b9-8c85-4ac8-951c-c55d08eedf12/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2fd7f8b9-8c85-4ac8-951c-c55d08eedf12/Untitled.png)

- Seq2Seq 모델은 Encoder와 Decoder가 서로 다른 네트워크로 구성된 반면, Transformer 모델은 하나의 네트워크로 구성되어 있음Seq2Seq 모델은 Encoder와 Decoder가 서로 다른 네트워크로 구성된 반면, Transformer 모델은 하나의 네트워크로 구성되어 있음

## 컴페티션

## 적용해볼 것들

