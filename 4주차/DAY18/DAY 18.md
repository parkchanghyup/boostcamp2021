
# 수업 복습
---

## Seq2Seq Model
---
![S2S.PNG](S2S.PNG)

- 위 구조에서 입력 문장을 읽어내는 RNN  모델을 인코더라고 부르고
- 문장을 한 단어씩 출력 하는 RNN 모델을 디코더 라고 한다.
- Rnn모델로는 LSTM을 사용


- RNN기반의 모델 구조 이기 떄문에 hidden state의 dim이 고정된 상태로 계속해서 정보를 누적한다. 길이가 짧을 때는 괜찮지만 길어지게 되면 앞부분의 정보를 잃을 수 있다.
- decoder에서 SOS가 들어오면 encoder의 최종 output을 고려해 첫 단어를 생성해야하는데, 위와 같은 문제로 encoder의 앞부분 정보를 잃어 첫 단어부터 제대로 생성할 수 없을 수 있다.
- 문제를 해결하기 위해 attention module를 사용한다. attention module는 decoder의 hidden state하나와 encoder의 각 time step의 hidden state를 입력을 받아 현재 decoder에서 어떤 time step의 encoder hidden  state를 어느정도 사용할지 encoder state vector의 가중 평균을 구해준다.


## Seq2Seq Model with Attentnion
---
![S2S.PNG](S2S.PNG)
- Encoder 의 입력에 decoder 값을 사용 .
- Decoder의 hidden state와 Encoder의 각 time step의 hidden state 유사도를 구한다. 이후 softmax를 취해 각 hidden state의 비율을 알 수 있는데, 이러한 벡터를 attention vector라 한다( 위 그림에서 Attention distribution vector )


`Attention Score(유사도) 구하는 방법`

1. 단순히 두 vector의 내적으로 계산 
    - $score(h_t, \bar{h_s})=h_t^T\bar{h_s}$
2. 단순한 내적을 구하는 것이아닌 학습 가능한 행렬을 추가하는 방법 
    - $score(h_t,\bar{h}_s)=h_t^TW_a\bar{h_s}$
3. 두 hidden state를 cocat 하여 FC_Layer의 입력으로 넣어 유사도를 구하는 방법
    - $score(h_t, \bar{h_s})=v_a^Ttanh(W_a[h_t;\bar{h_s}])$

` Teacher forcing`
    - 디코더에 input 데이터를 넣을때 이전 스텝에서 잘못된 단어를 예측하더라도
    - 올바른 단어를 다음 디코더에 넣어주면서 학습 .

## Attention is great
---
- attention 모듈이 추가되면서 NMT에서 성능을 많이 올림
- 원인은 디코더의 매 타입스텝마다 입력시퀀스에서 어떤 부분에 정보를 집중해서 직접적으로 그 정볼를 사용할지 를 활용
- 인코더의 마지막 타입 스탭의 
- 기울기 소실 묹 ㅔ해결 .
- 모델이 어떤 단어에 집중했는지를 알수 있음  -> 해석가능 . . ? 


# Beam search
---

## Greedy decoding
---
- 시퀀스로서의 전체적인 문장의확률값을 포는게아니라 현재 타입스탭에서 가장 좋아보이는 단어를 선택하는 것
- 

## Exhaustive search
---
- 전체 문장의 확률값을 보는거 -> 너무 복잡하고 오래걸림


그래서 나온게 `Beam search`

디코더의 매타입스탭하다 단하나의 후보만 고려한느것도아니고 그렇다고 모든 후보를 고려하는 것도아님 우리가 정한 k개의 후보를 고려.

모든 경우의 수를 다 따지는 건 아니지만 효율적인 계산가능


그리디 디코딩의경우 엔드 토큰을 만날때 종료되고
빔 서치 디코딩은 서로다른 경로 혹은 가정이 존재하고 각각 서로 다른 시점에서 end 토큰을 생성할 수 있음 .

빔서치는 언제까지진행되는가
우리가 정한 time step 의 최댓값 까지 하거나 

## Beam search
---
- beam search는 디코더의 각 time step마다 k개의 가능한 경우를 고려해 최종k(beam size)개의 output중에서 가장 확률이 높은것을 선택하는 방식이다.(보통 beam size는 5~10)
- k개의 출력은 hypothesis라고 한다. 각 확률을 곱하는 것으로 각 hypothesis의 확률을 구해야 하지만 log를 사용해 곱셈을 덧셈으로 계산할 수 있다.
- $
score(y_1,...,y_t)=logP_{LM}(y_1,...,t_t|x)=\sum_{i = 1}^t logP_{LM}(y_i|y_1,...,y_{i-1},x)$


###  예시 : k = 2 일떄 beam search
![beam_search.PNG](beam_search.PNG)


- 가장 높은 확률을 가지는 k개를 계속해서 업데이트 해 가져간다.
위의 예에서 처음 문장을 생성할 때 가장 확률이 높은 2개 he, i가 선택되고 다음 tiem step에서는 he,i에서 각각 가장 확률이 높은 2개씩을 선택해 확률을 계산하고 가장 확률 이 높은 2개를 사용한다.
- beam search decoding에서는 각 hypothesis가 다른 timestep에 <END\>로 문장을 끝낼 수 있다. 이런 경우 임시 저장해놓고 위의 과정을 계산하고 가장 확률이 높은 2개를 사용한다.
- beam search decoding 종료시기
    - 미리 정해놓은 timestep T까지 도달할 때 종료
    - 저장해놓은 완성본이 n개 이상이 되었을 때 종료
- completed hypothesis중에서 선택을 해야하는데 길이가 짧을 수록 확률이 높게 나오기 떄문에 Normalize를 해주어야 한다. 
$score(y_1,...,y_t)=\frac{1}{t}\sum_{i=1}^tlogP_{LM}(y_i|y_1,...,y_{i-1},x)$
   

## BLEU score
---

- 


```python

```


```python

```
