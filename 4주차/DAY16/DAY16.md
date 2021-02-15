
# 수업복습
---

## NLP(Natural language processing)
- 딥러닝 기술의 발전을 선도
- 가령 low-level parsing을예로들면
    - 문장을 이루는 각 단어들을 token이라 부르고 단어 단위로 쪼개나가는 과정을 `tokenizaiton`이라 부름
    - 문장은 이런 토큰들이 특정 순서로 이뤄진 sequence라 볼 수 있다.
    - `stemming` : 단어의 어근 추출
- Word ans phrase level
        - Named entity recognition(NER)은 단일 단어 혹은 여러단어로 이뤄진  어떤 고유명사를 인식하는 task
            - newyork times 를 하나의 단어로 인식해야됨 (NewYork / times (x))
        - POS tagging : word들이 문장내에서 품사나 성분이 뭔지 알아내는 task
- sentece level(문장단위 분석)
        - 감정 분석, 기계 변역(i study math -> 나는 수학을 공부한다) 
- multi-sence and parapraph level
        - `Entaliment predciont`: 두 문장 간의 논린적인 내포, 모순관계를 예측, 
          - ex. 어제 존이 결혼을 했다. 어제 최소한 한명이 결혼을 했다. -> 첫문장이 참이면 두번째 문장도 참 ,
          - 그런데 두번째 문장이 거짓이라고 하면 첫번째 문장도 당연히 거짓
        - 
        - `questoin answereing` : 구글에 어떤 질문을 검색하면 그 질문이 포함된 키워드를 결과창에 뛰우지않고, 질문자체에 답을함.
        - dialog systems : 챗봇 대화 
        - summarization : 주어진 문서를 요약

## text mining 
  - 빅데이터 분석과 관련된 경우가 많음
  - 과거 1년 간의 뉴스기사를 모아서 거기에 나타난 특정 단어의 빈도수를 분석하거나 기사의 트렌드를 분석. 어떤 유명인의 이미지가 -> 사건이 터져서 어떻게 안좋게 이미지가 변했는지,
  - 어떤 상품이 출시 되었을 때 그 상품에 대한 소비자 반응을 얻음
  - `topic modeling `
      - 어떤 상품에 대해 사람들이 세부적인 요소를 이야기 하고있다면 그중 어떤것이 유용한 정보인지 알아낼 수 있다.
  - twiter나 facebook을 분석해서 현대인들의 패턴 분석, 사회과학분야에서의 인사이트 발견
    
## Inforamtion retrieval
  - 검색기술 연구
  - 현재 검색기술의 성능은 점차 고도화되면서 검색기술은 어느정도 완성됬다고 할 수 있음.
  - 따라서 기술발전도 앞서 말한 텍스트마이닝이나 nlp보다 느림 (거의 완성이니까)
  - 그러나 검색 기술에서도 추천 시스템 분야는 좀 활발하게 발전됨 .
    

## Trends of NLP

- 대부분의 머신러닝 기술은 숫자로 이뤄진 데이터를 필요로 하기때문에
텍스트 데이터를 먼저 단어단위로 분리하고, 단어를 벡터화 시켜줌  
그래서 워드를 단어를 벡터화 시키는 것을 `워드 임베딩`이라고 한다.  
<br/>

- 그리고 단어들의 순서가 중요하기 때문에 시퀀스 데이터를 처리하는데 특화된 모델 구조로 `RNN`이 자연어 처리의 핵심모델로 자리잡음
RNN 계열중에서 `LSTM`과 `GRU (LSTM을 단순화시킨 모델)` 모델이 많이 사용됨



- 2017년 구글에서 발표한 `attention is all you need` 이후
기존의 RNN 기반의 자연어 처리 모델의 구조를 `self-attetion` 모듈로 완전히 대체할 수 있는 `transformer` 가 등장 -> 성능도 다좋음
> 요즘 그래서 대부분 자연어 처리에서는 `transformer` 를 사용

- Transforemr는 원래 기계번역을 위하 등장.
- 기계번역을 위해서 많은것들을 직접 설정해줘야했지만 언어의 다양한 패턴에 일일히 대응하는것에 한계가 있었음 
> -> 딥러닝 기반으로 기계번역을 해보니 성능이 월등히 올라감. 
> 구글,파파고 -> 기계변역 

기계번역에서  새로운 성능 향상을 가져온게 `transformer`
tranformer은 처음에는 기계번역을 위해 만들어졌지만 시계열 ,영상처리, 신물질 개발등 다양한 분야에 활발히 적용중 


- 자연어 처리에서 self- superviser leanring 은 입력문장이 주어질때 입력문장중 한 단어를 가리고 그 단어를 맞출수있게, 하는 task 에 해당.
  - 대표모델 bert, gtp - 3 등

이것들은 특정 task에서만 적용이 가능한게 아니라 범용으로 적용가능하다
따라서 인공지능 기술은 한단계 발전을 거듭하게됨.

- 최근 기술적 트렌드로볼때 self- superviser leanring을 위해서는 대규모 데이터와, gpu 가 필요함
- 그래서 facebook, google등에서 함 (대규모 자산 필요->개인이 힘듦)

## bag-of words
--- 
1.  문장이 주어졌을때 문장을 단어단위로 나눠서 사전을 만듦
2.  사전을 원-핫 인코딩 해준다.
    - bag-of-words는 각 단어간 거리는 루트 2
    - 코사인 유사도는 0
3. 문장이나 누서를 대표하는 원핫 벡터는 다음과 같이 나타낼 수 있음.
![bag_of_word_1.PNG](bag_of_word_1.PNG)

## NaiveBayes Classifier for Document Classification
---
![NaiveBayes_1.PNG](NaiveBayes_1.PNG)
![NaiveBayes_2.PNG](NaiveBayes_2.PNG)



## word embedding
---
- 단어를 벡터로 변환해주는 기법
- 비슷한 의미를 가지는 단어가 좌표상에서 비슷한 위치에 위치하게됨.


### word2 vec
---
- 비슷한 의미를 가진단어가 좌표공간상에서 가까운 위치에 표시
- 가까운 위치에 있는 단어끼리 비슷할것이라는 가정을 사용
![word2vec.PNG](word2vec.PNG)

https://wikidocs.net/22660

### prorperty of word2vec - Intrusion Detection
---
- 여러단어가 주어질떄 나머지 단어와 의미가 가장 상의한 단어를 찾아내는것
- word2vec의 임베딩 벡터 활용
    - 유클리드거리를 구해서 평균 거리가 가장 먼 단어 를구함
    - 그단어가 가장 의미가 상의한 단어


## Glove : word embedding model
---
- word2vec과 가장 큰차이점은 loss functio을 달리함 ?
- 특정한 입출력 쌍이 자주등장한ㄴ경우
- word2vec그경우 그 데이터 아이템이 자연스레 여러 번학습됨 -> 내적값이 즈악
- 중복되는 계산 줄여줌 -> 학습이 더빠름 -> 적응량의 데이터에도 적합
? ? ?

성능도 비등비등 둘다 임베딩 알고리즘

## CBOW (Continuous Bag-of-Words)
---
- 주변 단어들을 가지고 중심 단어를 예측하는 방식으로 학습합니다.
- 주변 단어들의 one-hot encoding 벡터를 각각 embedding layer에 projection하여 각각의 embedding 벡터를 얻고 이 embedding들을 element-wise한 덧셈으로 합친 뒤, 다시 linear transformation하여 예측하고자 하는 중심 단어의 one-hot encoding 벡터와 같은 사이즈의 벡터로 만든 뒤, 중심 단어의 one-hot encoding 벡터와의 loss를 계산합니다.
- 예) A cute puppy is walking in the park. & window size: 2
    - Input(주변 단어): "A", "cute", "is", "walking"
    - Output(중심 단어): "puppy"
---    
## Skip-gram

- 중심 단어를 가지고 주변 단어들을 예측하는 방식으로 학습합니다.
- 중심 단어의 one-hot encoding 벡터를 embedding layer에 projection하여 해당 단어의 embedding 벡터를 얻고 이 벡터를 다시 linear transformation하여 예측하고자 하는 각각의 주변 단어들과의 one-hot encoding 벡터와 같은 사이즈의 벡터로 만든 뒤, 그 주변 단어들의 one-hot encoding 벡터와의 loss를 각각 계산합니다.
- 예) A cute puppy is walking in the park. & window size: 2
    - Input(중심 단어): "puppy"
    - Output(주변 단어): "A", "cute", "is", "walking"



```python

```


```python

```


```python

```
