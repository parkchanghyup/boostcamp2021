# 수업복습
---

## Generative Models
---
참고 : https://deepgenerativemodels.github.io/

- genereative model ? 

무언 가를 만들어 내는 모델이지만 단순히 그것이 전부는 아님 .


강아지 이미지를 우리가 가지고 있다면 우리는 샘플링 하는 모델을 만들고 싶을 수 있다.
거기에 더해 우리는 강아지 같은 데이터를 더 만들어 낼수도 있고,
- 어떤 이미지가 강아지인지 아닌지도 구분해 내줌 (explicit model)
- 비지도 학습의 대표 모델 ?



- Bernoulli distribution
    - 숫자가 한개 필요 ( 앞이 나올 확률(P) 한개만있으면  뒷면이 나올확률은 1- p로 구할 수 있음)
- Categorical distribution
    - 6개의 주사위를 던지려면 5개의 파라미터가 필요 나머지 하나의 확률은 1에서 다빼주면됨 
    


- 숫자를 분류하는 이진 분류 문제의 경우의 수는 2 x 2 x ... x 2 = $2^n$
- 파라미터의 숫자는 $2^n$ - 1 개 
 
기계학습에서 파라미터의 숫자가 많아 질 수록 학습은 더욱 어렵다. -> 파라미터를 줄여야됨 

### 각각의 표본이 독립이라고 가정 ?(Markov assumption)
각각의 표본이 독립이라고 가정하면 음 표현할 수 있는 경우의 수는 $2^n$ 으로 똑같지만,
파라미터의 갯수는 n개로 줄어듦 
그러나 이러한 가정은 좀 말이 안됨 그래서 중간지점을 찾아내야됨 .


## Condition Independence
---
- 3개의 중요한 룰
    - Chain rule
    - Bayes' rule:
    - Conditional independece 
        - $if$  $x \perp y|z$,   $then$   $p(x|y,z)$   =   $p(x|z)$

chain rule 을 쓰면 몇개의 파라미터 필요 할까?
$p(x_1,...,x_n) = p(x_1)p(x_2|x_1)...p(x_n|x_1,...,x_{n-1})$

- $p(x_1)$ = 파라미터 한개
- $p(x_2|x_1)$ = 파라미터 2개
- $p(x_3|x_1,x_2)$ = 파라미터 4개
- 결론적으로 $1 + 2+ 2^2+...+2^{n-1} = 2^n-1$ 개의 파라미터 필요

여기에 Markov assumption 를 적용하면 몇개의 파라미터 필요 ?

$p(x_1,...,x_n) = p(x_1)p(x_2|x_1)p(x_3|x_2)...p(x_n|x_{n-1})$
-> ```2n-1``` 개 필요


그래서 이런 방법을 Auto-regressive model 이라고 부름.


## Auto-regressive Model
---

- 우리의 목표는 28 x 28 픽셀을 학습시키는 거다
- 얘를 어떻게 $p(x)$ 로 표현 할 수 있을까 ?
    - chain rule을 가지고 join distribution으로 나누고
    - p(x_{1:784} = p(x_1)p(x_2|x_1)p(x_3|x_1:2)...
    - 이게 autoregressive model
    - 그리고 임의의 데이터를 이용하려면 순서를 매겨야함
    - 근데 이미지 순서를 매기는 거에 따라 전체 모델의 구조나 성능이 달라짐 .
    
    

### NADE : Neural Aitoregressive Denstiy Estimator
---

i번째 픽셀을 첫번째부터 i-1번째 픽셀까지 의존 되게 하고 
두번째 픽셀에 대한 확률을 첫번째 픽셀에만 의존되게
첫번째 픽셀값을 입력으로 받는 뉴럴네트워크를 만들고 싱글 스칼라가 나온다음에 시그모이를 통과해서 0~-1로 바꾸고 
다섯번째는 1~4번째에대한 값을 다바꿔서 nn을 통과해서 sigmode를 통과시켜서 나온 확률 값을 사용


i 번재 픽셀은 i-1개의 픽셀에 의존 ? - > 3번째 픽셀ㅇ된 nn만들때는 2개의 입력을 받는 weight가 필요 100번째는 99개의 입력을 받을 수 있는 nn가 필요.

- NADE는 explict model 임 
    - n개으 ㅣ픽셀이 주어지면 
    - 우리의 모델이 첫번째 픽셀에 대한 확률 분포를 알 고 있고 
   첫번째를 알면 두번째를 알수있다.
    - 그래서 이값들을 다곱하면 작지만 확률 값을 하나도출 할 수 있음.
- 연속형 확률 분포일때는 gaussian mixture 모델을 활용하면됨 -> 어떻게 ?

### Pixel RnN
--- 
- n x n image 가 있을때 
R G B 를 순서대로 만듦

앞에서 봤던거는 fully connected model을 통해서 만들 었고 
근데 pixel rnn은 rnn을 통해서 generate 하겠다는 차이 ?
order를 어떻게 하냐에 따라
- row lstm
- diagnal biLstm 으로 나눔

row는 i번쨰 픽셀을 만들떄 그 위쪽에 있는 정보를 확ㄹ용
dia는 자기 이전 정보를 다 활용 .




## Latent Variable Models
---
autoencoder는 generative model 은

### GAN
--- 
- 


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
