## Entropy

엔트로피란 불확실성(uncertainty)에 대한 척도이다. 큐브를 보면, 모든 면이 제대로 맞추어진 경우의 수는 1이지만 이를 흐뜨러뜨리기 시작하면 그 수를 바로 가늠하기 어려울정도로 많아진다. 이 때 큐브 입장에서 나중의 엔트로피가 더 높다고 볼 수 있다. 그 불확실성의 정도가 커지기 때문이다.

따라서 엔트로피의 경우 예측하기 어려운 일에서 더 높아진다. 예를 들어 주사위와 동전이 있을 때 엔트로피 식 $H(x)=−\sum _{i=1}^{n}  p(x_i) \log{p(x_i)}$
에 따라 그값은 아래와 같다.  
<br/>

$H(x)= -\left( \frac{1}{2} \log{\frac{1}{2}} + \frac{1}{2} \log{\frac{1}{2}} \right) \approx 0.693$  
$H(x)= - \left( \frac{1}{6} \log{\frac{1}{6}} + \cdots + \frac{1}{6} \log{\frac{1}{6}} \right) \approx 1.79$  

보다시피 불확실성이 더 큰 주사위의 엔트로피가 더 높다. 여담으로 위와 같이 모든 사건이 일어날 확률이 동일한 경우, 엔트로피는 결국 
$H(x) = -\log p(x_i)$가 된다. 즉, 모든 사건의 확률이 uniform하면 일어날 수 있는 사건의 갯수가 많을수록 엔트로피 값도 크다.

## Cross-Entropy

cross-Entropy는 모델링을 통해 얻어낸 분포 $p$를 통해 실제 데이터의 분포 $q$를 예측하는 식이다. 
$H_p (q)=−\sum_{i=1}^{n}  q(x_i) log{p(x_i)}$  
예를 들어, 예측을 하고 싶은 주머니에 0.8/0.1/0.1의 비율로 삼색구슬이 들어있다고 하자. 그런데 우리가 가지고 있는 샘플 데이터상으로는 0.2/0.2/0.6의 비율로 색구슬이 들어있엇다. 그렇다면 실제 엔트로피와 $p$에 대한 $q$의 크로스 엔트로피는 아래와 같다.  

### $H(q) = -[0.8log(0.8) + 0.1log(0.1) + 0.1log(0.1)] = 0.63$
### $H_p(q) = -[0.8log(0.2) + 0.1log(0.2) + 0.1log(0.6)] = 1.50$  
cross entropy는 감소하는 방향으로 학습을 진행할 것이다.  

그런데 위와 같은 크로스 엔트로피 값을 구하려면 우리는 구하고자하는 task에 대한 확률분포를 미리 알고 있어야 한다. 당연히 우리는 그걸 모르기 때문에 에측을 하고 있는 것이다. 그래서 우리는 $q$의 자리에 위 주머니 예시처럼 실제 분포를 넣지는 못한다.  
<br/>

모델을 학습할때는 실제 데이터의 분포, 즉 **q가 바로 학습 데이터의 레이블이 된다.** 우리의 목표는 **학습 데이터를 통해 얻어낸 모델의 예측 p의 분포가 가지고 있는 데이터의 레이블 q와 비슷해지도록 학습을 진행하는 것**이다.  
그래서 cross-entropy를 loss로 사용할 때 label로 원-핫 벡터를 활용했던 것이, 괜히 사용한게 아니고 이게 확률에 대한 식이므로 해당 label에 확률 1을 몰아주는 것이었다고 이해해볼 수 있다.
<br/>

overfitting 현상이 어떨 때 발생하는가? 너무 trainig dataset에만 초점이 맞춰진 학습이 진행될 때 발생하게 된다. 다른 것들도 그렇지만, cross entropy라는 손실 함수 역시 계속 학습을 진행하다보면 모델링으로 얻은 예측 $P$의 확률분포가 학습 데이터의 레이블 $q$의 확률 분포랑 너무 똑같아져버려서 이와 같은 현상이 발생하게 된다.

## Kullback-Leibler Divergence (KL divergence)

KL Divergence는 이전에도 다루었지만, **서로 다른 두 분포 간의 차이를 계산하기 위해** 이용한다.  식을 간단한 형태로 보면 아래와 같다.  
### $\begin{aligned} D_{KL}(q \Vert p) &= -\sum_{c=1}^{C} q(y_c) [log(p(y_c)) - log(q(y_c))]  \\&= H_p(q) - H(q)\end{aligned}$  
사실 cross-entropy 식만 봐서는 이게 분포 간 차이를 최소화하는 것이 맞는지 직관적으로 와닿지는 않지만, `KL Divergence`를 보면 그 의미가 더 확실해진다.  
<br/>

일단 **cross-entropy는 반드시 entropy보다는 크다는 것을 기억하자.**Jensen’s Inequality(젠센 부등식)을 이용하여 이를 증명할 수 있으나, 당장 직관적으로 봐도 학습을 왜하는지에 대해 조금만 생각해보면 cross entropy가 entropy보다 클 수가 없다.  
<br/>

따라서 KL-Divergence의 값은 반드시 양수이며 따라서 이를 0에 가깝게 만드는 것이 $p$와 $q$의 분포를 근사시키는 일과 같다.  그런데 뒷항 H(q)는 q의 엔트로피라서 모델의 파라미터를 수정해도 건들 수가 없다. 결국 중요한 것은 모델을 건드리면 값이 바뀌는 $H_p(q)$이며, 학습이 cross entropy를 최소화하는 방향으로 진행되는 이유를 여기서 찾을 수 있다.  
<br/>

한편 KL-Divergence는 두 분포간의 거리를 찾는 방법 중 하나라고 잘 알려져있지만, 실제로는 식 자체가 asymmetric(비대칭적)이라서 거리라고 부르기 어렵다. 대신, Jensen-Shannon divergence라는 방법이 있는데 두 KL divergence의 평균을 내는 방식이다. 이 방법을 이용하여 얻은 값은 거리에 가까운 의미를 가진다.
<br/>

### $JSD(p\|q) = \frac{1}{2}KL(p\|M)  + \frac{1}{2}KL(q\|M)$



잘 쓰이지는 않는다고 하나, 그 유명한 GAN loss에도 Jensen-Shannon divergence가 내포되어있다.
<br/>

위와 같이 cross-entropy의 의미를 찾을 때 KL-Divergence를 살펴볼 수 있는 한편, 실제 여러 모델의 loss에도 KL Divergence가 직접적으로 관여하는 경우가 많으므로 이 개념에 대해 잘 숙지하고 넘어가도록 하자.

