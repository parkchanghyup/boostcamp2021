## 가지치기(Pruning)

사람의 시냅스는 태어난 직후 가장 저고, 영유아기에 가장 많았다가 이후 **필요 없는 시냅스들이 없어지고** 성인이 되서는 시냅스의 수가 어느정도 안정된 상태로 유지된다.  
<br/>

딥러닝에 쓰이는 여러 기법들이 그러하듯, pruning 기법도 실제 사람의 신경망과 같은 맥락에서 동작한다. 여러 가중치(parameter)들 중 그 중요도가 낮은 것들을 잘라내어 정확도는 비슷하게 유지하는 한편, 속도 및 메모리를 최적화 하는 기법이다.  
<br/>

1989년에 처음 그 concept이 제시된 이후, 2015년 `Learning both weights and connections for efficient neural network` 라는 논문에서 딥러닝에서의 pruning의 포문을 열었다.    

![](https://olenmg.github.io/img/posts/38-4.png)

그 형태는 앞서 설명했듯이 중요도가 낮은 뉴런/시냅스들으 죽이는 방식으로 이루어진다. 드롭아웃과 어떻게 보면 비슷하지만, 차이점을 보자면  
1. drop out은 매번 없앤 뉴런을 다시 복원시키지만 pruning은 그렇지 않다.
2. pruning은 inference도 뉴런을 없앤 상태로 진행하지만 drop out은 역시 그렇지 않다.  
보통 이렇게 해서 몇 개의 뉴런을 없애면 parameter수를 급격히 떨어뜨릴 수 있다. 그런데 없어진 paramter의 분포를 보면 그 형태가 꽤 흥미롭다.  
  

![](https://olenmg.github.io/img/posts/38-5.png)

위와 같이 pruning을 적용하면 대부분 양상이 비슷한데 **0주 변의 파라미터들은 급격히 감소하게 된다** 즉, 앞서 말한 중요도를 측정할 때 0 주변의 파라미털르 우선적으로 없애는 것이 정확도를 살리면서 parameter수를 감소시키는 데에 큰 도움이 될 것이라는 점을 알 수 있다. 그리고 그냥 직관적으로 생각해봐도 값이 0 주변인 파라미터는 모델의 결정에 큰 영향을 안주기 때문에 이를 제거해도 큰 문제가 없을 것이다. 그래서 보통은 **magnitude(absolute value of parameters)를 파라미터의 중요도를 측정하는 데에 많이 활용** 한다.  
<br/>

Pruning 기법도 지금까지 여러가지가 제시되어왔다. 앞서 말한 2015년에 제시된 가장 기초적인 형태는 아래와 같다.    

![](https://olenmg.github.io/img/posts/38-6.png)

1. original 네트워크를 학습
2. 중요도에 의거하여 마스크를 씌운 모델을 만들어 fine-tuning을 한다.
3. 1과 2를 반복한다.  
여담으로 원 논문에서는 여기어 L1 norm/L2 norm을 적용했을 때 더 성능이 좋았다고 한다.  
<br/>

아무튼 지금에 와서도 많은 pruning 기법들이 결국 위 방법에서 조금의 변형을 거쳐서 개발되고 있다. 다만 pruning이라는 기법의 특성상 당연히 많이 가지치기를 할 수록 정확도는 감소하는 `trade-off`의 경향성이 있어 정확도 감소는 최소화하되 parameter는 줄이는 것이 여러 연구의 목표가 될 것이다. 


# Reference 

[Blalock, Davis, et al. “What is the state of neural network pruning?” (2020)](https://arxiv.org/pdf/1803.03635.pdf)  
[Frankle, Jonathan, et al. “Linear Mode Connectivity and the Lottery Ticket Hypothesis” (2020)](https://arxiv.org/pdf/1912.05671.pdf)