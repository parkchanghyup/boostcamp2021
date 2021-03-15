# Object Detection
---
![](https://olenmg.github.io/img/posts/33-1.png)

Object Detection의 주요 목적은 주어진 물체 전부에 대한 구분선을 긋는 것이다.  
더 진보된 형태로는 instance segmentation, panoptic segmentation 등이 있는데 여기서는 **category 뿐만 아니라 instacne도 구분해낸다.**  
panoptic segmentation은 instance segmentation에서 좀더 진보된 형태이다.  

<br/>

보통 사진은 2차원 형태이므로 bounding box는 왼쪽 위 꼭짓점과 오른쪽 아래 꼭짓점의 x,y 좌표, 그리고 해당 물체의 class 까지 총 5가지 정보를 저장 하게 된다.   

<br/>

object dection은 regional proposal과 classification 단계가 분리되어 있는 two-stage detection과 별도의 regional proposal 추출 단계 없이 한번에 detection이 이루어지는 `one-stage detecion`으로 나눌 수 있다.

## Two-stage detctor

two-stage detctor의 앞단에서 regional proposal을 제시 해줄 수 있는 알고리즘에 대해 알아 보자.
![](https://olenmg.github.io/img/posts/33-2.png)
고전적인 알고리즘의 대표적인 예로 HOG 방법이 있다. 이미지의 local gradient를 해당 영ㅇ상의 특징으로 활용하는 방법이다. 

<br/>

간단하게만 보면, 이 알고리즘에서는 픽셀별로 x축, y축 방향 gradient(edge)를 계산하고 각 픽셀별 orientation을 hitogram으로 표현한다. 그리고 인접한 픽셀들끼리 묶어 블록 단위 특징 벡터를 구한다. 이렇게 하면 의미있는 특징을 가진 regional proposal을 구할 수 있게 된다.

<br/>

HOG 알고리즘을 제시한 논문에서는 보행자 검출을 수행하였는데, 보행 중인 사람과 가만히 있는 사람 두 가지 클래스를 SVM을 통해 분류하였다.

![](https://olenmg.github.io/img/posts/33-3.png)

한편, regional proposal을 위한 또다른 알고리즘으로 `Selective search(SS)`가 있다. 여기서는 먼저 비슷한 특징값(질감, 색, 강도 등)을 가진 픽셀들을 sub-segmentation한다. 그래서 초기에는 아주 많은 영역을 생성해낸다. 그 다음 greedy algorithm으로 작은 영역을 큰 영역으로 통합해 나간다. 여기서도 역시 비슷한 특징값을 가진 것들이 우선순위이다. 이렇게 후보군이 1개가 될 때까지 반복한다.

<br/>

그럼 이제 regional proposal을 만드는 방법을 알아보았으니 classfication 방법론들에 대해 알아보자. 여기서는 근본적으로는 딥러닝, 즉 CNN을 활용한 기법들에 대해 알아 볼 것이다. 

<br/>

아래 제시하는 초기 two-stage 방법론들은 regional proposal 단계에서 selectiv search를 활용하였다. 다만 two-stage detector도 후속 모델들은 위에서 제시한 알고리즘 없이 `regional proposal`을 스르로 찾아낸다.

## R-CNN 

![](https://olenmg.github.io/img/posts/33-4.png)

R-CNN 모델에서는 먼저 regional proposal을 뽑아낸다. 박스 후보군은 다른 말로 Rol(region of Interest)라고 한다.

<br/>

다음으로, 추출한 Rol들을 CNN에 통과시키기 위해 모두 동일 input size로 만들어 준다. 물론 CNN의 input은 가변적일 수 있지만 여기서는 최종 classification을 위해 FC layer를 활용하였기 때문에 고정된 input size가 필요하다.

<br/>

마지막으로 Warped Rol image를 각각 CNN 모델에 넣고 출력으로 나오는 feature vector로 svm classification을 통해 결과를 얻는다. 여기서는 먼저 이 bounding box가 객체가 맞는지 판별하고 객체가 맞다면 어떤 객체인지까지 판별하는 역할을 수행한다.

<br/>

추가적으로 SS algorithm으로 만들어진 bounding box가 정확하지 않기 때문에 물체를 정확히 감싸도록 만들어주는 bounding box regression(선형회귀 모델)도 마지막에 활용 한다.

<br/>

이 방법론이 제시될 당시에는 데이터가 비교적 적었는지 softmax를 통한 분류보다 SVM이 더좋은 성능을 보였기 때문에 svm이 사용되었다. 그런데 현재와 같이 데이터가 많을 때 R-CNN 방법론을 다시 활용 한다고 하면 softmax와 SVM중 어떤 방법이 더 좋은 정확도를 보일지도 긍금해지는 대목이다.

<br/>

이 방법은 Rol 수가 2000개가 넘기 때문에 CNN을 너무 많이 돌려야 하고 무엇보다 CNN, SVM, bounding box regression 각각의 pipeline이 다르기 때문에 이를 end-to-end로 학습시키는 것이 불가능하다는 단점이 있다.  

그래서 이 두문제를 해결한 Fast R-CNN이 등장하게 된다.

## Fast R-CNN

Fast R-CNN은 Rol pooling 기법을 활용하여 이 두문제를 해결하였다.  

![](https://olenmg.github.io/img/posts/33-5.png)  
여기서도 먼저 SS를 통해 Rol을 찾는다. 그 **다음 전체 이미지를 먼저 CNN에 통과**시켜 전체에 대한 feature map를 얻는다. 그리고 이전에 찾았던 Rol을 feature map 크기에 맞춰 projection 시킨 후 여기에 Rol Pooling을 적용하여 각 Rol에 대한 고정된 크기의 feature vector를 얻는다.

<br/>

마지막으로 이 feature vector 각각을 FC layer에 통과시킨 후 이번에는 **softmax로 분류**를 하고 앞서 언급한 **bounding box regression**도 함께 적용하여 최종적인 bounding box 및 class 분류 출력을 내놓는다.

<br/>

방금 중간에 각 Rol에 대한 고정된 크기의 feature vector를 얻어내는 과정이 있었다. 즉, 여기서는 input size에 대한 제약을 타파하여 warp 과정이 없어진다.

<br/>

좀 더 자세히 보면, 이 부분에서는 max pooling이 되는데 정확히는 **고정된 크기의 출력이 나오게끔 max pooling이 된다.** 이 부분은 우리가 원래 알던 CNN의 max pooling과 조금 다르다.

![](https://olenmg.github.io/img/posts/33-6.png)

맨 왼쪽 그림이 feature map이고 검은 바운딩이 Rol일 때 **고정된 크기의 출력 $H$ x $W$가 나오게끔 max pooling을 해야한다.

<br/>

만약 Rol 크기가 h x w이면 $H$ x $W$의 feature를 얻기 위해 Rol를 $\frac{h}{H} \times \frac{w}{W}$ 크기만큼 grid를 만들어 각 그리드에서 max-pooling을 수행하게 된다.  
결국 의도된 크기로 풀링이 되었기 때문에 고정된 크기의 feature vector를 얻을 수 있게 된다.  

<br/>

요약하면 CNN 연산이 1번밖에 사용되지 않아 연산량 측면에서 이점을 확실히 가져갔으며 **CNN을 먼저 적용하고 warp 없이 Rol를 projection시키고 연산한 것이라서 calssification 단계에서 end-to-end 학습이 가능**하다.

<br/>

다만 아직 regional proposal을 별도의 알고리즘을 통해 수행하기 때문에 완벽한 end-to-end 학습이 불가능하다는 단점이 있다. 또한 Fast R-CNN은 사실 Rol 추출 단계가 대부분의 실행시간을 차지한다는 점이 critical 하므로 이 부분에 대한 개선이 필요하다. 따라서 이를 해결한 Faster R-CNN 모델이 제안된다.

## Faster R-CNN

Faster R-CNN은 ROl 추출을 위해 RPN(Region Proposal Network)단을 도입한다. 참고로, 그 뒷단은 Fast R-CNN과 완전히 동일하다. 물론 학습이 end-to-end로 이루어지기 때문에 region proposal을 찾는 단과 calssfication을 수행하는 단이 동시에 학습된다는 차이점도 있다.  

아무튼 그래서 RPN에 대해 알아보자  

![](https://olenmg.github.io/img/posts/33-8.png)  
여기서는 **anchor box**라는 개념이 잇는데 그냥 미리 rough하게 정해놓은 후보군의 크기 정도로 이해하면 된다. hyperparameter이며, 원 논문에서는 박스의 scale과 비율을 각각 3종류씩 주어 총 9개의 ancohr box를 활용하였다.  

<br>

Faster R-CNN에서는 미리 학습 데이터를 정하게 되는데, 모든 픽셀에 대하여 anchor box를 다 구해놓고 **ground truth와의 IoU score를 계산하여 0.7보다 크면 positive sample, 0.3보다 작으면 negative sample로 활용하게 된다.  

<br/>

IoU(Intersection over Union) score는 주어진 두 영역의 `교집합 영역 / 합집합 영역` 이다. 즉 영역의 overlap이 많으면 이 score가 높게 나오게 된다. 참고로 IoU score가 0.3에서 0.7사이인 샘플은 학습에 도움이 되지 않는다고 판단하여 위 논문에서는 이를 학습에 활용하지 않는다.  
![](https://olenmg.github.io/img/posts/33-7.png)

RPN의 input은 CNN을 먼저 통과한 feature map이다. input에 3 x 3 conv를 하여 256d로 channel을 늘린 후 cls layer와 reg layer에서는 각각 1 x 1 conv를 통해 2k, 4k channel을 가지는 feature를 얻게 된다.(k는 anchor box의 수이다) cls layer의 output 2k개에서는 해당 위치의 k개의 anchor가 각각 개체가 맞는지 아닌지에 대한 예측값을 담고, reg layer의 output 4k개에서는 box regression 예측 좌표 값을 담는다.

<br/>

또한 논문에서 제시하는 바로는 그냥 cls layer에서 logistic regression을 적용하기 위해 output k chaanel로 conv를 수행해도 된다고 한다. 

<br/>

여기까지 하고 나면 대충 물체(object)가 맞는지에 대한 확률 값을 알 수 있게 되는데, 이제 이를 내림차순으로 정렬한 후 높은 순으로 K개의 ancohr만 추려낸다. 그 다음 K개의 ancohr에 대한 box regression을 해준다. 그러면 이제 개체일 확률이 높은 anchor에 대한 위치 값까지 알게 되는데 여기서 `Non-Maximum Suppresion(NMS)`를 적용한다.  
![](https://olenmg.github.io/img/posts/33-9.png)  

NMS는 후보군 anchor box에서 허수/중복 데이터를 필터링하는 역할을 한다. 여기서는 IoU가 0.7이상인 anchor box에서 확률 값이 낮은 박스를 지워낸다. 논문에서는 NMS를 적용해도 성능 하락이 얼망 없었지만 효과적으로 proposal의 수를 줄 일 수 있다고 언급하고 있다.

<br/>

이렇게까지 하면 최종적인 anchor 후보군을 선정할 수 있게 된다. 남은 것은 앞서 본 Fast R-CNN을 위에서 만들어낸 proposal에 적용하는 것 뿐이다.

<br/>

R-CNN family의 구조를 전체적으로 요약하면 아래와 같다.
![](https://olenmg.github.io/img/posts/33-10.png)


# Single-satge detector(One-stage detection)

single stage detector는 정확도는 조금 떨어질 수 있지만 그 속도가 매우 빠를 것으로 예상할 수 있다. two-stage detector에서는 region proposal에 대한 Rol pooling 단계가 필요했지만 역시서는 그런 단게 없이 곧바로 box regression과 classification을 수행 하게 된다. 다만 최근 연구에는 two stage detecion임에도 속도가 빠르다고 기술되어있는 thunderNet 등의 architecture도 있다. 따라서 속도든 정확도든 어느 것이 우워하다라는 것을 이분법적으로 가리기는 어려울듯 하다.

자세한 내용은 [블로그](https://olenmg.github.io/2021/03/10/boostcamp-day33.html) 참고. . 하면 좋을거 같다.

# CNN visualization

신경망 모델을 설계하고 성능을 평가하고 나면 왜 성능이 이렇게 나오는지, 어떻게 하면 모델의 성능을 개선할 수 있을지 쉽게 감이 잡히지 않는다. 왜냐하면 신경망 모델은 대부분 깊게 쌓여있고 그 가중치를 하나하나 쳐다보면서 의미있는 결과를 도출해내는 것은 현실적으로 어렵기 때문이다.

<br/>

그래서 우리는 이 신경망 내부에서 어떤 일이 벌어지고 있는지 살펴보기 위해 visualization 기법을 활용해 볼수 있다. 여기서는 그중에서도 CNN의 Visualization 방법에 대해 알아 보자.  
![](https://olenmg.github.io/img/posts/33-17.png)

CNN은 위와 같이 level별 feature가 담고있는 정보가 다르다.(물론 다른 신경망들도 레이어 깊이별 나타나는 정보가 다르다) 따라서 CNN에서는 각 레벨별로 어떤 feature를 분석해볼 수 있는지 고려햐 보야아 한다.

<br/>

신경망을 시각화 한다고 하면 크게 model beghavior(행동) 분석을 위한 시각화와 model decision(결과) 분석을 위한 시각화로 나뉜다.

![](https://olenmg.github.io/img/posts/33-16.png)

## Analysis of model behaviors

먼저 high level feature(분류 직전)를 살펴보도록 하자. 가장 간단하게 여러 이미지의 마지막 feature들의 Nearest Neighbors를 생각해볼 수 있다. feature들간의 거리를 측정하고 거리가 이웃한 이미지들이 사람이 직접 봐도 비슷한 모습을 보이는지, 속하는지 살펴보겠다는 뜻이다.

<br/>

feature들 간의 거리는 그냥 계산해도 되지만, 고차원 벡터는 시각화하기 어렵다는 단점이 있다. 그래서 feature들을 이해하기 쉬운 2차원 공간에 나타내기 위한 시도가 존재하였다.

<br/>

feature의 차원 축소를 위한 많은 기법이 있지만, 그 중에서도 `t-SNE(t-distributed stochastic neighbor embedding)`가 가장 좋은 임베딩을 보이는 것으로 알려져 있다. 이를 통해 우리는 2차원 평면에서 여러 피처들의 거리를 직접 보면서 모델이 보는 이미지간 유사도와 사람이 보는 이미지간 유사도의 차이를 생각해볼 수 있다.  
![](https://olenmg.github.io/img/posts/33-18.png)  

위 그림은 t-SNE를 이용한 MNIST dataset의 시각화 모습이다. 사람이 보기에도 비슷하게 생긴 숫자 3과 8이 모델이 내놓은 feature space에서도 비슷한 거리에 위치하고 있음을 확인할 수 있다.  

<br/>

다음으로, mid level과 high level 사이에서 나오는 featur맵을 가지고 layer activation을 생각해볼 수 있다. 여기서는 보고있는 레이어의 출력(hidden node)에서 각 채널이 이미지의 어느 부분에서 활성화가 되는지 살펴본다.  
![](https://olenmg.github.io/img/posts/33-19.png)  

그 결과 위와 같이 hidden node에서 channel들 각각이 어느 부분을 중점적으로 보는지 확인할 수 있다. 이를 통해 CNN은 중간중간 hidden layer들이 물체를 여러 부분으로 나누어 보고 이들을 조합하여 물체를 인식하는 것이라고 추측할 수 있다.  

<br/>

또 다른 방법으로 `maximally activation patches`, 즉 활성화된 것들 중에서도 최대 크기로 활성화된 뉴런의 주변을 시각화해볼 수 있다.  
이 방법은 (1) focus할 특정 레이어와 채널을 정하고 (2) 여러 이미지를 통과시킨 후 (3) **activation이 최대인 부분의 receptive field를 crop해온다**

![](https://olenmg.github.io/img/posts/33-20.png)

위 그림은 hidden node 별 activation이 큰 값을 가진 patch를 가져온 모습이다. 각 hidden node별로 활성화가 잘되는 형태가 다른 것을 알 수 있다. 어찌보면 각각의 hidden node가 자신이 담당하는 특정 모양을 찾는 detector 역할을 하고 있다고 볼 수 있다. 

<br/>

마지막 방법으로, 데이터를 넣어주지 않고 모델 자체가 내재하고있는(기억하고 있는)이미지를 분석하는 `class visualizaion` 방법이 있다.  
여기서는 원하는 class의 이미지가 나오도록 **입력값을 최적화 해준다.**
이것은 사실살 인공의 이미지를 generate 하는 과정이다. 

$I^{*} = \underset{I}{\text{argmax}} \, S_C(I) - \lambda \Vert I \Vert ^2 _2$  
결국 우리가 원하는 것은 이미지 $I$ 이고 이를 모델이 원한느 클래스 $C$로 분류하기를 원하므로 위 식에 따라 $I$를 최적화 해주면 우리가 워하는 이미지를 얻을 수 있다. 뒤에는 L2 Regularization이 들어가는데, 이는 이미지에 극단저거인 값이 들어가는 것을 방지하기 위해 넣어준다.  

<br/>

maximize하는 과정이므로 `gradient ascent`가 쓰인다. 다만, 부호를 반대로 해주면 당연히 gradient descent과정이 될 것이고 우리가 전에역전파로 하던 것처럼 최적화를 해주면 된다.  

<br/>

방법을 좀더 자세히 살펴보면, 처음에는 그냥 blank/random image를 첫 input으로 넣고 output을 도출해 낸다. 이후 **모델은 학습하지 않고, input에 gradient를 전달하여 이미지를 수정해나간다.** 물론 목표함수를 최대화하는 것이 목표이며 원하는 target class의 score가 높게 나오도록 input에 역전파를 반복적으로 전달하면서 워하는 이미지를 만들어 낸다. 이 generate 과정을 살펴보면 당연히 초깃값 설정이 중요하다는것을 알 수 있다. 

## Model decision explanation

여기서는 모델이 원하는 출력을 내기까지 입력을 어떠한 각도로 보고 있는지 알아보기 위해 이를 시각화해본다.

<br/>

먼저 saliency test(특징 추출) 계열의 방법 중 하나인 occlusion experiments부터 알아보자.

![](https://olenmg.github.io/img/posts/33-21.png)  

여기서는 간단하게 input 이미지의 일부를 마스킹한채로 모델에 통과시켜 운하는 클래스에 대한 확률 값을 구한다. 그리고 이 확률값을 가린 부분의 heat라고 생각하고 전체 부분에 대한 heatmap(coolusion map)를 그린다.  

<br/>

그럼 최종적으로 이 **heat가 강한 부분(salient part)**이 model의 decision에 큰 영향을 준다고 생각해 볼 수 있다.

<br/>

다음으로 backpropagation을 통해 saliency map를 그려볼 수도 있다.
![](https://olenmg.github.io/img/posts/33-22.png)  

이 방법의 기본 가정은 **역전파가 큰 scale로 전달되는 부분이 decision에 큰 영향을 준다**는 것이다. 

<br/>

여기서는  
첫번째로 target source image를 모델에 넣어 **원하는 클래스의 score를 찾고**
두번째로 역전파를 input단까지 전달하여 **input까지 전달된 gradient magnitude의 mpa을 그린다.**  
magnitude map을 그릴때 부호는 중요하지 않으므로 gradient에 절댓값을 취하거나 제곱을 취하여 이를 map으로 그리면 된다.  

<br/>

이 방법은 class visualization과 하는 작업은 비슷하지만 input이 다르고 목적하는 바가 다르다는 점에서 조금의 차이가 있다.  

<br/>

backpropagation을 통해 saliency map을 구하는 방법에는 이것보다 advanced한 방법이 있다. 바로 `guided backpropagation`이라는 방법인데, 여기에 앞서 deconvolution 연산에 대해 먼저 짚고 넘어 가자.

![](https://olenmg.github.io/img/posts/33-23.png)

일반적인 conv net의 backward pass에서는 forward 단의 ReLU가 활성화된 부분에서만 gradient가 흐르게 된다. 그런데 **deconvnet에서는 forward pass와 무관하게 backward pass의 부호만고려(즉, backward에 ReLU 적용)한다.**   그리고 guided backpropagation에서는 backward pass에서 forward단의 ReLU와 backward단의 ReLU 둘 모두를 고려한 gradient가 흐른다.  
  
정리하면 $h^{l+1} = max(0,h^l)$ 일 때, 아래와 같다.  
$\begin{aligned}
& \text{(standard)} \;\; \frac{\partial L}{\partial h^l} = [(h^l > 0)]\frac{\partial L}{\partial h^{l+1}} \\ 
& \text{(deconvnet)} \;\; \frac{\partial L}{\partial h^l} = [(h^{l+1} > 0)]\frac{\partial L}{\partial h^{l+1}} \\ 
& \text{(guided)} \;\; \frac{\partial L}{\partial h^l} = [(h^l > 0) \& (h^{l+1} > 0)]\frac{\partial L}{\partial h^{l+1}} \\
\end{aligned}$  

## Model decision explanation - CAM/Grad-CAM

visualization 방법으로 널리 통용되는 CAM(Class activation mapping) 방법에 대해 알아보자.

![](https://olenmg.github.io/img/posts/33-25.png)  

이 방법을 통해 위와 같이 decision에 큰 영향을 주는 부분에 대한 heatmap을 얻을 수 있다. threshold를 잘 조정하면 semantic segmentation에도 활용할 수 있을 것으로 보인다.

![](https://olenmg.github.io/img/posts/33-26.png)

이 방법을 쓰려면 반드시 **모델의 마지막 decision part에서 GAP(global average pooling)과 FC layer가 순차적으로 활용되어야 한다.** heatmap을 찍기 위해 이 GAP의 결과 $F_k$와 FC layer의 weight가 $w_k$가 활용되기 때문이다.  

일단 위 조건을 만족하는 모델의 pre-training이 완료되었으면, 이제 target image를 모델에 넣고 *(1) GAP의 결과인 $F_k$들과 (2) 그 값들이 원하는 클래스의 값으로 연결되는 FC layer의 weight 값 $w_k^c$들을 가져와서 클래스 c에 대한 score 값 $S_c$를 구한다. score 값을 구하는 구체적인 식은 아래와 같다.  

$S_c$   
$= \sum_{k} \mathrm{w}_{k}^{c}F_k$  
    $= \sum_{k} \mathrm{w}_{k}^c \sum_{(x,y)}f_k(x,y)$  
        $= \sum_{(x,y)} \sum_{k} \mathrm{w}_{k}^{c}f_k(x,y)$


(6)은 scoring 식을 나타낸 것이고, (7)은 GAP 연산을 풀어서 쓴 것이다. (8)은 식의 순서를 변경한 것이다. 우리가 필요한 것은 **GAP 연산을 하기 전, 공간 정보까지 남아있는 class activation이다**. 따라서 최종적으로 우리가 활용할 부분은 공간정보를 합치기 이전 $\sum\limits_{k} \mathrm{w} _{k}^{c}f_k(x,y)$ 항이다. 이 부분을 `CAM`이라 부르며, 우리가 원하던 값이다.

<br/>

이렇게 하면 위에서 봤던 그림처럼 각 feature map 별로 가중합이 연산되어 최종적으로 class activation map을 얻을 수 있게 된다.

<br/>

이 방법의 장점은 따로 위치 정보에 대한 anootation 없이도 사진의 주요한 부분의 위치정보를 쉽게 찾을 수 있다는 것이다. 하지만, 앞서 말했듯이 대상 모델이 GAP 구조를 활용해야 하기 때문에, targat 모델에 GAP이 없다면 GAP을 삽입하고 새로 재학습시켜야한다는 단점이 있다. 게다가 이 과정에서 기존 모델과 decision 성능이 다르게 나올수 있다는 문제도 있다.  

<br/>

마지막으로 이러한 CAM의 단점을 해결한 `Grad-CAM`을 살펴보도록 하자.
![](https://olenmg.github.io/img/posts/33-27.png)  
Gard-CAM은 모델의 구조 변경이나 이에 따른 재학습 과정이 필요하지 않다.  

직관적으로 이해해보면, 결국 우리한테 필요한 것은 $\sum\limits_{k} \mathrm{w}_{k}^{c}f_k(x,y)$ 이고 여기서 $f$는 Gap 이전의 feature map이므로 여타 모델에서도 충분히 얻어낼 수 있다.  

<br/>

그럼 결국 필요한 것은 w인데, 이 부분도 사실 기존 모델은 놔두고 **이전에 얻은 feature map에 따로 GAP을 취하여 얻을 수 있다.

![](https://olenmg.github.io/img/posts/33-29.png)

즉, decision task를 위한 layer를 모델에서 제외하고 마지막으로 conv 연산을 통해 나온 feature에 GAP를 취하고 이를 다시 feature와 곱해서 최종적으로 원하는 heatmap을 얻을 수 있다.  

<br/>

결국 가중치 값도 얻을 수 있는데, 중요한 것은 Grad-CAM에서는 그 이름에 걸맞게 **그냥 feature map 대신 backpropagation에서 나온 해당 feature의 gradient map을 활용한다는 점이다.** 왜 이것을 활용하는지 그 이유에 대해서는 앞서 backpropagation을 통해 얻었던 heatmap이 시각화가 잘 되었다는 점에서 찾을 수 있다. 한편 마지막 feature의 gradient만을 활용하기 때문에 backprop도 그부분까지만 흘려주면 원하는 값을 얻을 수 있다.  

<br/>

결론적으로 이를 식으로 나타내면 아래와 같다. $\alpha ^c _k$가 standard CAM에서의 $w_k^c$와 같은 역할(가중치)를 하고 $A^k$가 앞에서의 $f^k$와 같은 역할(feature)를 한다.  

$\alpha_{k}^{c} = \frac{1}{Z}\sum_i \sum_j\, \frac{\partial y^c}{\partial A_{ij}^k}$  
$L_{\text{Grad-CAM}}^{c} = \mathrm{ReLU}(\sum_{k} \alpha_{k}^{c}A^{k})$

결국 이전과 다른 점은 gradient를 활용했다는 점, 그리고 앞서 본 gradient에 ReLU를 적용했을 때의 장점을 취한다는 점 뿐이다.  
gradient를 활용한다고 해도 원래 CAM과 방법론면에서는 달라진 것이 없다.
![](https://olenmg.github.io/img/posts/33-28.png)
Grad-CAM에서는 위와 같이 Guided Backpropagation 방법도 함께 활용한다. guided backprop에서는 sensitive, high frequecny의 정보를 얻을 수 있고, Grad-CAM에서는 class에 민감한 정보를 얻을 수 있으므로 이들 결과를 모두 활용하려면 여러 관점에서의 정보를 모두 고려할 수 있게 된다. 최종적으로 이 둘 값의 dot연산을 통해 결과를 도출 해낸다.  

<br/>

위에서 오른쪽 그림은 다양한 task에 대해 이를 적용할 수 있다는 것을 나타낸 그림이다. 



