# R-CNN 계열 모델 소개

R-CNN은 CNN에 Region Proposal 을 추가하여 물체가 있을법한 곳을 제안하고, 그 구역에서 object detection을 하는 것이다. R-CNN 계열 모델은 R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN 까지 총 4가지 종류가 있다.

출처: https://mylifemystudy.tistory.com/82 [ENCAPSULATION]
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F1wDMd%2FbtqxpACZC67%2FFMztXdpZW0XsX3GRLHo4gk%2Fimg.png)  

R-CNN은 분류 후 회귀, Fast R-CNN, Faster R-CNN은 분류와 회귀를 병렬로, Mask R-CNN은 여기에 Masking 까지 병렬로 수행  
## mask R-CNN 구조
![](https://olenmg.github.io/img/posts/34-1.png)




### BackBone
- `이미지의 feature 추출`을 위해 사용
    - ResNet과 ResNeXt의 50, 101 layer과 FPN(Feature Pyramid Network)을 backbone으로 사용

    - FPN(Feature Pyramid Network)
        - Object Detection 분야에서 풀리지 않던 고질적인 난제이니 작은 물체 탐지를 해결
        ![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc5D2i4%2FbtqEfqUaK1s%2Fk5kgInuWo7qu1ik0IP6Tz1%2Fimg.png)
        - D를 보시면 먼저 신경망을 통과하면서 단계별로 피쳐맵들을 생성.
        - 가장 상위 레이어에서부터 거꾸로 내려오면서 피처를 합쳐준 뒤, Object Detection을 진행 
        - 이러한 방식을 통해서 상위 레이어의 추상화된 정보와 하위 레이어의 작은 물체들에 대한 정보를 동시에 살리면서 Object Detection 수행

### Head
- Bounding Box Recognition (Classification and Regression)과 Mask Prediction을 위해 사용됨
- Faster R-CNN의 Head(Classification and Regression)에 Mask branch를 추가
- backbone(ResNet, FPN)에 따라 Head의 구조가 달라짐

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbqAIDx%2FbtqUUrwWvvU%2FQBx3SXKaX5XEja5SnmKKtK%2Fimg.png)

## Fast R-CNN과 다른점
---
1. Faster R-    CNN에 존재하는 bounding-box 인식을 위한 branch에 **병렬로** object mask branch 추가.   
    - Mask R-CNN은 기존의 Faster R-CNN을 Object Detectoin 역할을 하도록 하고 각각의 ROI에 Mask segmentation을 해주는 작은 FCN(Fully Connected Network)추가.
    -  각 class별로 binary mask prediction을 수행한다. 먼저 upsampling을 한 후 클래스 수만큼의 mask(여기서는 80개)를 모조리 생성한 후 위쪽 head에서 해당 이미지의 classification을 완료하면 이를 참조하여 그 class에 해당하는 mask를 최종적으로 출력하게 된다.
    - 내가 이해한것은 위쪽에서 자동차와 사람을 구분해주면 사람 1 2 3 을 mask branch에서 구분
2. ROI Pooling 대신 ROI Align을 사용
    - 기존 RoI Pooling은 RoI가 소수점 좌표를 가지고 있을 경우 반올림하여 Pooling을 해준다는 특징이 있다. (딱 grid 단위로 쪼개서 본다) 즉, RoI Pooling은 정수 좌표만을 처리할 수 있다. 이러한 처리는 classification 처리에서는 문제가 없지만 segmentation task에서는 위치가 왜곡되기 때문에 문제가 발생한다.
    ![](https://olenmg.github.io/img/posts/34-2.png)
    - Mask R-CNN에서는 RoIAlign이라는 기법을 활용한다. 현재 문제는 정수 좌표만 볼 수 있다는 점이다. RoIAlign은 정수 좌표(즉, 점선으로 되어있는 grid 좌표)를  bilinear interpolation 연산을 사용하여 각 RoI bin의 샘플링된 4개의 위치에서 input feature의 정확한 value를 계산.   
    - 그 후 결과를 max 혹은 average하여 집계함
    
`bilinear interpolation` : 선형보간법,알려진 지점의 값 사이(중간)에 위치한 값을 알려진 값으로부터 추정하는 것을 말한다.


3. RPN 전에 FPN(feature pyramid network)가 추가됐다.



<br/>
<br/>
<br/>

## Reference 
[FPN](https://yeomko.tistory.com/44)  
[Mask R-CNN 리뷰](https://ropiens.tistory.com/76)