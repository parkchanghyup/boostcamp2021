

## loss.backward()


```python
for epoch in range(2):

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        input, labels = data

        # zero the parameter gradients
        optimizer.zero__grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```



코드를 보면 outputs, labels, model,input 까지 하나의 체인으로 연결 되어 있는 것을 알 수 있다.
loss.backward() 함수를 호출하면 뒤에서 부터(outputs) 역전파로 계산하여 결국 모델의 파라미터의 grad 값을 업데이트 된다.

이후 `Optimizer.step()` 를 통해 파라미터값 자체가 업데이트 된다.




## 조금 특별한 loss

쉽게 말해서 Error를 만들어내는 과정에 양념을 치는것
<br/>

`Focal Loss` 


클래스 불균형 문제가 있는 경우, 맞춘 확률이 높은 class는 조금의 loss를 맞춘 확률이 낮은 class는 LOss를 훨씬 높게 부여





`Label Smoothing Loss`


Class target label을 Onehot 표현으로 사용하기 보다는 조금 soft 하게 표현하는 일반화 성능을 높이기 위함


[0, 1, 0, 0,...] - > [0.0025, 0.9, 0.025, ...]




## Optimizer


 - 어느 방향으로, 얼마나 움직일 지 ?


    - 영리하게 움직일 수록 수렴은 빨라 진다.

[![loss image](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/2fdbbf56b81c4a7a807fe1b1b2bcbd33/f2052869-3773-49d0-88cd-1e5151fe561b.png)](https://slid.cc/vdocs/2fdbbf56b81c4a7a807fe1b1b2bcbd33?v=5e3d0f492a0e47e0a6876796971064c2&start=1540.123936)


 - LR Scheduler


    - 학습 시에 Learning rate를 동적으로 조절. - 조금더 영리하게 움직이게 해줌

[![loss image](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/2fdbbf56b81c4a7a807fe1b1b2bcbd33/7e249fa1-70da-4e5a-8ab7-fe69fe51741d.png)](https://slid.cc/vdocs/2fdbbf56b81c4a7a807fe1b1b2bcbd33?v=5e3d0f492a0e47e0a6876796971064c2&start=1584.243368)    

    - StepLR : 특정 Step 마다 LR 감소

[![loss image](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/2fdbbf56b81c4a7a807fe1b1b2bcbd33/39b16f12-edc3-4f55-a417-e0d24526dcf2.png)](https://slid.cc/vdocs/2fdbbf56b81c4a7a807fe1b1b2bcbd33?v=5e3d0f492a0e47e0a6876796971064c2&start=1698.702083)  


    - CosineAnnealingLR : Cosine 함수 형태처럼 LR을 급격히 변경

[![loss image](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/2fdbbf56b81c4a7a807fe1b1b2bcbd33/2ec233a4-e540-44dc-a731-f92616ac18dd.png)](https://slid.cc/vdocs/2fdbbf56b81c4a7a807fe1b1b2bcbd33?v=5e3d0f492a0e47e0a6876796971064c2&start=1755.540043)  


    - ReduceLROnPlateau : 더 이상 성능 향상이 없을 때 LR 감소

[![loss image](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/2fdbbf56b81c4a7a807fe1b1b2bcbd33/88d93e95-76f1-40e1-a13b-d6c18bb2b54f.png)](https://slid.cc/vdocs/2fdbbf56b81c4a7a807fe1b1b2bcbd33?v=5e3d0f492a0e47e0a6876796971064c2&start=1873.045586)  



## 올바른 MeTric의 선택





데이터 상태에 따라 적절한 Metrice를 선택하는 것이 필요

[![loss image](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/2fdbbf56b81c4a7a807fe1b1b2bcbd33/2c214d12-1efa-4bae-8d2f-a4d393efc35c.png)](https://slid.cc/vdocs/2fdbbf56b81c4a7a807fe1b1b2bcbd33?v=5e3d0f492a0e47e0a6876796971064c2&start=2621.53451)




## 정리


`loss` :  역전파 알고리즘을 진행하는 과정에러를 발생시키고 그에러를 하나하나 파라미터에 업데이트하기위한 grad 생성


`optimizer` : 생성된 grad 를 바탕으로 파라미터를 어떤 방향으로 얼마나 업데이트 할지 정의를 내릴 수있고 학습률을 어떻게 동적으로 조정할지 까지 정할 수있다. (lr_schedule)


`matrix` : 모델의 객관적 지표를 어떤 것을 이용할지 판단 할 수있다.



## 마스터님의 Tip!!

1. 레이블 스무딩

2. Knowledge distillation
    - 이미지의 랜덤 crop 
    - Teacher model의 예측 값을 student model의 예측값으로
    - 작은 student모델이 teacher 보다 성능이 좋은 경우도 있다.
3. MixUp
    고양이와 개가 섞인 이미지를 줘서 레이블을 비율로 나눔
4. Cosine Learing rate Decay
    - step decay : 특정 epoch 마다 학습률 감소
    - Cosine Decay : Cosine 함수로 학습률 감소 -> 요즘 논문에서 많이 사용
5. CutMix 
    - 사진을 네모로 오려서 붙이는 기법 -> 효과가 좋아서 거의다 사용
6. FMix 
    - 사진을 네모로 자르지 않고 object 모양을 유지해서 자름


## Efficient Training

### Learinig rate
    - 배치가 클 수록 학습 속도는 빠르나 수렴이 느림
    - Learing rate = 0.1 x battch / 256

### Zero Gamma in BatchNormalization
    - $\gamma$는 batch Norm의 scaling factor
    - 0으로 시작하여 서서히 늘림!
    - 좀 더 수렴하기 좋다

### Low Precision Training

- 학습을 가볍고 빠르게! 32bit ⇒ 16 bit (순전 역전파만)
- 가중치 업데이트는 32bit 유지
- 비슷한 성능을 유지하면서 빨라지고 가벼워짐, 오히려 성능이 높기도 함
- 토치에서 기능 찾아보기
- No bias decay? 28분


































