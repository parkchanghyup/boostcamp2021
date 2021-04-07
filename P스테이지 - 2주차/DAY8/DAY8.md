
# 프로젝트 회고 (일요일)

1. Competition을 위한 코드 진행 과정

    - 현재 최고 f1 score : 0.7373 
    - 적용한 기법 : K-fold(k=3) 앙상블, TTA,loss 함수에 weight , b0, cosine, Centercrop(350), age 58, CosineAnnealingWarmRestarts
    - 하이퍼 파라미터 : epcoh 10 , lr = 0.0001 ,train / val = 16, 8



2. 오늘 적용한 기법
    
    - SAM(두번의 optimizer.step 를 통해 모델의 일반화 성능을 높여줌) : `f1-score : 0.6507`
    - Adamp + SAM + effcient b3 : `f1-score : 0.6620`
    - 데이터셋을 사람 기준으로 나눔 + b0: `f1 - score : 0.6995`
    - 데이터셋을 사람 기준으로 나눔 + b1: `f1 - score : 0.6215`
    - 데이터셋을 사람 기준으로 나눔 + b3 : `f1 - score : 0.6854`
    - 데이터셋을 사람 기준으로 나눔 + b3 : `f1 - score : 0.7110` - *colab*
    - SAM + 사람 기준 + b0 : `f1-score : 0.6908`

    - Augmentation -> Center Crop 말고는 적용 하지 않음 아무것도 효과가 없음
        
    
    

데이터 셋을 사람 기준으로나누고 + b0 모델로 돌린 경우 f1 score는 낮지만 정확도가 79.4286%정도나옴.  
앙상블을 통해 f1 - score를 최대로 올려 볼 예정.






3. 추후 진행할 사항
    - 대회 하루남음 앙상블을 통해 f1 score를 최대로 올려보는게 목표
    - 23:46분 augmentation 적용이되지 않을것을 발견하였다 . .
