
# 프로젝트 회고 

1. Competition을 위한 코드 진행 과정

    - 현재 최고 f1 score : 0.7373 
    - 적용한 기법 : K-fold(k=3) 앙상블, TTA,loss 함수에 weight , b0, cosine, Centercrop(300), age 58, CosineAnnealingWarmRestarts
    - 하이퍼 파라미터 : epcoh 10 , lr = 0.0001 ,train / val = 16, 8



2. 오늘 적용한 기법
    - mutli loss (CEL + f1 ) : `f1-score : 0.7105`
    - mutli loss (CEL + focal ) : `f1-score : 0.7067`
    - f1 + focal 은 실험 하지 않음 -> 두개 개별적으로 했을 때도 별로임..
    - Augmentation 
        - A.RandomBrightnessContrast(p=0.2),A.HueSaturationValue(p=0.2) : `f1-score : 0.6063`
        - A.flip(p=0.2)
    
    

개인적으로 acc는 괜찮게 나오는데 f1-score가 너무 안나온다 ..

3. 추후 진행할 사항
    - 다른 사람들은 multi loss 썻을때 다 성능 향상 있었는데 난왜 없지 . .
    - 성능 향상에 효과적인 Augmentation 찾아 보기
    - VScode 연결
    - `.py` 형식으로 나열해보기

