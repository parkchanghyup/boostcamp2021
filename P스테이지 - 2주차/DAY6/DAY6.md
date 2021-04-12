
# 프로젝트 회고 

1. Competition을 위한 코드 진행 과정

    - 현재 최고 f1 score : 0.7373 
    - 적용한 기법 : K-fold(k=3) 앙상블, TTA,loss 함수에 weight , b0, cosine, Centercrop(300), age 58, CosineAnnealingWarmRestarts
    - 하이퍼 파라미터 : epcoh 10 , lr = 0.0001 ,train / val = 16, 8



2. 오늘 적용한 기법
    - efficient net b4 앙상블 -> `f1-score : 0.7289`
    - Normalization -> `f1-score : 0.6059`
    
오늘은 다른 수업을 진행한다고 실험을 많이 하지 못하였다..

3. 추후 진행할 사항
    - TTA 적용
    - b3 모델을 통한 앙상블
    - 성능 향상에 효과적인 Augmentation 찾아 보기


그리고 몰랏는데 v100이랑 p40이랑 성능 차이가 많이남... 한 2~3배정도 ...
그래서 다른 사람들은 한 epoch 당 2분에 끝나는데 나는 6~7분 걸림,,, 다음 stage에서는 V100을 먹을 수 있도록 해봐야 겠다. . .
    

# 수업 리뷰

오늘 시각화 기초에 대해 배웟는데, 아직 matplotlib 정도 밖에 하지 않아서 따로 정리를 하지 않았다. 그러나 다음 수업부터는 꽤 유익한 강의가 많아 보여서 기대된다.