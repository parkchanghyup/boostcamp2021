
# 프로젝트 회고 

1. Competition을 위한 코드 진행 과정

    - 현재 최고 f1 score : 0.7373 
    - 적용한 기법 : K-fold(k=3) 앙상블, TTA,loss 함수에 weight , b0, cosine, Centercrop(350), age 58, CosineAnnealingWarmRestarts
    - 하이퍼 파라미터 : epcoh 10 , lr = 0.0001 ,train / val = 16, 8



2. 오늘 적용한 기법

- K-fold (k = 5) 앙상블 -> 성능 향상 x ( 시간이 부족하여 제대로 학습을 못함 ..)
- CenterCrop (387) , multi loss 적용


오늘은 마지막 날이라서 2주간 실험했던 최적의 모델로 앙상블을 시킬 예정이였다. 그러나 어제 밤 12시 경에 내가 이떄까지 data augmentation적용을 안하고 있었던 것을 발견했고,
관련하여 여러가지 실험을 진행하느라 시간을 다 써버렸다. 

너무너무 아쉽고 부끄럽다. 내가 짠 코드가 작동을 한다고해서 안심 하지말고 두번 세번 제대로 확인해야 겠다고 생각하였다. 오늘일을 계기로 조금더 성장 항 수 있도록 해야겠다.