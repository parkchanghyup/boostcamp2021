# 프로젝트 회고

1. Competition을 위한 코드 진행 과정
    - 다른 사람들과 달리 내 모델은 loss 가 2에서 수렴해버리는 문제 발생
    - 애초에 custom dataset 부터 다시 뜯어 고치는 중인데, 아직 정확히 해결 못하였다.
    - 그리고 코드를 수정하던중 Silu 활성화 함수를 쓰고 싶어서 pytorch 를 업데이트 하였다.
    - 여기서 문제가 발생했는데 pytorch 를 업데이트하고 torch vision 버전을 맞춰 줘야하는데 잘 알지 못해서 계속 빙빙 해맸다.
    - 애초에 torch vision 버전 문제인데 다른 곳이 문제인가 싶어서 코드를 이곳저곳 수정하던중 코드가 꼬여서 custom dataset 뿐만아니라 train 하는 코드까지 다시 수정 해야 될 거 같다.
    
2. modeling
    - 오늘 Daily mission은 모델을 구축해보는 것이다.
    - 제공된 모델은 dark net 이 었는데, 모델 구축만 해보고 사용하지 않을 것 같다. 애초에 effcient net이나 resnet이 성능이 좋기 때문.
    
3. 그외
    - Cutmix 를 알아보려 하였지만 다른 자잘한 문제들이 너무 많아서 손대지 못하였다.
    - 기본적으로 70%이상의 성능을 확보한 후 KFold도 진행 한뒤 불균형 클래스 문제를 해결 해야 할 것 같다.
    

# Daily mission 코드 
---