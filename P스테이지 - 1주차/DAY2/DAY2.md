# 프로젝트 회고

1. Competition을 위한 베이스라인 작성
    - 어제 작성한 베이스라인에서 라벨링 코드에 오류가 존재 -> 수정
    - VGG11 모델 학습 결과 정확도가 낮았던 이유도 라벨링 코드에 문제가 있었기 때문.
    - efficient net - b4, epoch 5 로 학습한 결과 : `정확도 : 0.49`
    
2. Custom Dataset
    - 항상 pytorch를 공부할 때 CustomDataset 이 잘 이해되지 않아서 힘들었는데, 이제는 좀 이해가됨
    - 처음 Custom Dataset의 경우 image를 cv2로 불러왔었는데, augementation 적용시 에러 발생 
    - Custom Dataset에서 image 를 PIL로 읽는 것으로 수정함.
    - 라벨링 코드를 검증해보지 않았는데 오류투성이 .. 즉각 수정..

3. 그외
    - 데이터 셋에서 60대 이상의 비율이 너무 적음. 어떻게 오버샘플링을 해야 할 지 고민중.
    - 클래스에 weight를 주거나, cutMix를 사용해 볼 예정
    - 오늘 피어세션에서 조원이 CutMix사용 해봤는데 성능 향상 그닥 ? 그래서 이것도 사실 고민중.

    
<br/>
<br/>

# CustomDataset 코드 
---
