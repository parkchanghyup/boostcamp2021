# 프로젝트 회고

1. Competition을 위한 코드 진행 과정

    - 기존 Custom Dataset을 갈아엎음
    - 기존에는 Custom dataset에는 아래와 같이 `img_path` 를 파라미터로 넘겨주면, class 내부에서 라벨을 만들고 리턴 하는 형식이였지만
    - 그렇게하니 train data 와 validation data 를 나누는게 애매해져서 그냥 class 외부로 뺴주고 'img_path' 와 'label'을 파라미터로 넘겨주는 형식으로 수정.

```python

```
    - 드디어 정상적으로 작동하는 코드를 완성함.
    - 정확도가 70%를 겨우 넘겼는데 f1 score가 너무낮아서 다시 순위가 낮아짐 ㅎ ㅎ.. f1 score를 올릴 수 있는 방법을 연구하는중.
    - 오늘 오후에 모든 base line 코드가 배포되기때문에, 내 코드와 base line 코드를 조금 섞어서 완전한 코드를 완성할 계획 -> 주말
    
2. Daily mission
    - K-fold 적용 -> clear
    - TTA 적용 시켜보기 -> 보류
    
3. 그외
    - 클래스 불균형 문제 -> 58세 이미지 까지 60세로 라벨링하여 학습 (이현규 피어님 의견)
    - f1 score를 위한 loss를 찾았는데 오히려 성능이 별로 ? 왜그런지 모르겠다.