### Code Tree
    code  
    ├── README.md  
    ├── args.py  
    ├── baseline.ipynb  
    ├── dkt  
    │   ├── criterion.py  
    │   ├── dataloader.py  
    │   ├── metric.py  
    │   ├── model.py  
    │   ├── scheduler.py  
    │   ├── trainer.py  
    │   └── utils.py  
    ├── evaluation.py  
    ├── inference.py  
    ├── requirements.txt  
    └── train.py

### Data
주요 데이터는 .csv 형태로 제공되며, train/test 합쳐서 총 7,442명의 사용자가 존재합니다. 이 때 이 사용자가 푼 마지막 문항의 정답을 맞출 것인지 예측하는 것이 최종 목표입니다.  
데이터는 아래와 같은 형태이며, 한 행은 한 사용자가 한 문항을 풀었을 때의 정보와 그 문항을 맞췄는지에 대한 정보가 담겨져 있습니다. 데이터는 모두 Timestamp 기준으로 정렬되어 있습니다.

![image](https://user-images.githubusercontent.com/38639633/119285303-b6933780-bc7c-11eb-865b-ae5d8f4e3727.png)

- `userID` 사용자의 고유번호입니다. 총 7,442명의 고유 사용자가 있으며, train/test셋은 이 userID를 기준으로 90/10의 비율로 나누어졌습니다.
- `assessmentItemID` 문항의 고유번호입니다. 총 9,454개의 고유 문항이 있습니다. 이 일련 번호에 대한 규칙은, P stage 4 - DKT 2강 EDA에서 다루었으니 강의 들어보시면 좋을 것 같습니다.
- `testId` 시험지의 고유번호입니다. 문항과 시험지의 관계는 아래 그림을 참고하여 이해하시면 됩니다. 총 1,537개의 고유한 시험지가 있습니다.  
![image](https://user-images.githubusercontent.com/38639633/119285319-beeb7280-bc7c-11eb-876f-3c98125e0381.png)
- `answerCode` 사용자가 해당 문항을 맞췄는지 여부에 대한 이진 데이터이며 0은 사용자가 해당 문항을 틀린 것, 1은 사용자가 해당 문항을 맞춘 것입니다.
- `Timestamp` 사용자가 해당문항을 풀기 시작한 시점의 데이터입니다.
- `KnowledgeTag` 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할을 합니다. 태그 자체의 정보는 비식별화 되어있지만, 문항을 군집화하는데 사용할 수 있습니다. 912개의 고유 태그가 존재합니다.

Test data에 대해서도 마찬가지이며, 이 때 Timestamp상 가장 마지막에 푼 문항의 answerCode는 모두 -1로 표시되어 있습니다. 여러분들의 과제는 이 -1로 처리되어 있는 interaction의 정답 여부를 맞추는 것입니다.
