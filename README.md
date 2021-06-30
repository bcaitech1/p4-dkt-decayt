# :question: 지식 상태 추론(Deep Knowledge Tracing)

## 🏆최종 성적

- `Public LB`: AUROC 0.8422 | ACC 0.7823 | 2등:2nd_place_medal:
- `Private LB` : AUROC 0.8456 | ACC 0.7715 | 4등



## 📚Task Description

사용자(학생) 개개인이 푼 문제 리스트와 정답 여부가 담긴 데이터로부터 최종 문제를 맞출지 틀릴지 예측하는 **지식 상태 추론** Task

- ***기간*** : 2021.05.24 ~ 2021.06.15(4주)

- ***Deep Knowledge Tracing(DKT) description*** :

	- `Input` : Dialogue 내에서 User와 System 발화 쌍 (1 Turn 단위)

	- `Output` : 해당 turn까지 누적된 Domain-Slot-Value의 pair

		![image](https://user-images.githubusercontent.com/38639633/122345725-23030d00-cf83-11eb-8023-e31719205950.png)

- ***Dataset Overview :*** 

	- 주요 데이터는 `.csv` 형태로 제공되며, train/test 합쳐서 총 7,442명의 사용자가 존재합니다. 이 때 이 사용자가 푼 마지막 문항의 정답을 맞출 것인지 예측하는 것이 최종 목표입니다.

		![image](https://user-images.githubusercontent.com/38639633/122147484-f3c1a280-ce93-11eb-8e42-2d8d6ad0fb83.png)

		- `userID` 사용자의 고유번호입니다. 총 7,442명의 고유 사용자가 있으며, train/test셋은 이 `userID`를 기준으로 9 : 1의 비율로 나누어졌습니다.

		- `testId` 시험지의 고유번호입니다. 문항과 시험지의 관계는 아래 그림을 참고하여 이해하시면 됩니다. 총 1,537개의 고유한 시험지가 있습니다.

		- `assessmentItemID` 문항의 고유번호입니다. 총 9,454개의 고유 문항이 있습니다. "A+앞 6자리"는 `testId`의 정보를 나타내고 있으며, 뒤 3자리는 문제의 번호를 의미합니다.

			![img](https://s3-ap-northeast-2.amazonaws.com/prod-aistages-public/app/Users/00000068/files/593ab0f9-a096-451b-86ea-086fc5575118..png)

		- `answerCode` 사용자가 해당 문항을 맞췄는지 여부에 대한 이진 데이터이며 0은 사용자가 해당 문항을 틀린 것, 1은 사용자가 해당 문항을 맞춘 것입니다.

		- `Timestamp` 사용자가 해당문항을 풀기 시작한 시점의 데이터입니다.

		- `KnowledgeTag` 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할을 합니다. 태그 자체의 정보는 비식별화 되어있지만, 문항을 군집화하는데 사용할 수 있습니다. 912개의 고유 태그가 존재합니다.

		

- ***Metric*** : 

	- DKT는 주어진 마지막 문제를 맞았는지 틀렸는지로 분류하는 이진 분류 문제입니다. 

	- 평가를 위해 **AUROC**(Area Under the ROC curve)와 **Accuracy**를 사용합니다. 

	- 리더보드에 두 지표가 모두 표시되지만, **최종 평가는 AUROC 로만** 이루어집니다.

		![image](https://user-images.githubusercontent.com/38639633/122149543-32a52780-ce97-11eb-8384-ed1de4ad58d5.png)

		

## :computer:Team Strategy

- ***Git Projects 활용***

	- 칸반보드를 활용한 협업

		![image](https://user-images.githubusercontent.com/38639633/122527657-7d20d280-d056-11eb-8ab1-d9786260776e.png)

- ***Notion 활용***

	- notion을 활용한 팀 실험 결과 공유 및 feature engineering EDA 공유
	- 피어세션 기록 등
	- **[여기](https://www.notion.so/Home-b263b1f24c3147ac9f8f2544178d66f6)**에서 확인할 수 있습니다.

	

<br><br>

## 📁프로젝트 구조

```
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
```



<br><br>



## :handshake:Reference 

- [이곳](https://www.notion.so/8f643763c8d94a6b95fa18d188a95b89?v=506161d2c96648bc9f56b0519592acaf)에서 확인할 수 있습니다.

<br/>

## :man_technologist: Contributors

[윤도연(ydy8989)](https://github.com/ydy8989) | [전재열(Jayten)](https://github.com/jayten-jeon) | [설재환(anawkward)](https://github.com/anawkward) | [민재원(ekzm8523)](https://github.com/ekzm8523) | [김봉진(BongjinKim)](https://github.com/BongjinKim) | [태영돈(taepd)](https://github.com/taepd)







