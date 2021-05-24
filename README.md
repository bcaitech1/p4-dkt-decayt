### components

code  
├── README.md  
├── args.py  
├── baseline.ipynb  
├── dkt  
│ ├── criterion.py  
│ ├── dataloader.py  
│ ├── metric.py  
│ ├── model.py  
│ ├── optimizer.py  
│ ├── scheduler.py  
│ ├── trainer.py  
│ └── utils.py  
├── evaluation.py  
├── inference.py  
├── requirements.txt  
└── train.py  



- `args.py`
	- 훈련에 필요한 파라미터들을 설정할 수 있는 파일입니다.
- `inference.py`
	- 학습이 완료된 모델을 이용해 test 데이터를 기반으로 예측된 csv파일을 생성하는 파일입니다.
- `train.py`
	- 메인 파일로 훈련을 시작할 때 사용됩니다.
- `dkt/trainer.py`
	- train, validation 등과 같은 실제로 모델이 훈련이 되는 로직이 포함된 파일입니다.
- `dkt/dataloader.py`
	- 데이터의 전처리 및 모델에 학습가능한 input 을 만드는 파일입니다.
- `dkt/model.py`
	- 3가지 베이스라인 모델이(LSTM, LSTM+ATTN, BERT) 포함된 파일입니다.
- `dkt/metric.py`
	- 평가 지표가 정의 된 파일입니다.
	- roc_auc, accuracy
- `dkt/criterion.py`
	- loss함수가 정의된 파일입니다.
	- BCELoss
- `dkt/optimizer.py`
	- 훈련에 사용될 optimizer가 정의된 파일입니다.
	- Adam, AdamW
- `dkt/scheduler.py`
	- learing rate을 조절하기 위한 scheduler가 포함된 파일입니다.
	- ReduceLROnPlateau, get*linear*schedule*with*warmup
- `dkt/utils.py`
	- 그외 훈련과 직접적으로 관련은 없는 로직들을 포함할 수 있는 파일입니다.
	- 현재는 random seed를 고정하는 함수만 작성되어 있습니다.

### How to use

- 파이썬 스크립트에서 실행
	1. `pip install -r requirements.txt`
		1. 이미 패키지들이 설치되어 있을 겁니다!
	2. `python train.py`
	3. `python inference.py`
- 주피터 노트북 실행
	- `baseline.ipynb` 를 처음 부터 실행시켜주시면 됩니다.
- 제출
	- output 폴더 안에 있는 output.csv를 제출해주시면 됩니다.