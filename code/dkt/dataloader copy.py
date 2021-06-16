import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(0)  # fix to default seed 0
            random.shuffle(data)

        cut_size = int(len(data) * ratio)
        train_data = data[:cut_size]
        test_data = data[cut_size:]

        return train_data, test_data

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag', 'grade']  # 문항, 시험지, 문항 태

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            label_encoder = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                label_encoder.fit(a)
                self.__save_labels(label_encoder, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col+'_classes.npy')
                label_encoder.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if x in label_encoder.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = label_encoder.transform(df[col])
            df[col] = test
            

        def convert_time(s):
            """
            2020-03-24 00:17:11 이런식으로 시간을 param으로 받아서 1585009031.0 이렇게 초단위로 변경해서 timestamp로 반환
            여기서 전처리의 대부분 시간을 아먹음
            """
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        # df['Timestamp'] = df['Timestamp'].apply(convert_time)
        if self.args.fversion >=2:
            cont_cols = ['test_mean', 'ItemID_mean', 'tag_mean']
            df[cont_cols] = df[cont_cols].astype(np.float32)
        return df

    def __feature_engineering(self, df, is_train, train_df=None):
        if self.args.fversion >= 1:
            df = self.__feature_split_user(df, is_train)
        if self.args.fversion >= 2:
            df['grade'] = df['testId'].apply(lambda row: int(row[2]))
            df = self.__continuous_feature_engineering(df, is_train, train_df)
        return df

    def __feature_split_user(self, df, is_train):
        if is_train:
            # UserID를 시험지로 나눔
            arr = []
            for ele in df.assessmentItemID.values:
                arr.append(ele[:3])
            newID = [str(e1) + str(e2) for e1, e2 in zip(arr, df["userID"])]
            df["userID"] = newID
        else:
            temp = df.copy()
            # UserID를 시험지로 나눔
            grade_arr = []
            for ele in temp.assessmentItemID.values:
                grade_arr.append(ele[:3])

            newID = [str(e1) + str(e2) for e1, e2 in zip(grade_arr, temp["userID"])]
            temp["userID"] = newID

            test_data = temp[temp['answerCode'] == -1]
            bool_arr = []

            for id in temp['userID']:
                if id in test_data['userID'].values:
                    bool_arr.append(True)
                else:
                    bool_arr.append(False)

            df = df[bool_arr]

        return df

    def __continuous_feature_engineering(self, df, is_train, train_df=None):
        if is_train:
            train_df = df
        correct_t = train_df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
        correct_t.columns = ["test_mean", 'test_sum']
        correct_k = train_df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
        correct_k.columns = ["tag_mean", 'tag_sum']
        correct_a = train_df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
        correct_a.columns = ["ItemID_mean", 'ItemID_sum']

        df.sort_values(by=['userID','Timestamp'], inplace=True)

        df.Timestamp = pd.to_datetime(df.Timestamp)
        diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().shift(-1).fillna(pd.Timedelta(seconds=37))
        # diff = diff.fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

        df['time_diff'] = diff
        # # 아웃라이어를 보정
        df['time_diff'] = df['time_diff'].apply(lambda x: x if x < 50 else 37)  

        #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
        df['user_acc'].fillna(0.65, inplace=True)
         

        # 봉진님 FE

        # user 별 마지막으로 푼 tag로부터 지난 시간, NaN값은 300으로 한다.
        prev_timestamp_ac = df.groupby(['userID', 'KnowledgeTag'])[['Timestamp']].shift()
        # df['diff_time_btw_KnowledgeTag_ids'] = (df['Timestamp'] - prev_timestamp_ac['Timestamp']).fillna(pd.Timedelta(seconds=300))
        
        # 각 tag 별 마지막으로 풀었을때 정답 여부
        prev_correct_ac = df.groupby(['userID', 'KnowledgeTag'])[['answerCode']].shift()        
        df['prev_answered_correctly'] = prev_correct_ac['answerCode'].fillna(0)
        
        # #test, item, tag 별 평균 정답률
        # df["test_mean"] = df.testId.map(testId_mean_sum['mean'])
        # df["ItemID_mean"] = df.assessmentItemID.map(assessmentItemID_mean_sum['mean'])
        # df["tag_mean"] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['mean'])
        
        #test, Item, tag 별 상대적 정답률
        # df['relative_test_answer'] = df['answerCode'] - df['test_mean']
        # df['relative_ItemID_answer'] = df['answerCode'] - df['ItemID_mean']
        # df['relative_tag_answer'] = df['answerCode'] - df['tag_mean']
        
        # #이동평균선 5, 10, 15, 20, 25, 30
        # df['ma5'] = df['user_acc'].fillna(0).rolling(window=5).mean().fillna(0.65)  # nan 평균 정답률로 처리
        # df['ma10'] = df['user_acc'].fillna(0).rolling(window=10).mean().fillna(0.65)
        # df['ma15'] = df['user_acc'].fillna(0).rolling(window=15).mean().fillna(0.65)
        # df['ma20'] = df['user_acc'].fillna(0).rolling(window=20).mean().fillna(0.65)
        # df['ma25'] = df['user_acc'].fillna(0).rolling(window=25).mean().fillna(0.65)
        # df['ma30'] = df['user_acc'].fillna(0).rolling(window=30).mean().fillna(0.65)
        
        # #MACD
        # df['MACD'] = df['ma15'] - df['ma25']
        
        # #Standard Deviation 5,10, 15, 20, 25, 30
        # df['sd5'] = df['user_acc'].fillna(0).rolling(window=5).std().fillna(0.0)  # nan 0으로 처리
        # df['sd10'] = df['user_acc'].fillna(0).rolling(window=10).std().fillna(0.0)
        # df['sd15'] = df['user_acc'].fillna(0).rolling(window=15).std().fillna(0.0)
        # df['sd20'] = df['user_acc'].fillna(0).rolling(window=20).std().fillna(0.0)
        # df['sd25'] = df['user_acc'].fillna(0).rolling(window=25).std().fillna(0.0)
        # df['sd30'] = df['user_acc'].fillna(0).rolling(window=30).std().fillna(0.0)
        
        # #볼린저 밴드
        # df['Upper BollingerBand'] = df['ma10'] + (df['sd10'] * 3).fillna(0.5)  # 볼린저밴드가 중심일 때 0.5
        # df['Lower BollingerBand'] = df['ma10'] - (df['sd10'] * 3).fillna(0.5) 
        
        #이전에 같은 item, tag 몇 번 풀었는지
        df['prior_ItemID_frequency'] = df.groupby(['userID', 'assessmentItemID']).cumcount()
        df['prior_tag_frequency'] = df.groupby(['userID', 'KnowledgeTag']).cumcount()

        # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
        # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")
        
        return df


    def load_data_from_file(self, file_name, is_train=True, train_df=None):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)     #, nrows=100000)
        if not is_train:
            train_df = pd.read_csv(os.path.join(self.args.data_dir, train_df))
        df = self.__feature_engineering(df, is_train, train_df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir, 'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir, 'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir, 'KnowledgeTag_classes.npy')))
        self.args.n_grade = len(np.load(os.path.join(self.args.asset_dir, 'grade_classes.npy')))

        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag', 'grade']
        if self.args.fversion >=2:
            cont_cols = ['time_diff', 'user_correct_answer', 'user_total_answer', 'user_acc','prev_answered_correctly',  
                        # 'ma5', 'ma10', 'ma15', 'ma20', 'ma25', 'ma30', 'MACD', 'sd5', 'sd10', 'sd15', 'sd20', 'sd25', 'sd30', 'Upper BollingerBand', 'Lower BollingerBand', 
                        'prior_ItemID_frequency']
                        # 'diff_time_btw_KnowledgeTag_ids', 타입에러로 잠시 뺌
                        # 'relative_test_answer', 'relative_ItemID_answer', 'relative_tag_answer'
            # cont_cols = ['time']
            df[cont_cols] = df[cont_cols].astype(np.float32)  # 추후 cont_fe 메서드에 통합할 것
            columns += cont_cols
            self.args.n_cont = len(cont_cols)
            group = df[columns].groupby('userID').apply(
                    lambda r: (
                        r['testId'].values,
                        r['assessmentItemID'].values,
                        r['KnowledgeTag'].values,
                        r['answerCode'].values,
                        r['grade'].values,
                        r[cont_cols].values,
                    )
                )
        else:
            group = df[columns].groupby('userID').apply(
                    lambda r: (
                        r['testId'].values,
                        r['assessmentItemID'].values,
                        r['KnowledgeTag'].values,
                        r['answerCode'].values,
                    )
                )
        return group.values

    def custom_load_data_from_file(self, df, is_train=True, train_df=None):
        
        if not is_train:
            train_df = pd.read_csv(os.path.join(self.args.data_dir, train_df))
        df = self.__feature_engineering(df, is_train, train_df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir, 'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir, 'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir, 'KnowledgeTag_classes.npy')))
        self.args.n_grade = len(np.load(os.path.join(self.args.asset_dir, 'grade_classes.npy')))

        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag', 'grade']
        if self.args.fversion >=2:
            cont_cols = ['time_diff', 'user_total_answer', 'user_acc', #'prev_answered_correctly',   'user_correct_answer',
                        # 'ma5', 'ma10', 'ma15', 'ma20', 'ma25', 'ma30', 'MACD', 'sd5', 'sd10', 'sd15', 'sd20', 'sd25', 'sd30', 'Upper BollingerBand', 'Lower BollingerBand', 
                        #'prior_ItemID_frequency']
            ]
                        # 'diff_time_btw_KnowledgeTag_ids', 타입에러로 잠시 뺌
                        # 'relative_test_answer', 'relative_ItemID_answer', 'relative_tag_answer'
            # cont_cols = ['time']
            df[cont_cols] = df[cont_cols].astype(np.float32)  # 추후 cont_fe 메서드에 통합할 것
            columns += cont_cols
            self.args.n_cont = len(cont_cols)
            group = df[columns].groupby('userID').apply(
                    lambda r: (
                        r['testId'].values,
                        r['assessmentItemID'].values,
                        r['KnowledgeTag'].values,
                        r['answerCode'].values,
                        r['grade'].values,
                        r[cont_cols].values,
                    )
                )
        else:
            group = df[columns].groupby('userID').apply(
                    lambda r: (
                        r['testId'].values,
                        r['assessmentItemID'].values,
                        r['KnowledgeTag'].values,
                        r['answerCode'].values,
                    )
                )
        return group.values    
    

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)
    
    def custom_load_train_data(self, file):
        self.train_data = self.custom_load_data_from_file(file)

    def load_test_data(self, file_name, train_df=None):
        self.test_data = self.load_data_from_file(file_name, is_train= False, train_df=train_df)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        if self.args.fversion >=2:
            # test, question, tag, correct, grade, conts = row[0], row[1], row[2], row[3], row[4], row[5]
            test, question, tag, correct, grade, conts = row
        else:
            test, question, tag, correct = row[0], row[1], row[2], row[3]

        cate_cols = [test, question, tag, correct, grade, conts] if self.args.fversion >=2 else [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            if len(col.size()) == 1:
                pre_padded = torch.zeros(max_seq_len)
            else:
                # print(i, col, col.size())

                pre_padded = torch.zeros(max_seq_len, len(col[0]), dtype=torch.float)
            pre_padded[-len(col):] = col

            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train_data, valid_data):
    pin_memory = True
    train_loader, valid_loader = None, None
    
    if train_data is not None:
        train_set = DKTDataset(train_data, args)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate)

    if valid_data is not None:
        valid_set = DKTDataset(valid_data, args)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate)

    return train_loader, valid_loader