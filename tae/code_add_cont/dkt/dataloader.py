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
            random.seed(0) # fix to default seed 0
            random.shuffle(data)

        cut_size = int(len(data) * ratio)
        train_data = data[:cut_size]
        test_data = data[cut_size:]

        return train_data, test_data

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']  # 문항, 시험지, 문항 태그

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

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __feature_engineering(self, df, is_train):
        if self.args.fversion == 1:
            df = self.__feature_split_user(df, is_train)
        if self.args.fversion == 2:
            df = self.__feature_tag_mean(df, is_train=is_train)
            df = self.__feature_split_user(df, is_train)
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

    def __percentile(self, s):
        return np.sum(s) / len(s)

    def __feature_tag_mean(self, df, is_train):
        
        if is_train:
            testId_mean_sum = df.groupby(['testId'])['answerCode'].agg(['mean','sum'])
            assessmentItemID_mean_sum = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
            KnowledgeTag_mean_sum = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])


            # tag_groupby = df.groupby('KnowledgeTag').agg({
            # 'userID': 'count',
            # 'answerCode': self.__percentile
            # })
            
            if not os.path.exists(self.args.asset_dir):
                os.makedirs(self.args.asset_dir)
            # train의 수치 데이터를 test에서 사용하기 위해 직렬화 저장
            # tag_groupby.to_pickle(os.path.join(self.args.asset_dir,'tag_groupby.pkl'))
            KnowledgeTag_mean_sum.to_pickle(os.path.join(self.args.asset_dir,'KnowledgeTag_mean_sum.pkl'))
            assessmentItemID_mean_sum.to_pickle(os.path.join(self.args.asset_dir,'assessmentItemID_mean_sum.pkl'))
            testId_mean_sum.to_pickle(os.path.join(self.args.asset_dir,'testId_mean_sum.pkl'))
        else:
            # tag_groupby = pd.read_pickle(os.path.join(self.args.asset_dir,'tag_groupby.pkl'))
            KnowledgeTag_mean_sum = pd.read_pickle(os.path.join(self.args.asset_dir,'KnowledgeTag_mean_sum.pkl'))
            assessmentItemID_mean_sum = pd.read_pickle(os.path.join(self.args.asset_dir,'assessmentItemID_mean_sum.pkl'))
            testId_mean_sum = pd.read_pickle(os.path.join(self.args.asset_dir,'testId_mean_sum.pkl'))

        # df['tag_mean'] = df['KnowledgeTag'].apply(lambda x: tag_groupby.loc[x, 'answerCode'])
        df["test_mean"] = df.testId.map(testId_mean_sum.to_dict()['mean'])
        df['test_sum'] = df.testId.map(testId_mean_sum.to_dict()['sum'])
        df["ItemID_mean"] = df.assessmentItemID.map(assessmentItemID_mean_sum['mean'])
        df['ItemID_sum'] = df.assessmentItemID.map(assessmentItemID_mean_sum['sum'])
        df["tag_mean"] = df.KnowledgeTag.map(KnowledgeTag_mean_sum.to_dict()['mean'])
        df['tag_sum'] = df.KnowledgeTag.map(KnowledgeTag_mean_sum.to_dict()['sum'])

        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)     #, nrows=100000)
        df = self.__feature_engineering(df, is_train)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir, 'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir, 'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir, 'KnowledgeTag_classes.npy')))

        self.args.n_cont = 4  # 수치형 컬럼 갯수를 지정해야 하는데 우선 하드코딩

        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)

        diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().shift(-1).fillna(0)
        diff = diff.fillna(pd.Timedelta(seconds=0))

        df['time'] = diff
        # 아웃라이어를 보정
        df['time'] = df['time'].apply(lambda x: x if x < 650 else 650)              

        #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
        df = df.fillna(0)

        # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag' ,'tag_mean', 'user_correct_answer', 'user_total_answer', 'user_acc']
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag' ,'tag_mean', 'tag_sum' ,'test_mean', 'test_sum', 'ItemID_mean', 'time']

        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['testId'].values, 
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['answerCode'].values,
                    r['tag_mean'].values,
                    # r['tag_sum'].values,
                    r['ItemID_mean'].values,
                    r['test_mean'].values,
                    r['time'].values,
                    # r['user_correct_answer'].values,
                    # r['user_total_answer'].values,
                    # r['user_acc'].values,
                )
            )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        # row 예시 : load_data_from_file의 return인 group 참조
        # test, question, tag, correct, tag_mean, user_correct_answer, user_total_answer, user_acc = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        test, question, tag, correct, tag_mean, ItemID_mean, test_mean, time = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        # test, question, tag, correct = row[0], row[1], row[2], row[3]

        # cate_cols = [test, question, tag, correct]
        # cate_cols = [test, question, tag, correct, tag_mean, user_correct_answer, user_total_answer, user_acc]  # tag_mean은 수치형이지만 임시적으로 추가
        cate_cols = [test, question, tag, correct, tag_mean, ItemID_mean, test_mean, time]  # tag_mean은 수치형이지만 임시적으로 추가

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
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)


    for i, _ in enumerate(col_list):
        col_list[i] =torch.stack(col_list[i])
    
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