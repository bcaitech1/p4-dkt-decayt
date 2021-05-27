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
            random.seed(self.args.seed) # fix to default seed 0
            random.shuffle(data)

        cut_size = int(len(data) * ratio)
        train_data = data[:cut_size]
        test_data = data[cut_size:]

        return train_data, test_data

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']  # 문항, 시험지, 문항 태

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

    def __feature_engineering(self, df, is_train, fversion):
        if fversion == 1:
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
    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)     #, nrows=100000)
        df = self.__feature_engineering(df, is_train, self.args.fversion)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir, 'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir, 'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir, 'KnowledgeTag_classes.npy')))

        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['testId'].values, 
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['answerCode'].values
                )
            )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct = row[0], row[1], row[2], row[3]
        

        cate_cols = [test, question, tag, correct]

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
    pin_memory = False
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