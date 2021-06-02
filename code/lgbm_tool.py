import numpy as np
import pandas as pd
import os
import random
import pickle
from pycaret.classification import *
from pycaret.utils import check_metric
from datetime import timedelta, timezone, datetime
import torch


class lgbm_processor:

    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test
        self.setup()
        self.df_train_ori = df_train.copy()
        self.df_test_ori = df_test.copy()


    def setup(self):
        answerCode2bool = {'userID': object, 'answerCode': 'int16', 'KnowledgeTag': object}
        self.df_train = self.df_train.astype(answerCode2bool)
        self.df_test = self.df_test.astype(answerCode2bool)
        self.df_train.Timestamp = pd.to_datetime(self.df_train.Timestamp)
        self.df_test.Timestamp = pd.to_datetime(self.df_test.Timestamp)

        df_train_test = pd.concat([self.df_train, self.df_test['answerCode'], self.df_test[self.df_test['answerCode'] != -1]])
        self.testId_mean_sum = df_train_test.groupby(['testId'])['answerCode'].agg(['mean', 'sum']).to_dict()
        self.assessmentItemID_mean_sum = df_train_test.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum']).to_dict()
        self.KnowledgeTag_mean_sum = df_train_test.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum']).to_dict()
        self.set_assessmentItemID = set(self.df_test.loc[self.df_test.answerCode == -1, 'assessmentItemID'].values) # 문제별 ID


    def __feature_engineering(self, df):
        # 문항이 중간에 비어있는 경우를 파악 (1,2,3,,5)
        def assessmentItemID2item(x):
            return int(x[-3:]) - 1  # 0 부터 시작하도록

        df['item'] = df.assessmentItemID.map(assessmentItemID2item)

        item_size = df[['assessmentItemID', 'testId']].drop_duplicates().groupby('testId').size() # test_size -> item_size
        testId2maxlen = item_size.to_dict()  # 중복해서 풀이할 놈들을 제거하기 위해

        item_max = df.groupby('testId').item.max()
        print(len(item_max[item_max + 1 != item_size]), '개의 시험지가 중간 문항이 빈다. item_order가 올바른 순서')  # item_max는 0부터 시작하니까 + 1
        shit_index = item_max[item_max + 1 != item_size].index
        shit_df = df.loc[df.testId.isin(shit_index), ['assessmentItemID', 'testId']].drop_duplicates().sort_values(
            'assessmentItemID')
        shit_df_group = shit_df.groupby('testId')

        shitItemID2item = {}
        for key in shit_df_group.groups:
            for i, (k, _) in enumerate(shit_df_group.get_group(key).values):
                shitItemID2item[k] = i

        def assessmentItemID2item_order(x):
            if x in shitItemID2item:
                return int(shitItemID2item[x])
            return int(x[-3:]) - 1  # 0 부터 시작하도록

        df['item_order'] = df.assessmentItemID.map(assessmentItemID2item_order)

        # 유저가 푼 시험지에 대해, 유저의 전체 정답/풀이횟수/정답률 계산 (3번 풀었으면 3배)
        df_group = df.groupby(['newUserID', 'testId'])['answerCode']
        df['user_total_correct_cnt'] = df_group.transform(lambda x: x.cumsum().shift(1))
        df['user_total_ans_cnt'] = df_group.cumcount()
        df['user_total_acc'] = df['user_total_correct_cnt'] / df['user_total_ans_cnt']

        df['user_acc'] = df.newUserID.map(self.newUserID_mean_sum['mean'])
        df["test_mean"] = df.testId.map(self.testId_mean_sum['mean'])
        df["ItemID_mean"] = df.assessmentItemID.map(self.assessmentItemID_mean_sum['mean'])
        df["tag_mean"] = df.KnowledgeTag.map(self.KnowledgeTag_mean_sum['mean'])


    def feature_engineering(self):
        self.__feature_engineering(self.df_train)
        self.__feature_engineering(self.df_test)


    def my_train_vali_split(self, filter_option=None, ratio=0.5, seed=0):
        random.seed(seed)
        # 리더보드와 동일 조건의 컬럼 수집
        vali_full = self.df_train[
            (self.df_train['newUserID'] != self.df_train['newUserID'].shift(-1)) &
            (self.df_train.assessmentItemID.isin(self.set_assessmentItemID))
            ].copy()  # size 4366

        # 리더보드와 동일 조건의 컬럼을 나누기
        ratio_r = (1 - ratio)
        vali_1 = vali_full.sample(frac=ratio_r, random_state=seed)  # ratio가 1이면, ratio_r이 0이 되어, vali_1에 아무것도 할당되지 않는다.
        vali_2 = vali_full.drop(vali_1.index)

        # vali에 포함된 유저 목록 확인하기
        vali_1_userID = set(vali_1.newUserID.values)
        vali_2_userID = set(vali_2.newUserID.values)

        # vali에 없는 유저들만 train으로 데려오기
        train_1 = self.df_train[self.df_train['newUserID'].isin(vali_1_userID) == False].copy()
        train_2 = self.df_train[self.df_train['newUserID'].isin(vali_2_userID) == False].copy()

        # 마지막 응답만 가져올지 여부
        if filter_option == '시험지마지막응답':
            train_1 = train_1[train_1['testId'] != train_1['testId'].shift(-1)].copy()
            train_2 = train_2[train_2['testId'] != train_2['testId'].shift(-1)].copy()


        return train_1, vali_1, train_2, vali_2, vali_full



    def exam_full(datasets, FEATS, categorical_features=[], numeric_features=[], seed=0):
        train_1, vali_1, train_2, vali_2, vali_full = datasets
        random.seed(seed)
        settings = setup(data=train_1[FEATS], target='answerCode', train_size=0.9,
                         categorical_features=categorical_features, numeric_features=numeric_features)

        lgbm = create_model('lightgbm', sort='AUC')
        tuned_lgbm = tune_model(lgbm, optimize='AUC', fold=10)
        final_lgbm = finalize_model(tuned_lgbm)

        predict_model(lgbm)
        predict_model(tuned_lgbm)
        predict_model(final_lgbm)

        log = []
        prediction = predict_model(final_lgbm, data=vali_full[FEATS], raw_score=True)
        log.append(
            f"모든 vali 데이터:    {check_metric(prediction['answerCode'], prediction['Label'], metric='Accuracy')} ,{check_metric(prediction['answerCode'], prediction['Score_1'], metric='AUC')}")
        return final_lgbm, log

