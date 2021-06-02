import numpy as np
import pandas as pd
import os
import random
import pickle
from pycaret.classification import *
from pycaret.utils import check_metric
from datetime import timedelta, timezone, datetime
import torch
from lgbm_tool import lgbm_processor


def setSeeds(seed = 42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def feature_split_user(df):
    new_df = df.copy()
    new_df["newUserID"] = df['assessmentItemID'].apply(lambda x:x[:3]) + df['userID'].astype(str)
    return new_df


def get_remain_test_data(df_test):
    get_new_id = set(df_test.loc[df_test.answerCode == -1, 'newUserID'])
    test_data = df_test[df_test.newUserID.isin(get_new_id)]
    remain_data = df_test.drop(test_data.index)
    return test_data, remain_data



if __name__ == "__main__":
    setSeeds(42)
    train_path = '/opt/ml/input/data/train_dataset/train_data.csv'
    test_path = '/opt/ml/input/data/train_dataset/test_data.csv'

    df_train_ori = pd.read_csv(train_path)
    df_test_ori = pd.read_csv(test_path)

    df_train = feature_split_user(df_train_ori)
    df_test = feature_split_user(df_test_ori)

    df_test, remain_data = get_remain_test_data(df_test)
    df_train = pd.concat([df_train, remain_data])

    # 여기부터 시작
    processor = lgbm_processor(df_train, df_test)
    processor.feature_engineering()

    filter_option = '시험지마지막응답'  # 시험지마지막응답, 사용자마지막응답, None
    train_must_exist_leaderboard = False  # True, False

    FEATS = ['user_acc', 'ItemID_mean', 'answerCode', 'user_total_acc']
    FEATS += ['test_mean', 'tag_mean', 'KnowledgeTag']

    categorical_features = []
    categorical_features += ['KnowledgeTag']

    numeric_features = []
    numeric_features += []

    # datasets = processor.my_train_vali_split(filter_option=filter_option, ratio=0.5)
    datasets_full = processor.my_train_vali_split(filter_option=filter_option, ratio=1.0)

    model, log = processor.exam_full(datasets_full, FEATS, categorical_features, numeric_features)


    ### inference
    df_test_shift = df_test[df_test['userID'] != df_test['userID'].shift(-1)]  # 맞춰야하는 row만 모아놓은 것
    prediction = predict_model(model, data=df_test_shift[FEATS], raw_score=True)
    total_preds = prediction.Score_1.values

    prediction_name = datetime.now(timezone(timedelta(hours=9))).strftime('%m%d_%H%M')

    output_dir = './'
    write_path = os.path.join(output_dir, f"{prediction_name}.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id, p))

    name = ('filter_option', 'train_must_exist_leaderboard', 'FEATS', 'log1', 'log2', 'log_full')
    logs = (filter_option, train_must_exist_leaderboard, FEATS, log)
    write_path = os.path.join(output_dir, f"{prediction_name}_log.txt")
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        for n, l in zip(name, logs):
            w.write(f'{n}: {l}\n')

