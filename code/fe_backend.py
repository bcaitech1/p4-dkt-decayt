import pandas as pd
import args
import os
from pandas import Series, DataFrame
import numpy as np
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.core.strings.accessor import StringMethods
from pandas import Timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from typing import *
from numpy import NaN
from cyclical import cyclical
from sklearn import linear_model
from tqdm import tqdm
from collections import namedtuple
import scipy

################################# region 재료 #################################

def features_declaration(df:DataFrame):
    lines = []
    for c in df.columns:
        line = f"{c} = df['{c}']"
        lines.append(line)
    print('\n'.join(lines))

def lin_regress(data, yvar, xvars):
    # output, input variables
    Y = data[yvar]
    X = data[xvars]
    # Create linear regression object
    linreg = linear_model.LinearRegression()
    # Fit the linear regression model
    model = linreg.fit(X, Y)
    # Get the intercept and coefficients
    intercept = model.intercept_
    coef = model.coef_
    result = [intercept, coef]
    return result

def lgbm():
    # 실제 태스크는 첫번째, 두번째를 맞추는게 아니라 마지막을 맞추는거니까
    # 한 유저의 시퀀스중에서 후반부를 lgbm 데이터로 써야하는거 아닐까?

    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score

    df_train = load_upgraded_df('./upgraded_df_train.csv')
    cols = discern_cols(df_train)

    x_train = df_train.groupby('userID').nth(-1).reset_index().drop(
        cols.not_oomean + cols.cate + cols.labels + ['Timestamp'], axis=1)
    y_train = df_train.groupby('userID').nth(-1).reset_index()[cols.labels]['answerCode']

    df_test = load_upgraded_df('./upgraded_df_test.csv')

    x_test = df_test.groupby('userID').nth(-2).reset_index().drop(
        cols.not_oomean + cols.cate + cols.labels + ['Timestamp'], axis=1)
    y_test = df_test.groupby('userID').nth(-2).reset_index()[cols.labels]['answerCode']

    model = LGBMClassifier().fit(x_train, y_train)
    predict = model.predict(x_test)
    accuracy_score(y_test, predict)

def get_95_quantile(x:Series, method = 'mean'):
    if method.lower() == 'mode':
        center = x.mode()
    elif method.lower() == 'median':
        center = x.median()
    else: # 'mean'
        center = x.mean()
    upper = int(center + 1.96* x.std())
    lower = max(0, int(center - 1.96* x.std()))
    rate = len(x[x.isin(range(lower, upper))]) / len(x)
    print(f"in range({lower} ~ {upper}) : {round(rate*100)} %")
    return x[x.isin(range(lower, upper))]

def left_merge(df, right, ln, rn):
    right = Series(right, name = rn)
    left = df
    return pd.merge(left, right, 'left', left_on = ln, right_index = True)[rn]

def get_grade(df:DataFrame) -> Series:
    g = df['testId'].apply(lambda row: int(row[2]))
    return g
def get_school(df:DataFrame) -> Series:
    g = df['testId'].apply(lambda row: int(row[2]))
    school = g.apply(lambda row: 'elementary_school' if row // 7 == 0 else 'middle_school')
    return school
def get_familySize(df):
    right = Series(df.groupby('familyID')['userID'].unique().apply(lambda r: len(r)), name = 'familySize')
    left = df
    return pd.merge(left, right, 'left', on = 'familyID')['familySize']

def userID_to_familyUserID(df):
    # 1. generate new_userID dictionary
    '''
    school, grade 가 다르면 familyID 를 생성하고, userID 를 변경
    '''
    df.set_index(['userID', 'school', 'grade'], inplace=True)

    new_users = df.index.unique()
    userID_dict = dict.fromkeys(new_users, "17752-1")
    k = 0
    before_uid = None
    for i in range(len(new_users)):
        item = new_users[i]
        if before_uid != item[0]:
            before_uid = item[0]
            k = 0
        k += 1
        userID_dict[item] = f"{item[0]}-{k}"

    # 2. assign familyID & replace userID with new_userID
    df.reset_index(inplace=True)
    df['familyID'] = df['userID'].astype('str')
    tqdm.pandas()
    new_userID = df.progress_apply(lambda row: userID_dict[row['userID'], row['school'], row['grade']], axis=1)

    df['userID'] = new_userID
    cols = df.columns.to_list()
    new_cols = [cols.pop(-1), *cols]
    df = df[new_cols]

    df = df.sort_values(by=['original_order']).reset_index(drop=True)

    return df

def get_time_diff(df:DataFrame, outlier_thres = 80) -> Series:
    dg: DataFrameGroupBy = df.groupby('userID')
    df['Timestamp_after'] = dg['Timestamp'].shift(-1)
    td: Series = df.groupby('userID').apply(lambda row: row['Timestamp_after'] - row['Timestamp'])
    df.drop(columns='Timestamp_after', inplace=True)
    td = td.reset_index(drop=True)
    td = td.dt.seconds
    if isinstance(outlier_thres, int):
        td.loc[td > outlier_thres] = NaN  # outlier 처리
    time_diff = td
    return time_diff

def get_time_diff_toolong(df:DataFrame, outlier_thres = 79):
    assert 'time_diff' in df.columns, "time_diff doesnt exist"
    return df.apply(lambda row: 1 if row['time_diff'] > outlier_thres else 0, axis = 1)

def get_time_diff_userChange(df:DataFrame):
    assert 'time_diff' in df.columns, "time_diff doesnt exist"
    return df.apply(lambda row: 1 if np.isnan(row['time_diff']) else 0, axis = 1)

def get_encoded_hour(df: DataFrame) -> tuple:
    hour = df['Timestamp'].dt.hour
    hr_sin, hr_cos = cyclical.encode(hour, 24)
    en_hour = (Series(hr_sin), Series(hr_cos))
    return en_hour

def get_encoded_dayoftheweek(df: DataFrame) -> tuple:
    dw = df['Timestamp'].dt.dayofweek
    dw_sin, dw_cos = cyclical.encode(dw, 7)
    en_dw = Series(dw_sin), Series(dw_cos)
    return en_dw

def get_encoded_dayofthemonth(df: DataFrame) -> tuple:
    dm = df['Timestamp'].dt.day
    dm_sin, dm_cos = cyclical.encode(dm, 31)
    en_dm = Series(dm_sin), Series(dm_cos)
    return en_dm

def get_encoded_month(df: DataFrame) -> tuple:
    m = df['Timestamp'].dt.month
    m_sin, m_cos = cyclical.encode(m, 12)
    en_m = Series(m_sin), Series(m_cos)
    return en_m

def get_user_enterDay(df:DataFrame) -> Series:
    dg: DataFrameGroupBy = df.groupby('userID')
    dg['Timestamp'].min()
    df = pd.merge(left=df, right=DataFrame(Series(dg['Timestamp'].min(), name='Timestamp_start')), how='left',
                  left_on="userID", right_index=True)
    timestamp_the_first_start = df['Timestamp'].min()
    tss = df.groupby('userID')['Timestamp_start']
    day_elapse = tss.apply(lambda row: row - timestamp_the_first_start).dt.days
    return day_elapse

def get_user_day_progress(df: DataFrame) -> Series:
    dg: DataFrameGroupBy = df.groupby('userID')
    dg['Timestamp'].min()
    df = pd.merge(left=df, right=DataFrame(Series(dg['Timestamp'].min(), name='Timestamp_start')), how='left',
                  left_on="userID", right_index=True)
    tqdm.pandas()
    user_day_progress = df.progress_apply(lambda row: row['Timestamp'] - row['Timestamp_start'], axis=1)
    return user_day_progress.dt.days

def get_KnowledgeTag_count(df):
    right = df.groupby('KnowledgeTag')['answerCode'].count()
    return left_merge(df, right, 'KnowledgeTag', 'KnowledgeTag_count')

def get_KnowledgeTag_acc_mean(df):
    right = Series(df.groupby('KnowledgeTag')['answerCode'].mean(), name = 'KnowledgeTag_acc_mean')
    left = df
    return pd.merge(left, right, 'left', left_on = 'KnowledgeTag', right_index = True)['KnowledgeTag_acc_mean']

def get_KnowledgeTag_countU(df):
    right = Series(df.groupby(['userID', 'KnowledgeTag'])['answerCode'].count(), name = "KnowledgeTag_countU")
    left = df
    return pd.merge(left, right, 'left', left_on=['userID', 'KnowledgeTag'], right_index=True)['KnowledgeTag_countU']

def get_KnowledgeTag_acc_meanU(df):
    right = Series(df.groupby(['userID', 'KnowledgeTag'])['answerCode'].mean(), name = "KnowledgeTag_acc_meanU")
    left = df
    return pd.merge(left, right, 'left', left_on = ["userID", "KnowledgeTag"], right_index = True)['KnowledgeTag_acc_meanU']

def get_assessmentItemID_count(df):
    right = df.groupby('assessmentItemID')['answerCode'].count()
    return left_merge(df, right, 'assessmentItemID', 'assessmentItemID_count')

def get_assessmentItemID_acc_mean(df):
    right = df.groupby('assessmentItemID')['answerCode'].mean()
    return left_merge(df, right, 'assessmentItemID', 'assessmentItemID_acc_mean')

def get_assessmentItemID_acc_oomean(df):
    g = df.groupby('assessmentItemID')['answerCode']
    l = g.count()
    ool = l - 1
    right = DataFrame(dict(sum = g.sum(), ool = ool))
    left = df
    md = pd.merge(left, right, 'left', left_on='assessmentItemID', right_index=True)
    oos = md["sum"] - df['answerCode']
    ool = md["ool"]
    return oos/ool

def get_assessmentItemID_countU(df):
    right = Series(df.groupby(['userID', 'assessmentItemID'])['answerCode'].count(), name="assessmentItemID_countU")
    left = df
    return pd.merge(left, right, 'left', left_on=['userID', 'assessmentItemID'], right_index=True)['assessmentItemID_countU']

def get_assessmentItemID_acc_meanU(df):
    right = Series(df.groupby(['userID', 'assessmentItemID'])['answerCode'].mean(), name="assessmentItemID_acc_meanU")
    left = df
    return pd.merge(left, right, 'left', left_on=['userID', 'assessmentItemID'], right_index=True)['assessmentItemID_acc_meanU']

def get_userID_count(df):
    right = df.groupby('userID')['answerCode'].count()
    return left_merge(df, right, 'userID', 'userID_count')

def get_userID_acc_mean(df):
    right = df.groupby('userID')['answerCode'].mean()
    return left_merge(df, right, 'userID', 'userID_acc_mean')

def get_userID_acc_oomean(df):
    g = df.groupby('userID')['answerCode']
    l = g.count()
    ool = l - 1
    right = DataFrame(dict(sum = g.sum(), ool = ool))
    left = df
    md = pd.merge(left, right, 'left', left_on='userID', right_index=True)
    oos = md["sum"] - df['answerCode']
    ool = md["ool"]
    return oos/ool

def get_userID_acc_cummean(df):
    return df.groupby('userID')['answerCode'].expanding().mean().reset_index(drop = True)

def get_userID_acc_mean_recentK(df, K):
    m:DataFrame = df.groupby('userID')['answerCode'].rolling(min_periods=K//2, window=K).mean().reset_index()
    m.sort_values('level_1', inplace = True)
    m.drop('level_1', axis = 1, inplace = True)
    m.reset_index(drop = True, inplace = True)
    return m['answerCode']

def get_userID_acc_oomean_recentK(df, K):
    s:DataFrame = df.groupby('userID')['answerCode'].rolling(min_periods=(K+1)//2, window=K+1).sum().reset_index()

    s.sort_values('level_1', inplace = True)
    s.drop(['userID', 'level_1'], axis = 1, inplace = True)
    s.reset_index(drop = True, inplace = True)
    ool = K
    oos = Series(s['answerCode'], dtype = 'float16') - Series(df['answerCode'], dtype = 'float16')
    return oos/ool

def get_userID_acc_ooWeightedMovingAverage(df, weights):
    ret = 0

    ma1 = get_userID_acc_oomean_recentK(df, 3)
    ma2 = get_userID_acc_oomean_recentK(df, 8)
    ma3 = get_userID_acc_oomean_recentK(df, 15)
    ma4 = get_userID_acc_oomean_recentK(df, 30)
    for i in range(len(weights)):
        ret += [ma1,ma2,ma3,ma4][i] * weights[i]
    return ret

def get_userID_countT(df):
    right = df.groupby(['userID', 'testId'])['answerCode'].count()
    return left_merge(df, right, ['userID', 'testId'], 'userID_countT')

def get_userID_acc_meanT(df):
    right = Series(df.groupby(['userID', 'testId'])['answerCode'].mean(), name = "userID_acc_meanT")
    left = df
    return pd.merge(left, right, 'left', left_on=['userID', 'testId'], right_index=True)["userID_acc_meanT"]

def get_userID_acc_oomeanT(df):
    g = df.groupby(['userID', 'testId'])['answerCode']
    l = g.count()
    ool = l - 1
    right = DataFrame(dict(sum = g.sum(), ool = ool))
    left = df
    md = pd.merge(left, right, 'left', left_on=['userID', 'testId'], right_index=True)
    oos = md["sum"] - df['answerCode']
    ool = md["ool"]
    return oos/ool

def get_testId_count(df):
    right = df.groupby('testId')['answerCode'].count()
    return left_merge(df, right, 'testId', 'testId_count')

def get_testId_acc_mean(df):
    right = df.groupby('testId')['answerCode'].mean()
    return left_merge(df, right, 'testId', 'testId_acc_mean')

def get_testId_acc_oomean(df):
    g = df.groupby('testId')['answerCode']
    l = g.count()
    ool = l - 1
    right = DataFrame(dict(sum=g.sum(), ool=ool))
    left = df
    md = pd.merge(left, right, 'left', left_on='testId', right_index=True)
    oos = md["sum"] - df['answerCode']
    ool = md["ool"]
    return oos / ool
def get_testId_post(df:DataFrame) -> Series:
    g = df['testId'].apply(lambda row: int(row[-3:]))
    return g

# 환경설정
# 경로설정, 타입 지정
args = args.parse_args()
file_name = "train_data.csv"
csv_file_path = os.path.join(args.data_dir, file_name)

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}
df = None
def reset_df(mode='fid'):
    global df
    if mode == 'test':
        df = pd.read_csv(os.path.join(args.data_dir, 'test_data.csv'), dtype=dtype, parse_dates=['Timestamp'])
    else:
        df = pd.read_csv(csv_file_path, dtype=dtype, parse_dates=['Timestamp'])
    df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

    if mode.lower() == 'fid':
        df['original_order'] = df.index
        df['school'] = get_school(df)
        df['grade'] = get_grade(df)
        df = userID_to_familyUserID(df)
    return df

def discern_cols(df):
    cols_not_oomean = []
    cols_cate = []
    for cname, cdtype in zip(df.columns, df.dtypes):
        if 'mean' in cname and not 'oomean' in cname:
            cols_not_oomean.append(cname)
        if cdtype == 'object':
            cols_cate.append(cname)
    Cols = namedtuple("Cols", ["not_oomean", "cate", "labels"])
    retDict = Cols(cols_not_oomean, cols_cate, ["answerCode"])
    return retDict

def min_max_normalization(df):
    return (df-df.min())/(df.std())
def mean_std_normalization(df):
    return (df-df.mean())/df.std()

# endregion

################################# region 프로세스(프론트) #################################
def load_upgraded_df(path='./upgraded_df.csv'):
    dtypes = {"original_order": "int64", "userID": "object", "school": "object", "grade": "int64", "familyID": "object",
              "assessmentItemID": "object", "testId": "object", "answerCode": "int8", "KnowledgeTag": "int16",
              "familySize": "int64", "time_diff": "float64", "time_diff_toolong": "int64",
              "time_diff_userChange": "int64", "hour_sn": "float64", "hour_cs": "float64", "dayoftheweek_sn": "float64",
              "dayoftheweek_cs": "float64", "dayofthemonth_sn": "float64", "dayofthemonth_cs": "float64",
              "month_sn": "float64", "month_cs": "float64", "user_enterDay": "int64", "user_day_progress": "int64",
              "KnowledgeTag_count": "int64", "KnowledgeTag_acc_mean": "float64", "KnowledgeTag_countU": "int64",
              "KnowledgeTag_acc_meanU": "float64", "assessmentItemID_count": "int64",
              "assessmentItemID_acc_mean": "float64", "assessmentItemID_acc_oomean": "float64",
              "assessmentItemID_countU": "int64", "assessmentItemID_acc_meanU": "float64", "userID_count": "int64",
              "userID_acc_mean": "float64", "userID_acc_oomean": "float64", "userID_acc_cummean": "float64",
              "userID_acc_mean_recentK": "float64", "userID_acc_oomean_recentK": "float16",
              "userID_acc_ooWeightedMovingAverage": "float16", "userID_countT": "int64", "userID_acc_meanT": "float64",
              "userID_acc_oomeanT": "float64", "testId_count": "int64", "testId_acc_mean": "float64"}
    df = pd.read_csv(path, dtype=dtypes, parse_dates=['Timestamp'], index_col=0)
    return df


def quick_start():
    msg = '''
    import fe_backend as feb
    # df_train = feb.load_upgraded_df('./upgraded_df_train.csv')
    df_train = feb.reset_df(mode = 'base')
    df_train = feb.upgrade_df(df_train)
    df_train.to_csv('./upgraded_df_train.csv')
    cols = feb.discern_cols(df_train)

    x_train = df_train.drop(cols.not_oomean + cols.cate + cols.labels + ['Timestamp'], axis = 1)
    y_train = df_train[cols.labels]['answerCode']

    # df_test = feb.load_upgraded_df('./upgraded_df_test.csv')
    df_test = feb.reset_df('test')
    df_test = feb.upgrade_df(df_test)
    df_test.to_csv('./upgraded_df_test.csv')

    x_test = df_test.drop(cols.not_oomean + cols.cate + cols.labels + ['Timestamp'], axis = 1)
    y_test = df_test[cols.labels]['answerCode']
    '''
    print(msg)


def upgrade_df(df) -> DataFrame:
    df['original_order'] = df.index
    df['school'] = get_school(df)
    df['grade'] = get_grade(df)
    df = userID_to_familyUserID(df) # 이 부분까지는 reset에 포함돼있음
    df['familySize'] = get_familySize(df)
    df['time_diff'] = get_time_diff(df, outlier_thres = 122)
    df['time_diff_toolong'] = get_time_diff_toolong(df, outlier_thres = 120)
    df['time_diff_userChange'] = get_time_diff_userChange(df)
    df['time_diff'] = df.set_index('userID')['time_diff'].fillna(df.groupby('userID')['time_diff'].mean()).reset_index(drop = True)
    df['time_diff'] = df.set_index('testId')['time_diff'].fillna(df.groupby('testId')['time_diff'].mean()).reset_index(drop = True)
    df['hour_sn'], df['hour_cs'] = get_encoded_hour(df)
    df['dayoftheweek_sn'], df['dayoftheweek_cs'] = get_encoded_dayoftheweek(df)
    df['dayofthemonth_sn'], df['dayofthemonth_cs'] = get_encoded_dayofthemonth(df)
    df['month_sn'], df['month_cs'] = get_encoded_month(df)
    df['user_enterDay'] = get_user_enterDay(df)
    df['user_day_progress'] = get_user_day_progress(df)
    df['KnowledgeTag_count'] = get_KnowledgeTag_count(df)
    df['KnowledgeTag_acc_mean'] = get_KnowledgeTag_acc_mean(df)
    df['KnowledgeTag_countU'] = get_KnowledgeTag_countU(df)
    df['KnowledgeTag_acc_meanU'] = get_KnowledgeTag_acc_meanU(df)
    df['assessmentItemID_count'] = get_assessmentItemID_count(df)
    df['assessmentItemID_acc_mean'] = get_assessmentItemID_acc_mean(df)
    df['assessmentItemID_acc_oomean'] = get_assessmentItemID_acc_oomean(df)
    df['assessmentItemID_countU'] = get_assessmentItemID_countU(df)
    df['assessmentItemID_acc_meanU'] = get_assessmentItemID_acc_meanU(df)
    df['userID_count'] = get_userID_count(df)
    df['userID_acc_mean'] = get_userID_acc_mean(df)
    df['userID_acc_oomean'] = get_userID_acc_oomean(df)
    df['userID_acc_cummean'] = get_userID_acc_cummean(df)
    df['userID_acc_mean_recentK'] = get_userID_acc_mean_recentK(df, 10)
    df['userID_acc_mean_recentK'] = df.set_index('testId')['userID_acc_mean_recentK'].fillna(df.groupby('testId')['userID_acc_mean_recentK'].mean()).reset_index(drop = True)
    df['userID_acc_oomean_recentK'] = get_userID_acc_oomean_recentK(df, 10)
    df['userID_acc_oomean_recentK'] = df.set_index('testId')['userID_acc_oomean_recentK'].fillna(df.groupby('testId')['userID_acc_oomean_recentK'].mean()).reset_index(drop=True)
    df['userID_acc_ooWeightedMovingAverage'] = get_userID_acc_ooWeightedMovingAverage(df, [0.1,0.5,0.3,1])
    df['userID_acc_ooWeightedMovingAverage'] = df.set_index('testId')['userID_acc_ooWeightedMovingAverage'].fillna(df.groupby('testId')['userID_acc_ooWeightedMovingAverage'].mean()).reset_index(drop=True)
    df['userID_countT'] = get_userID_countT(df)
    df['userID_acc_meanT'] = get_userID_acc_meanT(df)
    df['userID_acc_oomeanT'] = get_userID_acc_oomeanT(df)
    df['testId_count'] = get_testId_count(df)
    df['testId_acc_mean'] = get_testId_acc_mean(df)
    df['testId_acc_oomean'] = get_testId_acc_oomean(df)
    df['testId_post'] = get_testId_post(df)
    return df
if __name__ == "__main__":
    # df = reset_df('fid') # fe_backend 에서 작업할 때 사용할 기본처리(grade,school,familyID,userID)만 된 df
    df_train = load_upgraded_df('./upgraded_df_train.csv')
    df_test = load_upgraded_df('./upgraded_df_test.csv')
#endregion

###
############################### 팩토리 #################################