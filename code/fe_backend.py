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
from numpy import NaN
from cyclical import cyclical
from sklearn import linear_model
from tqdm import tqdm

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

def get_grade(df:DataFrame) -> Series:
    g = df['testId'].apply(lambda row: int(row[2]))
    return g
def get_school(df:DataFrame) -> Series:
    g = df['testId'].apply(lambda row: int(row[2]))
    school = g.apply(lambda row: 'elementary_school' if row // 7 == 0 else 'middle_school')
    return school

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
    df['familyID'] = df['userID']
    tqdm.pandas()
    new_userID = df.progress_apply(lambda row: userID_dict[row['userID'], row['school'], row['grade']], axis=1)

    df['userID'] = new_userID
    cols = df.columns.to_list()
    new_cols = [cols.pop(-1), *cols]
    df = df[new_cols]

    return df

def get_time_diff(df:DataFrame, outlier_thres = 80) -> Series:
    dg: DataFrameGroupBy = df.groupby('userID')
    df['Timestamp_after'] = dg['Timestamp'].shift(-1)
    td: Series = df.groupby('userID').apply(lambda row: row['Timestamp_after'] - row['Timestamp'])
    df.drop(columns='Timestamp_after', inplace=True)
    td = td.reset_index(drop=True)
    td = td.dt.seconds
    td.loc[td >= outlier_thres] = NaN  # outlier 처리

    time_diff = td
    return time_diff

def get_encoded_hour(df: DataFrame) -> Series:
    hour = df['Timestamp'].dt.hour
    hr_sin, hr_cos = cyclical.encode(hour, 24)
    en_hour = Series(zip(hr_sin, hr_cos))
    return en_hour

def get_encoded_dayoftheweek(df: DataFrame) -> Series:
    dw = df['Timestamp'].dt.dayofweek
    dw_sin, dw_cos = cyclical.encode(dw, 7)
    en_dw = Series(zip(dw_sin, dw_cos))
    return en_dw

def get_encoded_dayofthemonth(df: DataFrame) -> Series:
    dm = df['Timestamp'].dt.day
    dm_sin, dm_cos = cyclical.encode(dm, 31)
    en_dm = Series(zip(dm_sin, dm_cos))
    return en_dm

def get_encoded_month(df: DataFrame) -> Series:
    m = df['Timestamp'].dt.month
    m_sin, m_cos = cyclical.encode(m, 12)
    en_m = Series(zip(m_sin, m_cos))
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
    return df.groupby('KnowledgeTag')['answerCode'].count()

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
    return df.groupby('assessmentItemID')['answerCode'].count()

def get_assessmentItemID_acc_mean(df):
    return df.groupby('assessmentItemID')['answerCode'].mean()

def get_assessmentItemID_countU(df):
    right = Series(df.groupby(['userID', 'assessmentItemID'])['answerCode'].count(), name="assessmentItemID_countU")
    left = df
    return pd.merge(left, right, 'left', left_on=['userID', 'assessmentItemID'], right_index=True)['assessmentItemID_countU']

def get_assessmentItemID_acc_meanU(df):
    right = Series(df.groupby(['userID', 'assessmentItemID'])['answerCode'].mean(), name="assessmentItemID_acc_meanU")
    left = df
    return pd.merge(left, right, 'left', left_on=['userID', 'assessmentItemID'], right_index=True)['assessmentItemID_acc_meanU']

def get_userID_count(df):
    return df.groupby('userID')['answerCode'].count()

def get_userID_acc_mean(df):
    return df.groupby('userID')['answerCode'].mean()

def get_userID_acc_cummean(df):
    return df.groupby('userID')['answerCode'].expanding().mean()

def get_userID_acc_mean_recentK(df, K):
    return df.groupby('userID')['answerCode'].rolling(min_periods = 1, window = K).mean()

def get_userID_acc_meanT(df):
    right = Series(df.groupby(['userID', 'testId'])['answerCode'].mean(), name = "userID_acc_meanT")
    left = df
    return pd.merge(left, right, 'left', left_on=['userID', 'testId'], right_index=True)["userID_acc_meanT"]

def get_userID_countT(df):
    return df.groupby(['userID', 'testId'])['answerCode'].count()

def get_testId_count(df):
    return df.groupby('testId')['answerCode'].count()

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
def reset():
    global df
    df = pd.read_csv(csv_file_path, dtype=dtype, parse_dates=['Timestamp'])
    df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    df['original_order'] = df.index
    df['school'] = get_school(df)
    df['grade'] = get_grade(df)
    df = userID_to_familyUserID(df)
    return df
# endregion

################################# region 프로세스 #################################
if __name__ == "__main__123":
    reset() # reset dataframe
    df['original_order'] = df.index
    df['school'] = get_school(df)
    df['grade'] = get_grade(df)
    df = userID_to_familyUserID(df) # 이 부분까지는 reset에 포함돼있음
    df['time_diff'] = get_time_diff(df, outlier_thres = 80)
    df['encoded_hour'] = get_encoded_hour(df)
    df['encoded_dayoftheweek'] = get_encoded_dayoftheweek(df)
    df['encoded_dayofthemonth'] = get_encoded_dayofthemonth(df)
    df['encoded_month'] = get_encoded_month(df)
    df['user_enterDay'] = get_user_enterDay(df)
    df['user_day_progress'] = get_user_day_progress(df)
    df['KnowledgeTag_count'] = get_KnowledgeTag_count(df)
    df['KnowledgeTag_acc_mean'] = get_KnowledgeTag_acc_mean(df)
    df['KnowledgeTag_countU'] = get_KnowledgeTag_countU(df)
    df['KnowledgeTag_acc_meanU'] = get_KnowledgeTag_acc_meanU(df)
    df['assessmentItemID_count'] = get_assessmentItemID_count(df)
    df['assessmentItemID_acc_mean'] = get_assessmentItemID_acc_mean(df)
    df['assessmentItemID_countU'] = get_assessmentItemID_countU(df)
    df['assessmentItemID_acc_meanU'] = get_assessmentItemID_acc_meanU(df)
    df['userID_count'] = get_userID_count(df)
    df['user_acc_mean'] = get_userID_acc_mean(df)
#endregion
################################# 팩토리 #################################
df = reset()

pd.set_option('display.float_format', "{:.2f}".format)
pd.set_option('display.max_rows', 10)

y = get_testId_count(df)
y.describe()
y.mode()
y.median()
sum(y == y.mode()[0]) / len(y)
sum(y.isin(range(1300,1400))) / len(y)

sns.displot(y, bins = 10)
# sns.regplot(y.index, y)
plt.show()