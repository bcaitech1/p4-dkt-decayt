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
    for i in tqdm(range(len(new_users))):
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
    return df
# endregion
################################# region 프로세스 #################################
reset() # reset dataframe
df['original_order'] = df.index
df['school'] = get_school(df)
df['grade'] = get_grade(df)
df = userID_to_familyUserID(df)
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
dev = False
if dev is True:
    #################### region school, grade
    # 1. element-wise
    '''
    testID 의 세번째 원소는 학년(1,2,3,4,5,6,7,8,9)
    '''
    g = df['testId'][0][2]
    g = int(g)
    school = g//7

    # 2. DataFrame-wise
    g = df['testId'].apply(lambda row: int(row[2]))
    school = g.apply(lambda row: 'elementary_school' if row // 7 == 0 else 'middle_school')

    # 3. distribution_correction
    sns.displot(school, bins = 2) # distribution
    plt.show()
    sns.displot(g, bins = 6)
    plt.show()

    # 5. Implementation
    def get_grade(df:DataFrame) -> Series:
        g = df['testId'].apply(lambda row: int(row[2]))
        return g
    def get_school(df:DataFrame) -> Series:
        g = df['testId'].apply(lambda row: int(row[2]))
        school = g.apply(lambda row: 'elementary_school' if row // 7 == 0 else 'middle_school')
        return school

    ############## endregion
    #################### region new_userID, familyID

    # 1. generate new_userID dictionary
    '''
    school, grade 가 다르면 familyID 를 생성하고, userID 를 변경
    '''
    df.set_index(['userID', 'school', 'grade'], inplace = True)

    new_users = df.index.unique()
    userID_dict = dict.fromkeys(new_users, "17752-1")
    k = 0
    before_uid = None
    for i in tqdm(range(len(new_users))):
        item = new_users[i]
        if before_uid != item[0]:
            before_uid = item[0]
            k = 0
        k += 1
        userID_dict[item] = f"{item[0]}-{k}"

    # 2. assign familyID & replace userID with new_userID
    df.reset_index(inplace = True)
    df['familyID'] = df['userID']
    tqdm.pandas()
    new_userID = df.progress_apply(lambda row: userID_dict[row['userID'], row['school'], row['grade']], axis = 1)

    df['userID'] = new_userID
    cols = df.columns.to_list()
    new_cols = [cols.pop(-1), *cols]
    df = df[new_cols]

    # 3. modification check
    ids_that_should_not_exist = ['3-1','4-1']
    for id in ids_that_should_not_exist:
        assert not id in df['userID'].unique(), "something wrong"
    print(len(df['userID'].unique()))

    # 4. Implementation
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
        for i in tqdm(range(len(new_users))):
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

    ############## endregion
    #################### region time_diff
    # 1. element-wise
    '''
    1번을 푸는데 걸린 시각 = 2번 문제 시작 - 1번 문제 시작
    '''
    time_after = df['Timestamp'][1]
    time_before = df['Timestamp'][0]
    td:pd.Timedelta = time_after - time_before

    # 2. DataFrame-wise
    dg:DataFrameGroupBy = df.groupby('userID')
    df['Timestamp_after'] = dg['Timestamp'].shift(-1)
    td:Series = df.groupby('userID').apply(lambda row: row['Timestamp_after'] - row['Timestamp'])
    td = td.reset_index(drop = True)
    td = td.dt.seconds
    df['time_diff'] = td

    # 3. distribution_correction
    outlier_thres = 80
    td[td >= outlier_thres] = NaN # outlier 처리
    print(f"결측율 : {sum(td.isna()) / len(td)}")
    sns.displot(td, bins = 10) # distribution
    plt.show()
    df['time_diff'] = td

    # 4. Imputation : 추후 구현
    # reg_data = df[df['time_diff'].isna() == False]
    # yvar = 'time_diff'
    # xvars = ['question_difficulty', 'person_proficiency']
    # lin_regress(reg_data, yvar, xvars)

    # 5. Implementation
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

    ############## endregion
    #################### region encoded_hour
    # 1. DataFrame-wise
    '''
    월화수목금토일
    '''
    hour = df['Timestamp'].dt.hour
    hr_sin, hr_cos = cyclical.encode(hour, 24)
    en_hour = Series(zip(hr_sin, hr_cos))

    # 3. distribution_correction
    sns.displot(hour, bins = 24)  # distribution
    plt.show()

    # 4. Imputation : 추후 구현
    # reg_data = df[df['time_diff'].isna() == False]
    # yvar = 'time_diff'
    # xvars = ['question_difficulty', 'person_proficiency']
    # lin_regress(reg_data, yvar, xvars)

    # 5. Implementation
    def get_encoded_hour(df: DataFrame) -> Series:
        hour = df['Timestamp'].dt.hour
        hr_sin, hr_cos = cyclical.encode(hour, 24)
        en_hour = Series(zip(hr_sin, hr_cos))
        return en_hour

    ############## endregion
    #################### region encoded_dayoftheweek
    # 1. DataFrame-wise
    '''
    월화수목금토일
    '''
    dw = df['Timestamp'].dt.dayofweek
    dw_sin, dw_cos = cyclical.encode(dw, 7)
    en_dw = Series(zip(dw_sin, dw_cos))

    # 3. distribution_correction
    sns.displot(dw, bins=7)  # distribution
    plt.show()

    # 4. Imputation : 추후 구현
    # reg_data = df[df['time_diff'].isna() == False]
    # yvar = 'time_diff'
    # xvars = ['question_difficulty', 'person_proficiency']
    # lin_regress(reg_data, yvar, xvars)

    # 5. Implementation
    def get_encoded_dayoftheweek(df: DataFrame) -> Series:
        dw = df['Timestamp'].dt.dayofweek
        dw_sin, dw_cos = cyclical.encode(dw, 7)
        en_dw = Series(zip(dw_sin, dw_cos))
        return en_dw

    ############## endregion
    #################### region encoded_dayofthemonth
    # 1. DataFrame-wise
    '''
    월화수목금토일
    '''
    dm = df['Timestamp'].dt.day
    dm_sin, dm_cos = cyclical.encode(dm, 31)
    en_dm = Series(zip(dm_sin, dm_cos))

    # 3. distribution_correction
    sns.displot(dm, bins=31)  # distribution
    plt.show()

    # 4. Imputation : 추후 구현
    # reg_data = df[df['time_diff'].isna() == False]
    # yvar = 'time_diff'
    # xvars = ['question_difficulty', 'person_proficiency']
    # lin_regress(reg_data, yvar, xvars)

    # 5. Implementation
    def get_encoded_dayofthemonth(df: DataFrame) -> Series:
        dm = df['Timestamp'].dt.day
        dm_sin, dm_cos = cyclical.encode(dm, 31)
        en_dm = Series(zip(dm_sin, dm_cos))
        return en_dm
    ############## endregion
    #################### region encoded_month
    # 1. DataFrame-wise
    '''
    1일~30일
    '''
    m = df['Timestamp'].dt.month
    m_sin, m_cos = cyclical.encode(m, 12)
    en_m = Series(zip(m_sin, m_cos))

    # 2. distribution_correction
    sns.displot(m, bins=12)  # distribution
    plt.show()

    # 3. Imputation : 추후 구현
    # reg_data = df[df['time_diff'].isna() == False]
    # yvar = 'time_diff'
    # xvars = ['question_difficulty', 'person_proficiency']
    # lin_regress(reg_data, yvar, xvars)

    # 4. Implementation
    def get_encoded_month(df: DataFrame) -> Series:
        m = df['Timestamp'].dt.month
        m_sin, m_cos = cyclical.encode(m, 12)
        en_m = Series(zip(m_sin, m_cos))
        return en_m

    ############## endregion
    #################### region user_enterDay
    # 1. DataFrame-wise
    '''
    1일차, 2일차, ...
    '''
    dg:DataFrameGroupBy = df.groupby('userID')
    dg['Timestamp'].min()
    df = pd.merge(left = df, right = DataFrame(Series(dg['Timestamp'].min(), name = 'Timestamp_start')), how = 'left', left_on = "userID", right_index=True)
    timestamp_the_first_start = df['Timestamp'].min()
    tss = df.groupby('userID')['Timestamp_start']
    day_elapse = tss.apply(lambda row: row - timestamp_the_first_start).dt.days

    # 2. distribution_correction
    sns.displot(day_elapse, bins = 20) # distribution
    plt.show()

    # 3. Imputation : 추후 구현
    # reg_data = df[df['time_diff'].isna() == False]
    # yvar = 'time_diff'
    # xvars = ['question_difficulty', 'person_proficiency']
    # lin_regress(reg_data, yvar, xvars)

    # 4. Implementation
    def get_user_enterDay(df:DataFrame) -> Series:
        dg: DataFrameGroupBy = df.groupby('userID')
        dg['Timestamp'].min()
        df = pd.merge(left=df, right=DataFrame(Series(dg['Timestamp'].min(), name='Timestamp_start')), how='left',
                      left_on="userID", right_index=True)
        timestamp_the_first_start = df['Timestamp'].min()
        tss = df.groupby('userID')['Timestamp_start']

        day_elapse = tss.apply(lambda row: row - timestamp_the_first_start).dt.days
        return day_elapse
    ############## endregion
    #################### region user_day_progress
    # 1. DataFrame-wise
    '''
    1일차, 2일차, ...
    '''
    dg: DataFrameGroupBy = df.groupby('userID')
    dg['Timestamp'].min()
    df = pd.merge(left=df, right=DataFrame(Series(dg['Timestamp'].min(), name='Timestamp_start')), how='left',
                  left_on="userID", right_index=True)
    tqdm.pandas()
    user_day_progress = df.progress_apply(lambda row: row['Timestamp'] - row['Timestamp_start'], axis = 1)

    # 2. distribution_correction
    df['user_day_progress'] = user_day_progress.dt.days
    sns.displot(df.groupby('userID')['user_day_progress'].max(), bins=30)  # distribution
    plt.show()


    # 3. Imputation : 추후 구현
    # reg_data = df[df['time_diff'].isna() == False]
    # yvar = 'time_diff'
    # xvars = ['question_difficulty', 'person_proficiency']
    # lin_regress(reg_data, yvar, xvars)

    # 4. Implementation
    def get_user_day_progress(df: DataFrame) -> Series:
        dg: DataFrameGroupBy = df.groupby('userID')
        dg['Timestamp'].min()
        df = pd.merge(left=df, right=DataFrame(Series(dg['Timestamp'].min(), name='Timestamp_start')), how='left',
                      left_on="userID", right_index=True)
        tqdm.pandas()
        user_day_progress = df.progress_apply(lambda row: row['Timestamp'] - row['Timestamp_start'], axis=1)
        return user_day_progress.dt.days

    ############## endregion