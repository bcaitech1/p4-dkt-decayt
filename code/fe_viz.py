from fe_backend import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import sleep
import fe_backend as feb
from scipy.stats import mannwhitneyu

if __name__ == "__main__":
    df = feb.load_upgraded_df('./upgraded_df_train.csv')
    df_train = df
    cols = discern_cols(df_train)
    x_train = df_train.drop(cols.not_oomean + cols.cate + cols.labels + ['Timestamp'], axis=1)
    y_train = df_train[cols.labels]['answerCode']

    df_test = load_upgraded_df('./upgraded_df_test.csv')
    x_test = df_test.drop(cols.not_oomean + cols.cate + cols.labels + ['Timestamp'], axis=1)
    y_test = df_test[cols.labels]['answerCode']

    pd.set_option('display.float_format', "{:.2f}".format)
    pd.set_option('display.max_rows', 1000)

def viz_feature(case = ""):
    if case == "userID_countT":
        k = 3303
        userID = df['userID'].unique()[k]
        z = df[df.userID == userID].testId.apply(lambda row: int(row[7:]))[:100]
        sns.regplot(z.index, z)
        plt.show()
    if case == "assessmentItemID_acc_oomean":
        y = df["assessmentItemID_acc_oomean"]
        y.describe(percentiles = [0.5])
        sum(y == y.mode()[0]) / len(y)
        sum(y.between(0.2,1.0)) / len(y)
        sns.displot(y)
        plt.show()
    if case == "userID_acc_oomean":
        y = df["userID_acc_oomean"]
        y.describe(percentiles=[0.5])
        sum(y == y.mode()[0]) / len(y)
        sum(y.between(0.2, 1.0)) / len(y)
        sns.displot(y)
        plt.show()
    if case == "userID_acc_oomean_recentK":
        familySize = 3
        userID_pool = df[(df['familySize'] == familySize) & (df['userID_count'] > 500)]['userID'].unique()
        for k in range(0, min(50, len(userID_pool))):
            plt.figure()
            userID = userID_pool[k]
            z1 = df[df.userID == userID].userID_acc_oomean_recentK[:500]
            # z2 = df[df.userID == userID].testId_acc_mean[50:500]
            xticklabels = df[df.userID == userID]['Timestamp'].dt.month[:500]
            ax = sns.regplot(z1.index, z1, fit_reg = False)
            # sns.regplot(z2.index, z2, fit_reg = False)
            freq = 30
            step = len(xticklabels)/freq
            xticklabels = [xticklabels.iloc[int(i*step)] for i in range(0, freq)]
            ax.set_yticks(np.arange(0, 1, 0.1))
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('month')
            ax.set_title(f"familySize = {familySize}")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_locator(plt.LinearLocator(numticks=10))
            ax.xaxis.set_major_locator(plt.LinearLocator(numticks=freq))
            # plt.show()
            plt.savefig(f'plots/{familySize}_{k}.png')

    if case == "answerCode":
        for k in range(0,100):
            userID = df[df['familySize'] == 1]['userID'].unique()[k]
            z = df[df.userID == userID].answerCode
            sns.regplot(z.index, z, scatter_kws={'s':0.01})
            plt.show();print(userID)
            sleep(1)
    if case == "userID_acc_mean_recentK":
        k = 17
        userID = df[df['familySize'] == 1]['userID'].unique()[k]
        z = df[df.userID == userID].userID_acc_mean_recentK
        sns.regplot(z.index, z)
        plt.show();print(userID)
    if case == 'userID_acc_oomeanT':
        y = df["userID_acc_oomeanT"]
        y.describe(percentiles=[0.5])
        sum(y == y.mode()[0]) / len(y)
        sum(y.between(0.2, 1.0)) / len(y)
        sns.displot(y)
        plt.show()
    if case == 'familySize':
        y = df['familySize']
        y.describe(percentiles=[0.5])
        sum(y == y.mode()[0]) / len(y)
        sum(y.between(0.2, 1.0)) / len(y)
        sns.displot(y)
        plt.show()
    if case == 'time_diff':
        outlier_thres = 1200
        y = get_time_diff(df, outlier_thres=outlier_thres)
        df['time_diff'] = y
        df.answerCode.mean()
        df[y > 300].answerCode.mean()
        df[['answerCode', 'time_diff']].corr()

        len(y[y < 1200]) / len(df)
        sns.displot(y[y<1200], bins = 20)
        plt.show()

        k = 0
        userID = df['userID'].unique()[k]
        z1 = df[df.userID == userID]
        z2 = df.groupby('testId')['time_diff'].mean()
        z3 = df.groupby('assessmentItemID')['time_diff'].mean()
        z4 = df.groupby('KnowledgeTag')['time_diff'].mean()
        sns.regplot(z1.index, z1['time_diff'])
        sns.regplot(z1.index, z2)
        sns.regplot(z1.index, z3)
        sns.regplot(z1.index, z4)
        plt.show()

    if case == 'userID_acc_ooWeightedMovingAverage':
        k = 0

        w1 = [1, 0, 0, 0]
        w2 = [0.1, 0.8, 0.1, 0]
        w3 = [0.1,0.5,0.3,0.1]
        w4 = [0, 0.1, 0.2, 0.7]

        s00 = "assessID_acc"
        s0 = "userID_acc"
        s1 = "3"
        s2 = "w_short"
        s3 = "w_mid"
        s4 = "w_long"

        # 5 10 20 40
        '''
        df['userID_acc_ooWeightedMovingAverage1'] = get_userID_acc_ooWeightedMovingAverage(df, w1)
        df['userID_acc_ooWeightedMovingAverage2'] = get_userID_acc_ooWeightedMovingAverage(df, w2)
        df['userID_acc_ooWeightedMovingAverage3'] = get_userID_acc_ooWeightedMovingAverage(df, w3)
        df['userID_acc_ooWeightedMovingAverage4'] = get_userID_acc_ooWeightedMovingAverage(df, w4)
        '''

        familySize = 1
        userID_pool = df[(df['familySize'] == familySize) & (df['userID_count'] > 500)]['userID'].unique()
        userID = userID_pool[k]
        z00 = df[df.userID == userID].assessmentItemID_acc_meanU[:500]
        z0 = df[df.userID == userID].userID_acc_oomean[:500]
        z1 = df[df.userID == userID].userID_acc_ooWeightedMovingAverage1[:500]
        z2 = df[df.userID == userID].userID_acc_ooWeightedMovingAverage2[:500]
        z3 = df[df.userID == userID].userID_acc_ooWeightedMovingAverage3[:500]
        z4 = df[df.userID == userID].userID_acc_ooWeightedMovingAverage4[:500]
        zz = df[df.userID == userID].answerCode[:500]
        xticklabels = df[df.userID == userID]['Timestamp'].dt.month[:500]

        ax = sns.regplot(z00.index, z00, fit_reg=False, scatter_kws={'s': 1}, label=s00)
        # sns.regplot(z0.index, z0, fit_reg=False, scatter_kws={'s': 1}, label=s0)
        # sns.regplot(z1.index, z1, fit_reg=False, scatter_kws={'s':3}, label = s1)
        # sns.regplot(z2.index, z2, fit_reg = False, scatter_kws={'s':3}, label = s2)
        # sns.regplot(z3.index, z3, fit_reg=False, scatter_kws={'s':3}, label = s3)
        # sns.regplot(z4.index, z4, fit_reg=False, scatter_kws={'s': 3}, label=s4)
        sns.regplot(zz.index, zz, fit_reg=False, scatter_kws={'s': 3}, label="1")

        # sns.lineplot(z1.index, z1, label = s1)
        # sns.lineplot(z2.index, z2, label = s2)
        # sns.lineplot(z3.index, z3, label = s3)
        sns.lineplot(z4.index, z4, label=s4)

        freq = 30
        step = len(xticklabels) / freq
        xticklabels = [xticklabels.iloc[int(i * step)] for i in range(0, freq)]
        ax.set_yticks(np.arange(0, 1, 0.1))
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('month')
        ax.set_title(f"familySize = {familySize}")

        ax.set_ylim(-0.05,1.05)
        ax.yaxis.set_major_locator(plt.LinearLocator(10))
        ax.xaxis.set_major_locator(plt.LinearLocator(freq))

        plt.show()

    if case == 'testId_acc_oomean':
        y = df['testId_acc_oomean']
        sns.displot(y, bins = 100)
        plt.show()

def viz_corr_heatmap():
    cols = ["KnowledgeTag_acc_mean","KnowledgeTag_acc_meanU","assessmentItemID_acc_mean","assessmentItemID_acc_meanU","userID_acc_mean","userID_acc_mean_recentK","userID_acc_meanT","testId_acc_mean","assessmentItemID_countU","userID_count","answerCode"]
    df2 = df[cols]
    fig, ax = plt.subplots()
    fig.set_size_inches(12,12)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    fig.subplots_adjust(left=0.2, bottom=0.2)
    sns.heatmap(df2.corr(), annot = True, fmt = '.2f', cmap = 'Blues')
    fig.savefig('plot.png')

def viz_corr_heatmap2():
    df['time_diff'] = get_time_diff(df, outlier_thres = 15)

    cols = ['userID_acc_ooWeightedMovingAverage1', 'userID_acc_ooWeightedMovingAverage2', 'userID_acc_ooWeightedMovingAverage3', 'userID_acc_ooWeightedMovingAverage4', "time_diff", "assessmentItemID_acc_oomean","userID_acc_oomean","userID_acc_oomeanT", "userID_acc_oomean_recentK", "assessmentItemID_countU","userID_count","answerCode"]
    df2 = df[cols]
    fig, ax = plt.subplots()
    fig.set_size_inches(12,12)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    fig.subplots_adjust(left=0.2, bottom=0.2)
    sns.heatmap(df2.corr(), annot = True, fmt = '.2f', cmap = 'Blues')
    fig.savefig('plot.png')
    plt.show()

def viz_corr_heatmap3():
    df['time_diff'] = get_time_diff(df, outlier_thres = 15)

    cols = ['userID_acc_ooWeightedMovingAverage1', 'userID_acc_ooWeightedMovingAverage2', 'userID_acc_ooWeightedMovingAverage3', 'userID_acc_ooWeightedMovingAverage4', "assessmentItemID_acc_oomean","userID_acc_oomean", "userID_count","answerCode"]
    df2 = df[cols]
    fig, ax = plt.subplots()
    fig.set_size_inches(12,12)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    fig.subplots_adjust(left=0.2, bottom=0.2)
    sns.heatmap(df2.corr(), annot = True, fmt = '.2f', cmap = 'Blues')
    fig.savefig('plot.png')
    plt.show()

def viz_corr_heatmap4():
    cols = discern_cols(df)
    cols.keys()
    conti_features = df.drop(cols['cols_not_oomean']+cols['cols_cate']+cols['labels'], axis = 1)
    conti_features['answerCode'] = df['answerCode']
    fig, ax = plt.subplots()
    fig.set_size_inches(32,32)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    fig.subplots_adjust(left=0.2, bottom=0.2)
    sns.heatmap(conti_features.corr(), annot = True, fmt = '.2f', cmap = 'Blues')
    fig.savefig('plot.png')
    plt.show()

def logistic():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    cols = discern_cols(df)
    cols.keys()
    conti_features = df.drop(cols['cols_not_oomean'] + cols['cols_cate'] + cols['labels']+['Timestamp'], axis=1)
    labels = df['answerCode']
    model.fit(conti_features, labels)
    model.score(conti_features, labels)

def viz_train_test_dists():
    result = []
    i = 0
    f, axes = plt.subplots(1, 2, sharex=True)
    f: plt.Figure = f
    axes: plt.Axes = axes

    for x_kv, y_kv in zip(x_train.iteritems(), x_test.iteritems()):
        name = x_kv[0]
        x = x_kv[1]
        y = y_kv[1]

        f.suptitle(name)
        f.tight_layout()
        sns.histplot(x, ax=axes[0], bins=100)
        axes[0].set_title('train')
        sns.histplot(y, ax=axes[1], bins=100)
        axes[1].set_title('test')

        result.append((name, *mannwhitneyu(x, y)))

        i += 1
        plt.savefig(f'./plots/{i}_{name}.png')
        plt.cla()
        plt.clf()

def viz_model(save_dir:str, preds, x_test:DataFrame, y_test:DataFrame, x_train:DataFrame, x_reference:DataFrame):
    i = 0

    x_correct = x_test[y_test == preds]
    x_wrong = x_test[y_test != preds]

    for ref, train, test, correct, wrong in zip(x_reference.iteritems(), x_train.iteritems(), x_test.iteritems(), x_correct.iteritems(), x_wrong.iteritems()):
        f, axes = plt.subplots(1, 3, sharex=True)
        f: plt.Figure = f
        axes: plt.Axes = axes

        name = ref[0]
        x1 = ref[1]
        x2 = train[1]
        x3 = test[1]
        x4 = correct[1]
        x5 = wrong[1]

        f.suptitle(name)
        f.tight_layout()
        sns.histplot(x1, ax=axes[0], color = '#7aa0c4')
        axes[0].set_title('x_reference')

        sns.histplot(x2, ax=axes[1], color = '#8bcd50', label = "train")
        sns.histplot(x3, ax=axes[1], color = '#ca82e1', label = "test")
        axes[1].set_title('train / test')
        axes[1].legend(prop={'size': 8})

        sns.histplot(x4, ax=axes[2], color = '#8bcd50', label = "correct")
        sns.histplot(x5, ax=axes[2], color = '#ca82e1', label = "wrong")
        axes[2].set_title('correct / wrong')
        axes[2].legend(prop={'size': 8})

        i += 1
        if save_dir is None:
            save_dir = './plots'
        plt.savefig(os.path.join(save_dir, f'{i}_{name}.png'))
        plt.cla()
        plt.clf()