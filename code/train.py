import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds, increment_path
from sklearn.model_selection import KFold
import inference

import numpy as np
import pandas as pd
import random

def main(args):

    args.save = False
    if args.wandb_name:
        args.save_dir = f"{args.model_dir}/{args.wandb_name}"
        args.save = True
        args.save_dir = increment_path(args.save_dir)

    elif args.is_tensor_board:
        args.save_dir = f"{args.model_dir}/{args.model}"
        args.save = True
        args.save_dir = increment_path(args.save_dir)

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess = Preprocess(args)
    if args.custom_val:
        train_df = pd.read_csv(os.path.join(args.data_dir, args.file_name))
        test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file_name))
        # test_df에서 데이터 가져오기
        # tmp = test_df[test_df['answerCode'] != -1]
        # train_df = pd.concat([train_df, tmp]) 
        
        # pseudo labeling
        pseudo_df = pd.read_csv('/opt/ml/code/output/LGBM 31개 features 1000 boost.csv')['prediction']
        pseudo_labels = np.where(pseudo_df >= 0.5, 1, 0)
        tmp = test_df.copy()
        tmp.loc[tmp.answerCode ==-1, 'answerCode'] = pseudo_labels
        train_df = pd.concat([train_df, tmp]) 

        set_assessmentItemID = set(test_df.loc[test_df.answerCode == -1, 'assessmentItemID'].values)
        val_userids = list(set(train_df[(train_df['userID'] != train_df['userID'].shift(-1)) & (train_df.assessmentItemID.isin(set_assessmentItemID))]['userID']))
        random.seed(args.seed)
        random.shuffle(val_userids)
    else:
        preprocess.load_train_data(args.file_name)
        data = preprocess.get_train_data()

    if args.n_fold:
        if args.custom_val:
            length = len(val_userids)
            for i in range(args.n_fold):
                valid_df = train_df[train_df.userID.isin(val_userids[(length*i)//args.n_fold:length*(i+1)//args.n_fold])]
                train_df_ = train_df[~train_df.userID.isin(val_userids[(length*i)//args.n_fold:length*(i+1)//args.n_fold])]
                preprocess.custom_load_train_data(valid_df)
                valid_data = preprocess.get_train_data()
                preprocess.custom_load_train_data(train_df_)
                train_data = preprocess.get_train_data()
                trainer.run(args, train_data, valid_data, i)
        else:    
            k_fold = KFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)
            for fold, (train_idx, valid_idx) in enumerate(k_fold.split(data)):
                train_data = data[train_idx]
                valid_data = data[valid_idx]          

        inference.kfold_direct_inference(args)
    else:
        if args.custom_val:
            valid_df = train_df[train_df.userID.isin(val_userids[:(len(val_userids)//4)])]
            train_df = train_df[~train_df.userID.isin(val_userids[:(len(val_userids)//4)])]

            preprocess.custom_load_train_data(train_df)
            train_data = preprocess.get_train_data()
            preprocess.custom_load_train_data(valid_df)
            valid_data = preprocess.get_train_data()
        else:
            train_data, valid_data = preprocess.split_data(data, ratio=args.split_ratio)
        
        trainer.run(args, train_data, valid_data)
        inference.main(args)

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)