import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch

import pandas as pd

def kfold_direct_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    if args.custom_val:
        test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file_name))
        preprocess.custom_load_test_data(test_df, train_df=args.file_name)
        test_data = preprocess.get_test_data()
    else:
        preprocess = Preprocess(args)
        preprocess.load_test_data(args.test_file_name, train_df=args.file_name)
        test_data = preprocess.get_test_data()

    trainer.kfold_inference(args, test_data)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    preprocess = Preprocess(args)
    
    if args.custom_val:
        test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file_name))
        preprocess.custom_load_test_data(test_df, train_df=args.file_name)
        test_data = preprocess.get_test_data()
    else:
        
        preprocess.load_test_data(args.test_file_name, train_df=args.file_name)
        test_data = preprocess.get_test_data()
    
    trainer.inference(args, test_data)
    
if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)