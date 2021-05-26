import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds, increment_path
import wandb

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
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    
    train_data, valid_data = preprocess.split_data(train_data)

    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)