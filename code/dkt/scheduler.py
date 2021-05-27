
from torch.optim.lr_scheduler import *
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer, args):
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode='max', verbose=True)
    elif args.scheduler == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.total_steps)
    elif args.scheduler == 'steplr':
        #gamma : 한 스텝마다 현재 learning rate에 곱할 값
        scheduler = StepLR(optimizer, 1, gamma=0.9)  # 794) #gamma : 20epoch => lr x 0.01

    return scheduler