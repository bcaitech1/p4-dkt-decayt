import os
import torch
import numpy as np
import wandb
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .utils import get_lr
from .model import LSTM, LSTMATTN, Bert, LastQueryTransformer, LastQuery, Saint
from .augmentation import data_augmentation

def run(args, train_data, valid_data, fold=""):
    # tensorboard logger define
    logger = SummaryWriter(log_dir=args.save_dir+f'/{fold}')
    if args.aug:
        train_data = data_augmentation(train_data, args)
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10
            
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    # args에 wandb_name을 설정해주었을때만 wandb로 저장하도록 설정하였습니다.
    if args.wandb_name:
        wandb.init(project='dkt', config=vars(args))
        wandb.run.name = args.wandb_name

        wandb.watch(model)

    best_auc = -1
    best_epoch = 0
    early_stopping_counter = 0
    for epoch in tqdm(range(args.n_epochs)):
        print(f"[{fold}] Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        if args.scheduler == 'linear_warmup':
            train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args, fold, scheduler)
        else:
            train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args, fold)
        ### VALID
        auc, acc,_ , _ = validate(valid_loader, model, args, fold)

        ### LEARNING RATE CAPTURE
        current_lr = get_lr(optimizer)

        ### TODO: model save or early stopping
        if args.wandb_name:
            wandb.log({"learning_rate": current_lr,
                       "train_loss": train_loss,
                       "train_auc": train_auc,
                       "train_acc": train_acc,
                       "valid_auc": auc,
                       "valid_acc": acc})
        elif args.is_tensor_board:
            logger.add_scalar("Train/train_loss", train_loss, epoch * len(train_loader))
            logger.add_scalar("Train/train_auc", train_auc, epoch * len(train_loader))
            logger.add_scalar("Train/train_acc", train_acc, epoch * len(train_loader))
            logger.add_scalar("Valid/valid_auc", auc, epoch * len(train_loader))
            logger.add_scalar("Valid/valid_acc", acc, epoch * len(train_loader))
            logger.add_scalar("Train/Learning_Rate", current_lr, epoch * len(train_loader))
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            if args.save:
                model_to_save = model.module if hasattr(model, 'module') else model
                save_checkpoint(
                    state={
                        'epoch': epoch + 1,
                        'state_dict': model_to_save.state_dict(),
                    },
                    model_dir=args.model_dir,
                    model_filename=f"{args.model_name}{fold}.pt",
                )

            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'[{fold}] EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        

    if args.save:
        config_path = f"{args.model_dir}{args.model_name}_config.json"
        if os.path.exists(config_path):
            with open(config_path) as f:
                save_args = json.load(f)
        else:
            save_args = vars(args)
        save_args.update({f"[BEST AUC]{args.model_name}{fold}.pt": best_auc})
        save_args.update({f"best_epoch": best_epoch})
        save_args.update({f"end_epoch": epoch + 1})


        json.dump(
            save_args,
            open(f"{args.model_dir}{args.model_name}_config.json", "w"),
            indent=2,
            ensure_ascii=False,
        )

def train(train_loader, model, optimizer, args, fold="", scheduler=None):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        # items -> test, question, tag, correct, mask, interaction, gather_index
        items = process_batch(batch, args)
        preds = model(items)
        targets = items[3]  # correct

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)
        if args.scheduler == 'linear_warmup':
            scheduler.step()
        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        
        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        preds = preds.to('cpu').detach().numpy()
        targets = targets.to('cpu').detach().numpy()
        
        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'[{fold}]TRAIN AUC : {auc} ACC : {acc}')

    return auc, acc, loss_avg
    

def validate(valid_loader, model, args, fold=""):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        # items -> test, question, tag, correct, mask, interaction, gather_index
        items = process_batch(batch, args)
        preds = model(items)
        targets = items[3]  # correct

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]
    
        preds = preds.to('cpu').detach().numpy()
        targets = targets.to('cpu').detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    print(f'[{fold}]VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets


def inference(args, test_data):

    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []
    for step, batch in enumerate(test_loader):
        # items -> test, question, tag, correct, mask, interaction, gather_index
        items = process_batch(batch, args)
        preds = model(items)

        # predictions
        preds = preds[:, -1]
        preds = preds.to('cpu').detach().numpy()

        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, "output.csv")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, pred in enumerate(total_preds):
            w.write('{},{}\n'.format(id, pred))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm':
        model = LSTM(args)
    if args.model == 'lstmattn':
        model = LSTMATTN(args)
    if args.model == 'bert':
        model = Bert(args)
    if args.model == 'lastquery_old':
        model = LastQueryTransformer(args)
    if args.model == 'lastquery':
        model = LastQuery(args)
    if args.model == 'saint':
        model = Saint(args)
        
    return model.to(args.device)


# 배치 전처리
def process_batch(batch, args):
    if args.fversion >=2:
        test, question, tag, correct, conts, mask = batch
    else:
        test, question, tag, correct, mask = batch
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction[:, 0] = 0 # set padding index to the first sequence
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # device memory로 이동
    test = test.to(args.device)
    question = question.to(args.device)
    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)
    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)
    if args.fversion >=2:
        conts = conts.to(args.device)
        return (test, question, tag, correct,
                mask, interaction, gather_index, conts)
    else:
        return (test, question, tag, correct,
                mask, interaction, gather_index)

# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    #마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args):    
    model_path = os.path.join(args.model_dir, args.model_name + '.pt')
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
    print("Loading Model from:", model_path, "...Finished.")

    return model

# soft voting base
def kfold_inference(args, test_data):

    fold_total_preds = np.array([])
    for fold in range(args.n_fold):
        model = load_kfold_model(args, fold)
        model.eval()

        _, test_loader = get_loaders(args, None, test_data)

        total_preds = []
        for step, batch in enumerate(test_loader):
            # items -> test, question, tag, correct, mask, interaction, gather_index
            items = process_batch(batch, args)
            preds = model(items)

            # predictions
            preds = preds[:, -1]
            preds = preds.to('cpu').detach().numpy()

            total_preds += list(preds)
        print(f"[fold] inference complete")
        if fold == 0:
            fold_total_preds = np.array(total_preds)
        else:
            fold_total_preds += np.array(total_preds)
    fold_total_preds /= args.n_fold
    fold_total_pred = list(fold_total_preds)

    write_path = os.path.join(args.output_dir, f"{args.n_fold}_output.csv")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, pred in enumerate(fold_total_pred):
            w.write('{},{}\n'.format(id, pred))


def load_kfold_model(args, fold):
    model_name = f"{args.model_name}{fold}.pt"
    model_path = os.path.join(args.model_dir, model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
    print("Loading Model from:", model_path, "...Finished.")

    return model