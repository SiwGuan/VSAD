import json
from datetime import datetime
import os
import torch
import torch.nn as nn
import random
import numpy as np
from args import get_parser
from utils import *
from trainer import Trainer
from STAD import STAD
from torch.utils.tensorboard import SummaryWriter
random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
torch.cuda.manual_seed(44)
if torch.cuda.device_count()>1:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


if __name__ == "__main__":


    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()
    normalize = args.normalize
    dataset = args.dataset
    window_size = args.length
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    theta = args.theta
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    group_index = args.group[0]
    index = args.group[2:]
    training = args.train
    gamma = args.gamma
    log_tensorboard = args.use_tensorboard
    weight_decay = args.weight_decay
    clip = args.clip
    n_latent = args.n_latent
    d_model = args.d_model
    args_summary = str(args.__dict__)
    print(args_summary)

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    if dataset == 'SMD':
        output_path = f'./output/SMD/{args.group}'
        (x_train, _), (x_test, test_labels) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    else:
        output_path = f'./output/{dataset}'
        (x_train, _), (x_test, test_labels) = get_data(dataset, normalize=normalize)

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path) and training:
        os.makedirs(output_path)
    if not os.path.exists(log_dir) and training:
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"
    if not os.path.exists(save_path) and training:
        os.makedirs(save_path)
    if log_tensorboard:
        writer = SummaryWriter(f"{log_dir}")

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]


    train_dataset = SlidingWindowDataset(x_train, window_size, n_features)
    test_dataset = SlidingWindowDataset(x_test, window_size, n_features)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    model = STAD(n_features=n_features,window=window_size,d_model=d_model,n_latent=n_latent)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    Trainer = Trainer(model,
            dataset,
            lr,
            weight_decay,
            device,clip,
            train_loader,
            val_loader,
            test_loader,
            gamma,
            theta,
            x_train,
            x_test,
            n_epochs=n_epochs,
            window_size=window_size,
            n_features=n_features,
            save_path = save_path,
            )

    if training:
        Trainer.train()
    else:
        Trainer.load_model(output_path,model,device=device,model_name="/model.pth")
    Trainer.train_scores()
    Trainer.test_scores()

    Trainer.anomaly_detection(labels=test_labels[window_size:,:])

    args_path = f"{save_path}/config.txt"
    if training:
        with open(args_path,"w") as f:
            f.write(str(args))
            f.close()
