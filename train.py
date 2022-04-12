from pathlib import Path
from dataset import IDHDataset, SlideDataset, PatientDataset, get_transformation
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import random
import torch
import pickle
from tqdm import tqdm
from torchvision.transforms import (
    Compose, 
    RandomCrop, 
    Normalize, 
    Lambda, 
    CenterCrop
)
from read_config import load_config, BaseTrainConfig, read_yaml
import sys

import numpy as np
from model.fitter import HybridFitter, reshape_img_batch
from model.models import HybridModel
from model.helper import log_parlik_loss_cox, compose_logging, FlexLoss
import time
import os
from torch.nn import CrossEntropyLoss
import argparse

parser = argparse.ArgumentParser(description='load config file')
parser.add_argument('--config', default=None, help='path to config file')
sys_args = parser.parse_args()

config_file = sys_args.config
args = BaseTrainConfig(**read_yaml(config_file))

with open(args.data_split, 'rb') as handle:
    data_split = pickle.load(handle)
with open(args.data_stats, 'rb') as handle:
    data_stats = pickle.load(handle)

# basic check of argument validity
if args.stratify is not None:
    assert(args.sampling_ratio is not None)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

transform = get_transformation()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.train_level == 'patient':
        drop_last = False
        
        train_data = PatientDataset(args, pids=data_split['train'], num_patches = args.num_patches,
                                transforms = transform['train'])
        val_data = PatientDataset(args, pids=data_split['val'], num_patches = args.num_val, 
                                  transforms = transform['val'])
    elif args.train_level == 'slide':
        drop_last = True
        
        train_data = SlideDataset(args, pids=data_split['train'], num_patches = args.num_patches,
                                  transforms = transform['train'])
        val_data = SlideDataset(args,pids=data_split['val'], num_patches = args.num_val,
                                transforms = transform['val'])
        
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, num_workers=4, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size = 1, shuffle=True, num_workers=args.num_workers)
    loader = {'train':train_loader, 'val':val_loader}
    
    if len(args.timestr):
        TIMESTR = args.timestr
    else:
        TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    writer = compose_logging(TIMESTR)

    for arg, value in sorted(vars(args).items()):
        writer['meta'].info("Argument %s: %r", arg, value)

#     data_dir = '../IDH/Data'
    
    if args.outcome_type == 'survival':
        outcomes = ['time','status']
    else:
        outcomes = [args.outcome]

    if args.outcome_type in ['survival','regression']:
        num_classes = 1
        class_weights = None
    else:
#         num_classes = len(df_train[args.outcome].unique())
        num_classes = 2 # hardcoded change later
        if args.class_weights is not None:
            class_weights = list(map(float, args.class_weights.split(',')))
        else:
            class_weights = None

            
    model = HybridModel(
        backbone=args.backbone,
        pretrain=args.pretrain,
        outcome_dim=num_classes,
        outcome_type=args.outcome_type,
        random_seed=args.random_seed,
        dropout=args.dropout,
        device=device
    )

    writer['meta'].info(model)
    
#     criterion = FlexLoss(outcome_type=args.outcome_type, class_weights=class_weights, device=device)
    criterion = CrossEntropyLoss()
    
    # construct the model
    hf = HybridFitter(
        model=model,
        writer=writer,
        transform = transform,
        dataloader = loader,
        checkpoint_to_resume=args.resume,
        timestr=TIMESTR,
        args=args,
        model_name=TIMESTR,
        loss_function=criterion
    )
    
    
#     hf.fit(checkpoints_folder=os.path.join("checkpoints", TIMESTR))
    
    # fitting the model or evaluate
    if args.mode == 'test':
        pass
#         info_str = hf.evaluate(df_test, epoch=0)
#         writer['meta'].info(info_str)
       

    elif args.mode == 'train':
        hf.fit(
#             meta_pickle,
            checkpoints_folder=os.path.join(args.checkpoint_dir, TIMESTR))
    else:
        print("This mode has not been implemented!")