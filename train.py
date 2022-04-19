import sys
import time
import os
import argparse
import pickle
import numpy as np
import torch
from utils.read_config import load_config, BaseTrainConfig, read_yaml
from dataset import SlideDataset, PatientDataset, get_transformation
from torch.utils.data import DataLoader
from model.fitter import HybridFitter
from model.models import HybridModel
from model.helper import compose_logging
from model.loss import FlexLoss


### Read parameter configuration
parser = argparse.ArgumentParser(description='load config file')
parser.add_argument('--config', default=None, help='path to config file')
sys_args = parser.parse_args()
config_file = sys_args.config
args = BaseTrainConfig(**read_yaml(config_file))


### Read data split
with open(args.data_split, 'rb') as handle:
    data_split = pickle.load(handle)
    
    
### Read data mean and standard deviation if specified
if args.data_stats:
    with open(args.data_stats, 'rb') as handle:
        data_stats = pickle.load(handle)
else:
    data_stats = {'mean': None, 'std': None}

# argument validity assertion
if args.stratify is not None:
    assert(args.sampling_ratio is not None)
    
if args.outcome_type == 'classification':
    assert(args.num_classes is not None)

assert(args.train_level=='patient' or args.train_level=='slide')

os.chdir(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # data transformation
    transform = get_transformation(mean=data_stats['mean'], std=data_stats['std'])
    
    # Specify training level to patient or slide level, create correspoding dataloader
    if args.train_level == 'patient':
        drop_last = False
        
        train_data = PatientDataset(pids=data_split['train'], num_patches = args.num_patches,
                                transforms = transform['train'])
        val_data = PatientDataset(pids=data_split['val'], num_patches = args.num_val, 
                                  transforms = transform['val'])
        
    elif args.train_level == 'slide':
        drop_last = True
        
        train_data = SlideDataset(pids=data_split['train'], num_patches = args.num_patches,
                                  transforms = transform['train'])
        val_data = SlideDataset(pids=data_split['val'], num_patches = args.num_val,
                                transforms = transform['val'])
    
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, num_workers=4, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size = 1, shuffle=True, num_workers=args.num_workers)
    loader = {'train':train_loader, 'val':val_loader}
    
    
    ### Define name of the logs to pre-defined or current time
    if len(args.timestr):
        TIMESTR = args.timestr
    else:
        TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    writer = compose_logging(TIMESTR)

    for arg, value in sorted(vars(args).items()):
        writer['meta'].info("Argument %s: %r", arg, value)

    # Specify survival outcome (revisit this part later)
    if args.outcome_type == 'survival':
        outcomes = ['time','status']
    else:
        outcomes = [args.outcome]
    
    # Specify output size (1 for regression/survival and pre-defined for classification)
    if args.outcome_type in ['survival','regression']:
        num_classes = 1
        class_weights = None
    elif args.outcome_type == 'classification':
        num_classes = args.num_classes 
        if args.class_weights is not None:
            class_weights = list(map(float, args.class_weights.split(',')))
        else:
            class_weights = None

    # initialize model        
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
    
    # use FlexLoss
    criterion = FlexLoss(outcome_type=args.outcome_type, class_weights=class_weights, device=device)

    
    # initialize trainer
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
    
    
    # fitting the model or evaluate
    if args.mode == 'test':
        pass # will implement later or in a separate script
    
#         info_str = hf.evaluate(df_test, epoch=0)
#         writer['meta'].info(info_str)
       

    elif args.mode == 'train':
        hf.fit(checkpoints_folder=os.path.join(args.checkpoint_dir, TIMESTR))
        
    else:
        print("This mode has not been implemented!")