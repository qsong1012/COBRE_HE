import os
import time
import pandas as pd
import argparse
import torch
import torch.nn as nn
import glob
import numpy as np
from model.fitter import HybridFitter
from model.models import HybridModel
from model.helper import log_parlik_loss_cox, compose_logging, FlexLoss


parser = argparse.ArgumentParser(description='Patient Level Prediction Model')

# specify model structure
parser.add_argument('--backbone',
                    type=str, default='resnet-18',
                    help='backbone model, for example, resnet-50, mobilenet, etc')
parser.add_argument('--class-weights',
                    type=str, default=None,
                    help='weights for each class, separated by comma')
parser.add_argument('--outcome',
                    type=str, default='',
                    help='name of the outcome variable')
parser.add_argument('--outcome-type',
                    type=str, default='survival',
                    help='outcome type, choose from "survival", "classification", "regression"')

# specify the path of the meta files
parser.add_argument('--test-meta',
                    type=str, default='data/val-meta.pickle',
                    help='path to the meta file for the evaluation portion')
parser.add_argument('--train-meta',
                    type=str, default='data/train-meta.pickle',
                    help='path to the meta file for the training portion')
parser.add_argument('--patch-meta',
                    type=str, default='data/patch-meta.pickle',
                    help='path to the meta file for the training portion')

# specify patch manipulations
parser.add_argument('--crop-size',
                    type=int, default=224,
                    help='size of the crop')
parser.add_argument('--num-crops',
                    type=int, default=1,
                    help='number of crops to extract from one patch')
parser.add_argument('--num-patches',
                    type=int, default=8,
                    help='number of patches to select from one patient during one iteration')

# learning rate
parser.add_argument('--lr-backbone',
                    type=float, default=1e-5,
                    help='learning rate for the backbone model')
parser.add_argument('--lr-head',
                    type=float, default=1e-5,
                    help='learning rate for the head model')
parser.add_argument('--cosine-anneal-freq',
                    type=int, default=100,
                    help='anneal frequency for the cosine scheduler')
parser.add_argument('--cosine-t-mult',
                    type=int, default=1,
                    help='t_mult for cosine scheduler')

# specify experiment details
parser.add_argument('-m', '--mode',
                    type=str, default='train',
                    help='mode, train or test')
parser.add_argument('--patience',
                    type=int, default=100,
                    help='break the training after how number of epochs of no improvement')
parser.add_argument('--epochs',
                    type=int, default=100,
                    help='total number of epochs to train the model')
parser.add_argument('--pretrain',
                    action='store_true', default=False,
                    help='whether use a pretrained backbone')
parser.add_argument('--random-seed',
                    type=int, default=1234,
                    help='random seed of the model')
parser.add_argument('--resume',
                    type=str, default='',
                    help='path to the checkpoint file')

# data specific arguments
parser.add_argument('-b', '--batch-size',
                    type=int, default=8,
                    help='batch size')
parser.add_argument('--stratify',
                    type=str, default=None,
                    help='whether to use a stratify approach when splitting the train/val/test datasets')
parser.add_argument('--sampling-ratio',
                    type=str, default=None,
                    help='fixed sampling ratio for each class for each batch, for example 1,3')
parser.add_argument('--repeats-per-epoch',
                    type=int, default=100,
                    help='how many times to select one patient during each iteration')
parser.add_argument('--num-workers',
                    type=int, default=4,
                    help='number of CPU threads')

# model regularization
parser.add_argument('--dropout',
                    type=float, default=0,
                    help='dropout rate, not implemented yet')
parser.add_argument('--wd-backbone',
                    type=float, default=0.0001,
                    help='intensity of the weight decay for the backbone model')
parser.add_argument('--wd-head',
                    type=float, default=1e-5,
                    help='intensity of the weight decay for the head model')
parser.add_argument('--l1',
                    type=float, default=0,
                    help='intensity of l1 regularization')
parser.add_argument('--l2',
                    type=float, default=0,
                    help='intensity of l2 regularization')

# evaluation details
parser.add_argument('--sample-id',
                    action='store_true', default=False,
                    help='if true, sample patches by patient; otherwise evaluate the model on all patches')
parser.add_argument('--num-val',
                    type=int, default=100,
                    help='number of patches to select from one patient during validation')

# model monitoring
parser.add_argument('--timestr',
                    type=str, default='',
                    help='time stamp of the model')
parser.add_argument('--log-freq',
                    type=int, default=10,
                    help='how frequent (in steps) to print logging information')
parser.add_argument('--save-interval',
                    type=int, default=1,
                    help='how frequent (in epochs) to save the model checkpoint')



args = parser.parse_args()
# basic check of argument validity
if args.stratify is not None:
    assert(args.sampling_ratio is not None)

os.chdir(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if len(args.timestr):
        TIMESTR = args.timestr
    else:
        TIMESTR = time.strftime("%Y%m%d_%H%M%S")

    writer = compose_logging(TIMESTR)

    for arg, value in sorted(vars(args).items()):
        writer['meta'].info("Argument %s: %r", arg, value)

    data_dir = 'data'


    ########################################
    # prepare dataset
    df_patches = pd.read_pickle(args.patch_meta)
    df_patches['id_patient'] = df_patches.submitter_id.astype('category').cat.codes
    df_train   = pd.read_pickle(args.train_meta)
    df_test    = pd.read_pickle(args.test_meta )

    cols_to_use = df_patches.columns.difference(df_train.columns)
    cols_to_use = cols_to_use.tolist()
    cols_to_use.append('submitter_id')
    df_train = df_train.merge(df_patches[cols_to_use], on='submitter_id', how='left')
    df_test  = df_test.merge( df_patches[cols_to_use], on='submitter_id', how='left')

    if args.outcome_type == 'survival':
        outcomes = ['time','status']
    else:
        outcomes = [args.outcome]

    df_train.dropna(subset=outcomes, inplace=True)
    df_test.dropna( subset=outcomes, inplace=True)


    if args.outcome_type in ['survival','regression']:
        num_classes = 1
        class_weights = None
    else:
        num_classes = len(df_train[args.outcome].unique())
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

    # choose the loss function
    criterion = FlexLoss(outcome_type=args.outcome_type, class_weights=class_weights, device=device)

    # construct the model
    hf = HybridFitter(
        model=model,
        writer=writer,
        checkpoint_to_resume=args.resume,
        timestr=TIMESTR,
        args=args,
        model_name=TIMESTR,
        loss_function=criterion
    )

    # fitting the model or evaluate
    if args.mode == 'test':
        info_str = hf.evaluate(df_test, epoch=0)
        writer['meta'].info(info_str)

    elif args.mode == 'train':
        meta_pickle = {
            "train": df_train,
            "val": df_test
        }
        hf.fit(
            meta_pickle,
            checkpoints_folder=os.path.join("checkpoints", TIMESTR))
    else:
        print("This mode has not been implemented!")