import os
import time
import pandas as pd
import logging
import argparse
import torch
import torch.nn as nn
import glob
import numpy as np
from model import HybridModel, HybridFitter
from utils import log_parlik_loss_cox, get_filename_extensions


parser = argparse.ArgumentParser(description='Hybrid MIL Model')
parser.add_argument('--anneal-freq',
                    type=int, default=100,
                    help='anneal frequency for the cosine scheduler')
parser.add_argument('--backbone',
                    type=str, default='resnet-18',
                    help='backbone model, for example, resnet-50, mobilenet, etc')
parser.add_argument('--crop-size',
                    type=int, default=224,
                    help='size of the crop')
parser.add_argument('--checkpoints-folder',
                    type=str, default='checkpoints',
                    help='path to the checkpoints folder')
parser.add_argument('--dropout-p',
                    type=float, default=0,
                    help='dropout rate, not implemented yet')
parser.add_argument('--e-ne-ratio',
                    type=str, default=None,
                    help='ratio of patients with event to patients without event, for example 1to3')
parser.add_argument('--epochs',
                    type=int, default=100,
                    help='total number of epochs to train the model')
parser.add_argument('--gamma',
                    type=float, default=0.85,
                    help='rate to reduce learning rate')
parser.add_argument('--cancer',
                    type=str, default='LGG',
                    help='Cancer type, such as LGG, BRCA, etc')
parser.add_argument('--l1',
                    type=float, default=0,
                    help='intensity of l1 regularization')
parser.add_argument('--l2',
                    type=float, default=0,
                    help='intensity of l2 regularization')
parser.add_argument('--lr-backbone',
                    type=float, default=1e-5,
                    help='learning rate for the backbone model')
parser.add_argument('--lr-head',
                    type=float, default=1e-5,
                    help='learning rate for the head model')
parser.add_argument('--lr-decay-freq',
                    type=int, default=10,
                    help='how frequent (in epochs) to reduce learning rate')
parser.add_argument('--log-freq',
                    type=int, default=10,
                    help='how frequent (in steps) to print logging information')
parser.add_argument('--magnification',
                    type=str, default='20',
                    help='magnification level for the input images')
parser.add_argument('--num-crops',
                    type=int, default=1,
                    help='number of crops to extract from one patch')
parser.add_argument('--num-patches',
                    type=int, default=8,
                    help='number of patches to select from one patient during one iteration')
parser.add_argument('--num-val',
                    type=int, default=100,
                    help='number of patches to select from one patient during validation')
parser.add_argument('--num-workers',
                    type=int, default=8,
                    help='number of CPU threads')
parser.add_argument('--outcome',
                    type=str, default='',
                    help='name of the outcome variable')
parser.add_argument('--outcome-type',
                    type=str, default='survival',
                    help='outcome type, choose from "survival", "classification", "regression"')
parser.add_argument('--patch-size',
                    type=int, default=224,
                    help='size of the input image patch')
parser.add_argument('--patience',
                    type=int, default=100,
                    help='break the training after how number of epochs of no improvement')
parser.add_argument('--pretrain',
                    action='store_true', default=False,
                    help='whether use a pretrained backbone')
parser.add_argument('--root-dir',
                    type=str, default='./',
                    help='root directory')
parser.add_argument('--random-seed',
                    type=int, default=1,
                    help='random seed of generating the train/val/test splits')
parser.add_argument('--random-seed-torch',
                    type=int, default=1234,
                    help='random seed of the model')
parser.add_argument('--trainable-layers',
                    type=str, default='',
                    help='regex match for the trainable layers of the backbone, default is to train all layers')
parser.add_argument('--repeats-per-epoch',
                    type=int, default=100,
                    help='how many times to select one patient during each iteration')
parser.add_argument('--resume-model-name',
                    type=str, default='',
                    help='name of the model to be resumed')
parser.add_argument('--resume-epoch',
                    type=str, default='LAST',
                    help='epoch of the model to be resumed')
parser.add_argument('--resume-train',
                    action='store_true', default=False,
                    help='If true then continue training the resumed model using the same model name. If false, training using a new model name')
parser.add_argument('--round-month',
                    action='store_true', default=False,
                    help='round the follow up time to whole months')
parser.add_argument('--sample-id',
                    action='store_true', default=False,
                    help='if true, sample patches by patient; otherwise evaluate the model on all patches')
parser.add_argument('--save-interval',
                    type=int, default=1,
                    help='how frequent (in epochs) to save the model checkpoint')
parser.add_argument('--outer-fold',
                    type=int, default=5,
                    help='number of outer folds')
parser.add_argument('--inner-fold',
                    type=int, default=3,
                    help='number of inner folds')
parser.add_argument('--scale-lr',
                    action='store_true', default=False,
                    help='scale the learning rate by number of patches')
parser.add_argument('--scheduler',
                    type=str, default='cosine',
                    help='type of scheduler')
parser.add_argument('--weighted-loss',
                    action='store_true', default=False,
                    help='whether to use a weighted loss function for imbalanced classification')
parser.add_argument('--stratify',
                    type=str, default='status',
                    help='whether to use a stratify approach when spliting the train/val/test datasets')
parser.add_argument('--step-size',
                    type=int, default=5,
                    help='step size for the step learning rate scheduler')
parser.add_argument('--test-type',
                    type=str, default='val',
                    help='which split to use for testing the model')
parser.add_argument('--split',
                    type=str, default='00',
                    help='the identifier to locate the split the dataset, first digit is the outer fold, second digit is the inner fold')
parser.add_argument('--time-noise',
                    type=float, default=0.2,
                    help='standard deviation of the gaussian noise added to follow-up time')
parser.add_argument('--timestr',
                    type=str, default='',
                    help='time stamp of the model')
parser.add_argument('--train-type',
                    type=str, default='train',
                    help='which split used for training')
parser.add_argument('--wd-backbone',
                    type=float, default=0.0001,
                    help='intensity of the weight decay for the backbone model')
parser.add_argument('--wd-head',
                    type=float, default=1e-5,
                    help='intensity of the weight decay for the head model')
parser.add_argument('-b', '--batch-size',
                    type=int, default=8,
                    help='batch size')
parser.add_argument('-m', '--mode',
                    type=str, default='train',
                    help='mode, train or predict or test')

args = parser.parse_args()


def get_checkpoint_epoch(fname):
    return os.path.basename(fname).split(".")[0]


def get_resume_checkpoint(checkpoints_name, epoch_to_resume):
    files = glob.glob(os.path.join('checkpoints', checkpoints_name, "*.pt"))
    checkpoint_to_resume = [
        fname for fname in files if get_checkpoint_epoch(fname) == epoch_to_resume][0]
    return checkpoint_to_resume


def group_shuffle(df, random_seed):
    df['rand_group'] = df.index // 512
    dfr = df[['rand_group']].drop_duplicates().reset_index(drop=True)
    dfr = dfr.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return dfr.merge(df, on='rand_group', how='left')


def sample_rows(data, id_var='submitter_id', patches_per_id=8, ids_per_batch=64):
    ids = data[id_var].unique()
    ids_sel = np.random.choice(ids, ids_per_batch, replace=False)
    df_results_i = data.loc[data[id_var].isin(ids_sel), ].groupby(id_var, group_keys=False).apply(
        lambda x: x.sample(patches_per_id, replace=True)).reset_index(drop=True)
    return df_results_i


def prepare_data(
        mode_train=True,
        data_type='val',
        round_month=False,
        writer=None):

    data_file = '%s/meta_%s_%s_s-%s.pickle' % (data_dir, data_type,
                                               EXT_SPLIT, args.split[0] if data_type == 'test' else args.split)

    writer['meta'].info("Data type: %s; Loading data from %s" % (data_type, data_file))
    vars_to_keep = ['submitter_id']
    if args.outcome_type == 'survival':
        vars_to_keep.extend(['time', 'status'])
    else:
        vars_to_keep.append(args.outcome)
    df = pd.read_pickle(data_file)[vars_to_keep]

    df = df.merge(df_cmb[['submitter_id', 'file_original', 'file']], on='submitter_id', how='left')

    if round_month:
        df.time = (df.time * 12).round()

    df['id_patient'] = df.submitter_id.astype('category').cat.codes
    return df


def setup_logger(name, log_file, file_mode, to_console=False):
    """
            https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
        To setup as many loggers as you want
    """

    formatter = logging.Formatter('%(message)s')

    handler = logging.FileHandler(log_file, mode=file_mode)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    if to_console:
        logger.addHandler(logging.StreamHandler())

    return logger


def compose_logging(file_mode, model_name):
    writer = {}
    writer["meta"] = setup_logger("meta", os.path.join(
        "logs", "%s_meta.log" % model_name), 'w', to_console=True)
    writer["data"] = setup_logger("data", os.path.join(
        "logs", "%s_data.csv" % model_name), file_mode, to_console=True)
    return writer


class FlexLoss:
    def __init__(self, outcome_type, weight=None):
        assert outcome_type in ['survival', 'classification', 'regression']
        if outcome_type == 'survival':
            self.criterion = log_parlik_loss_cox
        elif outcome_type == 'classification':
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.MSELoss()
        self.outcome_type = outcome_type

    def calculate(self, pred, target):
        if self.outcome_type == 'survival':
            time = target[:, 0].float()
            event = target[:, 1].int()
            # print(pred, time, event)
            return self.criterion(pred, time, event)

        elif self.outcome_type == 'classification':
            return self.criterion(pred, target.long().view(-1))

        else:
            return self.criterion(pred, target.float())


if __name__ == '__main__':

    if len(args.timestr):
        TIMESTR = args.timestr
    else:
        TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    model_name = '%s-%s-%s-%s' % (TIMESTR, args.random_seed, args.split, args.random_seed_torch)

    writer = {}
    file_mode = 'w'  # write to a new log file
    # if we want to resume previous training
    if len(args.resume_model_name):
        resume_checkpoint = True

        checkpoint_to_resume = get_resume_checkpoint(args.resume_model_name, args.resume_epoch)
        if args.resume_train:
            # use the model name
            model_name = args.resume_model_name
            TIMESTR = model_name.split('-')[0]
            file_mode = 'a'

    # we don't want to resume
    else:
        resume_checkpoint = False
        checkpoint_to_resume = ''

    writer = compose_logging(file_mode, model_name)

    for arg, value in sorted(vars(args).items()):
        writer['meta'].info("Argument %s: %r", arg, value)

    checkpoints_folder = os.path.join("checkpoints", model_name)

    if args.scale_lr:
        scale_factor = np.sqrt(args.batch_size * args.num_patches * args.num_crops / 512)
        writer['meta'].info('scale learning rate by factor of %6.4f' % scale_factor)
        args.lr_backbone *= scale_factor
        args.lr_head *= scale_factor
        writer['meta'].info('new learning rate is %s,%s' % (args.lr_backbone, args.lr_head))

    EXT_DATA, EXT_EXPERIMENT, EXT_SPLIT = get_filename_extensions(args)
    ext = "%s_%s" % (EXT_DATA, EXT_EXPERIMENT)

    data_dir = 'data'

    patch_meta_file = os.path.join(args.root_dir, 'data/patches_meta_%s.pickle' % EXT_DATA)
    df_cmb = pd.read_pickle(patch_meta_file)

    ########################################
    # prepare dataset

    df_train = prepare_data(
        mode_train=True,
        data_type=args.train_type,
        round_month=args.round_month,
        writer=writer)

    df_test = prepare_data(
        mode_train=False,
        data_type=args.test_type,
        round_month=args.round_month,
        writer=writer)

    print("Training dataset:")
    print(df_train.head())
    print(df_train.columns)

    weight = None
    if args.outcome_type == 'classification':
        df_train_meta = df_train.drop_duplicates('submitter_id')
        num_classes = len(df_train_meta[args.outcome].unique().tolist())
        if args.weighted_loss:
            class_counts = df_train_meta[args.outcome].value_counts().sort_values()
            weight = torch.tensor(df_train_meta.shape[0] / class_counts).float()
            weight = weight.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            print("classification: ", class_counts, weight)
    else:
        num_classes = 1

    print('num_classes = ', num_classes)
    criterion = FlexLoss(outcome_type=args.outcome_type, weight=weight)

    model = HybridModel(
        backbone=args.backbone,
        pretrain=args.pretrain,
        outcome_dim=num_classes,
        outcome_type=args.outcome_type,
        random_seed=1,
        head='avg'
    )

    hf = HybridFitter(
        model=model,
        writer=writer,
        checkpoint_to_resume=checkpoint_to_resume,
        timestr=TIMESTR,
        args=args,
        loss_function=criterion
    )

    if args.mode == 'test':
        info_str = hf.evaluate(df_test, epoch=0)
        writer['meta'].info(info_str)

    elif args.mode == 'train':
        meta_pickle = {
            "train": df_train,
            "val": df_test
        }
        if len(args.trainable_layers) > 0:
            writer['meta'].info('### Freezing previous layers: %s' % args.trainable_layers)
            hf.freeze_conv_layers(regex=args.trainable_layers)

        hf.fit(
            meta_pickle,
            checkpoints_folder=checkpoints_folder)

    elif args.mode == 'predict':
        hf.predict()

    elif args.mode == 'extraction':
        hf.extract_features(df_test, root_dir=root_dir)
