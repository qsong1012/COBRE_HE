import os
import time
import pandas as pd
import sys
import glob
import socket
from maskhit.trainer.fitter import HybridFitter
from maskhit.trainer.losses import FlexLoss
from options.train_options import TrainOptions

opt = TrainOptions()
opt.initialize()
args = opt.parse()

args.all_arguments = ' '.join(sys.argv[1:])


assert not args.sample_all, "the argument --sample-all is deprecated, use --num-patches=0 instead"


if args.cancer == '.':
    args.cancer = ""

if args.wd is not None:
    args.wd_attn = args.wd_fuse = args.wd_pred = args.wd_loss = args.wd
if args.lr is not None:
    args.lr_attn = args.lr_fuse = args.lr_pred = args.lr_loss = args.lr

if args.resume_train:
    args.warmup_epochs = 0

if args.region_size is not None:
    args.region_length = args.region_size // args.patch_size
else:
    args.region_length = 0

if args.region_length is not None and args.region_length > 0:
    assert_message = "grid size is measured in patches and need to be a positive number no larger than the region size / patch size"
    assert args.grid_size <= args.region_length and args.grid_size > 0, assert_message

args.prop_mask = [int(x) for x in args.prop_mask.split(',')]
args.prop_mask = [x / sum(args.prop_mask) for x in args.prop_mask]

if args.sample_svs:
    args.id_var = 'id_svs_num'
else:
    args.id_var = 'id_patient_num'

if args.outcome_type == 'survival':
    args.outcomes = ['time', 'status']
else:
    args.outcomes = [args.outcome]

args.patch_spec = f"mag_{args.magnification}-size_{args.patch_size}"


args.mode_ops = {'train': {}, 'val': {}, 'predict': {}}

if args.num_patches > 0:
    args.mode_ops['train']['num_patches'] = args.num_patches
else:
    if args.region_length is None:
        args.mode_ops['train']['num_patches'] = 0
    else:
        args.mode_ops['train'][
            'num_patches'] = args.region_length * args.region_length

if args.num_patches_val is None:
    args.mode_ops['val']['num_patches'] = args.mode_ops['train']['num_patches']
elif args.num_patches_val > 0:
    args.mode_ops['val']['num_patches'] = args.num_patches_val
else:
    args.mode_ops['val'][
        'num_patches'] = args.region_length * args.region_length

args.mode_ops['predict']['num_patches'] = args.mode_ops['val']['num_patches']

args.mode_ops['train']['num_regions'] = args.regions_per_svs
if args.regions_per_svs_val is None:
    args.mode_ops['val']['num_regions'] = args.regions_per_svs
else:
    args.mode_ops['val']['num_regions'] = args.regions_per_svs_val
args.mode_ops['predict']['num_regions'] = args.mode_ops['val']['num_regions']

args.mode_ops['train']['svs_per_patient'] = args.svs_per_patient
args.mode_ops['val']['svs_per_patient'] = args.svs_per_patient
args.mode_ops['predict']['svs_per_patient'] = args.svs_per_patient

args.mode_ops['train'][
    'regions_per_patient'] = args.regions_per_svs * args.svs_per_patient
args.mode_ops['val']['regions_per_patient'] = args.mode_ops['val'][
    'num_regions'] * args.svs_per_patient
args.mode_ops['predict']['regions_per_patient'] = args.mode_ops['val'][
    'regions_per_patient']

args.mode_ops['train']['repeats_per_epoch'] = args.repeats_per_epoch
args.mode_ops['val']['repeats_per_epoch'] = 1
args.mode_ops['predict']['repeats_per_epoch'] = args.repeats_per_epoch

args.mode_ops['train']['batch_size'] = max(args.batch_size,
                                           args.svs_per_patient)
args.mode_ops['val']['batch_size'] = max(args.batch_size, args.svs_per_patient)
args.mode_ops['predict']['batch_size'] = max(args.batch_size,
                                             args.svs_per_patient)

if args.visualization:
    args.vis_spec = f"{args.timestr}-{args.resume}/{args.vis_layer}-{args.vis_head}"

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_checkpoint_epoch(fname):
    return os.path.basename(fname).split(".")[0]


def get_resume_checkpoint(checkpoints_name, epoch_to_resume):
    files = glob.glob(
        os.path.join(args.checkpoints_folder, checkpoints_name, "*.pt"))
    checkpoint_to_resume = [
        fname for fname in files
        if get_checkpoint_epoch(fname) == epoch_to_resume
    ][0]
    return checkpoint_to_resume


def prepare_data(meta_split, meta_file, round_month=False, vars_to_include=[]):

    vars_to_keep = ['id_patient']
    if args.outcome_type in ['survival', 'mlm']:
        vars_to_keep.extend(['time', 'status'])
    else:
        vars_to_keep.append(args.outcome)

    print("vars_to_keep = ", vars_to_keep)
    print("meta_file = ", meta_file.columns.tolist())
    print("meta_split = ", meta_split.columns.tolist())

    print(meta_file.id_patient.head(), meta_split.id_patient.head())
    meta_split = meta_split.merge(meta_file[vars_to_include],
                                  on='id_patient',
                                  how='inner')
    print("meta_split = ", meta_split.columns, meta_split.shape)
    meta_split['id_patient_num'] = meta_split.id_patient.astype(
        'category').cat.codes
    meta_split['id_svs_num'] = meta_split.id_svs.astype('category').cat.codes

    if args.outcome_type == 'classification':
        meta_split = meta_split.loc[~meta_split[args.outcome].isna()]
        meta_split[args.outcome] = meta_split[args.outcome].astype(
            'category').cat.codes

    elif args.outcome_type == 'survival':
        meta_split = meta_split.loc[~meta_split.status.isna()
                                    & ~meta_split.time.isna()]
    return meta_split


def main():

    if len(args.timestr):
        TIMESTR = args.timestr
    else:
        TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    model_name = str(TIMESTR)
    if args.meta_all is not None:
        model_name = f"{TIMESTR}-{args.fold}"

    # if we want to resume previous training
    if len(args.resume):
        checkpoint_to_resume = get_resume_checkpoint(args.resume,
                                                     args.resume_epoch)
        if args.resume_train:
            # use the model name
            model_name = args.resume
            TIMESTR = model_name.split('-')[0]

    # if we want to resume when error occurs
    elif args.resume_train:
        args.resume = model_name
        try:
            checkpoint_to_resume = get_resume_checkpoint(args.resume, "LAST")
        except Exception as e:
            print(e)
            checkpoint_to_resume = ''

    # we don't want to resume
    else:
        checkpoint_to_resume = ''

    args.model_name = model_name

    checkpoints_folder = os.path.join("checkpoints", model_name)
    args.hostname = socket.gethostname()

    # loading datasets
    meta_svs = pd.read_pickle(args.meta_svs)

    if args.ffpe_only:
        meta_svs = meta_svs.loc[meta_svs.slide_type == 'ffpe']
    if args.ffpe_exclude:
        meta_svs = meta_svs.loc[meta_svs.slide_type != 'ffpe']

    if args.meta_all is not None:
        meta_all = pd.read_pickle(args.meta_all)
        if args.mode == 'extract':
            meta_train = meta_val = meta_all
        elif 'fold' in meta_all.columns:
            val_fold = (args.fold + 1) % 5
            test_fold = args.fold
            train_folds = [
                x for x in range(5) if x not in [val_fold, test_fold]
            ]

            meta_train = meta_val = meta_all.loc[meta_all.fold.isin(
                train_folds)]
            if args.test_type == 'train':
                meta_val = meta_train
            elif args.test_type == 'val':
                meta_val = meta_all.loc[meta_all.fold == val_fold]
            elif args.test_type == 'test':
                meta_val = meta_all.loc[meta_all.fold == test_fold]
        else:
            meta_train = meta_all.loc[meta_all.train]
            if args.test_type == 'train':
                meta_val = meta_all.loc[meta_all.train]
            elif args.test_type == 'val':
                meta_val = meta_all.loc[meta_all.val]
            elif args.test_type == 'test':
                meta_val = meta_all.loc[meta_all.test]

    else:
        meta_train = pd.read_pickle(args.meta_train)
        meta_val = pd.read_pickle(args.meta_val)

    # select cancer subset
    if args.cancer == '':
        pass
    else:
        meta_svs = meta_svs.loc[meta_svs.cancer == args.cancer]
        meta_train = meta_train.loc[meta_train.cancer == args.cancer]
        meta_val = meta_val.loc[meta_val.cancer == args.cancer]

    print('shape of meta_svs = ', meta_svs.shape)
    meta_svs['folder'] = meta_svs['cancer']
    meta_svs['sampling_weights'] = 1
    vars_to_include = ['id_patient', 'folder', 'id_svs', 'sampling_weights']
    if 'svs_path' in meta_svs:
        vars_to_include = ['id_patient', 'folder', 'id_svs', 'sampling_weights', 'svs_path']

    print('=' * 30)
    print(meta_svs.columns)
    print('=' * 30)
    if args.visualization and 'pos' in meta_svs.columns:
        vars_to_include.append('pos')

    ########################################
    # prepare dataset
    df_test = prepare_data(meta_split=meta_val,
                           meta_file=meta_svs,
                           vars_to_include=vars_to_include)

    df_train = prepare_data(meta_split=meta_train,
                            meta_file=meta_svs,
                            vars_to_include=vars_to_include)

    if args.outcome_type == 'classification':
        num_classes = len(df_train[args.outcome].unique().tolist())
    else:
        num_classes = 1

    print('num_classes = ', num_classes)
    if args.weighted_loss:
        weight = df_train.shape[0] / df_train[
            args.outcome].value_counts().sort_index()
        print('weight is: ', weight)
    else:
        weight = None
    criterion = FlexLoss(outcome_type=args.outcome_type, weight=weight)

    if args.study is not None:
        model_name = f"{args.study}/{model_name}"

    hf = HybridFitter(timestr=TIMESTR,
                      num_classes=num_classes,
                      args=args,
                      loss_function=criterion,
                      model_name=model_name,
                      checkpoints_folder=checkpoints_folder,
                      checkpoint_to_resume=checkpoint_to_resume)

    data_dict = {"train": df_train, "val": df_test}

    # Simply call main_worker function
    if args.mode == 'test':
        hf.fit(data_dict, 'test')

    elif args.mode == 'train':
        hf.fit(data_dict)

    elif args.mode == 'predict':
        hf.predict(df_test)

    elif args.mode == 'extract':
        hf.fit(data_dict, procedure='extract')

    else:
        print(f"Mode {args.mode} has not been implemented!")


if __name__ == '__main__':
    main()
