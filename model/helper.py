import random
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, f1_score
from PIL import Image


###########################################
#         commonly used functions         #
###########################################


def get_filename_extensions(args):
    ext_data = '%s_mag-%s_size-%s' % (args.cancer, args.magnification, args.patch_size)
    ext_experiment = 'by-%s_seed-%s' % (args.stratify, args.random_seed)
    ext_split = '%s_by-%s_seed-%s_nest-%s%s' % (args.cancer, args.stratify,
                                                args.random_seed, args.outer_fold, args.inner_fold)
    return ext_data, ext_experiment, ext_split


###########################################
#             Loss function               #
###########################################


def log_parlik_loss_cox(scores, times=None, events=None):
    '''
    scores: values predicted by CNN model
    times: follow-up time
    events: 1 for event 0 for censor
    '''
    scores = scores.view(-1)
    times = times.view(-1)
    events = events.float().view(-1)

    scores = torch.clamp(scores, max=20)  # to avoid too large value after exponential
    scores_exp = torch.exp(scores)
    idx = torch.argsort(times, descending=True)
    times = times[idx]
    scores = scores[idx]
    scores_exp = scores_exp[idx]
    events = events[idx]
    log_scores = torch.log(torch.cumsum(scores_exp, dim=0) + 1e-5)
    uncensored_likelihood = scores - log_scores
    censored_likelihood = torch.mul(uncensored_likelihood, events.type(torch.float32)).sum()
    num_events = events.sum()
    loss = torch.div(-censored_likelihood, num_events)
    return loss


########################################
#               Early stopping         #
########################################
# modified from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = 0
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        print("Bad epochs: %s; Patience: %s; Best value: %6.4f" %
              (self.num_bad_epochs, self.patience, self.best))

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)

###########################################
#             MISC FUNCTIONS              #
###########################################


# instantiate the model
def create_model(num_layers, pretrain, num_classes):
    assert num_layers in [18, 34, 50, 101, 152]
    architecture = "resnet{}".format(num_layers)
    model_constructor = getattr(torchvision.models, architecture)
    model = model_constructor(num_classes=num_classes)

    if pretrain is True:
        print("Loading pretrained model!")
        pretrained = model_constructor(pretrained=True).state_dict()
        if num_classes != pretrained['fc.weight'].size(0):
            del pretrained['fc.weight'], pretrained['fc.bias']
        model.load_state_dict(pretrained, strict=False)
    return model


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MobileNetV2Updated(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV2Updated, self).__init__()
        self.model = torchvision.models.mobilenet.mobilenet_v2(pretrained=pretrained)
        self.model.classifier = Identity()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, 1000)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

########################################
#          balanced sampling           #
########################################
# On the patient level, each epoch might be too small (several hundred patients).
# We can repeat all the samples multiple times and randomize them to obtain a larger
# epoch size.
# Specifically, for survival analysis, we can fixed the ratio of e (events) and ne 
# (no events) to ensure each batch contains some events.


def grouped_sample(
        df,
        num_patches=4,
        num_repeats=100,
        e_ne_ratio='1to3',
        stratify=None):

    vars_to_drop = [stratify]
    if stratify is None:
        pass
    else:
        vars_to_drop.append(stratify)

    vars_to_drop = list(set(vars_to_drop))
    vars_to_keep = vars_to_drop.copy()
    vars_to_keep.append('submitter_id')
    df_meta = df.drop_duplicates('submitter_id')[vars_to_keep].reset_index(drop=True)

    if e_ne_ratio is not None and stratify == 'status':
        # oversampling for the under-represented group for suvival analysis
        num_e_per_batch, num_ne_per_batch = [int(x) for x in e_ne_ratio.split('to')]
        group_size = num_e_per_batch + num_ne_per_batch
        num_e = df_meta.loc[df_meta[stratify] == 1].shape[0]
        num_ne = df_meta.loc[df_meta[stratify] == 0].shape[0]

        df_e = pd.concat([shuffle(df_meta.loc[df_meta[stratify] == 1])
                          for _ in range(num_e_per_batch * num_repeats)])
        random_ids_e = np.concatenate(
            [np.array(range(num_e * num_repeats)) * group_size + x for x in range(num_e_per_batch)])
        df_e['random_id'] = random_ids_e

        num_patches_ne = df_e.shape[0] * num_ne_per_batch // num_e_per_batch
        num_repeats_ne = num_patches_ne // num_ne + 1
        df_ne = pd.concat([shuffle(df_meta.loc[df_meta[stratify] == 0])
                           for _ in range(num_repeats_ne)])
        random_ids_ne = np.concatenate(
            [np.arange(num_e * num_repeats) * group_size + x for x in range(num_e_per_batch, group_size)])
        random_ids_ne.sort()
        df_ne = df_ne.iloc[:num_patches_ne].copy()
        df_ne['random_id'] = random_ids_ne[:num_patches_ne]

        df_id = pd.concat([df_e, df_ne])
        df_id.sort_values('random_id', inplace=True)

    else:
        df_id = pd.concat([shuffle(df_meta) for _ in range(num_repeats)])

    df_id['dummy_count'] = 1
    df_id['id_of_patient'] = df_id.groupby('submitter_id').dummy_count.cumsum() - 1

    df_patches = df.groupby('submitter_id', as_index=False).sample(
        num_patches * (df_id.id_of_patient.max() + 1), replace=True)
    df_patches['id_of_patient'] = df_patches.groupby('submitter_id').file.transform(
        lambda x: np.arange(x.shape[0]) // num_patches)

    df_sel = df_id.drop(columns=vars_to_drop).merge(
        df_patches, on=['submitter_id', 'id_of_patient'], how='left')
    return df_sel


###########################################
#          CUSTOM DATALOADER              #
###########################################


PATH_MEAN = [0.66, 0.52, 0.73]
PATH_STD = [0.17, 0.20, 0.16]


########################################
# load slide patches
class SlidesDataset(Dataset):
    def __init__(
            self,
            data_file,
            image_dir,
            outcome,
            outcome_type,
            crop_size,
            num_crops):

        self.df = data_file
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.num_crops = num_crops

        if outcome_type == 'survival':
            self.outcomes = ['time', 'status']
        else:
            self.outcomes = [outcome]

    def __len__(self):
        return self.df.shape[0]

    def sample_patch(self, idx):
        idx = idx % self.df.shape[0]

        fname = self.df.iloc[idx]['file']
        img_name = os.path.join(self.image_dir, fname)
        img = np.array(Image.open(img_name))
        imgs = random_crops(img, self.crop_size, self.num_crops).reshape(-1, self.crop_size, 3)

        sample = (imgs, self.df.iloc[idx]['id_patient'],
                  self.df.iloc[idx][self.outcomes].to_numpy().astype(float))
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sample_patch(idx)


########################################
# DALI

def batchsplit(img, nrows, ncols):
    return np.concatenate(
        np.split(
            np.stack(
                np.split(img, nrows, axis=0)
            ), ncols, axis=2)
    )


def random_crop(img, height, width):
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img


def random_crops(img, crop_size, n_crops):
    h, w, c = img.shape
    if (h == crop_size) and (w == crop_size):
        return img
    nrows = h // crop_size
    ncols = w // crop_size
    if max(nrows % crop_size, ncols % crop_size):
        img = random_crop(img, nrows * crop_size, ncols * crop_size)
    splits = batchsplit(img, nrows, ncols)
    return splits[random.sample(range(splits.shape[0]), n_crops)]


class ExternalSourcePipeline(Pipeline):
    def __init__(self, data_iterator, batch_size, num_threads, device_id, mode, crop_size):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12)
        self.data_iterator = data_iterator
        self.input = ops.ExternalSource(num_outputs=3)
        self.flip = ops.Flip(device="gpu")
        self.should_flip_h = ops.CoinFlip(probability=0.5)
        self.should_flip_v = ops.CoinFlip(probability=0.5)

        self.color_twist = ops.ColorTwist(device="gpu")  # initializing of the brightness oparator
        self.brightness_param = ops.Uniform(range=(0.65, 1.35))
        # initializing of randomize parameters for ops.BrightnessContrast
        self.contrast_param = ops.Uniform(range=(0.5, 2))
        self.hue_param = ops.Uniform(range=(-30., 30.))
        self.saturation_param = ops.Uniform(range=(0.9, 1.1))
        self.cmn = ops.CropMirrorNormalize(
            device="gpu",
            # dtype=types.UINT8,
            std=[x * 255 for x in PATH_STD],
            mean=[x * 255 for x in PATH_MEAN],
            output_layout="CHW")

        self.mode = mode

    def define_graph(self):
        self.jpegs, self.ids, self.targets = self.input()
        images = self.jpegs.gpu()
        images = self.flip(images, horizontal=self.should_flip_h(), vertical=self.should_flip_v())
        if self.mode == 'train':
            images = self.color_twist(
                images,
                brightness=self.brightness_param(),
                contrast=self.contrast_param(),
                hue=self.hue_param(),
                saturation=self.saturation_param()
            )  # execution of the ops.BrightnessContrast transformation
        else:
            pass
        images = self.cmn(images)
        self.ids = self.ids.gpu()
        self.targets = self.targets.gpu()

        return (images, self.ids, self.targets)

    def iter_setup(self):
        # the external data iterator is consumed here and fed as input to Pipeline
        images, ids, targets = self.data_iterator.next()
        self.feed_input(self.jpegs, images)
        self.feed_input(self.ids, ids)
        self.feed_input(self.targets, targets)
        # self.feed_input(self.events, events)


###########################################
#   RESULTS COLLECTION AND EVALUATION     #
###########################################

########################################
# by patch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = []
        self.ns = []
        self.weights = []

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.values.append(val)
        self.ns.append(n)
        self.weights = [x / self.count for x in self.ns]
        self.avg = np.average(self.values, weights=self.weights)

    def get_avg(self):
        return self.avg

    def get_std(self):
        try:
            variance = np.average((np.array(self.values) - self.avg)**2, weights=self.weights)
            self.std = np.sqrt(variance)
        except:
            self.std = 0.0
        return self.std

    def final(self):
        std = self.get_std()
        avg = self.get_avg()
        return {'avg': avg, 'std': std}

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, verbose=True, prefix="", writer=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.verbose = verbose
        self.writer = writer

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.verbose:
            if self.writer is not None:
                self.writer.info('\t'.join(entries))
            else:
                print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

########################################
# all batches combined

def c_index(times, scores, events):
    try:
        cindex = concordance_index(times, scores, events)
    except Exception as e:
        print(e)
        cindex = 0.5
    return cindex


def xyear_auc(preds, times, events, cutpoint=5):
    preds = preds.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1)
    ebt = np.zeros_like(times) + 1
    ebt[(times < cutpoint) & (events == 0)] = -1
    ebt[times >= cutpoint] = 0
    ind = ebt >= 0
    try:
        auc = roc_auc_score(ebt[ind], preds[ind])
    except ValueError:
        auc = 0.5
    return auc

def calculate_metrics(preds, targets, outcome_type='survival'):

    if outcome_type == 'survival':
        times, events = targets[:, 0], targets[:, 1]
        c = c_index(times, -preds.reshape(-1), events)
        auc_2yr = xyear_auc(preds, times, events, cutpoint=2)
        auc_5yr = xyear_auc(preds, times, events, cutpoint=5)

        res = {
            'c-index': c,
            'auc-2yr': auc_2yr,
            'auc-5yr': auc_5yr
        }
        return res

    elif outcome_type == 'classification':
        f1 = f1_score(targets, preds.argmax(axis=1), average='weighted')
        try:
            if preds.shape[1] > 2:
                # multi-class
                auc = roc_auc_score(targets.reshape(-1),
                                    torch.softmax(torch.tensor(preds), dim=1), multi_class='ovr')
            else:
                # binary
                auc = roc_auc_score(targets.reshape(-1),
                                    torch.softmax(torch.tensor(preds), dim=1)[:, 1])
        except ValueError:
            auc = 0.5
        res = {
            'f1': f1,
            'auc': auc
        }

        return res


class ModelEvaluation(object):
    def __init__(
        self,
        outcome_type='survival',
        loss_function=None,
        mode='train',
        variables=['ids', 'preds', 'targets'],
        device=torch.device('cpu'),
        timestr=None
    ):

        self.outcome_type = outcome_type
        self.criterion = loss_function
        self.mode = mode
        self.timestr = timestr
        self.variables = variables
        self.device = device
        self.reset()

    def reset(self):
        self.data = {}
        for var in self.variables:
            self.data[var] = None

    def update(self, batch):
        for k, v in batch.items():
            if self.data[k] is None:
                self.data[k] = v.data.cpu().numpy()
            else:
                self.data[k] = np.concatenate([self.data[k], v.data.cpu().numpy()])

    def evaluate(self):
        metrics = calculate_metrics(
            self.data['preds'],
            self.data['targets'],
            outcome_type=self.outcome_type)

        loss_epoch = self.criterion.calculate(
            torch.tensor(self.data['preds']).to(self.device),
            torch.tensor(self.data['targets']).to(self.device))

        metrics['loss'] = loss_epoch.item()

        return metrics


