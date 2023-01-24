import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import math
from torch.utils.data import Dataset
from torchvision import transforms
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, f1_score
from PIL import Image
import re

########################################
#               Early stopping         #
########################################
# modified from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d


class EarlyStopping(object):

    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = -10000.
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

        if math.isnan(metrics):
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
                self.is_better = lambda a, best: a < best - (best * min_delta /
                                                             100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta /
                                                             100)


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
        self.model = torchvision.models.mobilenet.mobilenet_v2(
            pretrained=pretrained)
        self.model.classifier = Identity()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, 1000)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


###########################################
#          CUSTOM DATALOADER              #
###########################################
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


def mark_patch_on_tile(tile, patch_pos, size, ax, add_text=True):
    # Display the image
    ax.imshow(tile)

    # Create a Rectangle patch
    for i, pos_i in enumerate(patch_pos):
        rect = P.Rectangle(pos_i,
                           size,
                           size,
                           linewidth=1,
                           edgecolor='gray',
                           facecolor='blue',
                           alpha=0.2)
        ax.add_patch(rect)
        if add_text:
            ax.text(pos_i[0] + size // 2, pos_i[1] + size // 2, i)


def restore_mask(data):
    mask = np.zeros((data.pos_x.max() + 1, data.pos_y.max() + 1))
    for i, row in data.loc[data.valid == 1].iterrows():
        mask[row['pos_x'], row['pos_y']] = 1
    return mask


def load_features(features, fids, dim=512):
    return torch.cat([features, torch.zeros(1, dim).to(features.device)])[fids]


def get_image_fname(id_svs, loc, magnification=20, patch_size=224):
    return f"patches/mag_{magnification}-size_{patch_size}/{id_svs}/{loc[0]:05d}/{loc[1]:05d}.jpeg"


def load_images(id_svs, locs, is_valid, magnification=20, patch_size=224):
    imgs = []
    for loc, valid in zip(locs, is_valid):
        if valid:
            img_fname = get_image_fname(id_svs=id_svs,
                                        loc=loc,
                                        magnification=magnification,
                                        patch_size=patch_size)
            imgs.append(np.array(Image.open(img_fname)))
        else:
            imgs.append(np.zeros((3, 224, 224)))
    return imgs


PATH_MEAN = [0.7968, 0.6492, 0.7542]
PATH_STD = [0.1734, 0.2409, 0.1845]


def reverse_norm(mean, std):
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    return (-mean / std).tolist(), (1 / std).tolist()


def get_data_transforms(patch_size=224):

    data_transforms = {
        'train':
        transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.2,
                                   saturation=0.1,
                                   hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(
                PATH_MEAN, PATH_STD
            )  # mean and standard deviations for lung adenocarcinoma resection slides
        ]),

        # 'train':
        # transforms.Compose([
        #     transforms.Normalize(PATH_MEAN, PATH_STD)
        # ]),
        'val':
        transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(PATH_MEAN, PATH_STD)]),
        'predict':
        transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(PATH_MEAN, PATH_STD)]),
        'normalize':
        transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(PATH_MEAN, PATH_STD)]),
        'unnormalize':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*reverse_norm(PATH_MEAN, PATH_STD))
        ]),
    }
    return data_transforms


class SlidesDataset(Dataset):
    '''
    sample patches or features from the datasets
    '''

    def __init__(self,
                 data_file,
                 outcome,
                 writer=None,
                 mode='train',
                 transforms=transforms,
                 n_tiles=1,
                 args=None):

        self.df = data_file

        self.args = args
        self.writer = writer

        if self.args.outcome_type == 'survival':
            self.outcomes = self.df[['time',
                                     'status']].to_numpy().astype(float)
        elif self.args.outcome_type == 'mlm':
            self.outcomes = np.ones((self.df.shape[0], 1))
        else:
            self.outcomes = self.df[[outcome]].to_numpy().astype(float)

        if self.args.sample_svs:
            self.ids = self.df['id_svs_num'].tolist()
        else:
            self.ids = self.df['id_patient_num'].tolist()
        self.files = self.df['id_svs'].tolist()
        self.folders = self.df['folder'].tolist()
        if args.visualization:
            self.locs = self.df['pos'].tolist()
        else:
            self.locs = None

        self.wsi = None
        self.sample_size = self.df.shape[0]
        self.mode = mode
        self.transforms = transforms
        self.n_tiles = n_tiles

    def __len__(self):
        return self.sample_size

    def sample_features(self, folder, fname, loc):
        features_one = torch.load(
            f'data/{folder}/{fname}/mag_{self.args.magnification}-size_224/resnet_18/features.pt'
        ).detach()
        loc_tiles, locs_new, locs_orig, fids, labels, is_valid, pct_valid = self._get_patch_meta(
            folder=folder, fname=fname, loc=loc)
        imgs = load_features(features_one, fids.long(), dim=512)
        return imgs, loc_tiles, locs_new, labels, pct_valid

    def sample_patches(self, folder, fname):
        loc_tiles, locs_new, locs_orig, fids, labels, is_valid, pct_valid = self._get_patch_meta(
            folder=folder, fname=fname)
        imgs = load_images(fname, locs_orig, is_valid)
        processed = []
        if self.transforms is not None:
            for img, valid in zip(imgs, is_valid):
                if valid:
                    processed.append(self.transforms(img))
                else:
                    processed.append(torch.zeros(3, 224, 224))
        processed = torch.stack(processed)
        return processed, loc_tiles, locs_new, labels, pct_valid

    def _get_patch_meta(self, folder, fname, loc):
        # get all the patches for one wsi
        meta_one = pd.read_pickle(
            f'data/{folder}/{fname}/mag_{self.args.magnification}-size_224/meta.pickle'
        )
        meta_one['valid'].fillna(0, inplace=True)
        # get all the valid features forone wsi

        wsi = WsiTileSampler(meta_one.reset_index(drop=True),
                             sample_all=self.args.sample_all,
                             sample_twice=False,
                             mode=self.mode,
                             args=self.args)

        num_patches = self.args.num_patches if self.mode == 'train' else self.args.num_patches_val
        margin = int(
            self.args.tile_size //
            self.args.patch_size) if self.args.tile_size is not None else None

        try:
            output = wsi.sample(num_patches,
                                margin,
                                n_tiles=self.n_tiles,
                                threshold=self.args.sampling_threshold,
                                weighted_sample=False,
                                loc=loc)
            loc_tiles, locs_new, locs_orig, fids, pct_valid = output[
                'loc_tiles'], output['locs_new'], output['locs_orig'], output[
                    'fids'], output['pct_valid']
        except Exception as e:
            print(e)
            print(
                f'data/{folder}/{fname}/mag_{self.args.magnification}-size_224/meta.pickle'
            )
        fids = fids.squeeze(0)
        is_valid = fids != -1
        if f"c{self.args.num_clusters}" in meta_one.columns:
            labels = meta_one[f"c{self.args.num_clusters}"].iloc[fids.view(
                -1)].to_numpy()
            labels = labels.reshape(fids.size(0), -1)
            labels[fids.numpy() == -1] = -1
        else:
            labels = np.zeros_like(fids).astype(int)
        return loc_tiles, locs_new, locs_orig, fids, labels, is_valid, pct_valid

    def sample_patch(self, idx):
        tiles = torch.zeros(0)
        idx = idx % self.df.shape[0]
        fname = self.files[idx]
        folder = self.folders[idx]
        if self.locs is None:
            loc = None
        else:
            loc = self.locs[idx]

        if self.args.use_features:
            imgs, pos_tile, pos, labels, pct_valid = self.sample_features(
                folder, fname, loc)
        else:
            imgs, pos_tile, pos, labels, pct_valid = self.sample_patches(
                folder, fname, loc)

        id = self.ids[idx]
        outcome = self.outcomes[idx, :]
        sample = (imgs, id, outcome, pos, pos_tile, tiles, labels, pct_valid)
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sample_patch(idx)


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
            variance = np.average((np.array(self.values) - self.avg)**2,
                                  weights=self.weights)
            self.std = np.sqrt(variance)
        except Exception as e:
            print(e)
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

    def __init__(self,
                 num_batches,
                 meters,
                 verbose=True,
                 prefix="",
                 writer=None):
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


def calculate_metrics(ids, preds, targets, outcome_type='survival'):

    if outcome_type == 'survival':
        df = pd.DataFrame(np.concatenate([ids, targets, preds], axis=1))
        df.columns = ['id', 'time', 'event', 'pred']
        df = df.groupby('id').mean()
        c = c_index(df.time, -df.pred, df.event)
        auc_2yr = xyear_auc(df.pred.to_numpy(),
                            df.time.to_numpy(),
                            df.event.to_numpy(),
                            cutpoint=2)
        auc_5yr = xyear_auc(df.pred.to_numpy(),
                            df.time.to_numpy(),
                            df.event.to_numpy(),
                            cutpoint=5)

        res = {'c-index': c, 'auc-2yr': auc_2yr, 'auc-5yr': auc_5yr}
        return res

    elif outcome_type == 'classification':
        f1 = f1_score(targets, preds.argmax(axis=1), average='weighted')
        if len(np.unique(targets)) != preds.shape[1]:
            auc = 0.5
        else:
            try:
                if preds.shape[1] > 2:
                    # multi-class
                    auc = roc_auc_score(targets.reshape(-1),
                                        torch.softmax(torch.tensor(preds),
                                                      dim=1),
                                        multi_class='ovr')
                else:
                    # binary
                    auc = roc_auc_score(
                        targets.reshape(-1),
                        torch.softmax(torch.tensor(preds), dim=1)[:, 1])
            except Exception as e:
                print(e)
                auc = 0.5
        res = {'f1': f1, 'auc': auc}

        return res


class ModelEvaluation(object):

    def __init__(self,
                 outcome_type='survival',
                 loss_function=None,
                 mode='train',
                 variables=['ids', 'preds', 'targets'],
                 device=torch.device('cpu'),
                 timestr=None):

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
                self.data[k] = np.concatenate(
                    [self.data[k], v.data.cpu().numpy()])

    def evaluate(self):
        metrics = calculate_metrics(self.data['ids'],
                                    self.data['preds'],
                                    self.data['targets'],
                                    outcome_type=self.outcome_type)

        loss_epoch = self.criterion.calculate(
            torch.tensor(self.data['preds']).to(self.device),
            torch.tensor(self.data['targets']).to(self.device))

        metrics['loss'] = loss_epoch.item()

        return metrics

    def save(self, filename):
        values = []
        for k, v in self.data.items():
            values.append(v)
        df = pd.DataFrame(np.concatenate(values, 1))
        if filename is None:
            return df
        else:
            df.to_csv(filename)


########################################
# extract patches directly from WSIs
########################################


class WsiTileSampler:

    def __init__(self,
                 data,
                 sample_all=True,
                 sample_twice=False,
                 mode='train',
                 args=None):
        self.df = data
        self.df['fid'] = self.df.index.astype(int)
        self.sample_all = sample_all
        self.sample_twice = sample_twice
        self.mode = mode
        self.args = args
        self.num_patches = args.num_patches if mode == 'train' else args.num_patches_val

    def sample_blind(self, n):
        # sample from all the valid locations over the entire WSI
        df_valid = self.df.loc[self.df.valid == 1]
        try:
            df_sample_valid = df_valid.sample(n, replace=False)
        except Exception as e:
            print(e)
            df_sample_valid = df_valid.sample(n, replace=True)

        tile_loc = [0, 0]
        locs = df_sample_valid.pos.tolist()
        locs_orig = locs
        fids = df_sample_valid.fid.tolist()
        pct_valid = 1
        return tile_loc, locs, locs_orig, fids, pct_valid

    def sample_from_tile(self, n, margin, loc):
        if loc is None:
            n = margin * margin if self.sample_all else n
            return torch.zeros(n, 512), [-1,
                                         -1], [[0, 0] for _ in range(n)], None
        x, y = loc
        df_sel = self.df.loc[self.df.pos_x.isin(range(x, x + margin))
                             & self.df.pos_y.isin(range(y, y +
                                                        margin))].copy()

        # get all posible positions
        full_pos = np.indices((margin, margin)).reshape(2, -1).swapaxes(1, 0)
        df_full = pd.DataFrame(full_pos, columns=['x', 'y'])

        # original positions
        df_sel['x'] = df_sel.pos_x - x
        df_sel['y'] = df_sel.pos_y - y

        # combine them
        df_full = df_full.merge(df_sel, on=['x', 'y'], how='left')
        df_full.valid.fillna(0, inplace=True)
        df_full.loc[df_full.valid == 0, 'fid'] = -1

        fids = df_full.fid.to_numpy()
        try:
            _tmp = torch.tensor(fids)
        except Exception as e:
            print(e)
            print("None occurred")
            print(df_full.head())
        locs_orig = df_full.pos

        new_pos = np.indices((margin, margin))
        if self.args.mode == 'extract':
            pass
        else:
            # random rotate
            new_pos = np.rot90(new_pos, np.random.randint(4), (1, 2))

            # random vertical flip
            if np.random.rand(1) < 0.5:
                new_pos = np.flip(new_pos, axis=1)
            # random horizontal flip
            if np.random.rand(1) < 0.5:
                new_pos = np.flip(new_pos, axis=2)

        locs = new_pos.reshape(2, -1).swapaxes(1, 0)

        valid_locs = np.where(fids > -1)[0]
        n_valid = valid_locs.shape[0]

        invalid_locs = np.where(fids == -1)[0]

        if self.args.sample_all:
            pass
        elif self.num_patches < margin * margin:

            if self.num_patches <= n_valid:
                _sel = np.random.choice(valid_locs,
                                        self.num_patches,
                                        replace=False)
            else:
                _sel_valid = np.random.choice(valid_locs,
                                              n_valid,
                                              replace=False)
                _sel_invalid = np.random.choice(invalid_locs,
                                                self.num_patches - n_valid,
                                                replace=False)
                _sel = np.concatenate([_sel_valid, _sel_invalid])

            locs = locs[_sel]
            locs_orig = locs_orig[_sel]
            fids = fids[_sel]

        locs = locs.tolist()
        locs_orig = locs_orig.tolist()

        pct_valid = min(n_valid, self.num_patches) / self.num_patches

        return loc, locs, locs_orig, fids, pct_valid

    def sample(self,
               n,
               margin,
               threshold,
               n_tiles=1,
               loc=None,
               weighted_sample=False):
        var = f"counts_{margin}"

        if loc is not None:
            locs = [loc]
        # multi-block sampling
        elif self.args.visualization:
            locs = self.df.loc[(self.df.pos_x % 20 == self.args.offset)
                               & (self.df.pos_y % 20 == self.args.offset) &
                               (self.df[var] > 0)].pos.tolist()
            if len(locs) == 0:
                locs = self.df.loc[(self.df[var] > 0)].pos.tolist()

        elif self.sample_twice:
            _df = self.df
            criterion = min(threshold, _df[var].nlargest(20).min())
            sel = _df.loc[_df[var] >= criterion]
            locs = sel.sample(2, replace=True).pos.tolist()

        elif margin is None:
            locs = [[0, 0] for _ in range(n_tiles)]

        else:
            grid_sampling_mode = self.args.outcome_type != 'mlm' and n_tiles > 1

            dict_offsets = {}
            if grid_sampling_mode:
                max_patches = 0
                while max_patches == 0:
                    for _ in range(10):
                        offset_x = np.random.choice(10, 1).item()
                        offset_y = np.random.choice(10, 1).item()
                        nonzero_regions = self.df.loc[
                            (self.df.pos_x % 10 == offset_x)
                            & (self.df.pos_y % 10 == offset_y) &
                            (self.df[var] > 0)].shape[0]
                        max_patches = self.df.loc[
                            (self.df.pos_x % 10 == offset_x)
                            & (self.df.pos_y % 10 == offset_y)][var].max()
                        if math.isnan(nonzero_regions):
                            nonzero_regions = 0
                        if math.isnan(max_patches):
                            max_patches = 0
                        dict_offsets[(offset_x, offset_y,
                                      max_patches)] = nonzero_regions
                    offset_x, offset_y, max_patches = max(dict_offsets,
                                                          key=dict_offsets.get)
                    nonzero_regions = dict_offsets[(offset_x, offset_y,
                                                    max_patches)]
                    # print(offset_x, offset_y, nonzero_regions, max_patches)
                    if max_patches == 0:
                        print('<>' * 30)
                        print("Glitch found in sampling patches! Will retry")
                        print(dict_offsets)
            else:
                offset_x = None
                offset_y = None

            if weighted_sample:
                criterion = 1
            elif grid_sampling_mode:
                criterion = min(
                    threshold,
                    max(
                        self.df.loc[(self.df.pos_x % 10 == offset_x)
                                    & (self.df.pos_y % 10 == offset_y)]
                        [var].nlargest(20).min(), 1))
            else:
                criterion = min(threshold, self.df[var].nlargest(20).min())

            # will not sample from the edge
            if grid_sampling_mode:
                _df = self.df.loc[(self.df.pos_x % 10 == offset_x)
                                  & (self.df.pos_y % 10 == offset_y) &
                                  (self.df[var] >= criterion)]
            else:
                _df = self.df.loc[(self.df[var] >= criterion)]

            locs = []
            for _ in range(n_tiles):
                if weighted_sample:
                    loc = _df.pos.sample(1, weights=_df[var]).item()
                else:
                    loc = _df.pos.sample(1).item()
                locs.append(loc)

        # sample for given start locations
        loc_tiles = []
        locs_new = []
        locs_orig = []
        fids = []
        pct_valid = []
        for i, loc in enumerate(locs):

            if margin is None:
                _tile_loc, _loc, _loc_orig, _fid, _pct_valid = self.sample_blind(
                    n)
            else:
                _tile_loc, _loc, _loc_orig, _fid, _pct_valid = self.sample_from_tile(
                    n, margin, loc)
            loc_tiles.append(torch.tensor(_tile_loc))
            locs_new.append(torch.tensor(_loc))
            locs_orig.extend(_loc_orig)
            fids.append(torch.tensor(_fid))
            pct_valid.append(torch.tensor(_pct_valid))

        loc_tiles = torch.stack(loc_tiles)
        locs_new = torch.stack(locs_new)
        locs_orig = locs_orig
        fids = torch.stack(fids)
        pct_valid = torch.stack(pct_valid)
        if len(locs) == 1:
            fids = fids.unsqueeze(0)

        output = {
            'loc_tiles': loc_tiles,
            'locs_new': locs_new,
            'locs_orig': locs_orig,
            'fids': fids,
            'pct_valid': pct_valid
        }
        return output



def parse_archs(arch):
    assert len(arch.split('_')) <= 2
    if '_' in arch:
        name, specs = arch.split('_')
        hh = int(re.findall('(?<=h)\\d+', specs)[0])
        ll = int(re.findall('(?<=l)\\d+', specs)[0])
    else:
        name = arch
        hh, ll = 1, 1
    assert name in ['ap','attn','mhattn','deepattnmisl','vit']
    return {'name': name, 'h': hh, 'l': ll}

