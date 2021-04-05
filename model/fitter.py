import os
import re
import time
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from .helper import *
from warmup_scheduler import GradualWarmupScheduler
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import shutil

import warnings
warnings.filterwarnings("ignore", message=r'Please set `reader_name`.*')


class MultiSchedulers(object):
    def __init__(self):
        self.schedulers = {}
        self.types = {}

    def add(
            self,
            name,
            optimizer,
            optimizer_type,
            warmup_multiplier=1,
            warmup_epochs=40,
            **args):
        if optimizer_type == 'cosine':
            base_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **args)
        if optimizer_type == 'step':
            base_scheduler = lr_scheduler.StepLR(optimizer, **args)
        if optimizer_type == 'exponential':
            base_scheduler = lr_scheduler.ExponentialLR(optimizer, **args)
        if optimizer_type == 'plateau':
            base_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **args)

        self.types[name] = optimizer_type
        self.schedulers[name] = GradualWarmupScheduler(
            optimizer,
            multiplier=warmup_multiplier,
            total_epoch=warmup_epochs,
            after_scheduler=base_scheduler)

    def step(
        self,
        performance_measure,  # performance measurement
        epoch,  # current epoch
        enforce_epoch=100  # only for plateau scheduler
    ):

        for name in self.schedulers.keys():
            if self.types[name] == 'plateau':
                if epoch > enforce_epoch:
                    self.schedulers[name].step(performance_measure)
                else:
                    continue
            else:
                self.schedulers[name].step()


def format_results(res):
    line = ""
    for key in sorted(res):
        val = res[key]
        if isinstance(val, str) or isinstance(val, int):
            fmt = "%s: %s\t"
        else:
            fmt = "%s: %8.6f\t"
        line += (fmt % (key, val))
    return line


def reshape_img_batch(img_batch, crop_size):
    return img_batch.\
        view(img_batch.size(0), 3, -1, crop_size, crop_size).\
        permute(0, 2, 1, 3, 4).\
        contiguous().\
        view(-1, 3, crop_size, crop_size)


class HybridFitter:
    def __init__(
            self,
            model,
            checkpoint_to_resume='',
            writer=None,
            args=None,
            timestr='',
            model_name='model',
            loss_function=None):
        self.writer = writer
        self.args = args
        self.criterion = loss_function

        self.model = model
        self.device = self.model.device
        self.reset_optimizer()
        self.dataloaders = {}
        self.meta_df = {}
        self.es = EarlyStopping(patience=self.args.patience, mode='max')
        self.timestr = timestr
        self.meta_ds = {}
        self.model_name = model_name
        self.checkpoint_to_resume = checkpoint_to_resume

        self.current_epoch = 1
        if len(checkpoint_to_resume):
            self.resume_checkpoint()

        metrics = {
            'classification': 'auc',
            'survival': 'c-index',
            'regression': 'r2'
        }
        self.metric = metrics[self.args.outcome_type]

    def resume_checkpoint(self):
        ckp = torch.load(self.checkpoint_to_resume)
        self.writer['meta'].info("Loading model checkpoints ... Epoch is %s" % ckp['epoch'])
        self.model.load_state_dict(ckp['state_dict_model'])
        self.optimizers['adam'].load_state_dict(ckp['state_dict_optimizer_adam'])
        self.optimizers['sgd'].load_state_dict(ckp['state_dict_optimizer_sgd'])
        self.schedulers.schedulers['adam'].load_state_dict(ckp['state_dict_scheduler_adam'])
        self.schedulers.schedulers['sgd'].load_state_dict(ckp['state_dict_scheduler_sgd'])
        self.current_epoch = ckp['epoch'] + 1

    def get_datasets(
            self,
            pickle_file,
            mode='train',
    ):
        if mode == 'train':
            batch_size = self.args.num_patches * self.args.batch_size
            num_crops = self.args.num_crops
        elif mode == 'val':
            batch_size = self.args.num_val
            # num_crops = (self.args.patch_size // self.args.crop_size)**2
            num_crops = self.args.num_crops
        else:
            batch_size = self.args.batch_size
            num_crops = self.args.num_crops

        ds = SlidesDataset(
            data_file=pickle_file,
            image_dir=self.args.root_dir,
            crop_size=self.args.crop_size,
            num_crops=num_crops,
            outcome=self.args.outcome,
            outcome_type=self.args.outcome_type
        )
        dl = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            drop_last=False if self.args.mode == 'predict' else True
        )

        iterator = iter(dl)
        pipe = ExternalSourcePipeline(
            data_iterator=iterator,
            batch_size=batch_size,
            num_threads=1,
            device_id=0,
            mode=mode,
            crop_size=self.args.crop_size
        )
        pipe.build()
        self.dataloaders[mode] = DALIGenericIterator(
            [pipe], ['images', 'ids', 'targets'], pickle_file.shape[0])

        del iterator

    def prepare_datasets(
            self,
            pickle_file,
            mode='train',
            batch_size=256):
        self.batch_size = batch_size
        if isinstance(pickle_file, str):
            _df = pd.read_pickle(pickle_file)
        else:
            _df = pickle_file

        if mode == 'train':
            self.meta_df[mode] = grouped_sample(_df,
                                                num_patches=self.args.num_patches,
                                                num_repeats=self.args.repeats_per_epoch,
                                                stratify=self.args.stratify,
                                                e_ne_ratio=self.args.e_ne_ratio)

            self.writer['meta'].info(self.meta_df[mode].shape)
        elif self.args.sample_id:
            self.meta_df[mode] = _df.groupby('submitter_id', group_keys=False).apply(
                lambda x: x.sample(self.args.num_val, replace=True))
        else:
            self.meta_df[mode] = _df

    def reset_optimizer(self):
        conv_params = []
        for name, param in self.model.backbone.named_parameters():
            if re.search('fc', name):
                pass
            else:
                if param.requires_grad:
                    conv_params.append(param)
        self.optimizers = {}

        self.optimizers['adam'] = optim.Adam([
            {
                'params': conv_params,
                'lr': self.args.lr_backbone,
                'weight_decay': self.args.wd_backbone
            },
            {
                'params': self.model.head.parameters(),
                'lr': self.args.lr_head,
                'weight_decay': self.args.wd_head
            }])

        self.optimizers['sgd'] = optim.SGD(
            self.model.head.parameters(),
            lr=self.args.lr_head,
            weight_decay=self.args.wd_head)

        self.schedulers = MultiSchedulers()
        if self.args.scheduler == 'cosine':
            self.schedulers.add(
                'adam', self.optimizers['adam'],
                'cosine', T_0=self.args.anneal_freq)
            self.schedulers.add('sgd', self.optimizers['sgd'],
                'cosine', T_0=self.args.anneal_freq)
        elif self.args.scheduler == 'plateau':
            self.schedulers.add('adam', self.optimizers['adam'],
                'plateau', mode='max', factor=0.3, patience=self.args.lr_patience, min_lr=1e-8)
            self.schedulers.add(
                'sgd', self.optimizers['sgd'],
                'plateau', mode='max', factor=0.3, patience=self.args.lr_patience, min_lr=1e-8)
        elif self.args.scheduler == 'step':
            self.schedulers.add(
                'adam', self.optimizers['adam'],
                'step', gamma=self.args.gamma, step_size=self.args.step_size)
            self.schedulers.add(
                'sgd', self.optimizers['sgd'],
                'step', gamma=self.args.gamma, step_size=self.args.step_size)
        elif self.args.scheduler == 'exponential':
            self.schedulers.add(
                'adam', self.optimizers['adam'],
                'exponential', gamma=self.args.gamma)
            self.schedulers.add('sgd', self.optimizers['sgd'],
                'exponential', gamma=self.args.gamma)

    def freeze_conv_layers(self):
        for name, param in self.model.backbone.named_parameters():
            if re.search(self.args.trainable_layers, name):
                param.requires_grad = True
            else:
                param.requires_grad = False

            self.writer['meta'].info("Freezing convolutional layers ....")
            self.writer['meta'].info('%s-%s' % (name, param.requires_grad))

        # adam optimizer
        self.reset_optimizer()

    def unfreeze_conv_layers(self):
        for name, param in self.model.backbone.named_parameters():
            param.requires_grad = True
        # adam optimizer
        self.writer['meta'].info("All convolutional layers are available for training!")
        self.reset_optimizer()

    def train(self, df_train=None, epoch=0):
        self.prepare_datasets(
            df_train,
            'train',
            batch_size=self.args.batch_size * self.args.num_patches)
        self.get_datasets(
            self.meta_df['train'],
            'train')

        self.writer['meta'].info('Training from step %s' % epoch)
        # training phase
        self.model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        perfs = AverageMeter(self.metric, ':5.4f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':5.4f')
        progress = ProgressMeter(
            len(self.dataloaders['train']),
            [batch_time, data_time, losses, perfs],
            prefix="Epoch: [{}]".format(epoch),
            writer=self.writer['meta'],
            verbose=True)
        end = time.time()

        eval_t = ModelEvaluation(
            outcome_type=self.args.outcome_type,
            loss_function=self.criterion,
            mode='train',
            device=self.device,
            timestr=self.timestr)

        for group in self.optimizers['adam'].param_groups:
            current_lr = group['lr']
            self.writer['meta'].info("Learning rate is %s" % (current_lr))
        self.writer['meta'].info('-' * 30)

        # train over all training data
        for i, sample in enumerate(self.dataloaders['train']):

            batch_data = sample[0]
            train_imgs, train_ids, train_targets = list(
                map(lambda x: batch_data[x], ['images', 'ids', 'targets']))

            # if train_targets[:,1].max() == 0:
            #     print(train_targets)
            #     sys.exit()
            train_imgs = reshape_img_batch(train_imgs, self.args.crop_size)

            nbatches = train_imgs.size(0) // (self.args.num_patches * self.args.num_crops)

            data_time.update(time.time() - end)

            # forward and backprop
            self.optimizers['sgd'].zero_grad()
            with torch.set_grad_enabled(True):
                ppi = self.args.num_patches * self.args.num_crops
                model_outputs = self.model(train_imgs, ppi)
                train_preds = model_outputs['pred']

                train_targets = train_targets.view(nbatches, self.args.num_patches, -1)[:, 0, :]

                if self.args.outcome_type == 'survival' and self.args.time_noise > 0:
                    train_targets[:,0] = (train_targets[:,0] + torch.zeros_like(train_targets[:,0]).normal_(mean=0, std=self.args.time_noise)).exp()
                train_loss = self.criterion.calculate(train_preds, train_targets)
                train_loss += model_outputs.get('cls_loss', 0)

                train_loss.backward()

                eval_t.update(
                    {
                        "ids": train_ids,
                        "preds": train_preds,
                        "targets": train_targets
                    }
                )

                self.optimizers['adam'].step()
                self.optimizers['adam'].zero_grad()

                all_head_params = torch.cat([x.view(-1)
                                             for x in list(self.model.head.parameters())])
                l1_regularization = self.args.l1 * torch.norm(all_head_params.view(-1), 1)
                l2_regularization = self.args.l2 * torch.norm(all_head_params.view(-1), 2).pow(2)
                loss_regularization = l1_regularization + l2_regularization

                loss_regularization.backward()
                self.optimizers['sgd'].step()

            # update metrics
            losses.update(train_loss.item(), nbatches)

            train_metrics = calculate_metrics(
                train_preds.data.cpu().numpy(),
                train_targets.data.cpu().numpy(),
                outcome_type=self.args.outcome_type)
            perfs.update(train_metrics[self.metric], nbatches)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.log_freq == 0:
                progress.display(i + 1)

        train_res = eval_t.evaluate()
        train_res['epoch'] = epoch
        train_res['mode'] = 'train'
        return train_res

    def evaluate(self, df_val, epoch=0):
        self.prepare_datasets(
            df_val,
            'val',
            batch_size=self.args.batch_size)
        self.get_datasets(
            self.meta_df['val'],
            'val')

        self.writer['meta'].info('Starting evaluation')
        # validation phase
        self.model.eval()
        eval_v = ModelEvaluation(
            outcome_type=self.args.outcome_type,
            loss_function=self.criterion,
            mode='val',
            device=self.device,
            timestr=self.timestr)

        # forward prop over all validation data
        for i, sample in enumerate(tqdm(self.dataloaders['val'])):

            batch_data = sample[0]
            val_imgs, val_ids, val_targets = list(
                map(lambda x: batch_data[x], ['images', 'ids', 'targets']))

            val_imgs = reshape_img_batch(val_imgs, self.args.crop_size)
            nbatches = val_imgs.size(0) // (self.args.num_val * self.args.num_crops)
            # forward
            with torch.set_grad_enabled(False):
                val_preds = self.model(val_imgs, self.args.num_val*self.args.num_crops)['pred']
                val_targets = val_targets.view(nbatches, self.args.num_val, -1)[:, 0, :]
                eval_v.update(
                    {
                        "ids": val_ids,
                        "preds": val_preds,
                        "targets": val_targets
                    }
                )

        val_res = eval_v.evaluate()
        val_res['epoch'] = epoch
        val_res['mode'] = 'val'
        return val_res

    def fit_epoch(self, pickle_files, epoch=0):

        train_res = self.train(
            pickle_files['train'],
            epoch=epoch)
        val_res = self.evaluate(
            pickle_files['val'],
            epoch=epoch)

        self.writer['data'].info(format_results(train_res))
        self.writer['data'].info(format_results(val_res))

        ########################################
        # schedule step
        ########################################

        if self.args.outcome_type == 'survival':
            performance_measure = torch.tensor(val_res['c-index'])
        elif self.args.outcome_type == 'classification':
            performance_measure = torch.tensor(val_res['auc'])
        elif self.args.outcome_type == 'regression':
            performance_measure = torch.tensor(val_res['r2'])

        self.schedulers.step(performance_measure, epoch)

        is_best = False
        if performance_measure > self.es.best:
            self.es.best = performance_measure
            print("New best result: %6.4f" % self.es.best)
            is_best = True
        self.save_checkpoint(epoch, is_best, self.args.save_interval, self.checkpoints_folder)

        if epoch >= 100:
            if self.es.step(performance_measure):
                return 1  # early stop criterion is met, we can stop now
        return 0

    def save_checkpoint(
            self,
            epoch,
            is_best,
            save_freq,
            checkpoints_folder):

        state_dict = {
            'epoch': epoch,
            'state_dict_model': self.model.state_dict(),
            'state_dict_optimizer_adam': self.optimizers['adam'].state_dict(),
            'state_dict_optimizer_sgd': self.optimizers['sgd'].state_dict(),
            'state_dict_scheduler_adam': self.schedulers.schedulers['adam'].state_dict(),
            'state_dict_scheduler_sgd': self.schedulers.schedulers['sgd'].state_dict(),
        }
        # remaining things related to training
        os.makedirs(checkpoints_folder, exist_ok=True)
        epoch_output_path = os.path.join(checkpoints_folder, "LAST.pt")
        torch.save(state_dict, epoch_output_path)

        if is_best:
            print("Saving new best result!")
            fname_best = os.path.join(checkpoints_folder, "BEST.pt")
            if os.path.isfile(fname_best):
                os.remove(fname_best)
            shutil.copy(epoch_output_path, fname_best)

        if epoch % save_freq == 0:
            print("Saving new checkpoints!")
            shutil.copy(epoch_output_path, os.path.join(checkpoints_folder, "%04d.pt" % epoch))

    def fit(
        self,
        pickle_files,
        checkpoints_folder='checkpoints'
    ):

        self.checkpoints_folder = checkpoints_folder

        for epoch in range(self.current_epoch, self.args.epochs + 1):
            return_code = self.fit_epoch(
                pickle_files,
                epoch=epoch)
            if return_code:
                break
