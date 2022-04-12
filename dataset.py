from pathlib import Path
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import torch
from PIL import Image
import pickle
import torchvision.transforms.functional as tv_F
from torchvision import transforms

class IDHDataset(Dataset):
    def __init__(self, config, mode='train', transform = None):
        self.random_seed = 1
        self.config = config
        self.image_dir = config.image_dir
        self.patch_df = pd.read_pickle(config.patch_df)
        self.df = pd.read_pickle(config.id_df[mode])
        if mode == 'train':
            self.num_patches = config.num_patches
            num_obs = len(self.df[config.patient_var].unique())*config.repeats_per_epoch
            self.pids = weighted_sampling(self.df, num_obs, weights = [3,1])
#             self.pids = pd.concat([self.pids]*config.repeats_per_epoch, ignore_index=True)
        elif mode == 'val':
            self.pids = self.df
            self.num_patches = config.num_val
        self.transforms = transform
     
    def __getitem__(self, idx):
        pid = self.pids.iloc[idx]
        sid = pid['submitter_id']
        status = pid['status']
        days = pid['time']
        images = self.read_random_patches(sid)
        
        if self.transforms:
            images_transform = []
            for patch in images:
#                 random.seed(self.random_seed)
#                 torch.manual_seed(self.random_seed)
                images_transform.append(self.transforms(patch))
            
#             images = torch.stack([self.transforms(patch) for patch in images])
            images = torch.stack([patch for patch in images_transform])
        return images, sid, np.array([days, status], dtype='float')
    
    def __len__(self):
        return len(self.pids)
    
    def read_random_patches(self, sid):
        sid_df = self.patch_df[self.patch_df['submitter_id']== sid]
        sid_df = sid_df.sample(frac=1).iloc[:self.num_patches]
        image_path = sid_df['file'].to_list()
        image_path = [os.path.join(self.image_dir, path) for path in image_path]
        images = []
        for path in image_path:
            img = np.array(Image.open(path))
            img = random_crops(img, 224, 1).reshape(-1, 224, 3)
            img = torch.tensor(img).permute(2,0,1)/255.
            images.append(img)
#             images.append(tv_F.to_tensor(Image.open(path).convert("RGB")))
#         images = torch.stack([patch for patch in images])
        return images
    
class PatientDataset(Dataset):
    def __init__(self, args, pids, num_patches, transforms = None):
        self.random_seed = random.sample(range(0, 10000), len(pids))
        self.root_dir = args.root_dir
        self.pids = pids
        self.num_patches = num_patches
        self.transforms = transforms
        self.label_map = {'nodepos':1, 'nodeneg':0}
        
    def __getitem__(self, idx):
        pid = self.pids[idx]
        label = self.label_map[pid.parents[0].name]
        images = self.read_random_patches(list(pid.glob('*/*.png')))        
        if self.transforms:
            images_transform = []
            for patch in images:
                random.seed(self.random_seed[idx])
                torch.manual_seed(self.random_seed[idx])
                images_transform.append(self.transforms(patch))
            
            images = torch.stack([patch for patch in images_transform])
#             images = self.transforms(images)
        return images, pid.stem, label
    
    def __len__(self):
        return len(self.pids)
    
    def read_random_patches(self, patch_paths):
        random.shuffle(patch_paths)
        patch_paths = patch_paths[:self.num_patches]
        images = []
        for path in patch_paths:
#             images.append(tv_F.to_tensor(Image.open(path).convert("RGB")))
            img = np.array(Image.open(path))
#             img = random_crops(img, 224, 1).reshape(-1, 224, 3)
            img = torch.tensor(img).permute(2,0,1)/255.
            images.append(img)
        images = torch.stack([patch for patch in images])
        return images
    
    
class SlideDataset(Dataset):
    def __init__(self, args, pids, num_patches, transforms = None):
        self.pids = pids
        self.slides = self.get_slide_ids(self.pids)
        self.random_seed = random.sample(range(0, 10000), len(self.slides))
        self.num_patches = num_patches
        self.transforms = transforms
        self.label_map = {'nodepos':1, 'nodeneg':0}
        
    def __getitem__(self, idx):
        slide_id = self.slides[idx]
        label = self.label_map[slide_id.parent.parents[0].name]
        images = self.read_random_patches(list(slide_id.glob('*.png')))        
        if self.transforms:
            images_transform = []
            for patch in images:
                random.seed(self.random_seed[idx])
                torch.manual_seed(self.random_seed[idx])
                images_transform.append(self.transforms(patch))
            
#             images = torch.stack([self.transforms(patch) for patch in images])
            images = torch.stack([patch for patch in images_transform])
#             images = self.transforms(images)
        return images, str(slide_id), label
    
    def __len__(self):
        return len(self.slides)
    
    def get_slide_ids(self, pids):
        slide_ids = []
        for pid in pids:
            slide_ids.extend(list(pid.glob('*')))
        return slide_ids
    
    def read_random_patches(self, patch_paths):
        random.shuffle(patch_paths)
        patch_paths = patch_paths[:self.num_patches]
        images = []
        for path in patch_paths:
#             images.append(tv_F.to_tensor(Image.open(path).convert("RGB")))
            img = np.array(Image.open(path))
#             img = random_crops(img, 224, 1).reshape(-1, 224, 3)
            img = torch.tensor(img).permute(2,0,1)/255.
            images.append(img)
        images = torch.stack([patch for patch in images])
        return images
    
    def read_random_patches_modified(self, patch_paths):
        # sample without replacement when #available patches >= num_patchs
        if len(patch_paths) >= self.num_patches: 
            random.shuffle(patch_paths)
            patch_paths = patch_paths[:self.num_patches]
        # sample with replacement when #available patches < num_patchs
        else:     
            patch_paths = np.random.choice(patch_paths,self.num_patches)
        images = []
        for path in patch_paths:
#             images.append(tv_F.to_tensor(Image.open(path).convert("RGB")))
            img = np.array(Image.open(path))
#             img = random_crops(img, 224, 1).reshape(-1, 224, 3)
            img = torch.tensor(img).permute(2,0,1)/255.
            images.append(img)
        images = torch.stack([patch for patch in images])
        return images
    
def get_dataloader(config, transform):
    if config.level == 'patient':
        train_data = PatientDataset(config.data_split, 'train', num_patch=config.num_patches, transforms=transform['train'])
        val_data = PatientDataset(config.data_split, 'val', num_patch=config.num_patches, transforms=transform['val'])
        
    elif config.level == 'slide':
        train_data = SlideDataset(config.data_split, 'train', num_patch=config.num_patches, transforms=transform['train'])
        val_data = SlideDataset(config.data_split, 'val', num_patch=config.num_patches, transforms=transform['val'])
       
    train_loader = DataLoader(train_data, batch_size = config.batch_size, shuffle = True, num_workers = config.num_workers)
    val_loader = DataLoader(val_data, batch_size = config.batch_size, shuffle = False, num_workers = config.num_workers)

    loader = {'train': train_loader, 'val': val_loader}
    
    return loader


def weighted_sampling(df, num_obs, weights = [3,1], stratify_var='status', patient_var='submitter_id'):
    ids = df.groupby(stratify_var)[patient_var].apply(unique_shuffle_to_list).tolist()
    groups = []
    while len(groups) < num_obs:
        group = []
        for j,k in enumerate(weights):
            while k > 0:
                if min(list(map(len,ids))) == 0:
                    ids = df.groupby(stratify_var)[patient_var].apply(unique_shuffle_to_list).tolist()
                group.append(ids[j].pop())
                k -= 1
        np.random.shuffle(group)
        groups.extend(group)
        
    groups = groups[:num_obs]
    dfg = pd.DataFrame(groups,columns=[patient_var])
    dfg['queue_order'] = dfg.index
    dfg = dfg.merge(df,on=patient_var).sort_values('queue_order').reset_index(drop=True)
    return dfg

def unique_shuffle_to_list(x):
    x = x.unique()
    np.random.shuffle(x)
    return x.tolist()

# PATH_MEAN = [0.8523, 0.7994, 0.8636]
# PATH_STD = [0.1414, 0.2197, 0.0854]
PATH_MEAN = [0.5,0.5,0.5]
PATH_STD = [0.25, 0.25, 0.25]

def get_transformation(patch_size=224):

    data_transforms = {
        'train':
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(patch_size),
            transforms.ColorJitter(brightness=0.35,
                                   contrast=0.5,
                                   saturation=0.1,
                                   hue=0.16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                PATH_MEAN, PATH_STD
            )  # mean and standard deviations for lung adenocarcinoma resection slides
        ]),
        'val':
        transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.Normalize(PATH_MEAN, PATH_STD)
        ]),
        'predict':
        transforms.Compose([
            transforms.Normalize(PATH_MEAN, PATH_STD)
        ]),
        'normalize':
        transforms.Compose([
            transforms.Normalize(PATH_MEAN, PATH_STD)
        ]),
        'unnormalize':
        transforms.Compose([
            transforms.Normalize(*reverse_norm(PATH_MEAN, PATH_STD))
            ]),
    }
    return data_transforms

# get the data transforms:
def reverse_norm(mean, std):
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    return (-mean / std).tolist(), (1 / std).tolist()


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