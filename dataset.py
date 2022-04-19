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

##########################################
####        Dataset Class             ####
##########################################

class PatientDataset(Dataset):
    # TODO: Comment and check functions
    def __init__(self, pids, num_patches, transforms = None):
        self.random_seed = random.sample(range(0, 10000), len(pids))
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
    '''
    Dataset Class for Slide-level training (each datapoint is a slide)
    pids: path to patient folder
    num_patches: number of patches to sample from each slide
    transfroms: image augmentation
    '''
    def __init__(self, pids, num_patches, transforms = None):
        self.pids = pids
        self.slides = self.get_slide_ids(self.pids)
        self.random_seed = random.sample(range(0, 10000), len(self.slides))
        self.num_patches = num_patches
        self.transforms = transforms
        self.label_map = {'nodepos':1, 'nodeneg':0} # TODO: change this to not hard code
        
    def __getitem__(self, idx):
        slide_id = self.slides[idx] # get path to slide folder
        label = self.label_map[slide_id.parent.parents[0].name] # get label from folder path
        images = self.read_random_patches(list(slide_id.glob('*.png')))
        
        # apply transformation to each patch in the stack (revisit to optimize)
        if self.transforms:
            images_transform = []
            for patch in images:
                random.seed(self.random_seed[idx])
                torch.manual_seed(self.random_seed[idx])
                images_transform.append(self.transforms(patch))
            images = torch.stack([patch for patch in images_transform])
        return images, str(slide_id), label
    
    def __len__(self):
        return len(self.slides)
    
    def get_slide_ids(self, pids):
        '''
        Get path to slides from patient folders
        '''
        slide_ids = []
        for pid in pids:
            slide_ids.extend(list(pid.glob('*')))
        return slide_ids
    
    def read_random_patches(self, patch_paths):
         '''
        Sample random pateches from all available patches. 
        Sample without replacement if number of available patches > num_patches.
        Otherwise sample with replacement.
        '''
        # sample without replacement when #available patches >= num_patchs
        if len(patch_paths) >= self.num_patches: 
            random.shuffle(patch_paths)
            patch_paths = patch_paths[:self.num_patches]
        # sample with replacement when #available patches < num_patchs
        else:     
            patch_paths = np.random.choice(patch_paths,self.num_patches)
        images = []
        for path in patch_paths:
            img = np.array(Image.open(path))
            img = torch.tensor(img).permute(2,0,1)/255.
            images.append(img)
        images = torch.stack([patch for patch in images])
        return images
    
    def read_random_patches_deprecated(self, patch_paths):
        '''Deprecated read patch function that does only sample without replacement '''
        random.shuffle(patch_paths)
        patch_paths = patch_paths[:self.num_patches]
        images = []
        for path in patch_paths:
            img = np.array(Image.open(path))
            img = torch.tensor(img).permute(2,0,1)/255.
            images.append(img)
        images = torch.stack([patch for patch in images])
        return images


def get_transformation(patch_size=224, mean=[0.5,0.5,0.5], std=[0.25, 0.25, 0.25]):
    '''
    Get data augmentation for different dataset
    patch_size: size of the crops
    mean: sample mean, default [0.5,0.5,0.5] if not defined
    std: sample std, default [0.25, 0.25, 0.25] if not defined
    '''
    if mean==None or std==None:
        mean = PATH_MEAN
        std = PATH_STD
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
            transforms.Normalize(mean, std)]),
        'val':
        transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.Normalize(mean, std)
        ]),
        'predict':
        transforms.Compose([
            transforms.Normalize(mean, std)
        ]),
        'normalize':
        transforms.Compose([
            transforms.Normalize(mean, std)
        ]),
        'unnormalize':
        transforms.Compose([
            transforms.Normalize(*reverse_norm(mean, std))
            ]),
    }
    return data_transforms


# --------------------------------
# Unused functions, revisit later
# --------------------------------
def get_dataloader(config, transform):
    '''
    Constructed dataloader for training and validation data, currently called in main
    config: arguments passed from configuration file
    transform: data augmentation from get_transformation function
    '''
    
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
    '''
    Weighted sampling to sample both positive and negative class from batch
    Only used for Survival data
    TODO: revisit for suvival analysis
    '''
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

def grouped_sample(data, stratify_var, weights, num_obs=10000, num_patches=4, patient_var='submitter_id', patch_var='file'):
    #################################
    # step 1: sample patients
    if stratify_var is None:
        data_meta = data[[patient_var]].drop_duplicates()
        groups = []
        while len(groups) < num_obs:
            groups.extend(data_meta.sample(frac=1.)[patient_var].tolist())
    else:
        data_meta = data[[patient_var,stratify_var]].drop_duplicates()
        ids = data_meta.groupby(stratify_var)[patient_var].apply(unique_shuffle_to_list).tolist()
        groups = []
        while len(groups) < num_obs:
            group = []
            for j,k in enumerate(weights):
                while k > 0:
                    if min(list(map(len,ids))) == 0:
                        ids = data_meta.groupby(stratify_var)[patient_var].apply(unique_shuffle_to_list).tolist()
                    group.append(ids[j].pop())
                    k -= 1
            np.random.shuffle(group)
            groups.extend(group)
    # post processing
    groups = groups[:num_obs]

    dfg = pd.DataFrame(groups,columns=[patient_var])
    dfg['queue_order'] = dfg.index
    dfg = dfg.merge(data_meta,on=patient_var).sort_values('queue_order').reset_index(drop=True)

    #################################
    # step 2: sample patches
    # get the order of occurrence for each patient
    dfg['dummy_count'] = 1
    dfg['within_index'] = dfg.groupby(patient_var).dummy_count.cumsum() - 1

    # sample sufficient amount of patches
    dfp = data.groupby(patient_var, as_index=False).sample(
        num_patches * (dfg.within_index.max() + 1), replace=True)
    # for each patient, determine merge to which occurrence
    dfp['within_index'] = dfp.groupby(patient_var)[patch_var].transform(
        lambda x: np.arange(x.shape[0]) // num_patches)

    if stratify_var is not None:
        dfg.drop([stratify_var],axis=1,inplace=True)
    df_sel = dfg.merge(dfp, on=[patient_var, 'within_index'], how='left')
    return df_sel


########################################
# get the data transforms:
def reverse_norm(mean, std):
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    return (-mean / std).tolist(), (1 / std).tolist()

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
            num_crops,
            features=None,
            transform=None,
            ):

        self.df = data_file
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.transform = transform

        if outcome_type == 'survival':
            self.outcomes = self.df[['time', 'status']].to_numpy().astype(float)
        else:
            self.outcomes = self.df[[outcome]].to_numpy().astype(float)
        self.ids = self.df['id_patient'].tolist()
        self.files = self.df['file'].tolist()

        try:
            self.random_seeds = self.df['random_id'].tolist()
        except:
            self.random_seeds = np.ones(self.df.shape[0])

    def __len__(self):
        return self.df.shape[0]

    def sample_patch(self, idx):
        idx = idx % self.df.shape[0]
        fname = self.files[idx]
        random_seed = self.random_seeds[idx]

        img_name = os.path.join(self.image_dir, fname)
        imgs = np.array(Image.open(img_name))
        imgs = random_crops(imgs, self.crop_size, self.num_crops).reshape(-1, self.crop_size, 3)
        if self.transform is None:
            pass
        else:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            imgs = self.transform(torch.tensor(imgs).permute(2,0,1)/255.)
        sample = (
            imgs, 
            self.ids[idx], 
            # random_seed, 
            self.outcomes[idx,:]
            )
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sample_patch(idx)


def batchsplit(img, nrows, ncols):
    return np.concatenate(
        np.split(
            np.stack(
                np.split(img, nrows, axis=0)
            ), ncols, axis=2)
    )

# ----------------------------------------------------------------------------
# Helper functions for data transformation, currently not used in the pipeline
# ----------------------------------------------------------------------------
def reverse_norm(mean, std):
    '''Reverse normalization'''
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    return (-mean / std).tolist(), (1 / std).tolist()


def random_crop(img, height, width):
    '''random crop image'''
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img

def random_crops(img, crop_size, n_crops):
    '''Generate multiple random crops from the image (not used)'''
    h, w, c = img.shape
    if (h == crop_size) and (w == crop_size):
        return img
    nrows = h // crop_size
    ncols = w // crop_size
    if max(nrows % crop_size, ncols % crop_size):
        img = random_crop(img, nrows * crop_size, ncols * crop_size)
    splits = batchsplit(img, nrows, ncols)
    return splits[random.sample(range(splits.shape[0]), n_crops)]