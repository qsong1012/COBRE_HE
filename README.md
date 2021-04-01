# Whole Slide Image based Patient Level Prediction (WSI-PLP)
This repository contains the complete pipeline to predict patient outcome (categorical, time to event, or continuous) using whole slide images (WSIs). The method is described in detail in https://www.medrxiv.org/content/10.1101/2021.01.21.21250241v1 . 
![Model Structure](figure/model-structure.png)

# Requirements
* python ≥ 3.6
* Pytorch ≥ 1.7
* Pandas, Numpy
* sklearn
* tqdm
* warmup_scheduler (https://github.com/ildoonet/pytorch-gradual-warmup-lr)
* NVIDIA DALI (https://github.com/NVIDIA/DALI)
* lifelines (https://lifelines.readthedocs.io/en/latest/)

# Usage

## 1. Obtaining the datasets from TCGA

* Download WSIs from the TCGA repository, store in `data/WSI_TCGA`
* Download the meta file (with clinical and demographical information) from the TCGA repository in json, store in `data/meta_files`.


## 1. Create patches from the whole slide images

Example: 

```
python patch_extraction.py --cancer=LGG --num-cpus=8 --magnification=10 --patch-size=224 
```

## 2. Extract meta information; setup train/val/test splits

```
python create_meta_info.py --cancer=LGG --ffpe-only --magnification=10 --stratify=status
```

If the variable is not available in the meta json file, you will need to manually add this variable to the generated `data/meta_clinical_[].csv` file. If this variable is a genetic mutation, such as IDH, downloading this variable from TCGA and merge it to the .csv file can be done by running

```
Rscript obtain_gene_mutations.R
```

## 3. Training the deep learning model

Example:

* Classification
```
python train.py --cancer=LGG --magnification=10 --stratify=idh -b=8 --repeats-per-epoch=10 --num-patches=8 --num-val=100 --sample-id --save-interval=10 --outcome-type=classification --outcome=idh --inner-fold=3 --scale-lr --pretrain --split=00 --lr-backbone=1e-4 --lr-head=1e-4
```

* Survival
```
python train.py --cancer=LGG --magnification=10 --stratify=status -b=8 --repeats-per-epoch=10 --num-patches=8 --num-val=100 --sample-id --save-interval=10 --outcome-type=survival --inner-fold=3 --scale-lr --e-ne-ratio=1to3 --pretrain --split=00 --time-noise=1.0
```