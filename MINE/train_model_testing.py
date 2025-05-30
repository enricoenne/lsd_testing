#@title import packages

import albumentations as A
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import random
import torch

import torchvision
import torchvision.transforms as T
import skimage

from glob import glob
from lsd.train import local_shape_descriptor
from scipy.ndimage import binary_erosion
from skimage.measure import label
from skimage.io import imread, imsave
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import utils_2D

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# dataset creation
class CellDataset_single(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir):

        self.images = sorted(glob(image_dir))
        self.masks = sorted(glob(mask_dir))


        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def __len__(self):
        return len(self.images)


    # normalize raw data between 0 and 1
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)

    def __getitem__(self, idx):

        raw_path = self.images[idx]

        labels_path = self.masks[idx]

        raw = imread(raw_path)
        raw = self.normalize(raw)
        
        # slice first channel, relabel connected components
        labels = label(imread(labels_path)).astype(np.uint16)


        input = np.expand_dims(raw, axis=0)

        
        boundaries = skimage.segmentation.find_boundaries(labels)[None].astype(np.float32)
        output = boundaries

        return input, output



def model_training(input_dataset,
                   segmentation_dataset,
                   input_type,
                   output_type,
                   crop_size = 256,
                   training_steps = 200,
                   learning_rate = 1e-4,
                   batch_size = 4,
                   save_model = True,
                   show_metrics = True,
                   model_lsds = None,
                   lsd_sigma = 10,
                   output_folder = ''):
    





input_dataset = '_GFP_max_clahe'
segmentation_dataset = '_CELL_manual'


training_steps = 10000
batch_size = 14
crop_size = 128

lr = 1e-4



input_type, output_type = 'boundaries_d', 'boundaries'


output_folder = 'boundary_reconstruction/proper_degradation_random_bg/'



print()
print('training model: ' + input_type + ' -> ' + output_type)
print(f'steps: {training_steps}    batch: {batch_size}    crop: {crop_size}    lr: {lr}')

# create our network

d_factors = [[2,2],[2,2],[2,2]]

num_fmaps=12
fmap_inc_factor=5

unet = utils_2D.UNet(
    in_channels=1,
    num_fmaps=num_fmaps,
    fmap_inc_factors=fmap_inc_factor,
    downsample_factors=d_factors,
    padding='same',
    constant_upsample=True)

model = torch.nn.Sequential(
    unet,
    torch.nn.Conv2d(in_channels=num_fmaps,out_channels=1, kernel_size=1)
).to(device)

loss_fn = torch.nn.MSELoss().to(device)


#crop_size = 256
#training_steps = 200

# set optimizer
#learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# set activation
activation = torch.nn.Sigmoid()


### create datasets
train_dataset = utils_2D.CellDataset_single(
    image_dir='/group/jug/Enrico/TISSUE_roi_projection/training/*' + input_dataset + '.tif',
    mask_dir='/group/jug/Enrico/TISSUE_roi_projection/training/*' + segmentation_dataset + '.tif')

val_dataset = utils_2D.CellDataset_single(
    image_dir='/group/jug/Enrico/TISSUE_roi_projection/validation/*' + input_dataset + '.tif',
    mask_dir='/group/jug/Enrico/TISSUE_roi_projection/validation/*' + segmentation_dataset + '.tif')


# make dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset)

# training step

def model_step(model, loss_fn, optimizer, feature, gt_target, activation, train_step=True):
    
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()
        
    # forward
    target_logits = model(feature)

    loss_value = loss_fn(target_logits, gt_target)
    
    # backward if training mode
    if train_step:
        loss_value.backward()
        optimizer.step()

        
    target_output = activation(target_logits)

    outputs = {
        'pred_target': target_output,
        'target_logits': target_logits,
    }

    loss_value = loss_value.cpu().detach().numpy()

    return np.array(loss_value), outputs


# training loop

# set flags
model.train() 
loss_fn.train()
step = 0

np.random.seed(42)


with tqdm(total=training_steps) as pbar:
    while step < training_steps:
        # reset data loader to get random augmentations
        
        tmp_loader = iter(train_loader)

        for feature, gt_target in tmp_loader:
            feature = feature.to(device)
            gt_target = gt_target.to(device)
                                        
            loss_value, pred = model_step(model, loss_fn, optimizer, feature, gt_target, activation)
            step += 1
            pbar.update(1)
            


output_folder_name = f'output/test_model/step{str(training_steps)}_b{str(batch_size)}_c{str(crop_size)} _lr{str(lr)}'

if os.path.exists(output_folder_name) == False:
    os.makedirs(output_folder_name)

torch.save(model.state_dict(), f'{output_folder_name}/{input_type}-{output_type}.pth')

print('output: ' + output_folder_name)


