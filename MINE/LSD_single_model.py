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
    
    num_in_channels = possible_inputs[input_type]
    num_out_channels = possible_outputs[output_type]

    input_info = ''
    if input_type == 'lsds' or input_type == 'raw_lsds':
        if model_lsds is not None:
            input_info = '_pred'
        else:
            input_info = '_gt'
    
    print()
    print('training model: ' + input_type + input_info + ' -> ' + output_type)
    print(f'steps: {training_steps}    batch: {batch_size}    crop: {crop_size}    lr: {learning_rate}    lsd_s: {lsd_sigma}')

    # create our network

    d_factors = [[2,2],[2,2],[2,2]]

    num_fmaps=12
    fmap_inc_factor=5

    unet = utils_2D.UNet(
        in_channels=num_in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factors=fmap_inc_factor,
        downsample_factors=d_factors,
        padding='same',
        constant_upsample=True)

    model = torch.nn.Sequential(
        unet,
        torch.nn.Conv2d(in_channels=num_fmaps,out_channels=num_out_channels, kernel_size=1)
    ).to(device)

    loss_fn_boundaries = torch.nn.BCEWithLogitsLoss().to(device)
    loss_fn_lsds = torch.nn.MSELoss().to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    ### create datasets
    train_dataset = utils_2D.CellDataset_single(
        image_dir='/group/jug/Enrico/TISSUE_roi_projection/training/*' + input_dataset + '.tif',
        mask_dir='/group/jug/Enrico/TISSUE_roi_projection/training/*' + segmentation_dataset + '.tif',
        crop_size=crop_size,
        split='train',
        input_type=input_type, output_type=output_type,
        model=model_lsds,
        sigma = lsd_sigma)

    val_dataset = utils_2D.CellDataset_single(
        image_dir='/group/jug/Enrico/TISSUE_roi_projection/validation/*' + input_dataset + '.tif',
        mask_dir='/group/jug/Enrico/TISSUE_roi_projection/validation/*' + segmentation_dataset + '.tif',
        crop_size=crop_size,
        split='val',
        input_type=input_type, output_type=output_type,
        model=model_lsds,
        sigma = lsd_sigma)


    # make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset)

    # training step

    def model_step(model, optimizer, feature, gt_target, train_step=True):
        
        # zero gradients if training
        if train_step:
            optimizer.zero_grad()
            
        # forward
        target_logits = model(feature)
        # print(gt_target.min(), gt_target.max())
        # print(target_pred.min(), target_pred.max())
        
        target_pred = torch.sigmoid(target_logits)

        if output_type == 'lsds':
            loss_value = loss_fn_lsds(target_pred, gt_target)
        elif output_type == 'boundaries':
            # with Binary Cross Entropy Loss we need to use logits
            loss_value = loss_fn_boundaries(target_logits, gt_target)
        

        # backward if training mode
        if train_step:
            loss_value.backward()
            optimizer.step()
    
        outputs = {
            'loss': loss_value,
            'pred_target': target_pred,
        }

        # if train_step == False and num_out_channels == 7:
        #     loss_value_bound = loss_boundaries.cpu().detach().numpy()
        #     loss_value_lsd = loss_lsds.cpu().detach().numpy()

        #     loss_value = (loss_value, loss_value_bound, loss_value_lsd)

        
        return outputs

    # training loop

    # set flags
    model.train() 
    step = 0

    np.random.seed(42)

    if num_out_channels == 7:
        metrics = pd.DataFrame(columns=['step', *range(3)])
    else:
        metrics = pd.DataFrame(columns=['step', 0])

    with tqdm(total=training_steps) as pbar:
        while step < training_steps:
            # reset data loader to get random augmentations
            
            tmp_loader = iter(train_loader)

            for feature, gt_target in tmp_loader:
                feature = feature.to(device)
                gt_target = gt_target.to(device)
                                            
                loss_value, pred = model_step(model, optimizer, feature, gt_target)
                step += 1
                pbar.update(1)
                
                if step % 100 == 0:
                    model.eval()
                    tmp_val_loader = iter(val_loader)
                    acc_loss = []
                    for feature, gt_target in tmp_val_loader:
                        feature = feature.to(device)
                        gt_target = gt_target.to(device)

                        current_step = model_step(model, optimizer, feature, gt_target, train_step=False)
                        loss_value = current_step['loss'].cpu().detach().numpy()
                        acc_loss.append(loss_value)
                    model.train()
                    acc_loss = np.array(acc_loss)
                    acc_loss = np.mean(acc_loss, axis=0)

                    # if the loss is an array we need to unpack it
                    # 0: overall loss, 1: boundary loss, 2: lsds loss
                    if len(acc_loss.shape)>0:
                        new_row = {'step': step} | dict(enumerate(acc_loss))
                    else:
                        new_row = {'step': step, 0: acc_loss}

                    # adding the metrics to the dataframe

                    metrics.loc[len(metrics)] = new_row
                    metrics = metrics.reset_index(drop=True)

    output_folder_name = 'output/'+ output_folder +'/step' + str(training_steps) + '_b' + str(batch_size) + '_c' + str(crop_size) + '_lr' + str(learning_rate) + '_s' + str(lsd_sigma)
    if save_model:
        if os.path.exists(output_folder_name) == False:
            os.makedirs(output_folder_name)
        
        torch.save(model.state_dict(), output_folder_name + '/' + input_type + input_info + '-' + output_type +'.pth')

        metrics.to_csv(output_folder_name + '/' + input_type + input_info + '-' + output_type + '.csv', index=False)
        print('output: ' + output_folder_name)


    if show_metrics:
        plt.plot(metrics['step'], metrics['loss'])
        plt.xlabel('step')
        plt.ylabel('mean(acc_loss)')
        plt.ylim([0, min(100,np.max(metrics['loss']))])
        plt.show()
    
    return 0

possible_inputs = {'raw': 1,
                   'lsds': 6,
                   'raw_lsds': 7,
                   'boundaries_d': 1}

possible_outputs = {'labels': 1,
                    'lsds': 6,
                    'boundaries': 1,
                    'boundaries_lsds': 7,
                    'affinities': 2,
                    'affinities_lsds': 8}


input_dataset = '_GFP_max_clahe'
segmentation_dataset = '_CELL_manual'




testing = [('raw','boundaries'),
           ('raw','boundaries_lsds'),
           ('raw','lsds'),
           ('lsds','boundaries'),
           ('raw_lsds','boundaries')]


'''model_lsds_path = '/home/enrico.negri/github/lsd_testing/output/output_5000_14_256_s10/raw-lsds.pth'

model_lsd = model_loader(model_lsds_path)

print()
print('model loaded')'''

training_steps = 5000
batch_size = 14
crop_size = 256

lr = 1e-4

'''
if we are testing things for the first time
'''
testing = [('raw','lsds'),
           ('raw','boundaries_lsds')]
###########################################

'''
later
'''
testing = [('lsds','boundaries'),
           ('raw_lsds','boundaries'),
           ('raw','boundaries')]
###########################################

'''for k in testing:
    input_type, output_type = k

    if input_type == 'lsds' or input_type == 'raw_lsds':
        for model in (None, model_lsd):
            model_training(input_dataset,
                        segmentation_dataset,
                        input_type,
                        output_type,
                        training_steps = training_steps,
                        batch_size = batch_size,
                        crop_size = crop_size,
                        show_metrics = False,
                        model_lsds = model)
    else:
        model_training(input_dataset,
            segmentation_dataset,
            input_type,
            output_type,
            training_steps = training_steps,
            batch_size = batch_size,
            crop_size = crop_size,
            show_metrics = False)'''



'''input_type, output_type = 'lsds', 'affinities'

sigmas = [5, 10, 15, 20, 30]

for sigma in sigmas:
    model_name = '/home/enrico.negri/github/lsd_testing/output/different_sigmas/step10000_b14_c256_lr0.0001_s' + str(sigma) +'/raw-lsds.pth'
    model_lsd = utils_2D.model_loader(model_name)
    print()
    print(f'model s_{str(sigma)} loaded')
    model_training(input_dataset,
                segmentation_dataset,
                input_type,
                output_type,
                training_steps = training_steps,
                batch_size = batch_size,
                crop_size = crop_size,
                show_metrics = False,
                lsd_sigma = sigma,
                model_lsds=model_lsd,
                output_folder = 'lsd-bound_sigmas/test')'''

'''training_steps = 10000
batch_size = 14
crop_size = 256
lr = 1e-4

input_type, output_type = 'boundaries_d', 'boundaries'

model_training(input_dataset,
                segmentation_dataset,
                input_type,
                output_type,
                training_steps = training_steps,
                batch_size = batch_size,
                crop_size = crop_size,
                show_metrics = False,
                lsd_sigma = 15,
                output_folder = 'boundary_reconstruction',)'''



training_steps = 20000
batch_size = 14
crop_size = 256
lr = 1e-5

# simple raw -> lsds model

input_type, output_type = 'raw', 'lsds'
model_training(input_dataset,
                segmentation_dataset,
                input_type,
                output_type,
                training_steps = training_steps,
                learning_rate = lr,
                batch_size = batch_size,
                crop_size = crop_size,
                show_metrics = False,
                lsd_sigma = 15,
                output_folder = 'new_models',)


# predicted lsd -> boundaries

'''model_name = '/home/enrico.negri/github/lsd_testing/output/new_models/step5000_b14_c256_lr0.0001_s15/raw-lsds.pth'
model_lsd = utils_2D.model_loader(model_name)

input_type, output_type = 'lsds', 'boundaries'
model_training(input_dataset,
            segmentation_dataset,
            input_type,
            output_type,
            training_steps = training_steps,
            batch_size = batch_size,
            crop_size = crop_size,
            show_metrics = False,
            lsd_sigma = 15,
            model_lsds=model_lsd,
            output_folder = 'new_models/bce')'''