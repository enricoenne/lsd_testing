import matplotlib.pyplot as plt
import numpy as np


import albumentations as A
import torch
import skimage


from celldataset_module import CellDataset_single, CellDataset3D_single
from waterz import agglomerate

from scipy import ndimage


input_dataset = ' - C=1'
segmentation_dataset = '_CELLS'

train_dataset = CellDataset3D_single(
    image_dir='/group/jug/Enrico/TISSUE_roi/training/*' + input_dataset + '.tif',
    mask_dir='/group/jug/Enrico/TISSUE_roi/training/*' + segmentation_dataset + '.tif',
    crop_size=512,
    input_type='raw', output_type='affinities',
    split='train')




test_iter = iter(train_dataset)

raw, affinities = next(test_iter)
# affinities is a   [3,depth,height,width] numpy array of float32


binary_mask = affinities.mean(axis=0) > 0.5  # threshold affinities mean for seed mask
fragments, _ = ndimage.label(binary_mask)

fragments_copy = np.array(fragments, dtype=np.uint64)
print("fragments dtype:", fragments.dtype)
print("fragments_copy dtype:", fragments_copy.dtype)


thresholds = [0.5]

for segmentation in agglomerate(affinities, thresholds, fragments=fragments_copy):
    test = segmentation



plt.imshow(np.amax(test, axis=0), cmap='gray')
plt.show()
