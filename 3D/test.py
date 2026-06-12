import numpy as np, h5py, pandas as pd, pandas as pd, sys, os, ast
import xraylib_np as xrl_np, xraylib as xrl
# import  xrf_xrt_jxrft_file_util as futil_jxrft
from matplotlib import pyplot as plt
from itertools import combinations as combos
from scipy import ndimage as ndi
from imageio import v2 as iio2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure as meas

def downsample(array, downsample_factor_1, downsample_factor_2 = None, func = np.mean):
    if downsample_factor_2 is None:
        downsample_factor_2 = downsample_factor_1
    
    if downsample_factor_1 == 1 and downsample_factor_2 == 1:
        return array

    return meas.block_reduce(array, block_size = (downsample_factor_1, downsample_factor_2), func = func)

filename = '/Users/bwr0835/Documents/2_ide_realigned_data_03_27_2026_iter_reproj_cor_correction_only_final/xrt_od_xrf_realignment/recon_downsample_1_Fe_mlem.h5'

with h5py.File(filename, 'r') as f:
    xrf_data = f['MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec'][()]

xrf_data = xrf_data[0, :, :-2]

n_columns_orig = xrf_data.shape[1]

print(xrf_data.shape)

downsample_factors = [1, 2, 5, 10]

xrf_data_downsampled = np.zeros((len(downsample_factors), n_columns_orig, n_columns_orig))
    
for idx, downsample_factor in enumerate(downsample_factors):
    n_columns = n_columns_orig//downsample_factor
    print(xrf_data.shape)
    xrf_data_downsampled[idx, :n_columns, :n_columns] = downsample(xrf_data, downsample_factor)

fig, axs = plt.subplots(2, 2)

for idx, ax in enumerate(axs.flat):
    ax.imshow(xrf_data_downsampled[idx])
    ax.axis('off')
    ax.set_title(f'DS = {downsample_factors[idx]}')

fig.tight_layout()

plt.show()