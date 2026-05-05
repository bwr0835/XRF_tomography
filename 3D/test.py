import numpy as np, h5py, pandas as pd, pandas as pd, xrf_xrt_preprocess_utils as ppu, sys, os, xrf_xrt_preprocess_file_util as futil, ast
import xraylib_np as xrl_np, xraylib as xrl
# import  xrf_xrt_jxrft_file_util as futil_jxrft
from matplotlib import pyplot as plt
from itertools import combinations as combos
from scipy import ndimage as ndi
from imageio import v2 as iio2
from mpl_toolkits.axes_grid1 import make_axes_locatable

pre_existing_align_norm_file_path = '/home/bwr0835/3_id_realigned_data_04_19_2026_diff_cor_correction/xrt_od_xrf_realignment_003/raw_input_data.csv'

norm_array_xrt, \
norm_array_xrf, \
init_x_shift_array, \
init_y_shift_array, \
pixel_rad_adjacent_angle_jitter, \
pixel_rad_cor_correction, \
pixel_rad_iter_reproj, \
I0_photons, \
data_percentile_aux, \
aligning_element_aux = futil.extract_csv_raw_input_data(pre_existing_align_norm_file_path)

fig1, axs1 = plt.subplots(1, 2)