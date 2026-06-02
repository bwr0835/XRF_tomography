import numpy as np, h5py, pandas as pd, pandas as pd, xrf_xrt_preprocess_utils as ppu, sys, os, xrf_xrt_preprocess_file_util as futil, ast
import xraylib_np as xrl_np, xraylib as xrl
# import  xrf_xrt_jxrft_file_util as futil_jxrft
from matplotlib import pyplot as plt
from itertools import combinations as combos
from scipy import ndimage as ndi
from imageio import v2 as iio2
from mpl_toolkits.axes_grid1 import make_axes_locatable

filename = '/Users/bwr0835/Documents/2_ide_aggregate_xrt.h5'

with h5py.File(filename, 'r+') as f:
    f['exchange/data'].attrs['incident_energy_keV'] = 13.0