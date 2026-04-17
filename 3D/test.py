import numpy as np, h5py, pandas as pd, pandas as pd, xrf_xrt_preprocess_utils as ppu, sys, os, xrf_xrt_preprocess_file_util as futil
# import xrf_xrt_preprocess_file_util as futil_pp, xraylib_np as xrl_np, xraylib as xrl
# import  xrf_xrt_jxrft_file_util as futil_jxrft
from matplotlib import pyplot as plt
from itertools import combinations as combos
from scipy import ndimage as ndi
from imageio import v2 as iio2
from mpl_toolkits.axes_grid1 import make_axes_locatable

a = np.random.rand(50, 50)
plt.imshow(ppu.edge_gauss_filter(a, sigma = 10, alpha = 10, nx = 50, ny = 50))
plt.show()