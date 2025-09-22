import numpy as np, h5py, h5_util as util

from matplotlib import pyplot as plt

coarse_scan_filename = '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235675.h5'
fine_scan_filename = '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235676.h5'

elements, counts_coarse, _, x_um_coarse, y_um_coarse, nx_coarse, ny_coarse, dx_cm_coarse, dy_cm_coarse = util.extract_h5_xrf_data(coarse_scan_filename, synchrotron = 'nsls-ii', scan_coords = True)
_, counts_fine, _, x_um_fine, y_um_fine, nx_fine, ny_fine, dx_cm_fine, dy_cm_fine = util.extract_h5_xrf_data(coarse_scan_filename, synchrotron = 'nsls-ii', scan_coords = True)

desired_element = 'Fe_K'
desired_element_indx = elements.index(desired_element)
