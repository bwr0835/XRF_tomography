import numpy as np, h5_util as util, tkinter as tk

from matplotlib import pyplot as plt
from tkinter import filedialog as fd
from copy import copy

# file_path_np = '/Users/bwr0835/Downloads/recon_235327_1k_mode_object.npy'

# num = np.load(file_path_np)

# plt.imshow(np.log10(np.abs(num[0])))
# plt.show()
# file_path_array_xrf = ['/raid/users/roter/Jacobsen/img.dat/2xfm_0116.mda.h5',
#                        '/raid/users/roter/Jacobsen/img.dat/2xfm_0117.mda.h5']

# root = tk.Tk()

# root.withdraw()

file_path_array_xrt = ['/raid/users/roter/Jacobsen/img.dat/2xfm_0030.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0031.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0032.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0033.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0034.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0037.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0038.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0039.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0040.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0042.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0047.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0048.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0049.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0050.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0051.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0052.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0053.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0054.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0055.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0056.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0057.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0058.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0059.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0060.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0094.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0095.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0096.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0097.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0098.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0099.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0100.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0101.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0102.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0103.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0104.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0105.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0106.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0107.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0108.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0109.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0110.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0111.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0112.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0113.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0114.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0115.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0116.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0117.mda.h5',
                       '/raid/users/roter/Jacobsen/img.dat/2xfm_0118.mda.h5']
# file_path_array_xrf = list(fd.askopenfilenames(parent = root, title = 'Select XRF HDF5 files', filetypes = [('HDF5 Files', '*.h5')]))
# file_path_array_xrt = list(fd.askopenfilenames(parent = root, title = 'Select XRT HDF5 files', filetypes = [('HDF5 Files', '*.h5')]))
# file_path_array_xrt = copy(file_path_array_xrf)

# output_file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
output_file_path_xrt = '/home/bwr0835/2_ide_aggregate_xrt.h5'

# file_path = '/Users/bwr0835/Documents/GitHub/gradresearch/xrt/test_combined_file.h5'

# util.create_aggregate_xrf_h5(file_path_array_xrf, output_file_path_xrf, 'APS')
util.create_aggregate_xrt_h5(file_path_array_xrt, output_file_path_xrt, 'APS')

# elements, counts, theta, dataset_type = extract_h5_aggregate_xrf_data(file_path)


# print(theta)