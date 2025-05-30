import numpy as np, h5_util as util, tkinter as tk

from matplotlib import pyplot as plt
from tkinter import filedialog as fd

# file_path_np = '/Users/bwr0835/Downloads/recon_235327_1k_mode_object.npy'

# num = np.load(file_path_np)

# plt.imshow(np.log10(np.abs(num[0])))
# plt.show()
# file_path_array = ['/Users/bwr0835/Documents/GitHub/gradresearch/xrt/2xfm_0029.mda.h5',
                #    '/Users/bwr0835/Documents/GitHub/gradresearch/xrt/2xfm_0117.mda.h5']

root = tk.Tk()

root.withdraw()

file_path_array_xrf = fd.askopenfilenames(parent = root, title = 'Select XRF HDF5 files', filetypes = (('HDF5 Files', '*.h5')))
file_path_array_xrt = fd.askopenfilenames(parent = root, title = 'Select XRT HDF5 files', filetypes = (('HDF5 Files', '*.h5')))

output_file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
output_file_path_xrt = '/home/bwr0835/2_ide_aggregate_xrt.h5'

# file_path = '/Users/bwr0835/Documents/GitHub/gradresearch/xrt/test_combined_file.h5'

util.create_aggregate_xrf_h5(file_path_array_xrf, output_file_path_xrt, 'APS')
util.create_aggregate_xrt_h5(file_path_array_xrt, output_file_path_xrt, 'APS')

# elements, counts, theta, dataset_type = extract_h5_aggregate_xrf_data(file_path)


# print(theta)