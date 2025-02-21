import numpy as np

from h5_util import extract_h5_aggregate_xrf_data, create_aggregate_xrf_h5
# from h5_util import create_single_xrf_h5, extract_h5_xrf_data

from matplotlib import pyplot as plt

# file_path_np = '/Users/bwr0835/Downloads/recon_235327_1k_mode_object.npy'

# num = np.load(file_path_np)

# plt.imshow(np.log10(np.abs(num[0])))
# plt.show()
file_path_array = ['/Users/bwr0835/Documents/GitHub/gradresearch/xrt/2xfm_0029.mda.h5',
                   '/Users/bwr0835/Documents/GitHub/gradresearch/xrt/2xfm_0117.mda.h5']

file_path = '/Users/bwr0835/Documents/GitHub/gradresearch/xrt/test_combined_file.h5'

create_aggregate_xrf_h5(file_path_array, file_path, 'APS')

elements, counts, theta, dataset_type = extract_h5_aggregate_xrf_data(file_path)

print(dataset_type)

# print(theta)