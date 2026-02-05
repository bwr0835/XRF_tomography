import numpy as np, h5py

file_path = '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/3_id_aggregate_xrt_orig.h5'

with h5py.File(file_path, 'r+') as f:
    theta = f['exchange/theta']

    theta[35] = -69
    theta[53] = -15