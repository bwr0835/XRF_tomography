import numpy as np, h5py

file_path = '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/3_id_aggregate_xrt_orig.h5'

with h5py.File(file_path, 'r') as f:
    theta = f['exchange/theta'][()]
    filenames = f['filenames'][()]
    data = f['exchange/data'][()]
    elements = f['exchange/elements'][()]

idx = np.r_[:35, 36:53, 54:]

theta = theta[idx]
filenames = filenames[idx]
data = data[idx]
elements = elements[idx]

# with h5py.File(file_path, 'w') as f:
#     f.create_dataset('exchange/theta', data = theta)
#     f.create_dataset('filenames', data = filenames)
#     f.create_dataset('exchange/data', data = data)
#     f.create_dataset('exchange/elements', data = elements)