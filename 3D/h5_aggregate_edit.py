import numpy as np, h5py

file_path = '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/3_id_aggregate_xrt_orig.h5'

output_file_path = '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/3_id_aggregate_xrt.h5'

# with h5py.File(file_path, 'r+') as f:
#     theta = f['exchange/theta']

#     theta[35] = -69
#     theta[53] = -15

# with h5py.File(file_path, 'r') as f:
#     theta = f['exchange/theta'][()]
#     filenames = f['filenames'][()]
#     data = f['exchange/data'][()]
#     elements = f['exchange/elements'][()]

# theta_idx_sorted = np.argsort(theta)

# keep = (theta_idx_sorted != 35) & (theta_idx_sorted != 53)
# theta_sorted = theta[theta_idx_sorted][keep]
# data_sorted = data[:, theta_idx_sorted][:, keep]

# filenames_sorted = [filenames[i] for i in theta_idx_sorted if i != 35 and i != 53]

# print(filenames_sorted)

# with h5py.File(output_file_path, 'w') as f:
#     exchange = f.create_group('exchange')
    
#     intensity = exchange.create_dataset('data', data = data_sorted)
#     el = exchange.create_dataset('elements', data = elements)
#     angle = exchange.create_dataset('theta', data = theta_sorted)
    
#     fnames = f.create_dataset('filenames', data = filenames_sorted)

#     intensity.attrs['dataset_type'] = 'xrt'
#     intensity.attrs['us_ic_scaler_name'] = 'sclr1_ch4'
#     intensity.attrs['xrt_signal_name'] = 'stxm'

with h5py.File(output_file_path, 'r') as f:
    theta = f['exchange/theta'][()]
    intensity = f['exchange/data'][()]

print(np.where(theta == 0)[0][1])

intensity[:, np.where(theta == 0)[0][1]:] = \
    np.flip(intensity[:, np.where(theta == 0)[0][1]:], axis = 2)
    