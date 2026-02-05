import numpy as np, h5py

file_path = '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/3_id_aggregate_xrt_orig.h5'

output_file_path = '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/3_id_aggregate_xrt.h5'

with h5py.File(file_path, 'r+') as f:
    theta = f['exchange/theta']

    theta[35] = -72
    theta[53] = -18

# with h5py.File(file_path, 'r') as f:
#     theta = f['exchange/theta'][()]
#     filenames = f['filenames'][()]
#     data = f['exchange/data'][()]
#     elements = f['exchange/elements'][()]

# idx = np.r_[:35, 36:53, 54:]

# theta_idx_sorted = np.argsort(theta)

# theta_sorted = theta[theta_idx_sorted]
# filenames_sorted = filenames[theta_idx_sorted]
# data_sorted = data[:, theta_idx_sorted]

# print(elements)

# with h5py.File(output_file_path, 'w') as f:
#     exchange = f.create_group('exchange')
    
#     intensity = exchange.create_dataset('data', data = data_sorted)
#     el = exchange.create_dataset('elements', data = elements)
#     angle = exchange.create_dataset('theta', data = theta_sorted)
    
#     fnames = f.create_dataset('filenames', data = filenames_sorted)

#     intensity.attrs['dataset_type'] = 'xrt'
#     intensity.attrs['us_ic_scaler_name'] = 'sclr1_ch4'
#     intensity.attrs['xrt_signal_name'] = 'stxm'
    