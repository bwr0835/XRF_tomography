import numpy as np, \
       h5py, \
       os

dir_path = '/home/bwr0835'

output_path_xrf = os.path.join(dir_path, 'simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64.h5')
output_path_xrt = os.path.join(dir_path, 'simulated_proj_data_xrt_64_64_64.h5')

proj_data_xrf = np.zeros((4, 200, 64, 64))
proj_data_xrt = np.zeros((1, 200, 64, 64))

proj_data_xrt[0] = np.load(os.path.join(dir_path, 'simulated_proj_data_xrt_64_64_64.npy')).reshape(200, 64, 64)

theta = np.linspace(0, 360, 200)

elements_xrf = ['Ca', 'Ca_L', 'Sc', 'Sc_L']
elements_xrt = ['xrt_sig']

for theta_idx in range(200):
    file_path = f'{dir_path}/simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64_{theta_idx}.npy'

    proj_data_xrf[:, theta_idx] = np.load(file_path).reshape(4, 64, 64)

with h5py.File(output_path_xrt, 'w') as f:
    exchange = f.create_group('exchange')

    exchange.create_dataset('data', data = proj_data_xrt)
    exchange.create_dataset('elements', data = elements_xrt)
    exchange.create_dataset('theta', data = theta)

for theta_idx in range(200):
    file_path = f'{dir_path}/simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64_{theta_idx:03d}.npy'

    os.remove(file_path)

with h5py.File(output_path_xrf, 'w') as f:
    exchange = f.create_group('exchange')

    exchange.create_dataset('data', data = proj_data_xrf)
    exchange.create_dataset('elements', data = elements_xrf)
    exchange.create_dataset('theta', data = theta)