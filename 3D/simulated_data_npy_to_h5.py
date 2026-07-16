import numpy as np, \
       h5py, \
       os

from imageio import v2 as iio2
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

dir_path = '/home/bwr0835'

# output_path_xrf = os.path.join(dir_path, 'simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64.h5')
output_path_xrt = os.path.join(dir_path, 'simulated_proj_data_xrt_64_64_64.h5')

# proj_data_xrf = np.zeros((4, 200, 64, 64))
proj_data_xrt = np.zeros((1, 400, 64, 64))

proj_data_xrt[0] = np.load(os.path.join(dir_path, 'simulated_proj_data_xrt_64_64_64.npy')).reshape(400, 64, 64)

theta = np.linspace(0, 360, 400)

# elements_xrf = ['Ca', 'Ca_L', 'Sc', 'Sc_L']
elements_xrt = ['xrt_sig']

# for theta_idx in range(200):
    # file_path = f'{dir_path}/simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64_{theta_idx}.npy'

    # proj_data_xrf[:, theta_idx] = np.load(file_path).reshape(4, 64, 64)

with h5py.File(output_path_xrt, 'w') as f:
    exchange = f.create_group('exchange')

    exchange.create_dataset('data', data = proj_data_xrt)
    exchange.create_dataset('elements', data = elements_xrt)
    exchange.create_dataset('theta', data = theta)

fig1, axs1 = plt.subplots()

axs1.imshow(proj_data_xrt[0, :, 30], cmap = 'jet', aspect = 'auto', extent = [-0.5, 63.5, -180.45, 179.55])
axs1.tick_params(axis = 'both', which = 'major', labelsize = 14)
axs1.set_title(r'XRT')
axs1.set_ylabel(r'$\theta$ (degrees)')
axs1.set_xlabel(r'Scan position index')

plt.show()

# for theta_idx in range(200):
#     file_path = f'{dir_path}/simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64_{theta_idx}.npy'

#     os.remove(file_path)

# with h5py.File(output_path_xrf, 'w') as f:
#     exchange = f.create_group('exchange')

#     exchange.create_dataset('data', data = proj_data_xrf)
#     exchange.create_dataset('elements', data = elements_xrf)
#     exchange.create_dataset('theta', data = theta)