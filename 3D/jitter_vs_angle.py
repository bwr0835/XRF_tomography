import numpy as np

from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

data_file = '/Users/bwr0835/Documents/xrt_gridrec_6_iter_dynamic_ps_cor_correction_log_w_padding_gridrec_cor_idx_300_skimage_radon_aug_14_2025/net_x_shifts_ds_ic.npy'
theta_file = '/Users/bwr0835/Documents/xrt_gridrec_6_iter_dynamic_ps_cor_correction_log_w_padding_gridrec_cor_idx_300_skimage_radon_aug_14_2025/theta_array.npy'

data = np.load(data_file)
theta_array = np.load(theta_file)

fig1, axs1 = plt.subplots()

curve1, = axs1.plot(theta_array, data[0], '-o', markersize = 3, linewidth = 2)

axs1.set_xlabel(r'$\theta$ (\textdegree{})')
axs1.set_ylabel(r'$')