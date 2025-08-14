import numpy as np

from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

dx_file = '/Users/bwr0835/Documents/xrt_gridrec_6_iter_dynamic_ps_cor_correction_log_w_padding_gridrec_cor_idx_300_skimage_radon_aug_14_2025/dx_array_iter_ds_ic.npy'
theta_file = '/Users/bwr0835/Documents/xrt_gridrec_6_iter_dynamic_ps_cor_correction_log_w_padding_gridrec_cor_idx_300_skimage_radon_aug_14_2025/theta_array.npy'

dx_iter_array = np.load(dx_file)
theta_array = np.load(theta_file)

iter_idx_desired = 0

iteration_idx_array = np.arange(dx_iter_array.shape[0])

fig1, axs1 = plt.subplots()

curve1, = axs1.plot(theta_array, dx_iter_array[iter_idx_desired], 'k-o', markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[iter_idx_desired]))
curve2, = axs1.plot(theta_array, dx_iter_array[iter_idx_desired + 3], 'r-o', markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[iter_idx_desired + 3]))
curve3, = axs1.plot(theta_array, dx_iter_array[-1], 'b-o', markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[-1]))

axs1.tick_params(axis = 'both', which = 'major', labelsize = 14)
axs1.tick_params(axis = 'both', which = 'minor', labelsize = 14)
# axs1.set_title(r'Iteration index {0}'.format(iter_idx_desired), fontsize = 18)
axs1.set_xlabel(r'$\theta$ (\textdegree{})', fontsize = 16)
axs1.set_ylabel(r'$\delta x$', fontsize = 16)
axs1.legend(frameon = False, fontsize = 14)

fig1.tight_layout()

plt.show()