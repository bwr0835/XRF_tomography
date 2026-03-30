import numpy as np, os, sys, xrf_xrt_preprocess_file_util as futil, xrf_xrt_preprocess_utils as ppu

from matplotlib import pyplot as plt
from scipy import ndimage as ndi

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def normalize_array(array):
    return (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))

def create_cor_fig(init_proj, shifted_proj, theta_pair, vmin, vmax):
    fig, axs = plt.subplots(2, 3)

    init_proj_theta_0 = init_proj[theta_pair[0]]
    init_proj_theta_1 = init_proj[theta_pair[1]]
    shifted_proj_theta_0 = shifted_proj[theta_pair[0]]
    shifted_proj_theta_1 = shifted_proj[theta_pair[1]]

    init_proj_theta_0_norm = normalize_array(init_proj_theta_0)
    init_proj_theta_1_norm = normalize_array(init_proj_theta_1)
    shifted_proj_theta_0_norm = normalize_array(shifted_proj_theta_0)
    shifted_proj_theta_1_norm = normalize_array(shifted_proj_theta_1)

    

    im1_1 = axs[0, 0].imshow(init_proj_theta_0_norm, vmin = vmin, vmax = vmax)
    im1_2 = axs[1, 0].imshow(init_proj_theta_1_norm, vmin = vmin, vmax = vmax)
    im1_3 = axs[2, 0].imshow(shifted_proj_theta_0_norm, vmin = vmin, vmax = vmax)
    im1_4 = axs[0, 1].imshow(shifted_proj_theta_1_norm, vmin = vmin, vmax = vmax)

    



dir_path = '/Users/bwr0835/Documents'

init_aps_xrt_file_path = f'{dir_path}/2_ide_aggregate_xrt.h5'
init_hxn_xrt_file_path = f'{dir_path}/3_id_aggregate_xrt.h5'

init_aps_xrf_file_path = f'{dir_path}/2_ide_aggregate_xrf.h5'
init_hxn_xrf_file_path = f'{dir_path}/3_id_aggregate_xrf.h5'

elements_xrt_aps, intensity_xrt_aps, theta_xrt_aps, _, _, _, _ = futil.extract_h5_aggregate_xrt_data(init_aps_xrt_file_path)
elements_xrt_hxn, intensity_xrt_hxn, theta_xrt_hxn, _, _, _, _ = futil.extract_h5_aggregate_xrt_data(init_hxn_xrt_file_path)

elements_xrf_aps, intensity_xrf_aps, theta_xrf_aps, _, _, _, _ = futil.extract_h5_aggregate_xrf_data(init_aps_xrf_file_path)
elements_xrf_hxn, intensity_xrf_hxn, theta_xrf_hxn, _, _, _, _ = futil.extract_h5_aggregate_xrf_data(init_hxn_xrf_file_path)

xrt_sig_aps = intensity_xrt_aps[elements_xrt_aps.index('xrt_sig')]

aligning_element_aps = 'xrt_sig'
aligning_element_hxn = 'Ni'

aligning_element_idx_aps = elements_xrt_aps.index(aligning_element_aps)
aligning_element_idx_hxn = elements_xrf_hxn.index(aligning_element_hxn)

intensity_xrt_norm_aps, intensity_xrf_norm_aps, norm_array_xrt_aps, norm_array_xrf_aps, I0_photons_aps = ppu.joint_fluct_norm(xrt_sig_aps,
                                                                                                                             intensity_xrf_aps,
                                                                                                                             93,
                                                                                                                             False,
                                                                                                                             8.6776e8,
                                                                                                                             0.01)
theta_pair_aps = (np.where(theta_xrt_aps == -122)[0][0], np.where(theta_xrt_aps == 58)[0][0])

opt_dens_aps = -np.log(intensity_xrt_norm_aps/I0_photons_aps)

cor_shift_aps = -6.68587804108239

shifted_opt_dens_aps = ndi.shift(opt_dens_aps, shift = (0, 0, cor_shift_aps))

vmin_od = np.nanmin(opt_dens_aps, shifted_opt_dens_aps)
vmax_od = np.nanmax(opt_dens_aps, shifted_opt_dens_aps)



