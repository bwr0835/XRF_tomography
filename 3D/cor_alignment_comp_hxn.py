import numpy as np, os, sys, xrf_xrt_preprocess_file_util as futil, xrf_xrt_preprocess_utils as ppu, realignment_final as realign

from matplotlib import pyplot as plt
from scipy import ndimage as ndi, fft

import h5py

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def load_h5_data(file_path, get_edge_info = False):
    with h5py.File(file_path, 'r') as f:
        data = f['exchange/data/xrf'][()]

        if get_edge_info:
            top_edge_cropped_init = f['exchange/data'].attrs['top_edge_cropped_init']
            bottom_edge_cropped_init = f['exchange/data'].attrs['bottom_edge_cropped_init']

            edge_info = {'top': top_edge_cropped_init, 'bottom': bottom_edge_cropped_init}
    
            return data, edge_info
    
    return data

def rot_center(theta_sum):
    """
    Code written by E. Vacek (2021): 
    https://github.com/everettvacek/PhaseSymmetry/blob/master/PhaseSymmetry.py

    Calculates the center of rotation of a sinogram.

    Parameters
    ----------
    thetasum: array-like
        The 2D theta-sum array (z, t).

    Returns
    -------
    COR: float
        The center of rotation.
    """
    
    Nz = theta_sum.shape[0] # Number of slices
    Nt = theta_sum.shape[1] # Number of scan positions

    T = fft.rfft(theta_sum.ravel()) # Real FFT (no negative frequencies) of flattened 2D array of length Nt*Nz ('C'/row-major order)

    # Get real, imaginary components of the first AC spatial frequency for axis perpendicular to rotation axis.
    # Nt is the spatial period (there are Nt columns per row); Nz is the (fundamental) spatial frequency (thus, the first AC frequency)

    real, imag = T[Nz].real, T[Nz].imag

    # Get phase of thetasum and return center of rotation.
    
    # In a sinogram the feature may be more positive or less positive than the background (i.e. fluorescence vs
    # absorption contrast). This can mess with the T_phase value so we multiply by the sign of the even function
    # to account for this. (Comment from F. Marin's XRFTomo code)

    phase = np.arctan2(imag*np.sign(real), real*np.sign(real)) 
    
    COR = Nt//2 - Nt*phase/(2*np.pi)

    return COR

def rot_center_avg(proj_img_array, theta_pair_array, theta_array):
    n_columns = proj_img_array.shape[2]
   
    center_of_rotation_sum = 0
    
    for theta_pair in theta_pair_array:
        theta_sum = proj_img_array[theta_pair[0]] + proj_img_array[theta_pair[1]]

        center_of_rotation = rot_center(theta_sum)

        print(f'Center of rotation ({theta_array[theta_pair[0]]} degrees, {theta_array[theta_pair[1]]} degrees) = {ppu.round_correct(center_of_rotation, ndec = 3)}')

        center_of_rotation_sum += center_of_rotation
    
    center_rotation_avg = center_of_rotation_sum/len(theta_pair_array)

    geom_center = n_columns//2

    offset = center_rotation_avg - geom_center

    return center_rotation_avg, geom_center, offset

def normalize_array(array):
    return (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))

def create_cor_fig_hxn(init_proj, shifted_proj, theta_array, aligning_element):
    fig, axs = plt.subplots(2, 3)
    
    print(init_proj.shape, shifted_proj.shape)

    init_proj_theta_0 = init_proj[0]
    init_proj_theta_1 = np.fliplr(init_proj[-1])
    shifted_proj_theta_0 = shifted_proj[0]
    shifted_proj_theta_1 = np.fliplr(shifted_proj[-1])

    print(init_proj_theta_0.shape, init_proj_theta_1.shape, shifted_proj_theta_0.shape, shifted_proj_theta_1.shape)

    init_proj_theta_0_norm = normalize_array(init_proj_theta_0)
    init_proj_theta_1_norm = normalize_array(init_proj_theta_1)
    shifted_proj_theta_0_norm = normalize_array(shifted_proj_theta_0)
    shifted_proj_theta_1_norm = normalize_array(shifted_proj_theta_1)

    init_proj_theta_0_rgb = np.dstack((init_proj_theta_0_norm, np.zeros_like(init_proj_theta_0_norm), np.zeros_like(init_proj_theta_0_norm)))
    init_proj_theta_1_rgb = np.dstack((np.zeros_like(init_proj_theta_1_norm), init_proj_theta_1_norm, np.zeros_like(init_proj_theta_1_norm)))
    shifted_proj_theta_0_rgb = np.dstack((shifted_proj_theta_0_norm, np.zeros_like(shifted_proj_theta_0_norm), np.zeros_like(shifted_proj_theta_0_norm)))
    shifted_proj_theta_1_rgb = np.dstack((np.zeros_like(shifted_proj_theta_1_norm), shifted_proj_theta_1_norm, np.zeros_like(shifted_proj_theta_1_norm)))

    overlay_init = np.dstack((init_proj_theta_0_norm, init_proj_theta_1_norm, np.zeros_like(init_proj_theta_0_norm)))
    overlay_shifted = np.dstack((shifted_proj_theta_0_norm, shifted_proj_theta_1_norm, np.zeros_like(shifted_proj_theta_0_norm)))

    im1_1 = axs[0, 0].imshow(init_proj_theta_0_rgb, interpolation = 'none')
    im1_2 = axs[0, 1].imshow(init_proj_theta_1_rgb, interpolation = 'none')
    im1_3 = axs[0, 2].imshow(overlay_init, interpolation = 'none')
    im1_4 = axs[1, 0].imshow(shifted_proj_theta_0_rgb, interpolation = 'none')
    im1_5 = axs[1, 1].imshow(shifted_proj_theta_1_rgb, interpolation = 'none')
    im1_6 = axs[1, 2].imshow(overlay_shifted, interpolation = 'none')

    for ax in fig.axes:
        ax.axis('off')
        ax.axvline(x = init_proj_theta_0.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
        ax.axhline(y = init_proj_theta_0.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')

    text_1 = axs[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs[0, 0].transAxes, color = 'white')
    text_2 = axs[0, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[-1]), transform = axs[0, 1].transAxes, color = 'white')
    text_3 = axs[1, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs[1, 0].transAxes, color = 'white')
    text_4 = axs[1, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[-1]), transform = axs[1, 1].transAxes, color = 'white')

    axs[0, 0].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    axs[0, 1].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    axs[0, 2].set_title(r'{0} (overlay)'.format(aligning_element), fontsize = 14)
    axs[1, 0].set_title(r'{0} (shifted)'.format(aligning_element), fontsize = 14)
    axs[1, 1].set_title(r'{0} (shifted)'.format(aligning_element), fontsize = 14)
    axs[1, 2].set_title(r'{0} (shifted overlay)'.format(aligning_element), fontsize = 14)

    fig.suptitle(r'Center of rotation correction (Phase cross-correlation)', fontsize = 16)
    fig.tight_layout()

    plt.show()

def create_cor_fig_hxn_offset(init_proj, shifted_proj, theta_array, aligning_element):
    fig, axs = plt.subplots(2, 3)
    
    print(init_proj.shape, shifted_proj.shape)
    
    zero_deg_idx_array = np.where(theta_array == 0)[0]
    
    init_proj_theta_0 = init_proj[zero_deg_idx_array[0]]
    init_proj_theta_1 = np.fliplr(init_proj[-1])
    shifted_proj_theta_0 = shifted_proj[zero_deg_idx_array[0]]
    shifted_proj_theta_1 = np.fliplr(shifted_proj[-1])

    print(init_proj_theta_0.shape, init_proj_theta_1.shape, shifted_proj_theta_0.shape, shifted_proj_theta_1.shape)

    init_proj_theta_0_norm = normalize_array(init_proj_theta_0)
    init_proj_theta_1_norm = normalize_array(init_proj_theta_1)
    shifted_proj_theta_0_norm = normalize_array(shifted_proj_theta_0)
    shifted_proj_theta_1_norm = normalize_array(shifted_proj_theta_1)

    init_proj_theta_0_rgb = np.dstack((init_proj_theta_0_norm, np.zeros_like(init_proj_theta_0_norm), np.zeros_like(init_proj_theta_0_norm)))
    init_proj_theta_1_rgb = np.dstack((np.zeros_like(init_proj_theta_1_norm), init_proj_theta_1_norm, np.zeros_like(init_proj_theta_1_norm)))
    shifted_proj_theta_0_rgb = np.dstack((shifted_proj_theta_0_norm, np.zeros_like(shifted_proj_theta_0_norm), np.zeros_like(shifted_proj_theta_0_norm)))
    shifted_proj_theta_1_rgb = np.dstack((np.zeros_like(shifted_proj_theta_1_norm), shifted_proj_theta_1_norm, np.zeros_like(shifted_proj_theta_1_norm)))

    overlay_init = np.dstack((init_proj_theta_0_norm, init_proj_theta_1_norm, np.zeros_like(init_proj_theta_0_norm)))
    overlay_shifted = np.dstack((shifted_proj_theta_0_norm, shifted_proj_theta_1_norm, np.zeros_like(shifted_proj_theta_0_norm)))

    im1_1 = axs[0, 0].imshow(init_proj_theta_0_rgb, aspect = 'equal', interpolation = 'none')
    im1_2 = axs[0, 1].imshow(init_proj_theta_1_rgb, aspect = 'equal', interpolation = 'none')
    im1_3 = axs[0, 2].imshow(overlay_init, aspect = 'equal', interpolation = 'none')
    im1_4 = axs[1, 0].imshow(shifted_proj_theta_0_rgb, aspect = 'equal', interpolation = 'none')
    im1_5 = axs[1, 1].imshow(shifted_proj_theta_1_rgb, aspect = 'equal', interpolation = 'none')
    im1_6 = axs[1, 2].imshow(overlay_shifted, aspect = 'equal', interpolation = 'none')

    for ax in fig.axes:
        ax.axis('off')
        ax.axvline(x = init_proj_theta_0.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
        ax.axhline(y = init_proj_theta_0.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')

    text_1 = axs[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[zero_deg_idx_array[0]]), transform = axs[0, 0].transAxes, color = 'white')
    text_2 = axs[0, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[-1]), transform = axs[0, 1].transAxes, color = 'white')
    text_3 = axs[1, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[zero_deg_idx_array[0]]), transform = axs[1, 0].transAxes, color = 'white')
    text_4 = axs[1, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[-1]), transform = axs[1, 1].transAxes, color = 'white')

    axs[0, 0].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    axs[0, 1].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    axs[0, 2].set_title(r'{0} (overlay)'.format(aligning_element), fontsize = 14)
    axs[1, 0].set_title(r'{0} (shifted)'.format(aligning_element), fontsize = 14)
    axs[1, 1].set_title(r'{0} (shifted)'.format(aligning_element), fontsize = 14)
    axs[1, 2].set_title(r'{0} (shifted overlay)'.format(aligning_element), fontsize = 14)

    fig.suptitle(r'Center of rotation correction (Phase cross-correlation)', fontsize = 16)
    fig.tight_layout()

    plt.show()

dir_path = '/Users/bwr0835/Documents'
final_dir_path = f'{dir_path}/3_id_realigned_data_common_fov_cor_correction_only_03_30_2026_final'

init_hxn_xrt_file_path = f'{dir_path}/3_id_aggregate_xrt.h5'
init_hxn_xrf_file_path = f'{dir_path}/3_id_aggregate_xrf.h5'

elements_xrt_hxn, intensity_xrt_hxn, theta_xrt_hxn, _, _, _, _ = futil.extract_h5_aggregate_xrt_data(init_hxn_xrt_file_path)
elements_xrf_hxn, intensity_xrf_hxn, theta_xrf_hxn, _, _, _ = futil.extract_h5_aggregate_xrf_data(init_hxn_xrf_file_path)

input_csv_file_path = f'{final_dir_path}/xrt_od_xrf_realignment/raw_input_data.csv'
input_h5_file_path = f'{final_dir_path}/xrt_od_xrf_realignment/aligned_data/aligned_aggregate_xrf_xrt.h5'

# aligning_element_aps = 'opt_dens'
aligning_element_hxn = 'Ni'

xrt_sig_hxn = intensity_xrt_hxn[elements_xrt_hxn.index('xrt_sig')]

# aligning_element_idx_aps = elements_xrt_aps.index(aligning_element_aps)
# aligning_element_idx_hxn = elements_xrf_hxn.index(aligning_element_hxn)
_, \
_, \
init_x_shift_array, \
init_y_shift_array, \
_, \
_, \
_, \
_, \
_, \
_ = futil.extract_csv_raw_input_data(input_csv_file_path)

intensity_xrt_norm_hxn, intensity_xrf_norm_hxn, _, _, I0_photons_hxn, _ = ppu.joint_fluct_norm(xrt_sig_hxn,
                                                                                               intensity_xrf_hxn,
                                                                                               93,
                                                                                               True,
                                                                                               None,
                                                                                               None)
                                                                                            
element_to_align_with = 'Ni'
element_to_align_with_idx = elements_xrf_hxn.index(aligning_element_hxn)

aligned_proj_total_xrf, edge_info = load_h5_data(input_h5_file_path, get_edge_info = True)

start_slice, end_slice = edge_info['top'], edge_info['bottom']

init_proj = intensity_xrf_norm_hxn[element_to_align_with_idx]
aligned_proj_total_xrf = aligned_proj_total_xrf[element_to_align_with_idx]

for theta_idx in range(1, len(theta_xrt_hxn)):
    init_proj[theta_idx] = ndi.shift(init_proj[theta_idx], shift = (init_y_shift_array[theta_idx], 0))

init_proj_final = init_proj[:, start_slice:(intensity_xrt_norm_hxn.shape[1] - end_slice)]

print(init_proj_final.shape)

zero_deg_idx_array = np.where(theta_xrt_hxn == 0)[0]

theta_array_first_part = theta_xrt_hxn[:zero_deg_idx_array[1]]
theta_array_second_part = theta_xrt_hxn[zero_deg_idx_array[1]:]



theta_idx_pairs_first_part = [(0, -1)] # These remap to original -180° and 0° indices
theta_idx_pairs_second_part = [(0, -1)] # These remap to original 0° and +180° indices

# center_of_rotation_avg_first_part, geometric_center, offset_init_first_part = rot_center_avg(init_proj_final[:zero_deg_idx_array[1]], 
#                                                                                              theta_idx_pairs_first_part, 
#                                                                                              theta_array_first_part)

# center_of_rotation_avg_second_part, geometric_center, offset_final_second_part = rot_center_avg(init_proj_final[zero_deg_idx_array[1]:], 
#                                                                                                 theta_idx_pairs_second_part, 
#                                                                                                 theta_array_second_part)

# shifts_init_first_part, phase_xcorr_2d_first_part, _ = realign.phase_xcorr_manual(init_proj_final[0], np.fliplr(aligned_proj_total_xrf[zero_deg_idx_array[0]]), sigma = 25, alpha = 10, pixel_rad = 0, theta = np.array([-180, 0]))
# shifts_init_second_part, phase_xcorr_2d_second_part, _ = realign.phase_xcorr_manual(init_proj_final[zero_deg_idx_array[1]], np.fliplr(aligned_proj_total_xrf[-1]), sigma = 25, alpha = 10, pixel_rad = 0, theta = np.array([0, 180]))

# plt.imshow(phase_xcorr_2d_first_part, vmin = phase_xcorr_2d_first_part.min(), vmax = phase_xcorr_2d_first_part.max())
# plt.imshow(phase_xcorr_2d_second_part, vmin = phase_xcorr_2d_second_part.min(), vmax = phase_xcorr_2d_second_part.max())
# plt.show()

# print('Initial offset (first part):', shifts_init_first_part[1]/2)
# print('Initial offset (second part):', shifts_init_second_part[1]/2)


# offset_init_first_part = 2.6071765573317
# offset_init_second_part = 1.9150763284889
# pcc_offset = 0.0143047860342

center_of_rotation_avg_first_part, geometric_center, offset_final_first_part = rot_center_avg(aligned_proj_total_xrf[:zero_deg_idx_array[1]], 
                                                                                              theta_idx_pairs_first_part, 
                                                                                              theta_array_first_part)

center_of_rotation_avg_second_part, geometric_center, offset_final_second_part = rot_center_avg(aligned_proj_total_xrf[zero_deg_idx_array[1]:], 
                                                                                                theta_idx_pairs_second_part, 
                                                                                                theta_array_second_part)

# print(center_of_rotation_avg_first_part, geometric_center, offset_init_first_part)
# print(center_of_rotation_avg_second_part, geometric_center, offset_final_second_part)

# intensity_xrt_norm_hxn[0] = np.fliplr(intensity_xrt_norm_hxn[0])

# # plt.imshow(intensity_xrt_norm_hxn[0])
# plt.imshow(intensity_xrt_norm_hxn[:, 0], aspect = 'auto')
# plt.show()

create_cor_fig_hxn(init_proj_final[:zero_deg_idx_array[1]], aligned_proj_total_xrf[:zero_deg_idx_array[1]], theta_array_first_part, aligning_element_hxn)
# create_cor_fig_hxn(init_proj_final[zero_deg_idx_array[1]:], aligned_proj_total_xrf[zero_deg_idx_array[1]:], theta_array_second_part, aligning_element_hxn)
create_cor_fig_hxn_offset(init_proj_final, aligned_proj_total_xrf, theta_xrt_hxn, aligning_element_hxn)



