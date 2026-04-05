import numpy as np, pandas as pd,os, sys, xrf_xrt_preprocess_file_util as futil, xrf_xrt_preprocess_utils as ppu, realignment_final as realign

from matplotlib import pyplot as plt
from scipy import ndimage as ndi, fft

import h5py

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def load_h5_data(file_path, get_init_edge_info = False, get_final_edge_info = False):
    with h5py.File(file_path, 'r') as f:
        data = f['exchange/data/xrt'][()]

        if get_init_edge_info:
            top_edge_cropped_init = f['exchange/data'].attrs['top_edge_cropped_init']
            bottom_edge_cropped_init = f['exchange/data'].attrs['bottom_edge_cropped_init']
    
            return data, {'top': top_edge_cropped_init, 'bottom': bottom_edge_cropped_init}
    
        if get_final_edge_info:
            top_edge_cropped_final = f['exchange/data'].attrs['top_edge_cropped_final']
            bottom_edge_cropped_final = f['exchange/data'].attrs['bottom_edge_cropped_final']
    
            return data, {'top': top_edge_cropped_final, 'bottom': bottom_edge_cropped_final}

    return data, None

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

def create_cor_fig_aps(init_proj, shifted_proj, theta_array, theta_idx_pair, aligning_element):
    fig, axs = plt.subplots(2, 3)
    
    print(init_proj.shape, shifted_proj.shape)

    init_proj_theta_0 = init_proj[theta_idx_pair[0]]
    init_proj_theta_1 = np.fliplr(init_proj[theta_idx_pair[1]])
    shifted_proj_theta_0 = shifted_proj[theta_idx_pair[0]]
    shifted_proj_theta_1 = np.fliplr(shifted_proj[theta_idx_pair[1]])

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

    im1_1 = axs[0, 0].imshow(init_proj_theta_0_rgb)
    im1_2 = axs[0, 1].imshow(init_proj_theta_1_rgb)
    im1_3 = axs[0, 2].imshow(overlay_init)
    im1_4 = axs[1, 0].imshow(shifted_proj_theta_0_rgb)
    im1_5 = axs[1, 1].imshow(shifted_proj_theta_1_rgb)
    im1_6 = axs[1, 2].imshow(overlay_shifted)

    for ax in fig.axes:
        ax.axis('off')
        ax.axvline(x = init_proj_theta_0.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
        ax.axhline(y = init_proj_theta_0.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')

    text_1 = axs[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx_pair[0]]), transform = axs[0, 0].transAxes, color = 'white')
    text_2 = axs[0, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx_pair[1]]), transform = axs[0, 1].transAxes, color = 'white')
    text_3 = axs[1, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx_pair[0]]), transform = axs[1, 0].transAxes, color = 'white')
    text_4 = axs[1, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx_pair[1]]), transform = axs[1, 1].transAxes, color = 'white')

    axs[0, 0].set_title(r'{0}'.format(aligning_element), color = 'red', fontsize = 14)
    axs[0, 1].set_title(r'{0}'.format(aligning_element), color = 'green', fontsize = 14)
    axs[0, 2].set_title(r'{0} (overlay)'.format(aligning_element), fontsize = 14)
    axs[1, 0].set_title(r'{0} (shifted)'.format(aligning_element), color = 'red', fontsize = 14)
    axs[1, 1].set_title(r'{0} (shifted)'.format(aligning_element), color = 'green', fontsize = 14)
    axs[1, 2].set_title(r'{0} (shifted overlay)'.format(aligning_element), fontsize = 14)

    fig.tight_layout()

    plt.show()

dir_path = '/Users/bwr0835/Documents'
final_dir_path = f'{dir_path}/2_ide_realigned_data_03_27_2026_iter_reproj_cor_correction_only_final'

init_aps_xrt_file_path = f'{dir_path}/2_ide_aggregate_xrt.h5'
init_aps_xrf_file_path = f'{dir_path}/2_ide_aggregate_xrf.h5'

elements_xrt_aps, intensity_xrt_aps, theta_xrt_aps, _, _, _, _ = futil.extract_h5_aggregate_xrt_data(init_aps_xrt_file_path)
elements_xrf_aps, intensity_xrf_aps, theta_xrf_aps, _, _, _ = futil.extract_h5_aggregate_xrf_data(init_aps_xrf_file_path)

intensity_xrt_aps = ppu.pad_col_row(intensity_xrt_aps, 'xrt')
intensity_xrf_aps = ppu.pad_col_row(intensity_xrf_aps, 'xrf')

input_csv_file_path = f'{final_dir_path}/xrt_od_xrf_realignment/output_net_shift_data.csv'
input_h5_file_path = f'{final_dir_path}/xrt_od_xrf_realignment/aligned_data/aligned_aggregate_xrf_xrt.h5'

# aligning_element_aps = 'opt_dens'
aligning_element_aps = 'opt_dens'

xrt_sig_aps = intensity_xrt_aps[elements_xrt_aps.index('xrt_sig')]

# aligning_element_idx_aps = elements_xrt_aps.index(aligning_element_aps)
# aligning_element_idx_aps = elements_xrf_aps.index(aligning_element_aps)

x_shift_data = pd.read_csv(input_csv_file_path)
net_x_shifts = x_shift_data['net_x_pixel_shift'].to_numpy().astype(float)

intensity_xrt_norm_aps, intensity_xrf_norm_aps, _, _, I0_photons_aps, _ = ppu.joint_fluct_norm(xrt_sig_aps,
                                                                                               intensity_xrf_aps,
                                                                                               93,
                                                                                               True,
                                                                                               None,
                                                                                               None)

opt_dens = -np.log(intensity_xrt_norm_aps/I0_photons_aps)
# element_to_align_with = 'opt_dens'
# element_to_align_with_idx = elements_xrt_aps.index(element_to_align_with)``

aligned_proj_total_xrf, edge_info = load_h5_data(input_h5_file_path, get_final_edge_info = True)

# init_proj_final = intensity_xrt_norm_aps[element_to_align_with_idx]
init_proj_final = opt_dens
shifted_proj_final = np.zeros_like(init_proj_final)

for theta_idx in range(len(theta_xrt_aps)):
    shifted_proj_final[theta_idx] = ndi.shift(init_proj_final[theta_idx], shift = (0, net_x_shifts[theta_idx]))

init_proj_final = init_proj_final[:, :-2]
shifted_proj_final = shifted_proj_final[:, :-2]

theta_idx_pairs = ppu.find_theta_combos(theta_xrt_aps)
theta_idx_pair_desired = theta_idx_pairs[0]

create_cor_fig_aps(init_proj_final, shifted_proj_final, theta_xrt_aps, theta_idx_pair_desired, aligning_element_aps)
