import numpy as np, os, sys, xrf_xrt_preprocess_file_util as futil, xrf_xrt_preprocess_utils as ppu

from matplotlib import pyplot as plt
from scipy import ndimage as ndi, fft

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

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

def create_cor_fig_aps(init_proj, shifted_proj, theta_array,theta_pair, aligning_element, offset_init, offset_final):
    fig, axs = plt.subplots(2, 3)

    print(init_proj.shape, shifted_proj.shape)

    init_proj_theta_0 = init_proj[theta_pair[0]]
    init_proj_theta_1 = np.fliplr(init_proj[theta_pair[1]])
    shifted_proj_theta_0 = shifted_proj[theta_pair[0]]
    shifted_proj_theta_1 = np.fliplr(shifted_proj[theta_pair[1]])

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
    
    init_cor = init_proj_theta_0.shape[1]//2 - offset_init
    final_cor = shifted_proj_theta_0.shape[1]//2 + offset_final

    im1_1 = axs[0, 0].imshow(init_proj_theta_0_rgb)
    im1_2 = axs[0, 1].imshow(init_proj_theta_1_rgb)
    im1_3 = axs[0, 2].imshow(overlay_init)
    im1_4 = axs[1, 0].imshow(shifted_proj_theta_0_rgb)
    im1_5 = axs[1, 1].imshow(shifted_proj_theta_1_rgb)
    im1_6 = axs[1, 2].imshow(overlay_shifted)

    for ax in fig.axes:
        ax.axis('off')

    text_1 = axs[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_pair[0]]), transform = axs[0, 0].transAxes, color = 'white')
    text_2 = axs[0, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_pair[1]]), transform = axs[0, 1].transAxes, color = 'white')
    text_3 = axs[1, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_pair[0]]), transform = axs[1, 0].transAxes, color = 'white')
    text_4 = axs[1, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_pair[1]]), transform = axs[1, 1].transAxes, color = 'white')

    axs[0, 0].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    axs[0, 1].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    axs[0, 2].set_title(r'{0} (overlay)'.format(aligning_element), fontsize = 14)
    axs[1, 0].set_title(r'{0} (shifted)'.format(aligning_element), fontsize = 14)
    axs[1, 1].set_title(r'{0} (shifted)'.format(aligning_element), fontsize = 14)
    axs[1, 2].set_title(r'{0} (shifted overlay)'.format(aligning_element), fontsize = 14)

    axs[0, 2].hlines(y = init_proj_theta_0.shape[0]//2, xmin = init_cor - 20, xmax = init_cor - 3, color = 'white', linewidth = 2)
    axs[0, 2].hlines(y = init_proj_theta_0.shape[0]//2, xmin = init_cor + 3, xmax = init_cor + 20, color = 'white', linewidth = 2)
    axs[0, 2].vlines(x = init_cor, ymin = init_proj_theta_0.shape[0]//2 - 20, ymax = init_proj_theta_0.shape[0]//2 - 3, color = 'white', linewidth = 2)
    axs[0, 2].vlines(x = init_cor, ymin = init_proj_theta_0.shape[0]//2 + 3, ymax = init_proj_theta_0.shape[0]//2 + 20, color = 'white', linewidth = 2)
    
    axs[1, 2].hlines(y = shifted_proj_theta_0.shape[0]//2, xmin = final_cor - 20, xmax = final_cor - 3, color = 'white', linewidth = 2)
    axs[1, 2].hlines(y = shifted_proj_theta_0.shape[0]//2, xmin = final_cor + 3, xmax = final_cor + 20, color = 'white', linewidth = 2)
    axs[1, 2].vlines(x = final_cor, ymin = shifted_proj_theta_0.shape[0]//2 - 20, ymax = shifted_proj_theta_0.shape[0]//2 - 3, color = 'white', linewidth = 2)
    axs[1, 2].vlines(x = final_cor, ymin = shifted_proj_theta_0.shape[0]//2 + 3, ymax = shifted_proj_theta_0.shape[0]//2 + 20, color = 'white', linewidth = 2)

    fig.tight_layout()

    plt.show()

dir_path = '/Users/bwr0835/Documents'

init_aps_xrt_file_path = f'{dir_path}/2_ide_aggregate_xrt.h5'
init_hxn_xrt_file_path = f'{dir_path}/3_id_aggregate_xrt.h5'

init_aps_xrf_file_path = f'{dir_path}/2_ide_aggregate_xrf.h5'
init_hxn_xrf_file_path = f'{dir_path}/3_id_aggregate_xrf.h5'

elements_xrt_aps, intensity_xrt_aps, theta_xrt_aps, _, _, _, _ = futil.extract_h5_aggregate_xrt_data(init_aps_xrt_file_path)
elements_xrt_hxn, intensity_xrt_hxn, theta_xrt_hxn, _, _, _, _ = futil.extract_h5_aggregate_xrt_data(init_hxn_xrt_file_path)

elements_xrf_aps, intensity_xrf_aps, theta_xrf_aps, _, _, _ = futil.extract_h5_aggregate_xrf_data(init_aps_xrf_file_path)
elements_xrf_hxn, intensity_xrf_hxn, theta_xrf_hxn, _, _, _ = futil.extract_h5_aggregate_xrf_data(init_hxn_xrf_file_path)

n_slices_aps, n_columns_aps = intensity_xrt_aps.shape[2:]

intensity_xrt_aps = ppu.pad_col_row(intensity_xrt_aps, 'xrt')
intensity_xrf_aps = ppu.pad_col_row(intensity_xrf_aps, 'xrf')
            
n_slices_aps += 1
n_columns_aps += 1

aligning_element_aps = 'opt_dens'
# aligning_element_hxn = 'Ni'

xrt_sig_aps = intensity_xrt_aps[elements_xrt_aps.index('xrt_sig')]

# aligning_element_idx_aps = elements_xrt_aps.index(aligning_element_aps)
# aligning_element_idx_hxn = elements_xrf_hxn.index(aligning_element_hxn)

intensity_xrt_norm_aps, intensity_xrf_norm_aps, _, _, I0_photons_aps, _ = ppu.joint_fluct_norm(xrt_sig_aps,
                                                                                                 intensity_xrf_aps,
                                                                                                 93,
                                                                                                 False,
                                                                                                 8.6776e8,
                                                                                                 0.01)
                                                                                            
opt_dens_aps = -np.log(intensity_xrt_norm_aps/I0_photons_aps)

cor_shift_aps = -6.68587804108239

shifted_opt_dens_aps = ndi.shift(opt_dens_aps, shift = (0, 0, cor_shift_aps))

theta_idx_pairs_aps = ppu.find_theta_combos(theta_xrt_aps)

center_of_rotation_avg, center_geom, offset_init = rot_center_avg(shifted_opt_dens_aps, theta_idx_pairs_aps, theta_xrt_aps)
    
print(f'Average center of rotation after COR correction: {center_of_rotation_avg}')
print(f'Geometric center: {center_geom}')
print(f'Center of rotation error: {ppu.round_correct(offset_init, ndec = 3)}')

create_cor_fig_aps(opt_dens_aps, shifted_opt_dens_aps, theta_xrt_aps, theta_idx_pairs_aps[0], aligning_element_aps, offset_init, cor_shift_aps)




