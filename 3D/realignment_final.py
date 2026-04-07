import numpy as np, \
       tomopy as tomo, \
       xrf_xrt_preprocess_utils as ppu, \
       xrf_xrt_preprocess_file_util as futil, \
       sys

from matplotlib import pyplot as plt

from skimage import transform as xform, registration as reg
from scipy import ndimage as ndi, fft
from imageio import v2 as iio2

def normalize_array(array):
    return (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))

def normalize_array_for_gif(array):
    array_nonzero = array[array != 0]

    return np.clip((array - np.nanmin(array_nonzero))/(np.nanmax(array_nonzero) - np.nanmin(array_nonzero)), 0, 1)

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

def create_cor_fig_hxn_offset_for_gif(raw_proj, net_x_shift_array, net_y_shift_array, shift_array, theta_array, aligning_element):
    fig, axs = plt.subplots(3, 3)
    
    zero_deg_idx_array = np.where(theta_array == 0)[0]

    net_x_shift = net_x_shift_array.copy()
    shifted_proj = np.zeros_like(raw_proj)

    n_theta_first_part = len(theta_array[:zero_deg_idx_array[1]])
    n_theta_second_part = len(theta_array[zero_deg_idx_array[1]:])

    net_x_shift[0, zero_deg_idx_array[1]:] += shift_array[0]

    for theta_idx in range(n_theta_second_part):
        theta_idx_aux = theta_idx + n_theta_first_part

        shifted_proj[theta_idx_aux] = warp_shift(raw_proj[theta_idx_aux], net_x_shift[0, theta_idx_aux], net_y_shift_array[0, theta_idx_aux], cval = 0)
        
    shifted_proj_theta_0_0 = shifted_proj[zero_deg_idx_array[0]]
    shifted_proj_theta_0_1 = np.fliplr(shifted_proj[-1])

    shifted_proj_theta_1_0 = shifted_proj[zero_deg_idx_array[1]]
    shifted_proj_theta_1_1 = np.fliplr(shifted_proj[-1])

    shifted_proj_theta_2_0 = shifted_proj[zero_deg_idx_array[1] + 1]
    shifted_proj_theta_2_1 = np.fliplr(shifted_proj[-1])

    shifted_proj_theta_0_0_norm = normalize_array_for_gif(shifted_proj_theta_0_0)
    shifted_proj_theta_0_1_norm = normalize_array_for_gif(shifted_proj_theta_0_1)
    shifted_proj_theta_1_0_norm = normalize_array_for_gif(shifted_proj_theta_1_0)
    shifted_proj_theta_1_1_norm = normalize_array_for_gif(shifted_proj_theta_1_1)
    shifted_proj_theta_2_0_norm = normalize_array_for_gif(shifted_proj_theta_2_0)
    shifted_proj_theta_2_1_norm = normalize_array_for_gif(shifted_proj_theta_2_1)

    shifted_proj_theta_0_0_rgb = np.dstack((shifted_proj_theta_0_0_norm, np.zeros_like(shifted_proj_theta_0_0_norm), np.zeros_like(shifted_proj_theta_0_0_norm)))
    shifted_proj_theta_0_1_rgb = np.dstack((np.zeros_like(shifted_proj_theta_0_1_norm), shifted_proj_theta_0_1_norm, np.zeros_like(shifted_proj_theta_0_1_norm)))
    shifted_proj_theta_1_0_rgb = np.dstack((shifted_proj_theta_1_0_norm, np.zeros_like(shifted_proj_theta_1_0_norm), np.zeros_like(shifted_proj_theta_1_0_norm)))
    shifted_proj_theta_1_1_rgb = np.dstack((np.zeros_like(shifted_proj_theta_1_1_norm), shifted_proj_theta_1_1_norm, np.zeros_like(shifted_proj_theta_1_1_norm)))
    shifted_proj_theta_2_0_rgb = np.dstack((shifted_proj_theta_2_0_norm, np.zeros_like(shifted_proj_theta_2_0_norm), np.zeros_like(shifted_proj_theta_2_0_norm)))
    shifted_proj_theta_2_1_rgb = np.dstack((np.zeros_like(shifted_proj_theta_2_1_norm), shifted_proj_theta_2_1_norm, np.zeros_like(shifted_proj_theta_2_1_norm)))

    overlay_shifted_0 = np.dstack((shifted_proj_theta_0_0_norm, shifted_proj_theta_0_1_norm, np.zeros_like(shifted_proj_theta_0_0_norm)))
    overlay_shifted_1 = np.dstack((shifted_proj_theta_1_0_norm, shifted_proj_theta_1_1_norm, np.zeros_like(shifted_proj_theta_1_0_norm)))
    overlay_shifted_2 = np.dstack((shifted_proj_theta_2_0_norm, shifted_proj_theta_2_1_norm, np.zeros_like(shifted_proj_theta_2_0_norm)))

    im1_1 = axs[0, 0].imshow(shifted_proj_theta_0_0_rgb)
    im1_2 = axs[0, 1].imshow(shifted_proj_theta_0_1_rgb)
    im1_3 = axs[0, 2].imshow(overlay_shifted_0)
    im1_4 = axs[1, 0].imshow(shifted_proj_theta_1_0_rgb)
    im1_5 = axs[1, 1].imshow(shifted_proj_theta_1_1_rgb)
    im1_6 = axs[1, 2].imshow(overlay_shifted_1)
    im1_7 = axs[2, 0].imshow(shifted_proj_theta_2_0_rgb)
    im1_8 = axs[2, 1].imshow(shifted_proj_theta_2_1_rgb)
    im1_9 = axs[2, 2].imshow(overlay_shifted_2)

    for ax in fig.axes:
        ax.axis('off')
        ax.axvline(x = shifted_proj_theta_0_0.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
        ax.axhline(y = shifted_proj_theta_0_0.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')

    text_1 = axs[0, 0].text(0.02, 0.02, r'Shift = {0}'.format(shift_array[0]), transform = axs[0, 0].transAxes, color = 'white')

    axs[0, 0].set_title(r'$\theta = {0}^{{+}}$'.format(theta_array[zero_deg_idx_array[1]]), fontsize = 14)
    axs[0, 1].set_title(r'$\theta = {0}$\textdegree'.format(theta_array[-1]), fontsize = 14)
    axs[0, 2].set_title(r'{0} (overlay)'.format(aligning_element), fontsize = 14)
    axs[1, 0].set_title(r'$\theta = {0}^{{-}}$'.format(theta_array[zero_deg_idx_array[0]]), fontsize = 14)
    axs[1, 1].set_title(r'$\theta = {0}$\textdegree'.format(theta_array[-1]), fontsize = 14)
    axs[1, 2].set_title(r'{0} (shifted overlay)'.format(aligning_element), fontsize = 14)
    axs[2, 0].set_title(r'$\theta = {0}$\textdegree'.format(theta_array[zero_deg_idx_array[1] + 1]), fontsize = 14)
    axs[2, 1].set_title(r'$\theta = {0}$\textdegree'.format(theta_array[-1]), fontsize = 14)
    axs[2, 2].set_title(r'{0} (shifted overlay)'.format(aligning_element), fontsize = 14)

    fig.suptitle(r'Post-individual COR-corrected sample remount offset correction shifts ({0})'.format(aligning_element), fontsize = 16)
    
    frames = []
    for shift in shift_array:
        net_x_shift = net_x_shift_array.copy()
        net_x_shift[0, zero_deg_idx_array[1]:] += shift

        for theta_idx in range(n_theta_second_part):
            theta_idx_aux = theta_idx + n_theta_first_part

            shifted_proj[theta_idx_aux] = warp_shift(raw_proj[theta_idx_aux], net_x_shift[0, theta_idx_aux], net_y_shift_array[0, theta_idx_aux], cval = 0)
        

        shifted_proj_theta_0_0 = shifted_proj[zero_deg_idx_array[0]]
        shifted_proj_theta_0_1 = np.fliplr(shifted_proj[-1])

        shifted_proj_theta_1_0 = shifted_proj[zero_deg_idx_array[1]]
        shifted_proj_theta_1_1 = np.fliplr(shifted_proj[-1])

        shifted_proj_theta_2_0 = shifted_proj[zero_deg_idx_array[1] + 1]
        shifted_proj_theta_2_1 = np.fliplr(shifted_proj[-1])

        shifted_proj_theta_0_0_norm = normalize_array_for_gif(shifted_proj_theta_0_0)
        shifted_proj_theta_0_1_norm = normalize_array_for_gif(shifted_proj_theta_0_1)
        shifted_proj_theta_1_0_norm = normalize_array_for_gif(shifted_proj_theta_1_0)
        shifted_proj_theta_1_1_norm = normalize_array_for_gif(shifted_proj_theta_1_1)
        shifted_proj_theta_2_0_norm = normalize_array_for_gif(shifted_proj_theta_2_0)
        shifted_proj_theta_2_1_norm = normalize_array_for_gif(shifted_proj_theta_2_1)

        shifted_proj_theta_0_0_rgb = np.dstack((shifted_proj_theta_0_0_norm, np.zeros_like(shifted_proj_theta_0_0_norm), np.zeros_like(shifted_proj_theta_0_0_norm)))
        shifted_proj_theta_0_1_rgb = np.dstack((np.zeros_like(shifted_proj_theta_0_1_norm), shifted_proj_theta_0_1_norm, np.zeros_like(shifted_proj_theta_0_1_norm)))
        shifted_proj_theta_1_0_rgb = np.dstack((np.zeros_like(shifted_proj_theta_1_0_norm), shifted_proj_theta_1_0_norm, np.zeros_like(shifted_proj_theta_1_0_norm)))
        shifted_proj_theta_1_1_rgb = np.dstack((np.zeros_like(shifted_proj_theta_1_1_norm), shifted_proj_theta_1_1_norm, np.zeros_like(shifted_proj_theta_1_1_norm)))
        shifted_proj_theta_2_0_rgb = np.dstack((shifted_proj_theta_2_0_norm, np.zeros_like(shifted_proj_theta_2_0_norm), np.zeros_like(shifted_proj_theta_2_0_norm)))
        shifted_proj_theta_2_1_rgb = np.dstack((np.zeros_like(shifted_proj_theta_2_1_norm), shifted_proj_theta_2_1_norm, np.zeros_like(shifted_proj_theta_2_1_norm)))

        overlay_shifted_0 = np.dstack((shifted_proj_theta_0_0_norm, shifted_proj_theta_0_1_norm, np.zeros_like(shifted_proj_theta_0_0_norm)))
        overlay_shifted_1 = np.dstack((shifted_proj_theta_1_0_norm, shifted_proj_theta_1_1_norm, np.zeros_like(shifted_proj_theta_1_0_norm)))
        overlay_shifted_2 = np.dstack((shifted_proj_theta_2_0_norm, shifted_proj_theta_2_1_norm, np.zeros_like(shifted_proj_theta_2_0_norm)))

        im1_1.set_data(shifted_proj_theta_0_0_rgb)
        im1_2.set_data(shifted_proj_theta_0_1_rgb)
        im1_3.set_data(overlay_shifted_0)
        im1_4.set_data(shifted_proj_theta_1_0_rgb)
        im1_5.set_data(shifted_proj_theta_1_1_rgb)
        im1_6.set_data(overlay_shifted_1)
        im1_7.set_data(shifted_proj_theta_2_0_rgb)
        im1_8.set_data(shifted_proj_theta_2_1_rgb)
        im1_9.set_data(overlay_shifted_2)

        text_1.set_text(r'Shift = {0}'.format(shift))

        fig.canvas.draw()
        
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(frame)

    plt.close(fig)
  
    gif_filename = '/home/bwr0835/3_id_realigned_data_common_fov_cor_correction_only_03_30_2026_final/xrt_od_xrf_realignment/shifted_proj1.gif'

    iio2.mimsave(gif_filename, frames, fps = 10)

    return

def phase_xcorr_manual(ref_img,
                       mov_img, 
                       sigma, 
                       alpha, 
                       pixel_rad,
                       theta):
    
    n_slices = ref_img.shape[0]
    n_columns = ref_img.shape[1]
    
    ref_img_filtered = ppu.edge_gauss_filter(ref_img, sigma, alpha, nx = n_columns, ny = n_slices)
    mov_img_filtered = ppu.edge_gauss_filter(mov_img, sigma, alpha, nx = n_columns, ny = n_slices)

    ref_img_fft = fft.fft2(ref_img_filtered)
    mov_img_fft = fft.fft2(mov_img_filtered)

    phase_xcorr = fft.fftshift(np.abs(fft.ifft2(ref_img_fft*mov_img_fft.conjugate()/np.abs(ref_img_fft*mov_img_fft.conjugate()))))

    center_slice_idx = int(n_slices//2)
    center_column_idx = int(n_columns//2)

    # print(pixel_rad)

    if pixel_rad > 0:
        start_slice_idx = center_slice_idx - pixel_rad
        end_slice_idx = center_slice_idx + pixel_rad

        start_column_idx = center_column_idx - pixel_rad
        end_column_idx = center_column_idx + pixel_rad

        phase_xcorr_truncated = phase_xcorr[start_slice_idx:end_slice_idx, start_column_idx:end_column_idx]

    else:
        phase_xcorr_truncated = phase_xcorr

    pcc_max_idx = np.unravel_index(np.argmax(phase_xcorr_truncated), phase_xcorr_truncated.shape)

    n_rows_truncated = phase_xcorr_truncated.shape[0]
    n_columns_truncated = phase_xcorr_truncated.shape[1]
    
    if pcc_max_idx[0] + 1 < n_rows_truncated and pcc_max_idx[0] - 1 >= 0:
        pcc_p = phase_xcorr_truncated[pcc_max_idx[0] + 1, pcc_max_idx[1]]
        pcc_0 = phase_xcorr_truncated[pcc_max_idx[0], pcc_max_idx[1]]
        pcc_n = phase_xcorr_truncated[pcc_max_idx[0] - 1, pcc_max_idx[1]]
        
        subpix_shift_y = -0.5*(pcc_p - pcc_n)/(pcc_p + pcc_n - 2*pcc_0)

        if not np.isfinite(subpix_shift_y):
            print('Warning: Subpixel shift is not finite. Returning 0 for subpixel shift.')
            
            subpix_shift_y = 0
   
    else:
        print('Warning: Vertical parabolic fit failed (The peak is at an edge or corner of the truncated region) for theta = {0} and {1} degrees. Returning 0 for subpixel shift.'.format(theta[0], theta[1]))
        
        subpix_shift_y = 0

    if pcc_max_idx[1] + 1 < n_columns_truncated and pcc_max_idx[1] - 1 >= 0:
        pcc_p = phase_xcorr_truncated[pcc_max_idx[0], pcc_max_idx[1] + 1]
        pcc_0 = phase_xcorr_truncated[pcc_max_idx[0], pcc_max_idx[1]]
        pcc_n = phase_xcorr_truncated[pcc_max_idx[0], pcc_max_idx[1] - 1]
        
        subpix_shift_x = -0.5*(pcc_p - pcc_n)/(pcc_p + pcc_n - 2*pcc_0)

        if not np.isfinite(subpix_shift_x):
            print('Warning: Subpixel shift is not finite. Returning 0 for subpixel shift.')
            
            subpix_shift_x = 0
    else:
        print('Warning: Horizontal parabolic fit failed (The peak is at an edge or corner of the truncated region) for theta = {0} and {1} degrees. Returning 0 for subpixel shift.'.format(theta[0], theta[1]))
        
        subpix_shift_x = 0

    # Include integer peak offset: shift = (peak_position - center) + subpixel_refinement

    if pixel_rad > 0:
        shift_y = pcc_max_idx[0] - pixel_rad + subpix_shift_y
        shift_x = pcc_max_idx[1] - pixel_rad + subpix_shift_x
    
    else:
        shift_y = pcc_max_idx[0] - center_slice_idx + subpix_shift_y
        shift_x = pcc_max_idx[1] - center_column_idx + subpix_shift_x
    
    # print(shift_y)
    
    return np.array([shift_y, shift_x]), phase_xcorr, phase_xcorr_truncated

def correct_adjacent_angle_jitter_pre_cor_correction(init_proj_array,
                                                     net_y_shift_array,
                                                     sigma,
                                                     alpha,
                                                     pixel_rad,
                                                     theta,
                                                     return_aux_data):
    """
    Adjacent-angle vs cumulative convention
    ---------------------------------------
    Phase correlation measures one vertical shift per *pair* (θ_i, θ_{i+1}); call
    those δ_i (length n_theta - 1). This routine converts them to *per-projection*
    shifts Y_k (length n_theta) with projection 0 as reference: Y_0 is unchanged
    here, and Y_k for k >= 1 accumulates δ_0 + … + δ_{k-1} via np.cumsum, then
    adds that to net_y_shift_array[1:].

    init_y_shifts in raw_input_data.csv and ndi.shift must use the same Y_k
    (cumulative relative to angle 0), not the raw pair-wise δ_i. Mixing them
    misaligns the stack and changes min/max(Y), hence the common vertical FOV.
    """
    n_theta, n_slices, n_columns = init_proj_array.shape

    phase_xcorr_2d_aggregate = np.zeros((n_theta - 1, n_slices, n_columns))

    net_y_shift_cumsum_temp = np.zeros(n_theta - 1)

    if np.any(net_y_shift_array != 0):
        print('Applying initial vertical shifts...')

        shifted_proj_aux = np.zeros_like(init_proj_array)
        
        for theta_idx in range(n_theta):
            shifted_proj_aux[theta_idx] = ndi.shift(init_proj_array[theta_idx], shift = (net_y_shift_array[theta_idx], 0))
    
    else:
        shifted_proj_aux = init_proj_array
        
    if pixel_rad is None:
        print('Warning: \'pixel_rad\' not detected. Performing peak search without truncation...')

        pixel_rad = np.zeros(n_theta - 1)

        phase_xcorr_2d_truncated_aggregate = np.zeros_like(init_proj_array)
    
    else:
        if not isinstance(pixel_rad, np.ndarray):
            print('Error: \'pixel_rad\' must be a numpy array. Exiting program...')

            sys.exit()

        if pixel_rad.ndim != 1:
            print('Error: \'pixel_rad\' must be a 1D numpy array. Exiting program...')

            sys.exit()

        if pixel_rad.shape[0] != n_theta - 1:
            print('Error: \'pixel_rad\' must have the same number of elements as the number of theta angles. Exiting program...')

            sys.exit()
        
        if np.any(pixel_rad == 0):
            print('Warning: \'pixel_rad\' is 0. Performing peak search without truncation...')

            phase_xcorr_2d_truncated_aggregate = np.zeros((n_theta - 1, n_slices, n_columns))

        else:
            phase_xcorr_2d_truncated_aggregate = np.zeros((n_theta - 1, 2*pixel_rad.max(), 2*pixel_rad.max()))

    for theta_idx in range(n_theta - 1):
        shifts, phase_xcorr_2d, phase_xcorr_2d_truncated = phase_xcorr_manual(shifted_proj_aux[theta_idx],
                                                                              shifted_proj_aux[theta_idx + 1], 
                                                                              sigma, 
                                                                              alpha, 
                                                                              pixel_rad[theta_idx],
                                                                              theta[[theta_idx, theta_idx + 1]])

        net_y_shift_cumsum_temp[theta_idx] = shifts[0]

        if pixel_rad is not None and pixel_rad[theta_idx] > 0:
            phase_xcorr_2d_truncated_aggregate_midpt_idy, \
            phase_xcorr_2d_truncated_aggregate_midpt_idx = phase_xcorr_2d_truncated_aggregate.shape[1]//2, \
                                                           phase_xcorr_2d_truncated_aggregate.shape[2]//2
            
            start_y = int(phase_xcorr_2d_truncated_aggregate_midpt_idy - pixel_rad[theta_idx])
            start_x = int(phase_xcorr_2d_truncated_aggregate_midpt_idx - pixel_rad[theta_idx])

            end_y = int(phase_xcorr_2d_truncated_aggregate_midpt_idy + pixel_rad[theta_idx])
            end_x = int(phase_xcorr_2d_truncated_aggregate_midpt_idx + pixel_rad[theta_idx])

            phase_xcorr_2d_truncated_aggregate[theta_idx, start_y:end_y, start_x:end_x] = phase_xcorr_2d_truncated
        
        else:
            phase_xcorr_2d_truncated_aggregate[theta_idx] = phase_xcorr_2d_truncated
        
        phase_xcorr_2d_aggregate[theta_idx] = phase_xcorr_2d

    net_y_shift_cumsum = np.cumsum(net_y_shift_cumsum_temp) # Cumulative sum of net y shifts (registering one angle to the previous angle still has residual error due to previous angles)

    net_y_shift_array[1:] += net_y_shift_cumsum

    if return_aux_data:
        shifted_proj = np.zeros_like(init_proj_array)

        for theta_idx in range(n_theta):
            shifted_proj[theta_idx] = ndi.shift(init_proj_array[theta_idx], shift = (net_y_shift_array[theta_idx], 0))
    
    # Compute common field of view (FOV)

    start_slice = int(np.clip(np.ceil(np.max(net_y_shift_array)), 0, n_slices)) # Crop top: exclude rows where positive shifts pushed content out
    end_slice = int(np.clip(n_slices + np.floor(np.min(net_y_shift_array)), 0, n_slices)) # Crop bottom: exclude rows where negative shifts pushed content out. This index is exclusive.
        
    if return_aux_data:
        shifted_proj_final = shifted_proj[:, start_slice:end_slice]

    if end_slice <= start_slice:
        print('Error: Empty field of view detected - net shifts exceed the number of slices. Exiting program...')

        sys.exit()

    if return_aux_data:        
        return net_y_shift_array, \
               start_slice, \
               end_slice, \
               phase_xcorr_2d_aggregate, \
               phase_xcorr_2d_truncated_aggregate, \
               shifted_proj_final
    
    return net_y_shift_array, \
           start_slice, \
           end_slice, \
           None, \
           None, \
           None

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

def warp_shift(img, net_x_shift, net_y_shift, cval = 0):
    if (net_y_shift.ndim, net_x_shift.ndim) == (0, 0):
        return ndi.shift(img, shift = (net_y_shift, net_x_shift), cval = cval)
    
    elif (net_y_shift.ndim, net_x_shift.ndim) == (0, 1):
        ny, nx = img.shape

        # len(dx) = nx; dy is a scalar: different vertical shift per column; same horizontal shift
        
        rows = (np.arange(ny, dtype = float)[:, None] - net_y_shift) + np.zeros((1, nx)) # Shape: (ny, nx)
        cols = np.arange(nx, dtype = float)[None] - net_x_shift[:, None] # Shape: (ny, nx)
        
        coords = np.stack([rows, cols], axis = 0)  # Shape: (2, ny, nx)
 
        return ndi.map_coordinates(img, coords, cval = cval)

def realign_proj(cor_correction_only,
                 aligning_element,
                 xrf_element_list,
                 xrt_proj_img_array,
                 opt_dens_proj_img_array,
                 xrf_proj_img_array,
                 theta_array,
                 sample_flipped_remounted_mid_experiment,
                 n_iterations_cor_correction,
                 pixel_rad_cor_correction,
                 eps_cor_correction,
                 I0,
                 n_iterations_iter_reproj,
                 init_x_shift, 
                 init_y_shift,
                 sigma,
                 alpha,
                 pixel_rad_iter_reproj,
                 eps_iter_reproj,
                 edge_info,
                 return_aux_data):

    '''
    realign_proj: Perform phase symmetry and iterative reprojection on experimental optical density (OD) projection images 
    to correct for center of rotation (COR) error, jitter (per-projection translations), respectively, 
    in x-ray transmission, OD, and x-ray fluorescnece projection images

    Inputs
    ------
    opt_dens_proj_img_array: 3D optical density data derived from xrt_proj_img_array (projection angles, slices, scan positions) (array-like, dtype: float)
    xrf_proj_img_array: 4D XRF data (elements, projection_angles, slices, scan positions) (array-like; dtype: float)
    theta_array: Array of projection angles (array-like; dtype: float)
    n_iterations_iter_reproj: Maximum number of iterative reprojection iterations (dtype: int)
    init_x_shift: Initial x-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)    
    init_y_shift: Initial y-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)
    eps: Desired differential shift for convergence criterion (dtype: float)
    return_aux_data: Flag for returning per-iteration auxiliary data (dtype: bool; default: False)

    Outputs
    ------- 
    aligned_proj_total: 4D XRF tomography data (XRT data will come later) corrected for per-projection jitter, center of rotation misalignment (array-like; dtype: float)
    aligned_exp_proj_array: 1D array of experimental 3D XRF tomography data arrays for each iteration for ref_element (array-like; dtype: float)
    synth_proj_array: 1D array of synthetic 3D XRF tomography data arrays for each iteration for ref_element (array-like; dtype: float)
    recon_array: 1D array of 3D reconstruction slices for each iteration for ref_element (array-like; dtype: float)
    net_x_shifts_pcc_new: Array of net x shifts with dimensions (n_iterations_iter_reproj, n_theta) (array-like; dtype: float) (Note: n_iterations_iter_reproj can be smaller the more quickly iter_reproj() converges)
    net_y_shifts_pcc_new: Array of net y shifts with dimensions (n_iterations_iter_reproj, n_theta) (array-like; dtype: float) (Note: n_iterations_iter_reproj can be smaller the more quickly iter_reproj() converges)
    '''
    
    if sigma is None:
        print('Warning: Positive \'sigma\' not detected. Setting \'sigma\' to 5 pixels...')
        
        sigma = 5

    if alpha is None:
        print('Warning: Positive, \'alpha\' not detected. Setting \'alpha\' to 10 pixels...')
        
        alpha = 10

    if sigma < 0 or alpha < 0:
        print('Error: \'sigma\', \'alpha\', \'eps_cor_correction\' and \'eps_iter_reproj\' must all be positive numbers. Exiting program...')

        sys.exit()

    if aligning_element == 'opt_dens':
        proj_img_array_element_to_align_with = opt_dens_proj_img_array.copy()
        aligned_proj = opt_dens_proj_img_array.copy()

        cval = 0
    
    elif aligning_element in xrf_element_list:
        element_to_align_with_idx = xrf_element_list.index(aligning_element)
        
        proj_img_array_element_to_align_with = xrf_proj_img_array[element_to_align_with_idx].copy()
        aligned_proj = xrf_proj_img_array[element_to_align_with_idx].copy()

        cval = 0

    elif aligning_element == 'xrt':
        proj_img_array_element_to_align_with = xrt_proj_img_array.copy()
        aligned_proj = xrt_proj_img_array.copy()

        cval = I0
    
    else: 
        print('Error: \'aligning_element\' must be \'opt_dens\', an element in \'xrf_element_list\', or \'xrt\'. Exiting program...')

        sys.exit()
    
    n_elements_xrf, n_theta, n_slices, n_columns = xrf_proj_img_array.shape

    if edge_info is not None:
        if edge_info['top'] + edge_info['bottom'] > 0:
            net_x_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta, n_slices))
            net_y_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta))

            start_slice = edge_info['top'] # Second term is initial common field of view index
            end_slice = edge_info['bottom'] # First term is final common field of view index
            print(start_slice, end_slice)
            if start_slice >= end_slice:
                print('Error: Overlapping edge, field-of-view crops in y. Exiting program...')

                sys.exit()
        
        else:
            net_x_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta))
            net_y_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta))
    
    else:
        net_x_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta))
        net_y_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta))

    aligned_proj_total_xrt = np.zeros((n_theta, n_slices, n_columns))
    aligned_proj_total_opt_dens = np.zeros((n_theta, n_slices, n_columns))
    aligned_proj_total_xrf = np.zeros((n_elements_xrf, n_theta, n_slices, n_columns))

    if np.any(init_x_shift) or np.any(init_y_shift):
        if np.any(init_x_shift) and np.any(init_y_shift):
            print('Executing intial shift(s) in x and y')

            net_x_shifts_pcc[0] += init_x_shift
            # net_y_shifts_pcc[0] += init_y_shift
            net_y_shifts_pcc += init_y_shift

            if net_x_shifts_pcc.ndim == 3:
                for theta_idx in range(n_theta):
                    aligned_proj[theta_idx] = ndi.shift(proj_img_array_element_to_align_with[theta_idx], shift = (init_y_shift[theta_idx], init_x_shift[theta_idx]))

        elif np.any(init_x_shift):
            print('Executing initial shift(s) in x')
            
            net_x_shifts_pcc[0] += init_x_shift
            
            for theta_idx in range(n_theta):
                aligned_proj[theta_idx] = ndi.shift(proj_img_array_element_to_align_with[theta_idx], shift = (0, init_x_shift[theta_idx]))
                
        else:
            print('Executing initial shift(s) in y')
            
            net_y_shifts_pcc += init_y_shift

            for theta_idx in range(n_theta):
                aligned_proj[theta_idx] = ndi.shift(proj_img_array_element_to_align_with[theta_idx], shift = (init_y_shift[theta_idx], 0))


    if sample_flipped_remounted_mid_experiment: # This assumes that angles are order from -180° to +180° (360° range) AND that there are two zero degree angles
        if eps_cor_correction is None:
            print('Warning: Nonzero, positive \'eps_cor_correction\' not detected. Setting \'eps_cor_correction\' to 0.001 pixels...')

            eps_cor_correction = 0.001
        
        if eps_cor_correction < 0:
            print('Error: \'eps_cor_correction\' must be a positive number. Exiting program...')

            sys.exit()

        if n_iterations_cor_correction is None:
            print('Error: \'n_iterations_cor_correction\' not detected. Exiting program...')

            sys.exit()

        if n_iterations_cor_correction < 1 or not isinstance(n_iterations_cor_correction, int):
            print('Error: \'n_iterations_cor_correction\' must be a positive integer. Exiting program...')

            sys.exit()

        if np.count_nonzero(theta_array == 0) != 2:
            print('Error: Must have two 0° angles. Exiting program...')

            sys.exit()
        
        if pixel_rad_cor_correction is None and return_aux_data:
            print('Warning: \'pixel_rad_cor_correction\' not detected. Setting \'pixel_rad_cor_correction\' to zero pixels...')

            pixel_rad_cor_correction = 0

        zero_deg_idx_array = np.where(theta_array == 0)[0]

        theta_array_first_part = theta_array[:zero_deg_idx_array[1]]
        theta_array_second_part = theta_array[zero_deg_idx_array[1]:]

        theta_idx_pairs_first_part = [(0, -1)] # These remap to original -180° and 0° indices
        theta_idx_pairs_second_part = [(0, -1)] # These remap to original 0° and +180° indices

        dx_prev = 0

        # for i in range(n_iterations_cor_correction):
        # for i in range(1):
        #     print(f'COR correction iteration {i + 1}/{n_iterations_cor_correction}')

        #     center_of_rotation_avg_first_part, center_geom, offset_init_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1], start_slice:end_slice], 
        #                                                                                             theta_idx_pairs_first_part, 
        #                                                                                             theta_array_first_part)
                
        #     center_of_rotation_avg_second_part, _, offset_init_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:, start_slice:end_slice], 
        #                                                                                     theta_idx_pairs_second_part, 
        #                                                                                     theta_array_second_part)

        #     print(f'Average center of rotation (before flipping sample): {center_of_rotation_avg_first_part}')
        #     print(f'Average center of rotation (after flipping sample): {center_of_rotation_avg_second_part}\n')
        #     print(f'Geometric center: {center_geom}\n')
        #     print(f'Center of rotation error (before flipping sample): {ppu.round_correct(offset_init_first_part, ndec = 3)}')
        #     print(f'Center of rotation error (after flipping sample): {ppu.round_correct(offset_init_second_part, ndec = 3)}\n')

        #     if i == 0 and offset_init_first_part == 0 and offset_init_second_part == 0:
        #         print('No COR correction needed.')

        #         break

        #     else:
        #         print(f'Applying initial COR correction to pre-flipped, pre-remounted sample angles: {ppu.round_correct(-offset_init_first_part, ndec = 3)}')

        #         if net_x_shifts_pcc.ndim == 3:
        #             net_x_shifts_pcc[0, :zero_deg_idx_array[1], start_slice:end_slice] -= offset_init_first_part
        #             net_x_shifts_pcc[0, zero_deg_idx_array[1]:, start_slice:end_slice] -= offset_init_second_part
                
        #         else:
        #             net_x_shifts_pcc[0, :zero_deg_idx_array[1]] -= offset_init_first_part
        #             net_x_shifts_pcc[0, zero_deg_idx_array[1]:] -= offset_init_second_part

        #         for theta_idx in range(len(theta_array_first_part)):
        #             aligned_proj[theta_idx] = warp_shift(proj_img_array_element_to_align_with[theta_idx], net_x_shifts_pcc[0, theta_idx], net_y_shifts_pcc[0, theta_idx], cval = cval)
                        
        #         print(f'Applying initial COR correction to post-flipped, post-remounted sample angles: {ppu.round_correct(-offset_init_second_part, ndec = 3)}')
                    
        #         for theta_idx in range(len(theta_array_second_part)):
        #             theta_idx_aux = theta_idx + len(theta_array_first_part)
                    
        #             aligned_proj[theta_idx_aux] = warp_shift(proj_img_array_element_to_align_with[theta_idx_aux], net_x_shifts_pcc[0, theta_idx_aux], net_y_shifts_pcc[0, theta_idx_aux], cval = cval)

        #         if net_x_shifts_pcc.ndim == 3:
        #             center_of_rotation_avg_first_part, center_geom, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1], start_slice:end_slice], 
        #                                                                                                theta_idx_pairs_first_part, 
        #                                                                                                theta_array_first_part)
                
        #             center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:, start_slice:end_slice], 
        #                                                                                        theta_idx_pairs_second_part, 
        #                                                                                        theta_array_second_part)

        #         else:
        #             center_of_rotation_avg_first_part, center_geom, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1]], 
        #                                                                                                theta_idx_pairs_first_part, 
        #                                                                                                theta_array_first_part)
                    
        #             center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:], 
        #                                                                                        theta_idx_pairs_second_part, 
        #                                                                                        theta_array_second_part)

        #         print(f'New center of rotation (before flipping sample): {center_of_rotation_avg_first_part}')
        #         print(f'New center of rotation (after flipping sample): {center_of_rotation_avg_second_part}\n')
        #         print(f'Geometric center: {center_geom}\n')
        #         print(f'New center of rotation error (before flipping sample): {ppu.round_correct(offset_first_part, ndec = 3)}')
        #         print(f'New center of rotation error (after flipping sample): {ppu.round_correct(offset_second_part, ndec = 3)}\n')

        #         if pixel_rad_cor_correction is None:
        #             print('Warning: \'pixel_rad_cor_correction\' not detected. Performing peak search without truncation...')

        #             pixel_rad_cor_correction = 0

        #         if net_x_shifts_pcc.ndim == 3:
        #             shifts, pcc, pcc_truncated = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[0], start_slice:end_slice], 
        #                                                             aligned_proj[zero_deg_idx_array[1], start_slice:end_slice], 
        #                                                             sigma, 
        #                                                             alpha, 
        #                                                             pixel_rad_cor_correction,
        #                                                             theta = np.array([0, 0]))
        #             # print(shifts)
        #             # fig, axs = plt.subplots(2, 1)
        #             # axs[0].imshow(pcc, vmin = pcc.min(), vmax = pcc.max())
        #             # axs[1].imshow(pcc_truncated, vmin = pcc_truncated.min(), vmax = pcc_truncated.max())
        #             # plt.show()
                
        #         else:
        #             shifts, _, _ = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[0]], 
        #                                               aligned_proj[zero_deg_idx_array[1]], 
        #                                               sigma, 
        #                                               alpha, 
        #                                               pixel_rad_cor_correction,
        #                                               theta = np.array([0, 0]))
                
        #         dx = shifts[1]
                
        #         if np.abs(dx - dx_prev) <= eps_cor_correction or (i == 0 and dx == 0):
        #             print('No further COR correction needed.')

        #             break
                    
        #         print(f'Applying additional COR correction to flipped, remounted sample angles: {ppu.round_correct(dx, ndec = 3)}')

        #         if net_x_shifts_pcc.ndim == 3:
        #             net_x_shifts_pcc[0, zero_deg_idx_array[1]:, start_slice:end_slice] += dx
                
        #         else:
        #             net_x_shifts_pcc[0, zero_deg_idx_array[1]:] += dx

        #         for theta_idx in range(len(theta_array_second_part)):
        #             theta_idx_aux = theta_idx + len(theta_array_first_part)
                        
        #             aligned_proj[theta_idx_aux] = warp_shift(proj_img_array_element_to_align_with[theta_idx_aux], net_x_shifts_pcc[0, theta_idx_aux], net_y_shifts_pcc[0, theta_idx_aux], cval = cval)

        #         dx_prev = dx
        for i in range(1):
            print(f'COR correction iteration {i + 1}/{n_iterations_cor_correction}')

            # center_of_rotation_avg_first_part, center_geom, offset_init_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1], start_slice:end_slice], 
            #                                                                                         theta_idx_pairs_first_part, 
            #                                                                                         theta_array_first_part)
            
            # center_of_rotation_avg_second_part, _, offset_init_second_part = rot_center_avg(aligned_proj[(zero_deg_idx_array[1] + 1):, start_slice:end_slice], 
            #                                                                                 theta_idx_pairs_second_part, 
            #                                                                                 theta_array_second_part)
            shifts_init_first_part, phase_xcorr_first_part, _ = phase_xcorr_manual(aligned_proj[0, start_slice:end_slice], np.fliplr(aligned_proj[zero_deg_idx_array[0], start_slice:end_slice]), sigma = sigma, alpha = alpha, pixel_rad = 0, theta = np.array([-180, 0]))
            shifts_init_second_part, phase_xcorr_second_part, _ = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[1] + 1, start_slice:end_slice], np.fliplr(aligned_proj[-1, start_slice:end_slice]), sigma = sigma, alpha = alpha, pixel_rad = 0, theta = np.array([0, 180]))
            
            vert_shift_second_part = shifts_init_second_part[0]
            print(vert_shift_second_part)
            center_geom = aligned_proj.shape[2]//2

            offset_init_first_part = shifts_init_first_part[1]/2
            offset_init_second_part = shifts_init_second_part[1]/2

            center_of_rotation_avg_first_part = center_geom + offset_init_first_part
            center_of_rotation_avg_second_part = center_geom + offset_init_second_part

            print(f'Average center of rotation (before flipping sample): {ppu.round_correct(center_of_rotation_avg_first_part, ndec = 13)}')
            print(f'Average center of rotation (after flipping sample): {ppu.round_correct(center_of_rotation_avg_second_part, ndec = 13)}\n')
            print(f'Geometric center: {center_geom}\n')
            print(f'Center of rotation error (before flipping sample): {ppu.round_correct(offset_init_first_part, ndec = 13)}')
            print(f'Center of rotation error (after flipping sample): {ppu.round_correct(offset_init_second_part, ndec = 13)}\n')

            if i == 0 and offset_init_first_part == 0 and offset_init_second_part == 0:
                print('No COR correction needed.')

                break

            else:
                print(f'Applying initial COR correction to pre-flipped, pre-remounted sample angles: {ppu.round_correct(-offset_init_first_part, ndec = 13)}')

                if net_x_shifts_pcc.ndim == 3:
                    print('Yes')
                    ddx = 0
                    net_x_shifts_pcc[0, :zero_deg_idx_array[1], start_slice:end_slice] -= offset_init_first_part
                    net_x_shifts_pcc[0, zero_deg_idx_array[1]:, start_slice:end_slice] -= (offset_init_second_part + ddx)
                    # net_y_shifts_pcc[0, zero_deg_idx_array[1]:] += (vert_shift_second_part)
                else:
                    net_x_shifts_pcc[0, :zero_deg_idx_array[1]] -= offset_init_first_part
                    net_x_shifts_pcc[0, zero_deg_idx_array[1]:] -= offset_init_second_part

                for theta_idx in range(len(theta_array_first_part)):
                    aligned_proj[theta_idx] = warp_shift(proj_img_array_element_to_align_with[theta_idx], net_x_shifts_pcc[0, theta_idx], net_y_shifts_pcc[0, theta_idx], cval = cval)

                # fig1, axs1 = plt.subplots(2, 1)
                # axs1[0].imshow(phase_xcorr_first_part, vmin = phase_xcorr_first_part.min(), vmax = phase_xcorr_first_part.max())
                # axs1[1].imshow(phase_xcorr_second_part, vmin = phase_xcorr_second_part.min(), vmax = phase_xcorr_second_part.max())
                # axs1[0].set_title(r'$\theta = -180^{\circ}, 0^{-}$')
                # axs1[1].set_title(r'$\theta = 0^{+}, 180^{\circ}$')
                
                # for ax in fig1.axes:
                #     ax.axis('off')
                #     ax.axvline(x = phase_xcorr_first_part.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
                #     ax.axhline(y = phase_xcorr_first_part.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')
                #     ax.axvline(x = phase_xcorr_second_part.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
                #     ax.axhline(y = phase_xcorr_second_part.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')
                    
                # fig1.tight_layout()
                # plt.show()

                print(f'Applying initial COR correction to post-flipped, post-remounted sample angles: {ppu.round_correct(-offset_init_second_part, ndec = 13)}')
                    
                for theta_idx in range(len(theta_array_second_part)):
                    theta_idx_aux = theta_idx + len(theta_array_first_part)
                    
                    aligned_proj[theta_idx_aux] = warp_shift(proj_img_array_element_to_align_with[theta_idx_aux], net_x_shifts_pcc[0, theta_idx_aux], net_y_shifts_pcc[0, theta_idx_aux], cval = cval)

                if net_x_shifts_pcc.ndim == 3:
                    center_of_rotation_avg_first_part, center_geom, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1], start_slice:end_slice], 
                                                                                                       theta_idx_pairs_first_part, 
                                                                                                       theta_array_first_part)
                
                    center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:, start_slice:end_slice], 
                                                                                               theta_idx_pairs_second_part, 
                                                                                               theta_array_second_part)

                # else:
                #     center_of_rotation_avg_first_part, center_geom, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1]], 
                #                                                                                        theta_idx_pairs_first_part, 
                #                                                                                        theta_array_first_part)
                    
                #     center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:], 
                #                                                                                theta_idx_pairs_second_part, 
                #                                                                                theta_array_second_part)
                
                # shifts_first_part, _, _ = phase_xcorr_manual(aligned_proj[0, start_slice:end_slice], np.fliplr(aligned_proj[zero_deg_idx_array[0], start_slice:end_slice]), sigma = sigma, alpha = alpha, pixel_rad = 0, theta = np.array([-180, 0]))
                # shifts_second_part, _, _ = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[1], start_slice:end_slice], np.fliplr(aligned_proj[-1, start_slice:end_slice]), sigma = sigma, alpha = alpha, pixel_rad = 0, theta = np.array([0, 180]))

                # center_geom = aligned_proj.shape[2]//2

                # offset_first_part = shifts_first_part[1]/2
                # offset_second_part = shifts_second_part[1]/2

                # center_of_rotation_avg_first_part = center_geom + offset_first_part
                # center_of_rotation_avg_second_part = center_geom + offset_second_part
                # fig, axs = plt.subplots()
                
                # im1_norm = normalize_array(aligned_proj[zero_deg_idx_array[1] + 1, start_slice:end_slice])
                # im2_norm = normalize_array(np.fliplr(aligned_proj[-1, start_slice:end_slice]))
                
                # im = np.dstack((im1_norm, im2_norm, np.zeros_like(im1_norm)))
                # axs.imshow(im, aspect = 'equal')
                # axs.axvline(x = im.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
                # axs.axhline(y = im.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')
                # axs.axis('off')
                # axs.set_title(r'Phase cross-correlation ($\theta = 0^{+}, 180$\textdegree) (phase cross-correlation COR alignment)', fontsize = 16)
                # fig.tight_layout()
                shift_array = np.linspace(-20, 20, 41)
                # plt.show()
                create_cor_fig_hxn_offset_for_gif(aligned_proj, net_x_shifts_pcc, net_y_shifts_pcc, shift_array, theta_array, aligning_element)
                print(f'New center of rotation (before flipping sample): {center_of_rotation_avg_first_part}')
                print(f'New center of rotation (after flipping sample): {center_of_rotation_avg_second_part}\n')
                print(f'Geometric center: {center_geom}\n')
                print(f'New center of rotation error (before flipping sample): {ppu.round_correct(offset_first_part, ndec = 13)}')
                print(f'New center of rotation error (after flipping sample): {ppu.round_correct(offset_second_part, ndec = 13)}\n')

                if pixel_rad_cor_correction is None:
                    print('Warning: \'pixel_rad_cor_correction\' not detected. Performing peak search without truncation...')

                    pixel_rad_cor_correction = 0

                if net_x_shifts_pcc.ndim == 3:
                    # shifts, pcc, pcc_truncated = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[0], start_slice:end_slice], 
                    #                                                 np.fliplr(aligned_proj[-1])[start_slice:end_slice], 
                    #                                                 sigma, 
                    #                                                 alpha, 
                    #                                                 pixel_rad_cor_correction,
                    #                                                 theta = np.array([0, 180]))
                    # shifts, pcc, pcc_truncated = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[0], start_slice:end_slice], 
                    #                                                 np.fliplr(aligned_proj[-1, start_slice:end_slice]), 
                    #                                                 sigma, 
                    #                                                 alpha, 
                    #                                                 pixel_rad_cor_correction,
                    #                                                 theta = np.array([0, 180]))
                    # for theta_idx in range(len(theta_array_second_part)):
                    #     theta_idx_aux = theta_idx + len(theta_array_first_part)
                    #     net_x_shifts_pcc[0, theta_idx_aux] += -2

                    #     aligned_proj[theta_idx_aux] = warp_shift(proj_img_array_element_to_align_with[theta_idx_aux], net_x_shifts_pcc[0, theta_idx_aux], net_y_shifts_pcc[0, theta_idx_aux], cval = cval)
                    
                    dx_extra = np.linspace(-8, 8, 17)
                    net_x_shifts_pcc_copy = net_x_shifts_pcc.copy()
                    for ddx_extra in dx_extra:
                        net_x_shifts_pcc_copy[0, zero_deg_idx_array[1]:, start_slice:end_slice] += ddx_extra

                        for theta_idx in range(len(theta_array_second_part)):
                            theta_idx_aux = theta_idx + len(theta_array_first_part)

                            aligned_proj[theta_idx_aux] = warp_shift(proj_img_array_element_to_align_with[theta_idx_aux], net_x_shifts_pcc_copy[0, theta_idx_aux], net_y_shifts_pcc[0, theta_idx_aux], cval = cval)

                    # pixel_rad_cor_correction = 5
                    # shifts, pcc, pcc_truncated = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[0], start_slice:end_slice], 
                    #                                                 aligned_proj[-1, start_slice:end_slice], 
                    #                                                 sigma, 
                    #                                                 alpha, 
                    #                                                 pixel_rad_cor_correction,
                    #                                                 theta = np.array([0, 180]))               
                    print(shifts)
                    # fig, axs = plt.subplots()
                    # # fig, axs = plt.subplots(2, 1)
                    # # axs[0].imshow(pcc, vmin = pcc.min(), vmax = pcc.max())
                    # axs.imshow(pcc, vmin = pcc.min(), vmax = pcc.max(), aspect = 'equal', interpolation = 'none')
                    # axs.axvline(x = pcc.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
                    # axs.axhline(y = pcc.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')
                    # axs.axis('off')
                    # axs.set_title(r'Phase cross-correlation ($\theta = 0^{-}, 180$\textdegree) (phase cross-correlation COR alignment)', fontsize = 16)
                    # fig.tight_layout()
                    # axs[1].imshow(pcc_truncated, vmin = pcc.min(), vmax = pcc.max(), extent = [pcc.shape[1]//2 - pixel_rad_cor_correction, 
                                                                                            #    pcc.shape[1]//2 + pixel_rad_cor_correction, 
                                                                                            #    pcc.shape[0]//2 + pixel_rad_cor_correction, 
                                                                                            #    pcc.shape[0]//2 - pixel_rad_cor_correction])
                    plt.show()
                
                else:
                    shifts, _, _ = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[0]], 
                                                      np.fliplr(aligned_proj[-1]), 
                                                      sigma, 
                                                      alpha, 
                                                      pixel_rad_cor_correction,
                                                      theta = np.array([0, 180]))
                
                dx = shifts[1]
                    
                print(f'Applying additional COR correction to flipped, remounted sample angles: {ppu.round_correct(-dx, ndec = 13)}')

                if net_x_shifts_pcc.ndim == 3:
                    net_x_shifts_pcc[0, zero_deg_idx_array[1]:, start_slice:end_slice] += dx
                
                else:
                    net_x_shifts_pcc[0, zero_deg_idx_array[1]:] -= dx

                for theta_idx in range(len(theta_array_second_part)):
                    theta_idx_aux = theta_idx + len(theta_array_first_part)
                        
                    aligned_proj[theta_idx_aux] = warp_shift(proj_img_array_element_to_align_with[theta_idx_aux], net_x_shifts_pcc[0, theta_idx_aux], net_y_shifts_pcc[0, theta_idx_aux], cval = cval)
                
                xrf_proj_copy = xrf_proj_img_array[element_to_align_with_idx].copy()

                for theta_idx in range(n_theta):
                    xrf_proj_copy[theta_idx] = ndi.shift(xrf_proj_copy[theta_idx], shift = (net_y_shifts_pcc[0, theta_idx], 0))
                create_cor_fig_hxn(xrf_proj_copy[zero_deg_idx_array[1] + 1:, start_slice:end_slice], aligned_proj[zero_deg_idx_array[1] + 1:, start_slice:end_slice], theta_array_second_part, aligning_element)
                create_cor_fig_hxn_offset(xrf_proj_copy[:, start_slice:end_slice], aligned_proj[:, start_slice:end_slice], theta_array, aligning_element)
                # plt.show()

    else:
        theta_idx_pairs = ppu.find_theta_combos(theta_array)

        center_of_rotation_avg, center_geom, offset_init = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)
    
        print(f'Average center of rotation: {center_of_rotation_avg}')
        print(f'Geometric center: {center_geom}')
        print(f'Center of rotation error: {ppu.round_correct(offset_init, ndec = 3)}')
        print(f'Applying initial center of rotation correction: {ppu.round_correct(-offset_init, ndec = 3)}')
        
        if net_x_shifts_pcc.ndim == 3:
            net_x_shifts_pcc[0, :, start_slice:end_slice] -= offset_init
        
        else:
            net_x_shifts_pcc[0] -= offset_init
            
        for theta_idx in range(n_theta):
            aligned_proj[theta_idx] = warp_shift(proj_img_array_element_to_align_with[theta_idx], net_x_shifts_pcc[0, theta_idx], net_y_shifts_pcc[0, theta_idx], cval = cval)
        
        if net_x_shifts_pcc.ndim == 3:
            center_of_rotation_avg, _, _ = rot_center_avg(aligned_proj[:, start_slice:end_slice], theta_idx_pairs, theta_array)
       
        else:
            center_of_rotation_avg, _, _ = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

        offset = center_of_rotation_avg - center_geom

        print(f'Center of rotation after initial COR correction: {ppu.round_correct(center_of_rotation_avg, ndec = 3)}')
        print(f'Geometric center: {center_geom}')
        print(f'Center of rotation error: {ppu.round_correct(offset, ndec = 3)}')
    
    # add_shift = -0.8

    if cor_correction_only:
        print('Aligning XRT, optical density, XRF projection data after COR correction only...')

        print('Shifting all elements in cropped XRT, optical density aggregate projection arrays by current net shifts...')

        if net_x_shifts_pcc.ndim == 3:
            for theta_idx in range(n_theta):
                net_x_shift = net_x_shifts_pcc[0, theta_idx]
                net_y_shift = net_y_shifts_pcc[0, theta_idx]

                aligned_proj_total_xrt[theta_idx] = warp_shift(xrt_proj_img_array[theta_idx], net_x_shift, net_y_shift, cval = I0)
                aligned_proj_total_opt_dens[theta_idx] = warp_shift(opt_dens_proj_img_array[theta_idx], net_x_shift, net_y_shift)
        
        else:
            for theta_idx in range(n_theta):
                net_x_shift = net_x_shifts_pcc[0, theta_idx]
                net_y_shift = net_y_shifts_pcc[0, theta_idx]

                aligned_proj_total_xrt[theta_idx] = warp_shift(xrt_proj_img_array[theta_idx], net_x_shift, net_y_shift, cval = I0)
                aligned_proj_total_opt_dens[theta_idx] = warp_shift(opt_dens_proj_img_array[theta_idx], net_x_shift, net_y_shift)

        print('Shifting all elements in cropped XRF aggregate projection array by current net shifts...')
        
        if net_x_shifts_pcc.ndim == 3:
            for element_idx in range(n_elements_xrf):
                print(f'\rElement {element_idx + 1}/{n_elements_xrf}', end = '', flush = True)
                
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pcc[0, theta_idx]
                    net_y_shift = net_y_shifts_pcc[0, theta_idx]

                    aligned_proj_total_xrf[element_idx, theta_idx] = warp_shift(xrf_proj_img_array[element_idx, theta_idx].copy(), net_x_shift, net_y_shift)
        
        else:
            for element_idx in range(n_elements_xrf):
                print(f'\rElement {element_idx + 1}/{n_elements_xrf}', end = '', flush = True)
                
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pcc[0, theta_idx]
                    net_y_shift = net_y_shifts_pcc[0, theta_idx]

                    aligned_proj_total_xrf[element_idx, theta_idx] = warp_shift(xrf_proj_img_array[element_idx, theta_idx], net_x_shift, net_y_shift)

        if net_x_shifts_pcc.ndim == 3:
            print('\nTruncating projection images in y so object is in every projection image\'s field of view...')

            aligned_proj_total_xrf = aligned_proj_total_xrf[:, :, start_slice:end_slice]
            aligned_proj_total_xrt = aligned_proj_total_xrt[:, start_slice:end_slice]
            aligned_proj_total_opt_dens = aligned_proj_total_opt_dens[:, start_slice:end_slice]

            return aligned_proj_total_xrt, \
                   aligned_proj_total_opt_dens, \
                   aligned_proj_total_xrf, \
                   net_x_shifts_pcc[0], \
                   net_y_shifts_pcc[0], \
                   None, \
                   None, \
                   None, \
                   None, \
                   None, \
                   None, \
                   None, \

        return aligned_proj_total_xrt, \
               aligned_proj_total_opt_dens, \
               aligned_proj_total_xrf, \
               net_x_shifts_pcc[0], \
               net_y_shifts_pcc[0], \
               None, \
               None, \
               None, \
               None, \
               None, \
               None, \
               None
    
    if eps_iter_reproj is None:
        print('Warning: Nonzero, positive \'eps_iter_reproj\' not detected. Setting \'eps_iter_reproj\' to 1e-8...')

        eps_iter_reproj = 1e-3
    
    if eps_iter_reproj < 0:
        print('Error: \'eps_iter_reproj\' must be a positive number. Exiting program...')

        sys.exit()
    
    if pixel_rad_iter_reproj is None:
        print('Warning: \'pixel_rad_iter_reproj\' not detected. Setting \'pixel_rad_iter_reproj\' to 0 pixels for each projection angle...')

        pixel_rad_iter_reproj = np.zeros(n_theta)

    if return_aux_data:
        aligned_exp_proj_array = np.zeros((n_iterations_iter_reproj, n_theta, n_slices, n_columns))
        synth_proj_array = np.zeros((n_iterations_iter_reproj, n_theta, n_slices, n_columns))
        pcc_2d_array = np.zeros((n_iterations_iter_reproj, n_theta, n_slices, n_columns))
        
        if np.any(pixel_rad_iter_reproj == 0):
            pcc_2d_truncated_array = np.zeros((n_iterations_iter_reproj, n_theta, n_slices, n_columns))
        
        else:
            if not isinstance(pixel_rad_iter_reproj, np.ndarray) or pixel_rad_iter_reproj.ndim != 1:
                print('Error: \'pixel_rad_iter_reproj\' must be a 1D numpy array. Exiting program...')

                sys.exit()

            if pixel_rad_iter_reproj.shape[0] != n_theta:
                print('Error: \'pixel_rad_iter_reproj\' must have the same number of elements as the number of theta angles. Exiting program...')

                sys.exit()
        
            if np.any(pixel_rad_iter_reproj == 0):
                print('Warning: \'pixel_rad_iter_reproj\' is 0. Performing peak search without truncation...')

                pcc_2d_truncated_array = np.zeros((n_iterations_iter_reproj, n_theta, n_slices, n_columns))

            else:
                pcc_2d_truncated_array = np.zeros((n_iterations_iter_reproj, n_theta, 2*pixel_rad_iter_reproj.max(), 2*pixel_rad_iter_reproj.max()))

        recon_array = np.zeros((n_iterations_iter_reproj, n_slices, n_columns, n_columns))
    
    n_iter_converge = 2

    synth_proj = np.zeros((n_theta, n_slices, n_columns))
    dx_array_pcc = np.zeros((n_iterations_iter_reproj, n_theta))
    
    # if net_x_shifts_pcc.ndim != 3:
    dy_array_pcc = np.zeros((n_iterations_iter_reproj, n_theta))

    rms_net_shift_prev = 0

    for i in range(n_iterations_iter_reproj):
        print(f'Iteration {i + 1}/{n_iterations_iter_reproj}')
        
        if i > 0:
            if net_x_shifts_pcc.ndim == 3:
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pcc[i - 1, theta_idx]
                    net_y_shift = net_y_shifts_pcc[i - 1, theta_idx]

                    aligned_proj[theta_idx] = warp_shift(proj_img_array_element_to_align_with[theta_idx], net_x_shift, net_y_shift, cval = cval)
                
                if sample_flipped_remounted_mid_experiment:
                    center_of_rotation_avg_first_part, _, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1], start_slice:end_slice], theta_idx_pairs_first_part, theta_array_first_part)

                    center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:, start_slice:end_slice], theta_idx_pairs_second_part, theta_array_second_part)

                    shifts, _, _ = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[0], start_slice:end_slice], 
                                                      aligned_proj[zero_deg_idx_array[1], start_slice:end_slice], 
                                                      sigma, 
                                                      alpha, 
                                                      pixel_rad_cor_correction,
                                                      theta = np.array([0, 0]))

                    print(f'Experimental center of rotation after horizontal jitter correction (before flipping sample): {center_of_rotation_avg_first_part}')
                    print(f'Experimental center of rotation after horizontal jitter correction (after flipping sample): {center_of_rotation_avg_second_part}\n')
                    print(f'Geometric center: {center_geom}')
                    print(f'Center of rotation error after horizontal jitter correction (before flipping sample): {ppu.round_correct(offset_first_part, ndec = 3)}')
                    print(f'Center of rotation error after horizontal jitter correction (after flipping sample): {ppu.round_correct(offset_second_part, ndec = 3)}\n')
                    print(f'The two zero degree angles are offset from each other by {ppu.round_correct(np.abs(shifts[1]), ndec = 3)} pixels')
                
                else:
                    center_of_rotation_avg, _, offset = rot_center_avg(aligned_proj[:, start_slice:end_slice], theta_idx_pairs, theta_array)

                    print(f'Center of rotation after horizontal jitter correction: {ppu.round_correct(center_of_rotation_avg, ndec = 3)}')
                    print(f'Geometric center: {center_geom}')
                    print(f'Center of rotation error after horizontal jitter correction: {ppu.round_correct(offset, ndec = 3)}')

            else:
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pcc[i - 1, theta_idx]
                    net_y_shift = net_y_shifts_pcc[i - 1, theta_idx]

                    aligned_proj[theta_idx] = ndi.shift(proj_img_array_element_to_align_with[theta_idx], shift = (net_y_shift, net_x_shift))

                if sample_flipped_remounted_mid_experiment:
                    center_of_rotation_avg_first_part, _, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1], start_slice:end_slice], theta_idx_pairs_first_part, theta_array_first_part)

                    center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:, start_slice:end_slice], theta_idx_pairs_second_part, theta_array_second_part)

                    shifts, _, _ = phase_xcorr_manual(aligned_proj[zero_deg_idx_array[0], start_slice:end_slice], 
                                                      aligned_proj[zero_deg_idx_array[1], start_slice:end_slice], 
                                                      sigma, 
                                                      alpha,
                                                      pixel_rad_cor_correction,
                                                      theta = np.array([0, 0]))

                    print(f'Experimental center of rotation after horizontal jitter correction (before flipping sample): {center_of_rotation_avg_first_part}')
                    print(f'Experimental center of rotation after horizontal jitter correction (after flipping sample): {center_of_rotation_avg_second_part}\n')
                    print(f'Geometric center: {center_geom}')
                    print(f'Center of rotation error after horizontal jitter correction (before flipping sample): {ppu.round_correct(offset_first_part, ndec = 3)}')
                    print(f'Center of rotation error after horizontal jitter correction (after flipping sample): {ppu.round_correct(offset_second_part, ndec = 3)}\n')
                    print(f'The two zero degree angles are offset from each other by {ppu.round_correct(np.abs(shifts[1]), ndec = 3)} pixels')
                
                else:
                    center_of_rotation_avg, _, offset = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

                    print(f'Center of rotation after horizontal jitter correction: {ppu.round_correct(center_of_rotation_avg, ndec = 3)}')
                    print(f'Geometric center: {center_geom}')
                    print(f'Center of rotation error after horizontal jitter correction: {ppu.round_correct(offset, ndec = 3)}')
            
        if return_aux_data:
            aligned_exp_proj_array[i] = aligned_proj

        recon = tomo.recon(aligned_proj, theta_array*np.pi/180, algorithm = 'gridrec', filter_name = 'ramlak')
        # recon = tomo.recon(aligned_proj, theta_array*np.pi/180, algorithm = 'mlem', num_iter = 70)

        if return_aux_data:
            recon_array[i] = recon
        
        for slice_idx in range(n_slices):
            print(f'\rReprojecting slice {slice_idx + 1}/{n_slices}', end = '', flush = True)

            sinogram = (xform.radon(recon[slice_idx].copy(), theta_array)).T

            synth_proj[:, slice_idx, :] = sinogram

        if return_aux_data:
            synth_proj_array[i] = synth_proj
        
        for theta_idx in range(n_theta):
            if not return_aux_data:
                shifts, _, _ = phase_xcorr_manual(synth_proj[theta_idx], 
                                                  aligned_proj[theta_idx], 
                                                  sigma, 
                                                  alpha, 
                                                  pixel_rad_iter_reproj[theta_idx],
                                                  theta = np.array([theta_array[theta_idx], theta_array[theta_idx]]))
            
            else:
                shifts, phase_xcorr_2d, phase_xcorr_2d_truncated = phase_xcorr_manual(synth_proj[theta_idx], 
                                                                                      aligned_proj[theta_idx], 
                                                                                      sigma, 
                                                                                      alpha, 
                                                                                      pixel_rad_iter_reproj[theta_idx],
                                                                                      theta = np.array([theta_array[theta_idx], theta_array[theta_idx]]))
                
                pcc_2d_array[i, theta_idx] = phase_xcorr_2d

                if np.any(pixel_rad_iter_reproj[theta_idx] > 0):
                    start_x = int(phase_xcorr_2d_truncated.shape[1]//2 - pixel_rad_iter_reproj[theta_idx])
                    start_y = int(phase_xcorr_2d_truncated.shape[0]//2 - pixel_rad_iter_reproj[theta_idx])

                    end_x = int(phase_xcorr_2d_truncated.shape[1]//2 + pixel_rad_iter_reproj[theta_idx])
                    end_y = int(phase_xcorr_2d_truncated.shape[0]//2 + pixel_rad_iter_reproj[theta_idx])
                
                    pcc_2d_truncated_array[i, theta_idx, start_y:end_y, start_x:end_x] = phase_xcorr_2d_truncated
                
                else:
                    pcc_2d_truncated_array[i, theta_idx] = phase_xcorr_2d_truncated
            
            dy, dx = shifts[0], shifts[1]
            
            if i == 0:
                net_x_shifts_pcc[i, theta_idx] += dx
                net_y_shifts_pcc[i, theta_idx] += dy
                
            else:
                net_x_shifts_pcc[i, theta_idx] = net_x_shifts_pcc[i - 1, theta_idx] + dx
                net_y_shifts_pcc[i, theta_idx] = net_y_shifts_pcc[i - 1, theta_idx] + dy
            
            dx_array_pcc[i, theta_idx] = dx
            dy_array_pcc[i, theta_idx] = dy
            
            if (theta_idx % 7) == 0:
                if theta_idx == 0:
                    print(f'\nCurrent x-shift: {ppu.round_correct(dx, ndec = 3)} (theta = {ppu.round_correct(theta_array[theta_idx], ndec = 1)})')
                
                else:
                    print(f'Current x-shift: {ppu.round_correct(dx, ndec = 3)} (theta = {ppu.round_correct(theta_array[theta_idx], ndec = 1)})')
                
                print(f'Current y-shift: {ppu.round_correct(dy, ndec = 3)}')

        if not sample_flipped_remounted_mid_experiment:
            if net_x_shifts_pcc.ndim == 3:
                center_of_rotation_avg_synth, _, offset_synth = rot_center_avg(synth_proj[:, start_slice:end_slice], theta_idx_pairs, theta_array)
            
            else:
                center_of_rotation_avg_synth, _, offset_synth = rot_center_avg(synth_proj, theta_idx_pairs, theta_array)

            print(f'Average synthetic center of rotation after jitter correction: {ppu.round_correct(center_of_rotation_avg_synth, ndec = 3)}')
            print(f'Geometric center: {center_geom}')
            print(f'Center of rotation error: {ppu.round_correct(offset_synth, ndec = 3)}')
        
        else:
            if net_x_shifts_pcc.ndim == 3:                
                center_of_rotation_avg_first_part, center_geom, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1], start_slice:end_slice], 
                                                                                                   theta_idx_pairs_first_part, 
                                                                                                   theta_array_first_part)
                
                center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:, start_slice:end_slice], 
                                                                                           theta_idx_pairs_second_part, 
                                                                                           theta_array_second_part)

                shifts, _, _ = phase_xcorr_manual(synth_proj[zero_deg_idx_array[0], start_slice:end_slice], 
                                                  synth_proj[zero_deg_idx_array[1], start_slice:end_slice], 
                                                  sigma, 
                                                  alpha, 
                                                  pixel_rad_cor_correction,
                                                  theta = np.array([0, 0]))
            
            else:
                center_of_rotation_avg_first_part, center_geom, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1]], 
                                                                                                   theta_idx_pairs_first_part, 
                                                                                                   theta_array_first_part)
                
                center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:], 
                                                                                           theta_idx_pairs_second_part, 
                                                                                           theta_array_second_part)

                shifts, _, _ = phase_xcorr_manual(synth_proj[zero_deg_idx_array[0]], 
                                                  synth_proj[zero_deg_idx_array[1]], 
                                                  sigma, 
                                                  alpha, 
                                                  pixel_rad_cor_correction,
                                                  theta = np.array([0, 0]))

            print(f'Synthetic center of rotation after jitter correction (before flipping sample): {center_of_rotation_avg_first_part}')
            print(f'Synthetic center of rotation after jitter correction (after flipping sample): {center_of_rotation_avg_second_part}\n')
            print(f'Geometric center: {center_geom}\n')
            print(f'Center of rotation error after jitter correction (before flipping sample): {ppu.round_correct(offset_first_part, ndec = 3)}')
            print(f'Center of rotation error after jitter correction (after flipping sample): {ppu.round_correct(offset_second_part, ndec = 3)}\n')
            print(f'The two zero degree angles are offset from each other by {ppu.round_correct(np.abs(shifts[1]), ndec = 3)} pixels')

        rms_net_x_shift = np.sqrt((dx_array_pcc[i]**2).mean())
        rms_net_y_shift = np.sqrt((dy_array_pcc[i]**2).mean())
        
        rms_net_shift = np.sqrt(rms_net_x_shift**2 + rms_net_y_shift**2)

        if i > (n_iter_converge - 1):
            if np.abs(rms_net_shift - rms_net_shift_prev) <= eps_iter_reproj:
                net_x_shifts_pcc = net_x_shifts_pcc[:(i + 1)]
                net_y_shifts_pcc = net_y_shifts_pcc[:(i + 1)]

                if return_aux_data:
                    pcc_2d_array = pcc_2d_array[:(i + 1)]
                    pcc_2d_truncated_array = pcc_2d_truncated_array[:(i + 1)]
                    aligned_exp_proj_array = aligned_exp_proj_array[:(i + 1)]
                    recon_array = recon_array[:(i + 1)]
                    synth_proj_array = synth_proj_array[:(i + 1)]

                    dx_array_pcc = dx_array_pcc[:(i + 1)]
                    dy_array_pcc = dy_array_pcc[:(i + 1)]
                   
                print(f'Number of iterations taken: {i + 1}')
                print('Shifting all elements in cropped XRT, optical density aggregate projection arrays by current net shifts...')

                if net_x_shifts_pcc.ndim == 3:
                    for theta_idx in range(n_theta):
                        net_x_shift = net_x_shifts_pcc[i, theta_idx]
                        net_y_shift = net_y_shifts_pcc[i, theta_idx]

                        aligned_proj_total_xrt[theta_idx] = warp_shift(xrt_proj_img_array[theta_idx], net_x_shift, net_y_shift, cval = I0)
                        aligned_proj_total_opt_dens[theta_idx] = warp_shift(opt_dens_proj_img_array[theta_idx], net_x_shift, net_y_shift)
                    
                    print('Shifting all elements in cropped XRF aggregate projection array by current net shifts...')
                        
                    for element_idx in range(n_elements_xrf):
                        print(f'\rElement {element_idx + 1}/{n_elements_xrf}', end = '', flush = True)
                        
                        for theta_idx in range(n_theta):
                            net_x_shift = net_x_shifts_pcc[i, theta_idx]
                            net_y_shift = net_y_shifts_pcc[i, theta_idx]

                            aligned_proj_total_xrf[element_idx, theta_idx] = warp_shift(xrf_proj_img_array[element_idx, theta_idx], net_x_shift, net_y_shift)
                    
                    print('\nTruncating cropped XRT, OD, XRF projection images in y so object is in every projection image\'s field of view...')

                    aligned_proj_total_xrf = aligned_proj_total_xrf[:, :, start_slice:end_slice]
                    aligned_proj_total_xrt = aligned_proj_total_xrt[:, start_slice:end_slice]
                    aligned_proj_total_opt_dens = aligned_proj_total_opt_dens[:, start_slice:end_slice]
                
                else:
                    for theta_idx in range(n_theta):
                        net_x_shift = net_x_shifts_pcc[i, theta_idx]
                        net_y_shift = net_y_shifts_pcc[i, theta_idx]

                        aligned_proj_total_xrt[theta_idx] = warp_shift(xrt_proj_img_array[theta_idx], net_x_shift, net_y_shift, cval = I0)
                        aligned_proj_total_opt_dens[theta_idx] = warp_shift(opt_dens_proj_img_array[theta_idx], net_x_shift, net_y_shift)

                    for element_idx in range(n_elements_xrf):
                        print(f'\rElement {element_idx + 1}/{n_elements_xrf}', end = '', flush = True)
                        
                        for theta_idx in range(n_theta):
                            net_x_shift = net_x_shifts_pcc[i, theta_idx]
                            net_y_shift = net_y_shifts_pcc[i, theta_idx]

                            aligned_proj_total_xrf[element_idx, theta_idx] = warp_shift(xrf_proj_img_array[element_idx, theta_idx], net_x_shift, net_y_shift)
                    
                    if np.any(net_y_shifts_pcc[i]):
                        print("\nTruncating cropped XRT, OD, XRF projection images in y so object is in every projection image's field of view...")

                        dy_min, dy_max = np.min(net_y_shifts_pcc[i]), np.max(net_y_shifts_pcc[i])

                        start_slice = int(np.clip(np.ceil(dy_max), 0, n_slices))
                        end_slice = int(np.clip(n_slices + np.floor(dy_min), 0, n_slices))

                        if end_slice <= start_slice:
                            print('Error: Empty field of view detected - net shifts exceed the number of slices. Exiting program...')

                            sys.exit()

                        aligned_proj_total_xrt = aligned_proj_total_xrt[:, start_slice:end_slice]
                        aligned_proj_total_opt_dens = aligned_proj_total_opt_dens[:, start_slice:end_slice]
                        aligned_proj_total_xrf = aligned_proj_total_xrf[:, :, start_slice:end_slice]

                print('Done')

                break

        if i == n_iterations_iter_reproj - 1:
            print('Iterative reprojection complete. Shifting XRT intensities and optical densities by current net shifts...')
            
            if net_x_shifts_pcc.ndim == 3:
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pcc[i, theta_idx]
                    net_y_shift = net_y_shifts_pcc[i, theta_idx]

                    aligned_proj_total_xrt[theta_idx] = warp_shift(xrt_proj_img_array[theta_idx], net_x_shift, net_y_shift, cval = I0)
                    aligned_proj_total_opt_dens[theta_idx] = warp_shift(opt_dens_proj_img_array[theta_idx], net_x_shift, net_y_shift)

                print('Shifting all XRF elements by current net shifts...')

                for element_idx in range(n_elements_xrf):
                    print(f'\rElement {element_idx + 1}/{n_elements_xrf}', end = '', flush = True)

                    for theta_idx in range(n_theta):
                        net_x_shift = net_x_shifts_pcc[i, theta_idx]
                        net_y_shift = net_y_shifts_pcc[i, theta_idx]

                        aligned_proj_total_xrf[element_idx, theta_idx] = warp_shift(xrf_proj_img_array[element_idx, theta_idx], net_x_shift, net_y_shift)
                
                print('\nTruncating cropped XRT, OD, XRF projection images in y so object is in every projection image\'s field of view...')

                aligned_proj_total_xrf = aligned_proj_total_xrf[:, :, start_slice:end_slice]
                aligned_proj_total_xrt = aligned_proj_total_xrt[:, start_slice:end_slice]
                aligned_proj_total_opt_dens = aligned_proj_total_opt_dens[:, start_slice:end_slice]
            
            else:   
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pcc[i, theta_idx]
                    net_y_shift = net_y_shifts_pcc[i, theta_idx]

                    aligned_proj_total_xrt[theta_idx] = warp_shift(xrt_proj_img_array[theta_idx], net_x_shift, net_y_shift, cval = I0)
                    aligned_proj_total_opt_dens[theta_idx] = warp_shift(opt_dens_proj_img_array[theta_idx], net_x_shift, net_y_shift)

                print('Shifting all XRF elements by current net shifts...')

                for element_idx in range(n_elements_xrf):
                    print(f'\rElement {element_idx + 1}/{n_elements_xrf}', end = '', flush = True)
                    
                    for theta_idx in range(n_theta):
                        net_x_shift = net_x_shifts_pcc[i, theta_idx]
                        net_y_shift = net_y_shifts_pcc[i, theta_idx]

                        aligned_proj_total_xrf[element_idx, theta_idx] = warp_shift(xrf_proj_img_array[element_idx, theta_idx], net_x_shift, net_y_shift)
            
                if np.any(net_y_shifts_pcc[i]): # Any nonzero y shifts
                    print("\nTruncating cropped XRT, OD, XRF projection images in y so object is in every projection image's field of view...")

                    dy_min, dy_max = np.min(net_y_shifts_pcc[i]), np.max(net_y_shifts_pcc[i])

                    start_slice = int(np.clip(np.ceil(dy_max), 0, n_slices))
                    end_slice = int(np.clip(n_slices + np.floor(dy_min), 0, n_slices))

                    if end_slice <= start_slice:
                        print('Error: Empty field of view detected - net shifts exceed the number of slices. Exiting program...')

                        sys.exit()
                
                    aligned_proj_total_xrt = aligned_proj_total_xrt[:, start_slice:end_slice]
                    aligned_proj_total_opt_dens = aligned_proj_total_opt_dens[:, start_slice:end_slice]
                    aligned_proj_total_xrf = aligned_proj_total_xrf[:, :, start_slice:end_slice]

            print('Done')

            break
        
        rms_net_shift_prev = rms_net_shift

    if return_aux_data:
        if net_x_shifts_pcc.ndim == 3:
            return aligned_proj_total_xrt, \
                   aligned_proj_total_opt_dens, \
                   aligned_proj_total_xrf, \
                   net_x_shifts_pcc, \
                   net_y_shifts_pcc, \
                   recon_array, \
                   aligned_exp_proj_array, \
                   synth_proj_array, \
                   pcc_2d_array, \
                   pcc_2d_truncated_array, \
                   dx_array_pcc, \
                   None
        
        return aligned_proj_total_xrt, \
               aligned_proj_total_opt_dens, \
               aligned_proj_total_xrf, \
               net_x_shifts_pcc, \
               net_y_shifts_pcc, \
               recon_array, \
               aligned_exp_proj_array, \
               synth_proj_array, \
               pcc_2d_array, \
               pcc_2d_truncated_array, \
               dx_array_pcc, \
               dy_array_pcc
        
    return aligned_proj_total_xrt, \
           aligned_proj_total_opt_dens, \
           aligned_proj_total_xrf, \
           net_x_shifts_pcc, \
           net_y_shifts_pcc, \
           recon_array, \
           None, \
           None, \
           None, \
           None, \
           None, \
           None