import numpy as np, \
       xrf_xrt_preprocess_utils as ppu, \
       xrf_xrt_preprocess_file_util as futil, \
       os

from scipy import fft, ndimage as ndi
from matplotlib import pyplot as plt
from imageio import v2 as iio2

def phase_xcorr_manual(ref_img, mov_img, sigma, alpha):    
    n_slices = ref_img.shape[0]
    n_columns = ref_img.shape[1]
    
    ref_img_filtered = ppu.edge_gauss_filter(ref_img, sigma, alpha, nx = n_columns, ny = n_slices)
    mov_img_filtered = ppu.edge_gauss_filter(mov_img, sigma, alpha, nx = n_columns, ny = n_slices)

    ref_img_fft = fft.fft2(ref_img_filtered)
    mov_img_fft = fft.fft2(mov_img_filtered)

    phase_xcorr = np.abs(fft.ifft2(ref_img_fft*mov_img_fft.conjugate()/np.abs(ref_img_fft*mov_img_fft.conjugate())))

    return fft.fftshift(phase_xcorr)

def xcorr_vert_parabolic_fit(xcorr_img, pixel_rad, theta):
    # pixel_rad = int(np.atleast_1d(pixel_rad).flat[0])  # ensure scalar
    # pixel_rad = max(pixel_rad, 2)  # need at least 4x4 truncated region

    center_slice_idx = xcorr_img.shape[0]//2
    center_column_idx = xcorr_img.shape[1]//2

    start_slice_idx = center_slice_idx - pixel_rad
    end_slice_idx = center_slice_idx + pixel_rad

    start_column_idx = center_column_idx - pixel_rad
    end_column_idx = center_column_idx + pixel_rad

    xcorr_img_truncated = xcorr_img[start_slice_idx:end_slice_idx, start_column_idx:end_column_idx]

    pcc_max_idx = np.unravel_index(np.argmax(xcorr_img_truncated), xcorr_img_truncated.shape)

    n_rows = xcorr_img_truncated.shape[0]

    if pcc_max_idx[0] + 1 < n_rows and pcc_max_idx[0] - 1 >= 0:
        pcc_p = xcorr_img_truncated[pcc_max_idx[0] + 1, pcc_max_idx[1]]
        pcc_0 = xcorr_img_truncated[pcc_max_idx[0], pcc_max_idx[1]]
        pcc_n = xcorr_img_truncated[pcc_max_idx[0] - 1, pcc_max_idx[1]]
        
        denom = pcc_p + pcc_n - 2*pcc_0
        
        subpix_shift = -0.5*(pcc_p - pcc_n)/denom
        
        if not np.isfinite(subpix_shift):
            print('Warning: Subpixel shift is not finite for theta = {0} and {1} degrees. Returning 0 for subpixel shift.'.format(theta[0], theta[1]))
            
            subpix_shift = 0
    
    else:
        print('Warning: Parabolic fit failed (The peak is at an edge or corner of the truncated region) for theta = {0} and {1} degrees. Returning 0 for subpixel shift.'.format(theta[0], theta[1]))
        
        subpix_shift = 0
    # print(start_slice_idx)
    # print(subpix_shift)
    # print(pcc_max_idx[0])
    pcc_max_idx_remapped = start_slice_idx + pcc_max_idx[0] + subpix_shift
    # print(pcc_max_idx_remapped)

    # Match skimage convention: shift = peak - center (negate vs center-peak).
    dy = pcc_max_idx_remapped - center_slice_idx
    
    return xcorr_img_truncated, dy

def create_raw_img_fig(dir_path,
                       opt_dens,
                       counts_xrf_array, 
                       theta_array, 
                       element_array,
                       sigma,
                       alpha,
                       fps):
    
    fig, axs = plt.subplots(3, 2)
    
    n_slices = opt_dens.shape[1]
    n_columns = opt_dens.shape[2]
    
    im1_1 = axs[0, 0].imshow(opt_dens[0], vmin = opt_dens.min(), vmax = opt_dens.max())
    im1_2 = axs[1, 0].imshow(counts_xrf[0, 0], vmin = counts_xrf[0].min(), vmax = counts_xrf[0].max())
    im1_3 = axs[2, 0].imshow(counts_xrf[1, 0], vmin = counts_xrf[1].min(), vmax = counts_xrf[1].max())
    im1_4 = axs[0, 1].imshow(opt_dens[1], vmin = opt_dens.min(), vmax = opt_dens.max())
    im1_5 = axs[1, 1].imshow(counts_xrf[0, 1], vmin = counts_xrf[0].min(), vmax = counts_xrf[0].max())
    im1_6 = axs[2, 1].imshow(counts_xrf[1, 1], vmin = counts_xrf[1].min(), vmax = counts_xrf[1].max())

    for ax in fig.axes:
        ax.axis('off')
        ax.axvline(x = n_columns//2, color = 'red', linewidth = 2)
        ax.axhline(y = n_slices//2, color = 'red', linewidth = 2)
    
    text_1 = axs[1, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs[1, 0].transAxes, color = 'white')
    text_2 = axs[1, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[1]), transform = axs[1, 1].transAxes, color = 'white')

    axs[0, 0].set_title(r'Opt. Dens.', fontsize = 14)
    axs[1, 0].set_title(r'XRF ({0})'.format(element_array[0]), fontsize = 14)
    axs[2, 0].set_title(r'XRF ({0})'.format(element_array[1]), fontsize = 14)
    axs[0, 1].set_title(r'Opt. Dens.', fontsize = 14)
    axs[1, 1].set_title(r'XRF ({0})'.format(element_array[0]), fontsize = 14)
    axs[2, 1].set_title(r'XRF ({0})'.format(element_array[1]), fontsize = 14)

    theta_frames = []

    for theta_idx in range(1, n_theta):
        im1_1.set_data(opt_dens[theta_idx - 1])
        im1_2.set_data(counts_xrf_array[0, theta_idx - 1])
        im1_3.set_data(counts_xrf_array[1, theta_idx - 1])
        im1_4.set_data(opt_dens[theta_idx])
        im1_5.set_data(counts_xrf_array[0, theta_idx])
        im1_6.set_data(counts_xrf_array[1, theta_idx])

        text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx - 1]))
        text_2.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

        fig.canvas.draw()

        frame1 = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        theta_frames.append(frame1)
    
    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'raw_img_comp_sigma_{sigma}_alpha_{alpha}.gif')

    iio2.mimsave(gif_filename, theta_frames, fps = fps)

    return

def create_phase_xcorr_fig(dir_path,
                           phase_xcorr_array_xrf,
                           phase_xcorr_array_xrf_truncated,
                           phase_xcorr_array_opt_dens, 
                           phase_xcorr_array_opt_dens_truncated,
                           element_array,
                           theta_array,
                           sigma,
                           alpha,
                           fps):
    
    n_elements = phase_xcorr_array_xrf.shape[0]
    n_theta_angles = phase_xcorr_array_xrf.shape[1]
    n_slices = phase_xcorr_array_xrf.shape[2]
    n_columns = phase_xcorr_array_xrf.shape[3]

    # n_slices_truncated = phase_xcorr_array_opt_dens_truncated.shape[0]
    # n_columns_truncated = phase_xcorr_array_opt_dens_truncated.shape[1]
    
    
    fig, axs = plt.subplots(3, 2)

    im1_1 = axs[0, 0].imshow((phase_xcorr_array_opt_dens[1]), vmin = phase_xcorr_array_opt_dens.min(), vmax = phase_xcorr_array_opt_dens.max())
    im1_2 = axs[1, 0].imshow((phase_xcorr_array_xrf[0, 1]), vmin = phase_xcorr_array_xrf[0].min(), vmax = phase_xcorr_array_xrf[0].max())
    im1_3 = axs[2, 0].imshow((phase_xcorr_array_xrf[1, 1]), vmin = phase_xcorr_array_xrf[1].min(), vmax = phase_xcorr_array_xrf[1].max())
    im1_4 = axs[0, 1].imshow((phase_xcorr_array_opt_dens_truncated[0]), vmin = phase_xcorr_array_opt_dens.min(), vmax = phase_xcorr_array_opt_dens.max())
    im1_5 = axs[1, 1].imshow((phase_xcorr_array_xrf_truncated[0]), vmin = phase_xcorr_array_xrf[0].min(), vmax = phase_xcorr_array_xrf[0].max())
    im1_6 = axs[2, 1].imshow((phase_xcorr_array_xrf_truncated[n_theta_angles - 1]), vmin = phase_xcorr_array_xrf[1].min(), vmax = phase_xcorr_array_xrf[1].max())

    for ax in fig.axes:
        ax.axis('off')
        
        if ax == axs[0, 0] or ax == axs[1, 0] or ax == axs[2, 0]:
            ax.vlines(x = n_columns//2, ymin = n_slices//2 - 15, ymax = n_slices//2 - 5, color = 'red', linewidth = 2)
            ax.vlines(x = n_columns//2, ymin = n_slices//2 + 5, ymax = n_slices//2 + 15, color = 'red', linewidth = 2)
            ax.hlines(y = n_slices//2, xmin = n_columns//2 - 15, xmax = n_columns//2 - 5, color = 'red', linewidth = 2)
            ax.hlines(y = n_slices//2, xmin = n_columns//2 + 5, xmax = n_columns//2 + 15, color = 'red', linewidth = 2)
        
        # else:
            # ax.vlines(x = n_columns_truncated//2, ymin = n_slices_truncated//2 - 15, ymax = n_slices_truncated//2 - 5, color = 'red', linewidth = 2)
            # ax.vlines(x = n_columns_truncated//2, ymin = n_slices_truncated//2 + 5, ymax = n_slices_truncated//2 + 15, color = 'red', linewidth = 2)
            # ax.hlines(y = n_slices_truncated//2, xmin = n_columns_truncated//2 - 15, xmax = n_columns_truncated//2 - 5, color = 'red', linewidth = 2)
            # ax.hlines(y = n_slices_truncated//2, xmin = n_columns_truncated//2 + 5, xmax = n_columns_truncated//2 + 15, color = 'red', linewidth = 2)

    axs[0, 0].set_title(r'Opt. Dens.', fontsize = 14)
    axs[1, 0].set_title(r'XRF ({0})'.format(element_array[0]), fontsize = 14)
    axs[2, 0].set_title(r'XRF ({0})'.format(element_array[1]), fontsize = 14) 
    axs[0, 1].set_title(r'Opt. Dens. (Tr.)', fontsize = 14)
    axs[1, 1].set_title(r'XRF ({0}) (Tr.)'.format(element_array[0]), fontsize = 14)
    axs[2, 1].set_title(r'XRF ({0}) (Tr.)'.format(element_array[1]), fontsize = 14)

    text_1 = axs[1, 0].text(0.02, 0.02, r'$\theta = {0}^{{\circ}}, \theta^{{\prime}} = {1}^{{\circ}}$'.format(theta_array[0], theta_array[1]), transform = axs[1, 0].transAxes, color = 'white')
    
    theta_frames = []

    for theta_idx in range(1, n_theta_angles):
        im1_1.set_data(phase_xcorr_array_opt_dens[theta_idx])
        im1_2.set_data(phase_xcorr_array_xrf[0, theta_idx])
        im1_3.set_data(phase_xcorr_array_xrf[1, theta_idx])
        im1_4.set_data(phase_xcorr_array_opt_dens_truncated[theta_idx - 1])
        im1_5.set_data(phase_xcorr_array_xrf_truncated[theta_idx - 1])
        im1_6.set_data(phase_xcorr_array_xrf_truncated[theta_idx + n_theta_angles - 2])

        text_1.set_text(r'$\theta = {0}^{{\circ}}, \theta^{{\prime}} = {1}^{{\circ}}$'.format(theta_array[theta_idx - 1], theta_array[theta_idx]))

        fig.canvas.draw()

        frame1 = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        theta_frames.append(frame1)

    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'phase_xcorr_comp_sigma_{sigma}_alpha_{alpha}.gif')

    iio2.mimsave(gif_filename, theta_frames, fps = fps)

    return

def create_shifted_img_fig(dir_path,
                           counts_xrf,
                           opt_dens,
                           shifted_counts_xrf,
                           shifted_opt_dens,
                           desired_element_array,
                           theta_array,
                           sigma,
                           alpha,
                           fps):
    
    fig, axs = plt.subplots(3, 3)

    n_theta = counts_xrf.shape[1]
    n_slices = counts_xrf.shape[2]
    n_columns = counts_xrf.shape[3]

    im1_1 = axs[0, 0].imshow(opt_dens[0])
    im1_2 = axs[1, 0].imshow(counts_xrf[0, 0])
    im1_3 = axs[2, 0].imshow(counts_xrf[1, 0])
    im1_4 = axs[0, 1].imshow(opt_dens[1])
    im1_5 = axs[1, 1].imshow(counts_xrf[0, 1])
    im1_6 = axs[2, 1].imshow(counts_xrf[1, 1])
    im1_7 = axs[0, 2].imshow(shifted_opt_dens[0])
    im1_8 = axs[1, 2].imshow(shifted_counts_xrf[0, 0])
    im1_9 = axs[2, 2].imshow(shifted_counts_xrf[1, 0])

    for ax in fig.axes:
        ax.axis('off')
        ax.axvline(x = n_columns//2, color = 'red', linewidth = 2)
        ax.axhline(y = n_slices//2, color = 'red', linewidth = 2)

    axs[0, 0].set_title(r'Opt. Dens.', fontsize = 14)
    axs[1, 0].set_title(r'XRF ({0})'.format(desired_element_array[0]), fontsize = 14)
    axs[2, 0].set_title(r'XRF ({0})'.format(desired_element_array[1]), fontsize = 14)
    axs[0, 1].set_title(r'Opt. Dens. (Tr.)', fontsize = 14)
    axs[1, 1].set_title(r'XRF ({0}) (Tr.)'.format(desired_element_array[0]), fontsize = 14)
    axs[2, 1].set_title(r'XRF ({0}) (Tr.)'.format(desired_element_array[1]), fontsize = 14)
    axs[0, 2].set_title(r'Sh. Opt. Dens.', fontsize = 14)
    axs[1, 2].set_title(r'Sh. XRF ({0})'.format(desired_element_array[0]), fontsize = 14)
    axs[2, 2].set_title(r'Sh. XRF ({0})'.format(desired_element_array[1]), fontsize = 14)

    text_1 = axs[1, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs[1, 0].transAxes, color = 'white')
    text_2 = axs[1, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[1]), transform = axs[1, 1].transAxes, color = 'white')
    text_3 = axs[1, 2].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[1]), transform = axs[1, 2].transAxes, color = 'white')
    
    theta_frames = []
    
    for theta_idx in range(1, n_theta):
        im1_1.set_data(opt_dens[theta_idx - 1])
        im1_2.set_data(counts_xrf[0, theta_idx - 1])
        im1_3.set_data(counts_xrf[1, theta_idx - 1])
        im1_4.set_data(opt_dens[theta_idx])
        im1_5.set_data(counts_xrf[0, theta_idx])
        im1_6.set_data(counts_xrf[1, theta_idx])
        im1_7.set_data(shifted_opt_dens[theta_idx])
        im1_8.set_data(shifted_counts_xrf[0, theta_idx])
        im1_9.set_data(shifted_counts_xrf[1, theta_idx])
        
        text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx - 1]))
        text_2.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))
        text_3.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

        fig.canvas.draw()

        frame1 = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        theta_frames.append(frame1)
    
    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'shifted_img_comp_sigma_{sigma}_alpha_{alpha}.gif')

    iio2.mimsave(gif_filename, theta_frames, fps = fps)

    return

def create_common_fov_fig(dir_path,
                          counts_xrf,
                          counts_opt_dens,
                          shifted_counts_xrf,
                          shifted_opt_dens,
                          desired_element_array,
                          theta_array,
                          sigma,
                          alpha,
                          fps):
    
    fig, axs = plt.subplots(3, 1)

    n_theta, n_slices, n_columns = counts_opt_dens.shape

    im1_1 = axs[0].imshow(shifted_opt_dens[0], vmin = counts_opt_dens.min(), vmax = counts_opt_dens.max())
    im1_2 = axs[1].imshow(shifted_counts_xrf[0][0], vmin = counts_xrf[0].min(), vmax = counts_xrf[0].max())
    im1_3 = axs[2].imshow(shifted_counts_xrf[1][0], vmin = counts_xrf[1].min(), vmax = counts_xrf[1].max())

    for ax in fig.axes:
        ax.axis('off')
        ax.axvline(x = n_columns//2, color = 'red', linewidth = 2)
        ax.axhline(y = n_slices//2, color = 'red', linewidth = 2)

    axs[0].set_title(r'Opt. Dens.', fontsize = 14)
    axs[1].set_title(r'XRF ({0})'.format(desired_element_array[0]), fontsize = 14)
    axs[2].set_title(r'XRF ({0})'.format(desired_element_array[1]), fontsize = 14)

    text_1 = axs[1].text(0.02, 0.02, r'$\theta$ = {0}$\textdegree'.format(theta_array[0]), transform = axs[1].transAxes, color = 'white')
    
    theta_frames = []
    
    for theta_idx in range(n_theta):
        if np.any(counts_opt_dens[theta_idx] < 0):
            print('Negative values detected in opt. dens. for theta = {0}...'.format(theta_array[theta_idx]))

        im1_1.set_data(shifted_opt_dens[theta_idx])
        im1_2.set_data(shifted_counts_xrf[0][theta_idx])
        im1_3.set_data(shifted_counts_xrf[1][theta_idx])

        text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

        fig.canvas.draw()

        frame1 = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        theta_frames.append(frame1)

    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'common_fov_comp_sigma_{sigma}_alpha_{alpha}.gif')

    iio2.mimsave(gif_filename, theta_frames, fps = fps)

    return

def create_sinogram_fig(dir_path,
                        counts_xrf,
                        counts_opt_dens,
                        shifted_counts_xrf,
                        shifted_opt_dens,
                        desired_element_array,
                        sigma,
                        alpha,
                        fps):
    
    fig, axs = plt.subplots(3, 2, figsize = (7, 14))

    n_theta, n_slices, n_columns = counts_opt_dens.shape
    
    im1_1 = axs[0, 0].imshow(counts_opt_dens[:, 0], vmin = shifted_opt_dens.min(), vmax = shifted_opt_dens.max(), origin = 'lower', extent = [0, n_columns - 1, -180, 180])
    im1_2 = axs[1, 0].imshow(counts_xrf[0, :, 0], vmin = shifted_counts_xrf[0].min(), vmax = shifted_counts_xrf[0].max(), origin = 'lower', extent = [0, n_columns - 1, -180, 180])
    im1_3 = axs[2, 0].imshow(counts_xrf[1, :, 0], vmin = shifted_counts_xrf[1].min(), vmax = shifted_counts_xrf[1].max(), origin = 'lower', extent = [0, n_columns - 1, -180, 180])
    im1_4 = axs[0, 1].imshow(shifted_opt_dens[:, 0], vmin = shifted_opt_dens.min(), vmax = shifted_opt_dens.max(), origin = 'lower', extent = [0, n_columns - 1, -180, 180])
    im1_5 = axs[1, 1].imshow(shifted_counts_xrf[0, :, 0], vmin = shifted_counts_xrf[0].min(), vmax = shifted_counts_xrf[0].max(), origin = 'lower', extent = [0, n_columns - 1, -180, 180])
    im1_6 = axs[2, 1].imshow(shifted_counts_xrf[1, :, 0], vmin = shifted_counts_xrf[1].min(), vmax = shifted_counts_xrf[1].max(), origin = 'lower', extent = [0, n_columns - 1, -180, 180])

    axs[2, 0].set_xlabel(r'Pixel index', fontsize = 14)
    axs[2, 1].set_xlabel(r'Pixel index', fontsize = 14)
    axs[0, 0].set_ylabel(r'$\theta$ (\textdegree)', fontsize = 14)
    axs[1, 0].set_ylabel(r'$\theta$ (\textdegree)', fontsize = 14)
    axs[2, 0].set_ylabel(r'$\theta$ (\textdegree)', fontsize = 14)
        
    axs[0, 0].set_title(r'Opt. Dens.', fontsize = 14)
    axs[1, 0].set_title(r'XRF ({0})'.format(desired_element_array[0]), fontsize = 14)
    axs[2, 0].set_title(r'XRF ({0})'.format(desired_element_array[1]), fontsize = 14)
    axs[0, 1].set_title(r'Opt. Dens. (Sh.)', fontsize = 14)
    axs[1, 1].set_title(r'XRF ({0}) (Sh.)'.format(desired_element_array[0]), fontsize = 14)
    axs[2, 1].set_title(r'XRF ({0}) (Sh.)'.format(desired_element_array[1]), fontsize = 14)

    text_1 = axs[1, 0].text(0.02, 0.02, r'Slice index 0/{0}'.format(n_slices - 1), transform = axs[1, 0].transAxes, color = 'white')
    
    slice_frames = []
    
    for slice_idx in range(n_slices):
        im1_1.set_data(opt_dens[:, slice_idx])
        im1_2.set_data(counts_xrf[0, :, slice_idx])
        im1_3.set_data(counts_xrf[1, :, slice_idx])
        im1_4.set_data(shifted_opt_dens[:, slice_idx])
        im1_5.set_data(shifted_counts_xrf[0, :, slice_idx])
        im1_6.set_data(shifted_counts_xrf[1, :, slice_idx])
        
        text_1.set_text(r'Slice index {0}/{1}'.format(slice_idx, n_slices - 1))

        fig.canvas.draw()

        frame1 = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        slice_frames.append(frame1)
    
    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'shifted_sinogram_comp_sigma_{sigma}_alpha_{alpha}.gif')

    iio2.mimsave(gif_filename, slice_frames, fps = fps)

    return

sigma = 25
alpha = 10

aggregate_xrf_h5_file_path = '/Users/bwr0835/Documents/3_id_aggregate_xrf.h5'
aggregate_xrt_h5_file_path = '/Users/bwr0835/Documents/3_id_aggregate_xrt.h5'

dir_path = '/Users/bwr0835/Documents/3_id_realigned_data_02_10_2026'

elements_xrf, counts_xrf, theta, _, dataset_type = futil.extract_h5_aggregate_xrf_data(aggregate_xrf_h5_file_path)
elements_xrt, counts_xrt, theta_xrt, _, dataset_type = futil.extract_h5_aggregate_xrt_data(aggregate_xrt_h5_file_path)

_, n_theta, n_slices, n_columns = counts_xrf.shape

if (n_slices % 2) or (n_columns % 2):
    if (n_slices % 2) and (n_columns % 2):
        print('Odd number of slices (rows) and scan positions (columns) detected. Padding one additional slice and scan position column to XRF and XRT data...')

        counts_xrt = ppu.pad_col_row(counts_xrt, dataset_type)
        counts_xrf = ppu.pad_col_row(counts_xrf, dataset_type)
            
        n_slices += 1
        n_columns += 1
        
    elif n_slices % 2:
        print('Odd number of slices (rows) detected. Padding one additional slice to XRF and XRT data...')
                
        counts_xrt = ppu.pad_row(counts_xrt, dataset_type)
        counts_xrf = ppu.pad_row(counts_xrf, dataset_type)

        n_slices += 1

    else:
        print('Odd number of scan positions (columns) detected. Padding one additional scan position column to XRF and XRT data...')
                
        counts_xrt = ppu.pad_col(counts_xrt, dataset_type)
        counts_xrf = ppu.pad_col(counts_xrf, dataset_type)

        n_columns += 1

counts_xrt_sig_idx = elements_xrt.index('xrt_sig')
counts_xrt_sig = counts_xrt[counts_xrt_sig_idx]

desired_elements_xrf = ['Ni', 'Cu']
# desired_theta = [-150, -147]
desired_theta = [45, 55]

element_idx_array = []
theta_idx_array = []

for element_idx in desired_elements_xrf:
    element_idx_array.append(elements_xrf.index(element_idx))

for theta_idx in range(n_theta):
    if theta[theta_idx] in desired_theta:
        theta_idx_array.append(theta_idx)

desired_theta_array = theta_xrt[theta_idx_array]
desired_element_array = [elements_xrf[i] for i in element_idx_array]

counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts = ppu.joint_fluct_norm(counts_xrt_sig,
                                                                            counts_xrf,
                                                                            xrt_data_percentile = 80)

opt_dens = -np.log(counts_xrt_norm/I0_cts)
counts_xrf_norm_array = counts_xrf_norm[element_idx_array]

phase_xcorr_array_opt_dens = np.zeros((n_theta, n_slices, n_columns))
phase_xcorr_array_xrf = np.zeros((len(desired_elements_xrf), n_theta, n_slices, n_columns))

pixel_rad_array_xrf = np.full((len(desired_elements_xrf), n_theta), 15)

zero_theta_idx_array = np.where(theta == 0)[0]
theta_147_idx = np.where(theta == -147)[0][0]
theta_3_idx = np.where(theta == 3)[0][0]
theta_135_idx = np.where(theta == -135)[0][0]
theta_45_idx = np.where(theta == 45)[0][0]
theta_55_idx = np.where(theta == 55)[0][0]
theta_120_idx = np.where(theta == 120)[0][0]

pixel_rad_array_opt_dens = np.full(n_theta, 15)
pixel_rad_array_opt_dens[[theta_147_idx, theta_3_idx]] = 35 # Look at second angle in angle pairs only
pixel_rad_array_opt_dens[[theta_135_idx, theta_55_idx, zero_theta_idx_array[1]]] = 5
pixel_rad_array_opt_dens[theta_45_idx] = 6
pixel_rad_array_opt_dens[theta_120_idx] = 7

pixel_rad_array_xrf[:, [theta_147_idx, theta_3_idx]] = 35
pixel_rad_array_xrf[:, theta_55_idx] = 5
pixel_rad_array_xrf[:, theta_120_idx] = 6

dy_array_pcc = np.zeros(n_theta)

for theta_idx in range(1, n_theta):
    phase_xcorr_array_opt_dens[theta_idx] = phase_xcorr_manual(counts_xrt_norm[theta_idx - 1], counts_xrt_norm[theta_idx], sigma, alpha)

for element_idx in range(len(desired_elements_xrf)):
    for theta_idx in range(1, n_theta):
        phase_xcorr_array_xrf[element_idx, theta_idx] = phase_xcorr_manual(counts_xrf_norm_array[element_idx, theta_idx - 1], counts_xrf_norm_array[element_idx, theta_idx], sigma, alpha)

dy_xrf_array = np.zeros((len(desired_elements_xrf), n_theta))
dy_opt_dens_array = np.zeros(n_theta)

phase_xcorr_array_xrf_truncated_list = []
phase_xcorr_array_opt_dens_truncated_list = []

phase_corr_array_xrf_truncated_aggregate = np.zeros((2*pixel_rad_array_xrf.max(), 2*pixel_rad_array_xrf.max()))
phase_corr_array_opt_dens_truncated_aggregate = np.zeros((2*pixel_rad_array_opt_dens.max(), 2*pixel_rad_array_opt_dens.max()))

phase_corr_array_xrf_truncated_aggregate_midpt_idy, \
phase_corr_array_xrf_truncated_aggregate_midpt_idx = phase_corr_array_opt_dens_truncated_aggregate.shape[0]//2, \
                                                     phase_corr_array_opt_dens_truncated_aggregate.shape[1]//2

for element_idx in range(len(desired_elements_xrf)):
    for theta_idx in range(1, n_theta):
        phase_xcorr_array_xrf_truncated, dy_xrf_array[element_idx, theta_idx] = xcorr_vert_parabolic_fit(phase_xcorr_array_xrf[element_idx, theta_idx], pixel_rad_array_xrf[element_idx, theta_idx], theta[[theta_idx - 1, theta_idx]])
        
        start_y = phase_corr_array_xrf_truncated_aggregate_midpt_idy - pixel_rad_array_xrf[element_idx, theta_idx]
        start_x = phase_corr_array_xrf_truncated_aggregate_midpt_idx - pixel_rad_array_xrf[element_idx, theta_idx]

        end_y = phase_corr_array_xrf_truncated_aggregate_midpt_idy + pixel_rad_array_xrf[element_idx, theta_idx]
        end_x = phase_corr_array_xrf_truncated_aggregate_midpt_idx + pixel_rad_array_xrf[element_idx, theta_idx]
        
        phase_corr_array_xrf_truncated_aggregate[start_y:end_y, start_x:end_x] = phase_xcorr_array_xrf_truncated
        
        phase_xcorr_array_xrf_truncated_list.append(phase_xcorr_array_xrf_truncated)

for theta_idx in range(1, n_theta):
    phase_xcorr_array_opt_dens_truncated = np.zeros((2*pixel_rad_array_opt_dens[theta_idx], 2*pixel_rad_array_opt_dens[theta_idx]))
    
    phase_xcorr_array_opt_dens_truncated, dy_opt_dens_array[theta_idx] = xcorr_vert_parabolic_fit(phase_xcorr_array_opt_dens[theta_idx], pixel_rad_array_opt_dens[theta_idx], theta[[theta_idx - 1, theta_idx]])
    
    phase_xcorr_array_opt_dens_truncated_list.append(phase_xcorr_array_opt_dens_truncated)

# dy_pcc_xrf_1, _, _ = reg.phase_cross_correlation(counts_xrf_norm_array[0, 0], counts_xrf_norm_array[0, 1], upsample_factor = 100)
# dy_pcc_xrf_2, _, _ = reg.phase_cross_correlation(counts_xrf_norm_array[1, 0], counts_xrf_norm_array[1, 1], upsample_factor = 100)
# dy_pcc_od, _, _ = reg.phase_cross_correlation(opt_dens[0], opt_dens[1], upsample_factor = 100)


# print(dy_pcc_xrf_1[0])
# print(dy_pcc_xrf_2[0])
# print(dy_pcc_od[0])
# print('\n')
# print(dy_xrf_array)
# print(dy_opt_dens)

shifted_counts_xrf_norm_array = np.zeros((len(desired_elements_xrf), n_theta, n_slices, n_columns))
shifted_opt_dens = np.zeros((n_theta, n_slices, n_columns))

shifted_counts_xrf_norm_array[:, 0] = counts_xrf_norm_array[:, 0]
shifted_opt_dens[0] = opt_dens[0]

start_array_xrf = np.zeros((len(desired_elements_xrf), n_theta), dtype = int)
end_array_xrf = np.zeros((len(desired_elements_xrf), n_theta), dtype = int)

start_array_opt_dens = np.zeros(n_theta, dtype = int)
end_array_opt_dens = np.zeros(n_theta, dtype = int)

start_slice_xrf = np.zeros(len(desired_elements_xrf), dtype = int)
end_slice_xrf = np.zeros(len(desired_elements_xrf), dtype = int)

dy_xrf_cum = dy_xrf_array.copy()
dy_xrf_cum[:, 0] = 0
dy_xrf_cum[:, 1:] = np.cumsum(dy_xrf_array[:, 1:], axis=1)

dy_opt_dens_cum = dy_opt_dens_array.copy()
dy_opt_dens_cum[0] = 0
dy_opt_dens_cum[1:] = np.cumsum(dy_opt_dens_array[1:])

for element_idx in range(len(desired_elements_xrf)):
    for theta_idx in range(1, n_theta):
        # shifted_counts_xrf_norm_array[element_idx, theta_idx] = ndi.shift(counts_xrf_norm_array[element_idx, theta_idx], shift = (dy_xrf_array[element_idx, theta_idx], 0))

        shifted_counts_xrf_norm_array[element_idx, theta_idx] = ndi.shift(counts_xrf_norm_array[element_idx, theta_idx], shift = (dy_xrf_cum[element_idx, theta_idx], 0))

for theta_idx in range(1, n_theta):
    # shifted_opt_dens[theta_idx] = ndi.shift(opt_dens[theta_idx], shift = (dy_opt_dens_array[theta_idx], 0))

    shifted_opt_dens[theta_idx] = ndi.shift(opt_dens[theta_idx], shift = (dy_opt_dens_cum[theta_idx], 0))

per_element_crops = []
per_element_bounds = []

H = n_slices

for element_idx in range(len(desired_elements_xrf)):
    # dy_e = dy_xrf_array[element_idx]
    dy_e = dy_xrf_cum[element_idx]
    dy_min_e = float(np.nanmin(dy_e))
    dy_max_e = float(np.nanmax(dy_e))
    start_elem = int(np.clip(np.ceil(dy_max_e), 0, n_slices))
    end_elem   = int(np.clip(n_slices + np.floor(dy_min_e), 0, n_slices))

    crop_e = shifted_counts_xrf_norm_array[element_idx, :, start_elem:end_elem, :]
    per_element_crops.append(crop_e)          # shape: (n_theta, H_elem, n_columns)    per_element_bounds.append((start_elem, end_elem))


dy_min_opt = float(np.nanmin(dy_opt_dens_array))
dy_max_opt = float(np.nanmax(dy_opt_dens_array))
start_opt = int(np.clip(np.ceil(dy_max_opt), 0, H))
end_opt   = int(np.clip(H + np.floor(dy_min_opt), 0, H))  # exclusive

shifted_opt_dens_common_fov = shifted_opt_dens[:, start_opt:end_opt, :]

fps = 10

create_raw_img_fig(dir_path, opt_dens, counts_xrf_norm_array, theta, desired_element_array, sigma, alpha, fps)
# create_phase_xcorr_fig(dir_path, phase_xcorr_array_xrf, phase_xcorr_array_xrf_truncated_list, phase_xcorr_array_opt_dens, phase_xcorr_array_opt_dens_truncated_list, desired_element_array, theta, sigma, alpha, fps)
# create_shifted_img_fig(dir_path, counts_xrf_norm_array, opt_dens, shifted_counts_xrf_norm_array, shifted_opt_dens, desired_element_array, theta, sigma, alpha, fps)
# create_sinogram_fig(dir_path, counts_xrf_norm_array, opt_dens, shifted_counts_xrf_norm_array, shifted_opt_dens, desired_element_array, sigma, alpha, fps)
create_common_fov_fig(dir_path, counts_xrf_norm_array, opt_dens, per_element_crops, shifted_opt_dens_common_fov, desired_element_array, theta, sigma, alpha, fps)
# plt.show()