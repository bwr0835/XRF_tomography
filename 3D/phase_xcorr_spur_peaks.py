import numpy as np, \
       xrf_xrt_preprocess_utils as ppu, \
       xrf_xrt_preprocess_file_util as futil, \
       os

from scipy import fft, ndimage as ndi
from matplotlib import pyplot as plt
from skimage import registration as reg

def phase_xcorr_manual(ref_img, mov_img, sigma, alpha):    
    n_slices = ref_img.shape[0]
    n_columns = ref_img.shape[1]
    
    ref_img_filtered = ppu.edge_gauss_filter(ref_img, sigma, alpha, nx = n_columns, ny = n_slices)
    mov_img_filtered = ppu.edge_gauss_filter(mov_img, sigma, alpha, nx = n_columns, ny = n_slices)

    ref_img_fft = fft.fft2(ref_img_filtered)
    mov_img_fft = fft.fft2(mov_img_filtered)

    phase_xcorr = np.abs(fft.ifft2(ref_img_fft*mov_img_fft.conjugate()/np.abs(ref_img_fft*mov_img_fft.conjugate())))

    return fft.fftshift(phase_xcorr)

def xcorr_vert_parabolic_fit(xcorr_img, pixel_rad):
    center_slice_idx = xcorr_img.shape[0]//2
    center_column_idx = xcorr_img.shape[1]//2

    start_slice_idx = center_slice_idx - pixel_rad
    end_slice_idx = center_slice_idx + pixel_rad

    start_column_idx = center_column_idx - pixel_rad
    end_column_idx = center_column_idx + pixel_rad

    xcorr_img_truncated = xcorr_img[start_slice_idx:end_slice_idx, start_column_idx:end_column_idx]

    pcc_max_idx = np.unravel_index(np.argmax(xcorr_img_truncated), xcorr_img_truncated.shape)

    pcc_p = xcorr_img_truncated[pcc_max_idx[0] + 1, pcc_max_idx[1]]
    pcc_0 = xcorr_img_truncated[pcc_max_idx[0], pcc_max_idx[1]]
    pcc_n = xcorr_img_truncated[pcc_max_idx[0] - 1, pcc_max_idx[1]]
    
    subpix_shift = -0.5*(pcc_p - pcc_n)/(pcc_p + pcc_n - 2*pcc_0)
    # print(start_slice_idx)
    # print(subpix_shift)
    print(pcc_max_idx[0])
    pcc_max_idx_remapped = start_slice_idx + pcc_max_idx[0] + subpix_shift
    print(pcc_max_idx_remapped)

    dy = pcc_max_idx_remapped - center_slice_idx
    
    return xcorr_img_truncated, dy

def create_raw_img_fig(opt_dens,
                         counts_xrf, 
                         theta_array, 
                         element_array):
    
    fig, axs = plt.subplots(3, 2)
    
    n_slices = opt_dens.shape[1]
    n_columns = opt_dens.shape[2]
    
    im1_1 = axs[0, 0].imshow(opt_dens[0])
    im1_2 = axs[1, 0].imshow(counts_xrf[0, 0])
    im1_3 = axs[2, 0].imshow(counts_xrf[1, 0])
    im1_4 = axs[0, 1].imshow(opt_dens[1])
    im1_5 = axs[1, 1].imshow(counts_xrf[0, 1])
    im1_6 = axs[2, 1].imshow(counts_xrf[1, 1])

    for ax in fig.axes:
        ax.axis('off')
        ax.axvline(x = n_columns//2, color = 'red', linewidth = 2)
        ax.axhline(y = n_slices//2, color = 'red', linewidth = 2)
    
    text_1 = axs[2, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs[2, 0].transAxes, color = 'white')
    text_2 = axs[2, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[1]), transform = axs[2, 1].transAxes, color = 'white')

    axs[0, 0].set_title(r'Opt. Dens.', fontsize = 14)
    axs[1, 0].set_title(r'XRF ({0})'.format(element_array[0]), fontsize = 14)
    axs[2, 0].set_title(r'XRF ({0})'.format(element_array[1]), fontsize = 14)
    axs[0, 1].set_title(r'Opt. Dens.', fontsize = 14)
    axs[1, 1].set_title(r'XRF ({0})'.format(element_array[0]), fontsize = 14)
    axs[2, 1].set_title(r'XRF ({0})'.format(element_array[1]), fontsize = 14)

    return fig, axs

def create_phase_xcorr_fig(phase_xcorr_array_xrf,
                           phase_xcorr_array_xrf_truncated,
                           phase_xcorr_array_opt_dens, 
                           phase_xcorr_array_opt_dens_truncated,
                           element_array):
    
    n_slices = phase_xcorr_array_xrf.shape[1]
    n_columns = phase_xcorr_array_xrf.shape[2]
    
    fig, axs = plt.subplots(3, 2)

    # im1_1 = axs[0].imshow(np.log10(phase_xcorr_array_opt_dens), vmin = np.log10(phase_xcorr_array_opt_dens.min()))
    # im1_2 = axs[1].imshow(np.log10(phase_xcorr_array_xrf[0]), vmin = np.log10(phase_xcorr_array_xrf[0].min()))
    # im1_3 = axs[2].imshow(np.log10(phase_xcorr_array_xrf[1]), vmin = np.log10(phase_xcorr_array_xrf[1].min()))
    im1_1 = axs[0, 0].imshow((phase_xcorr_array_opt_dens))
    im1_2 = axs[1, 0].imshow((phase_xcorr_array_xrf[0]))
    im1_3 = axs[2, 0].imshow((phase_xcorr_array_xrf[1]))
    im1_4 = axs[0, 1].imshow((phase_xcorr_array_opt_dens_truncated), vmin = phase_xcorr_array_opt_dens.min())
    im1_5 = axs[1, 1].imshow((phase_xcorr_array_xrf_truncated[0]), vmin = phase_xcorr_array_xrf[0].min())
    im1_6 = axs[2, 1].imshow((phase_xcorr_array_xrf_truncated[1]), vmin = phase_xcorr_array_xrf[1].min())
            
    for ax in fig.axes:
        ax.axis('off')
        # axs.axvline(x = n_columns//2, color = 'red', linewidth = 2)
        # axs.axhline(y = n_slices//2, color = 'red', linewidth = 2)

    axs[0, 0].set_title(r'Opt. Dens.', fontsize = 14)
    axs[1, 0].set_title(r'XRF ({0})'.format(element_array[0]), fontsize = 14)
    axs[2, 0].set_title(r'XRF ({0})'.format(element_array[1]), fontsize = 14) 
    axs[0, 1].set_title(r'Opt. Dens (Tr.)', fontsize = 14)
    axs[1, 1].set_title(r'XRF ({0}) (Tr.)'.format(element_array[0]), fontsize = 14)
    axs[2, 1].set_title(r'XRF ({0}) (Tr.)'.format(element_array[1]), fontsize = 14)

    return fig, axs

sigma = 5
alpha = 10

aggregate_xrf_h5_file_path = '/Users/bwr0835/Documents/3_id_aggregate_xrf.h5'
aggregate_xrt_h5_file_path = '/Users/bwr0835/Documents/3_id_aggregate_xrt.h5'

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
desired_theta = [-150, -147]

element_idx_array = []
theta_idx_array = []

for element_idx in desired_elements_xrf:
    element_idx_array.append(elements_xrf.index(element_idx))

# print(element_idx_array)

for theta_idx in range(n_theta):
    if theta[theta_idx] in desired_theta:
        theta_idx_array.append(theta_idx)

desired_theta_array = theta_xrt[theta_idx_array]
desired_element_array = [elements_xrf[i] for i in element_idx_array]

counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts = ppu.joint_fluct_norm(counts_xrt_sig,
                                                                            counts_xrf,
                                                                            xrt_data_percentile = 80)

opt_dens = -np.log(counts_xrt_norm/I0_cts)[theta_idx_array]
counts_xrf_norm_array = counts_xrf_norm[element_idx_array][:, theta_idx_array]

phase_xcorr_array_xrf = np.zeros((len(desired_elements_xrf), n_slices, n_columns))

phase_xcorr_opt_dens = phase_xcorr_manual(opt_dens[0], opt_dens[1], sigma, alpha)

for element_idx in range(len(desired_elements_xrf)):
    phase_xcorr_array_xrf[element_idx] = phase_xcorr_manual(counts_xrf_norm_array[element_idx, 0], counts_xrf_norm_array[element_idx, 1], sigma, alpha)

pixel_rad = 35

dy_xrf_array = np.zeros(len(desired_elements_xrf))
phase_xcorr_array_xrf_truncated = np.zeros((len(desired_elements_xrf), 2 * pixel_rad, 2 * pixel_rad))

for element_idx in range(len(desired_elements_xrf)):
    phase_xcorr_array_xrf_truncated[element_idx], dy_xrf_array[element_idx] = xcorr_vert_parabolic_fit(phase_xcorr_array_xrf[element_idx], pixel_rad)

phase_xcorr_opt_dens_truncated, dy_opt_dens = xcorr_vert_parabolic_fit(phase_xcorr_opt_dens, pixel_rad)

dy_pcc_od, _, _ = reg.phase_cross_correlation(opt_dens[0], opt_dens[1], upsample_factor = 100)
dy_pcc_xrf_1, _, _ = reg.phase_cross_correlation(counts_xrf_norm_array[0, 0], counts_xrf_norm_array[0, 1], upsample_factor = 100)
dy_pcc_xrf_2, _, _ = reg.phase_cross_correlation(counts_xrf_norm_array[1, 0], counts_xrf_norm_array[1, 1], upsample_factor = 100)

print(dy_pcc_od[0])
print(dy_pcc_xrf_1[0])
print(dy_pcc_xrf_2[0])
print('\n')
print(dy_xrf_array)
print(dy_opt_dens)

fig1, axs1 = create_raw_img_fig(opt_dens, counts_xrf_norm_array, desired_theta_array, desired_element_array)
fig2, axs2 = create_phase_xcorr_fig(phase_xcorr_array_xrf, phase_xcorr_array_xrf_truncated, phase_xcorr_opt_dens, phase_xcorr_opt_dens_truncated, desired_element_array)

plt.show()