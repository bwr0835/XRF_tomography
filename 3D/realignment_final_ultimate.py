import numpy as np, \
       tomopy as tomo, \
       xrf_xrt_preprocess_utils as ppu, \
       sys

from skimage import transform as xform
from scipy import ndimage as ndi, fft
import matplotlib.pyplot as plt

def phase_xcorr_manual(ref_img,
                       mov_img, 
                       pixel_rad,
                       theta):
    
    n_slices = ref_img.shape[0]
    n_columns = ref_img.shape[1]

    ref_img_fft = fft.fft2(ref_img)
    mov_img_fft = fft.fft2(mov_img)

    phase_xcorr = fft.fftshift(np.abs(fft.ifft2(ref_img_fft*mov_img_fft.conjugate()/np.abs(ref_img_fft*mov_img_fft.conjugate()))))

    center_slice_idx = int(n_slices//2)
    center_column_idx = int(n_columns//2)

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
                                                     aligning_element,
                                                     net_x_shift_array,
                                                     net_y_shift_array,
                                                     sigma,
                                                     alpha,
                                                     pixel_rad,
                                                     theta,
                                                     cval_array,
                                                     cos_fit_enabled,
                                                     angle_range_to_fit,
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
   
    if sigma is None:
        print('Warning: Positive \'sigma\' not detected. Setting \'sigma\' to 5 pixels...')
        
        sigma = 5

    if alpha is None:
        print('Warning: Positive, \'alpha\' not detected. Setting \'alpha\' to 10 pixels...')
        
        alpha = 10

    if sigma < 0 or alpha < 0:
        print('Error: \'sigma\', \'alpha\', \'eps_cor_correction\' and \'eps_iter_reproj\' must all be positive numbers. Exiting program...')

        sys.exit()

    n_theta, n_slices, n_columns = init_proj_array.shape

    phase_xcorr_2d_aggregate = np.zeros((n_theta - 1, n_slices, n_columns))
    # net_x_shift_array, net_y_shift_array = np.zeros(n_theta), np.zeros(n_theta)
    net_y_shift_cumsum_temp = np.zeros(n_theta - 1)
    net_x_shift_cumsum_temp = np.zeros(n_theta - 1)

    if np.any(net_y_shift_array != 0) or np.any(net_x_shift_array != 0):
        print(f'Applying initial vertical and horizontal shifts...')
        
        shifted_proj_array = np.zeros_like(init_proj_array)
        
        for theta_idx in range(n_theta):
            shifted_proj_array[theta_idx] = ndi.shift(init_proj_array[theta_idx], shift = (net_y_shift_array[theta_idx], net_x_shift_array[theta_idx]), cval = cval_array[theta_idx])
    
    else:
        shifted_proj_array = init_proj_array.copy()
    
    shifted_proj_array_orig = shifted_proj_array.copy()

    # plt.imshow(shifted_proj_array[-1])
    # plt.show()
    
    if pixel_rad is None:
        print('Warning: \'pixel_rad\' not detected. Performing peak search without truncation...')

        pixel_rad = np.zeros(n_theta - 1)

        phase_xcorr_2d_truncated_aggregate = np.zeros((n_theta - 1, n_slices, n_columns))
    
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
        # print(pixel_rad)
        if np.any(pixel_rad == 0):
            print('Warning: \'pixel_rad\' is 0. Performing peak search without truncation...')

            phase_xcorr_2d_truncated_aggregate = np.zeros((n_theta - 1, n_slices, n_columns))

        else:
            phase_xcorr_2d_truncated_aggregate = np.zeros((n_theta - 1, 2*pixel_rad.max(), 2*pixel_rad.max()))

    for theta_idx in range(n_theta - 1):
        shifts, phase_xcorr_2d, phase_xcorr_2d_truncated = phase_xcorr_manual(shifted_proj_array[theta_idx],
                                                                              shifted_proj_array[theta_idx + 1], 
                                                                              pixel_rad[theta_idx],
                                                                              theta[[theta_idx, theta_idx + 1]])

        net_y_shift_cumsum_temp[theta_idx] = shifts[0]
        net_x_shift_cumsum_temp[theta_idx] = shifts[1]

        net_x_shift_array[theta_idx + 1] += net_x_shift_cumsum_temp[theta_idx]

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
    # net_x_shift_cumsum = np.cumsum(net_x_shift_cumsum_temp)

    net_y_shift_array[1:] += net_y_shift_cumsum
    # net_x_shift_array[1:] += net_x_shift_cumsum

    if cos_fit_enabled:
        theta_fit_idx_min = np.where(theta == angle_range_to_fit[0])[0][0]
        theta_fit_idx_max = np.where(theta == angle_range_to_fit[1])[0][0]

        cumulative_x_jitter_fit = ppu.cos_fit(theta[theta_fit_idx_min:theta_fit_idx_max + 1], net_x_shift_array[theta_fit_idx_min:theta_fit_idx_max + 1])
        # cumulative_y_jitter_fit = ppu.cos_fit(theta[theta_fit_idx_min:theta_fit_idx_max + 1], net_y_shift_cumsum[theta_fit_idx_min:theta_fit_idx_max + 1])
        
        net_x_shift_array[theta_fit_idx_min:theta_fit_idx_max + 1] -= cumulative_x_jitter_fit
        # net_y_shift_array[1:] += cumulative_y_jitter_fit

    if return_aux_data:
        print(f'Shifting original projection images of \'{aligning_element}\' post-adjacent angle jitter correction...')
        
        for theta_idx in range(n_theta):
            shifted_proj_array[theta_idx] = ndi.shift(init_proj_array[theta_idx], shift = (net_y_shift_array[theta_idx], net_x_shift_array[theta_idx]), cval = cval_array[theta_idx])
    
    if return_aux_data:        
        return net_x_shift_array, \
               net_y_shift_array, \
               phase_xcorr_2d_aggregate, \
               phase_xcorr_2d_truncated_aggregate, \
               shifted_proj_array_orig, \
               shifted_proj_array
    
    return net_x_shift_array, \
           net_y_shift_array, \
           None, \
           None, \
           None, \
           None

def rot_center_ps(theta_sum):
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

def rot_center_avg(proj_img_array, 
                   theta_pair_array, 
                   theta_array, 
                   cor_correction_alg,
                   sample_flipped_remounted_mid_experiment = False,
                   half_dataset_part = None):

    n_theta = proj_img_array.shape[0]
    n_columns = proj_img_array.shape[2]

    geom_center = n_columns//2

    center_of_rotation_sum = 0
    
    if cor_correction_alg == 'phase_xcorr':
        if sample_flipped_remounted_mid_experiment:
            if half_dataset_part == 'first':
                first_idx = theta_pair_array[0][0]
                second_idx = len(theta_array) - 1

            elif half_dataset_part == 'second':
                first_idx = n_theta - len(theta_array)
                second_idx = theta_pair_array[0][1]
            
            shifts, _, _ = phase_xcorr_manual(proj_img_array[first_idx], 
                                              np.fliplr(proj_img_array[second_idx]),
                                              pixel_rad = 0, 
                                              theta = np.array([theta_array[theta_pair_array[0][0]], theta_array[theta_pair_array[0][1]]]))
            offset = shifts[1]/2

            center_of_rotation = geom_center + offset

            print(f'Center of rotation ({theta_array[theta_pair_array[0][0]]} degrees, {theta_array[theta_pair_array[0][1]]} degrees) = {ppu.round_correct(center_of_rotation, ndec = 3)}')

            center_of_rotation_sum += center_of_rotation

        else:            
            for theta_pair in theta_pair_array:
                shifts, _, _ = phase_xcorr_manual(proj_img_array[theta_pair[0]], 
                                                  np.fliplr(proj_img_array[theta_pair[1]]),
                                                  pixel_rad = 0, 
                                                  theta = np.array([theta_array[theta_pair[0]], theta_array[theta_pair[1]]]))
            
                offset = shifts[1]/2

                center_of_rotation = geom_center + offset

                print(f'Center of rotation ({theta_array[theta_pair[0]]} degrees, {theta_array[theta_pair[1]]} degrees) = {ppu.round_correct(center_of_rotation, ndec = 3)}')

                center_of_rotation_sum += center_of_rotation
    
    elif cor_correction_alg == 'phase_symm':
        if sample_flipped_remounted_mid_experiment:
            if half_dataset_part == 'first':
                first_idx = theta_pair_array[0][0]
                second_idx = len(theta_array) - 1

            elif half_dataset_part == 'second':
                first_idx = n_theta - len(theta_array)
                second_idx = theta_pair_array[0][1]

            theta_sum = proj_img_array[first_idx] + proj_img_array[second_idx]

            center_of_rotation = rot_center_ps(theta_sum)

            print(f'Center of rotation ({theta_array[theta_pair_array[0][0]]} degrees, {theta_array[theta_pair_array[0][1]]} degrees) = {ppu.round_correct(center_of_rotation, ndec = 3)}')

            center_of_rotation_sum += center_of_rotation
        
        else:
            for theta_pair in theta_pair_array:
                theta_sum = proj_img_array[theta_pair[0]] + proj_img_array[theta_pair[1]]

                center_of_rotation = rot_center_ps(theta_sum)

                print(f'Center of rotation ({theta_array[theta_pair[0]]} degrees, {theta_array[theta_pair[1]]} degrees) = {ppu.round_correct(center_of_rotation, ndec = 3)}')

                center_of_rotation_sum += center_of_rotation
    
    center_rotation_avg = center_of_rotation_sum/len(theta_pair_array)

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

def correct_center_of_rotation(proj_img_array,
                               net_x_shift_array,
                               net_y_shift_array,
                               theta_array,
                               cor_correction_alg,
                               aligning_element,
                               cval_array,
                               pixel_rad_cor_correction = 0,
                               sample_flipped_remounted_mid_experiment = False,
                               sample_flipped_remounted_correction_type = 'differential',
                               return_aux_data = True):

    n_theta, n_slices, n_columns = proj_img_array.shape
    
    if pixel_rad_cor_correction is None:
        print('Warning: \'pixel_rad_cor_correction\' not detected. Setting \'pixel_rad_cor_correction\' to zero pixels...')

        pixel_rad_cor_correction = 0

    if np.any(net_x_shift_array) or np.any(net_y_shift_array):
        print('Applying initial vertical and/or horizontal shifts...')
        
        shifted_proj_img_array = np.zeros_like(proj_img_array)
        
        for theta_idx in range(n_theta):
            shifted_proj_img_array[theta_idx] = ndi.shift(proj_img_array[theta_idx], shift = (net_y_shift_array[theta_idx], net_x_shift_array[theta_idx]), cval = cval_array[theta_idx])

    else:
        shifted_proj_img_array = proj_img_array.copy()
    
    shifted_proj_img_array_orig = shifted_proj_img_array.copy()
    
    if sample_flipped_remounted_mid_experiment:
        if np.count_nonzero(theta_array == 0) != 2:
            print('Error: Must have two 0° angles. Exiting program...')

            sys.exit()

        zero_deg_idx_array = np.where(theta_array == 0)[0]
        
        theta_array_first_part = theta_array[:zero_deg_idx_array[1]]
        theta_array_second_part = theta_array[zero_deg_idx_array[1]:]

        theta_idx_pairs_first_part = [(0, -1)] # These remap to original -180° and 0° indices
        theta_idx_pairs_second_part = [(0, -1)] # These remap to original 0° and +180° indices

        if cor_correction_alg == 'phase_xcorr':
            center_of_rotation_avg_first_part, center_geom, offset_init_first_part = rot_center_avg(shifted_proj_img_array, theta_idx_pairs_first_part, theta_array_first_part, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment, half_dataset_part = 'first')
            center_of_rotation_avg_second_part, _, offset_init_second_part = rot_center_avg(shifted_proj_img_array, theta_idx_pairs_second_part, theta_array_second_part, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment, half_dataset_part = 'second')
            
        elif cor_correction_alg == 'phase_symm':
            center_of_rotation_avg_first_part, center_geom, offset_init_first_part = rot_center_avg(shifted_proj_img_array, theta_idx_pairs_first_part, theta_array_first_part, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment, half_dataset_part = 'first')
            center_of_rotation_avg_second_part, _, offset_init_second_part = rot_center_avg(shifted_proj_img_array, theta_idx_pairs_second_part, theta_array_second_part, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment, half_dataset_part = 'second')
            
        else:
            print('Error: Correction algorithm unavailable. Exiting program...')

            sys.exit()
        
        print(f'Average center of rotation (before flipping sample): {ppu.round_correct(center_of_rotation_avg_first_part, ndec = 13)}')
        print(f'Average center of rotation (after flipping sample): {ppu.round_correct(center_of_rotation_avg_second_part, ndec = 13)}\n')
        print(f'Geometric center: {center_geom}\n')
        print(f'Center of rotation error (before flipping sample): {ppu.round_correct(offset_init_first_part, ndec = 13)}')
        print(f'Center of rotation error (after flipping sample): {ppu.round_correct(offset_init_second_part, ndec = 13)}\n')
        
        print(f'Applying initial center of rotation correction to shifted projection images:\nBefore flipping sample: {ppu.round_correct(-offset_init_first_part, ndec = 13)}; After flipping sample: {ppu.round_correct(-offset_init_second_part, ndec = 13)}')

        net_x_shift_array[:zero_deg_idx_array[1]] -= offset_init_first_part
        net_x_shift_array[zero_deg_idx_array[1]:] -= offset_init_second_part
 
        for theta_idx in range(n_theta):
            shifted_proj_img_array[theta_idx] = ndi.shift(proj_img_array[theta_idx], shift = (net_y_shift_array[theta_idx], net_x_shift_array[theta_idx]), cval = cval_array[theta_idx])

        if return_aux_data:
            shifted_proj_img_array_aux = shifted_proj_img_array.copy()

        center_of_rotation_avg_first_part, center_geom, offset_first_part = rot_center_avg(shifted_proj_img_array[:zero_deg_idx_array[1]], theta_idx_pairs_first_part, theta_array_first_part, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment, half_dataset_part = 'first')
        center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(shifted_proj_img_array[zero_deg_idx_array[1]:], theta_idx_pairs_second_part, theta_array_second_part, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment, half_dataset_part = 'second')

        print(f'New average center of rotation (before flipping sample): {ppu.round_correct(center_of_rotation_avg_first_part, ndec = 13)}')
        print(f'New average center of rotation (after flipping sample): {ppu.round_correct(center_of_rotation_avg_second_part, ndec = 13)}\n')
        print(f'Geometric center: {center_geom}\n')
        print(f'New center of rotation error (before flipping sample): {ppu.round_correct(offset_first_part, ndec = 13)}')
        print(f'New center of rotation error (after flipping sample): {ppu.round_correct(offset_second_part, ndec = 13)}\n')

        # shifts, phase_xcorr_2d, phase_xcorr_2d_truncated = phase_xcorr_manual(shifted_proj_img_array[0], 
        #                                                                       shifted_proj_img_array[zero_deg_idx_array[1]], 
        #                                                                       pixel_rad_cor_correction, 
        #                                                                       theta = np.array([theta_array[0], theta_array[zero_deg_idx_array[1]]]))
        shifts, phase_xcorr_2d, phase_xcorr_2d_truncated = phase_xcorr_manual(shifted_proj_img_array[0], 
                                                                              shifted_proj_img_array[-1], 
                                                                              pixel_rad_cor_correction, 
                                                                              theta = np.array([theta_array[0], theta_array[-1]]))

        plt.imshow(phase_xcorr_2d_truncated)
        plt.show()
        dy, dx = shifts[0], shifts[1]
        
        if (dy, dx) != (0, 0):
            if sample_flipped_remounted_correction_type is None:
                print('Warning: Sample remounting correction type not detected. Setting \'sample_flipped_remounted_correction_type\' to \'differential\'...')

                sample_flipped_remounted_correction_type = 'differential'
        
            net_y_shift_array += dy
        
            if sample_flipped_remounted_correction_type == 'differential':
                print(f'Correcting sample remounting offset in shifted projection images: {ppu.round_correct(dy, ndec = 13)} pixels in y and {ppu.round_correct(dx/2, ndec = 13)} pixels in x for each half dataset...')

                net_x_shift_array[:zero_deg_idx_array[1]] -= dx/2
                net_x_shift_array[zero_deg_idx_array[1]:] += dx/2
            
            elif sample_flipped_remounted_correction_type == 'absolute':
                print(f'Correcting sample remounting offset in shifted projection images: {ppu.round_correct(dy, ndec = 13)} pixels in y and {ppu.round_correct(dx, ndec = 13)} pixels in x for second half dataset...')
                
                net_x_shift_array[zero_deg_idx_array[1]:] -= dx
            
            else:
                print('Error: Sample remounting correction type unavailable. Exiting program...')

                sys.exit()
            
            print(f'Shifting shifted projection images...')

            for theta_idx in range(n_theta):
                shifted_proj_img_array[theta_idx] = ndi.shift(proj_img_array[theta_idx], shift = (net_y_shift_array[theta_idx], net_x_shift_array[theta_idx]), cval = cval_array[theta_idx])

            print('Done')
        
    else:
        theta_idx_pairs = ppu.find_theta_combos(theta_array)

        if cor_correction_alg == 'phase_xcorr':
            center_of_rotation_avg, center_geom, offset_init = rot_center_avg(shifted_proj_img_array, theta_idx_pairs, theta_array, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment)
            
        elif cor_correction_alg == 'phase_symm':
            center_of_rotation_avg, center_geom, offset_init = rot_center_avg(shifted_proj_img_array, theta_idx_pairs, theta_array, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment)
            
        else:
            print('Error: Correction algorithm unavailable. Exiting program...')

            sys.exit()

        print(f'Average center of rotation: {ppu.round_correct(center_of_rotation_avg, ndec = 13)}')
        print(f'Geometric center: {center_geom}')
        print(f'Center of rotation error: {ppu.round_correct(offset_init, ndec = 13)}')

        print(f'Applying initial center of rotation correction to shifted projection images: {ppu.round_correct(-offset_init, ndec = 13)}')

        net_x_shift_array -= offset_init

        for theta_idx in range(n_theta):
            shifted_proj_img_array[theta_idx] = ndi.shift(proj_img_array[theta_idx], shift = (net_y_shift_array[theta_idx], net_x_shift_array[theta_idx]), cval = cval_array[theta_idx])

        if cor_correction_alg == 'phase_xcorr':
            center_of_rotation_avg, center_geom, offset_init = rot_center_avg(shifted_proj_img_array, theta_idx_pairs, theta_array, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment)
            
        elif cor_correction_alg == 'phase_symm':
            center_of_rotation_avg, center_geom, offset_init = rot_center_avg(shifted_proj_img_array, theta_idx_pairs, theta_array, cor_correction_alg = cor_correction_alg, sample_flipped_remounted_mid_experiment = sample_flipped_remounted_mid_experiment)

        print(f'New average center of rotation: {ppu.round_correct(center_of_rotation_avg, ndec = 13)}')
        print(f'Geometric center: {center_geom}')
        print(f'New center of rotation error: {ppu.round_correct(offset_init, ndec = 13)}')
            
    if return_aux_data:
        print(f'Shifting original projection images of \'{aligning_element}\' post-COR correction...')
        
        if sample_flipped_remounted_correction_type:
            return net_x_shift_array, net_y_shift_array, shifted_proj_img_array, shifted_proj_img_array_aux, shifted_proj_img_array_orig

        return net_x_shift_array, net_y_shift_array, shifted_proj_img_array, None, shifted_proj_img_array_orig
    
    return net_x_shift_array, net_y_shift_array, None, None, None

def iter_reproj(proj_img_array,
                theta_array,
                n_iterations_iter_reproj,
                init_x_shift, 
                init_y_shift,
                cval_array,
                pixel_rad_iter_reproj,
                eps_iter_reproj,
                return_aux_data):

    '''
    iter_reproj: Perform iterative reprojection on experimental optical density (OD) projection images 
    to correct for center of rotation (COR) error, jitter (per-projection translations), respectively, 
    in x-ray transmission, OD, and x-ray fluorescnece projection images

    Inputs
    ------
    proj_img_array: 3D XRF tomography projection images (projection angles, slices, scan positions) (array-like; dtype: float)
    theta_array: Array of projection angles (array-like; dtype: float)
    n_iterations_iter_reproj: Maximum number of iterative reprojection iterations (dtype: int)
    init_x_shift: Initial x-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)    
    init_y_shift: Initial y-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)
    eps: Desired differential shift for convergence criterion (dtype: float)
    return_aux_data: Flag for returning per-iteration auxiliary data (dtype: bool; default: False)

    Outputs
    ------- 
    aligned_proj: 3D XRF tomography data corrected for per-projection jitter, center of rotation misalignment (array-like; dtype: float)
    aligned_exp_proj_array: 1D array of experimental 3D XRF tomography data arrays for each iteration for ref_element (array-like; dtype: float)
    synth_proj_array: 1D array of synthetic 3D XRF tomography data arrays for each iteration for ref_element (array-like; dtype: float)
    recon_array: 1D array of 3D reconstruction slices for each iteration for ref_element (array-like; dtype: float)
    net_x_shifts_pcc_new: Array of net x shifts with dimensions (n_iterations_iter_reproj, n_theta) (array-like; dtype: float) (Note: n_iterations_iter_reproj can be smaller the more quickly iter_reproj() converges)
    net_y_shifts_pcc_new: Array of net y shifts with dimensions (n_iterations_iter_reproj, n_theta) (array-like; dtype: float) (Note: n_iterations_iter_reproj can be smaller the more quickly iter_reproj() converges)
    '''
    
    if eps_iter_reproj is None:
        print('Warning: Nonzero, positive \'eps_iter_reproj\' not detected. Setting \'eps_iter_reproj\' to 1e-8...')

        eps_iter_reproj = 1e-3
    
    if eps_iter_reproj < 0:
        print('Error: \'eps_iter_reproj\' must be a positive number. Exiting program...')

        sys.exit()
    
    if pixel_rad_iter_reproj is None:
        print('Warning: \'pixel_rad_iter_reproj\' not detected. Setting \'pixel_rad_iter_reproj\' to 0 pixels for each projection angle...')

        pixel_rad_iter_reproj = np.zeros(n_theta)

    n_theta, n_slices, n_columns = proj_img_array.shape

    net_x_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta))
    net_y_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta))

    if np.any(init_x_shift) or np.any(init_y_shift):
        print('Applying initial vertical and/or horizontal shifts...')

        net_x_shifts_pcc[0] += init_x_shift
        net_y_shifts_pcc[0] += init_y_shift

        for theta_idx in range(n_theta):
            aligned_proj[theta_idx] = ndi.shift(proj_img_array[theta_idx], shift = (net_y_shifts_pcc[0, theta_idx], net_x_shifts_pcc[0, theta_idx]), cval = cval_array[theta_idx])

    else:
        aligned_proj = proj_img_array.copy()
    
    aligned_proj_orig = aligned_proj.copy()
    
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
    dy_array_pcc = np.zeros((n_iterations_iter_reproj, n_theta))

    rms_net_shift_prev = 0

    for i in range(n_iterations_iter_reproj):
        print(f'Iteration {i + 1}/{n_iterations_iter_reproj}')
        
        if i > 0:
            for theta_idx in range(n_theta):
                net_x_shift = net_x_shifts_pcc[i - 1, theta_idx]
                net_y_shift = net_y_shifts_pcc[i - 1, theta_idx]

                aligned_proj[theta_idx] = ndi.shift(proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift), cval = cval_array[theta_idx])
                
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
                                                  pixel_rad_iter_reproj[theta_idx],
                                                  theta = np.array([theta_array[theta_idx], theta_array[theta_idx]]))
            
            else:
                shifts, phase_xcorr_2d, phase_xcorr_2d_truncated = phase_xcorr_manual(synth_proj[theta_idx], 
                                                                                      aligned_proj[theta_idx], 
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
            tomo.find_center_vo()
            if (theta_idx % 7) == 0:
                if theta_idx == 0:
                    print(f'\nCurrent x-shift: {ppu.round_correct(dx, ndec = 3)} (theta = {ppu.round_correct(theta_array[theta_idx], ndec = 1)})')
                
                else:
                    print(f'Current x-shift: {ppu.round_correct(dx, ndec = 3)} (theta = {ppu.round_correct(theta_array[theta_idx], ndec = 1)})')
                
                print(f'Current y-shift: {ppu.round_correct(dy, ndec = 3)}')

        rms_net_x_shift = np.sqrt((dx_array_pcc[i]**2).mean())
        rms_net_y_shift = np.sqrt((dy_array_pcc[i]**2).mean())
        
        rms_net_shift = np.sqrt(rms_net_x_shift**2 + rms_net_y_shift**2)

        if i > (n_iter_converge - 1):
            if np.abs(rms_net_shift - rms_net_shift_prev) <= eps_iter_reproj:
                print(f'Number of iterations taken: {i + 1}')

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

                break

        if i == n_iterations_iter_reproj - 1:
            print('Iterative reprojection complete.')
            
            break
        
        rms_net_shift_prev = rms_net_shift

    if return_aux_data:       
        return net_x_shifts_pcc, \
               net_y_shifts_pcc, \
               aligned_proj, \
               aligned_proj_orig, \
               recon_array, \
               aligned_exp_proj_array, \
               synth_proj_array, \
               pcc_2d_array, \
               pcc_2d_truncated_array, \
               dx_array_pcc, \
               dy_array_pcc
        
    return net_x_shifts_pcc, \
           net_y_shifts_pcc, \
           None, \
           None, \
           None, \
           None, \
           None, \
           None, \
           None, \
           None, \
           None

def realign_proj_final(xrf_proj_img_array,
                       xrt_proj_img_array,
                       opt_dens_proj_img_array,
                       theta_array,
                       net_x_shifts,
                       net_y_shifts,
                       I0):
    
    shifted_xrf_proj_img_array = np.zeros_like(xrf_proj_img_array)
    shifted_xrt_proj_img_array = np.zeros_like(xrt_proj_img_array)
    shifted_opt_dens_proj_img_array = np.zeros_like(opt_dens_proj_img_array)
    
    for theta_idx in range(theta_array.shape[0]):
        net_x_shift = net_x_shifts[theta_idx]
        net_y_shift = net_y_shifts[theta_idx]

        shifted_xrt_proj_img_array[theta_idx] = ndi.shift(xrt_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift), cval = I0)
        shifted_opt_dens_proj_img_array[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift))

        for element_idx in range(xrf_proj_img_array.shape[0]):
            shifted_xrf_proj_img_array[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))

    return shifted_xrf_proj_img_array, shifted_xrt_proj_img_array, shifted_opt_dens_proj_img_array