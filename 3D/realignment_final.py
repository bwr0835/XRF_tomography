import numpy as np, \
       tomopy as tomo, \
       xrf_xrt_preprocess_utils as ppu, \
       sys

from skimage import transform as xform, registration as reg
from scipy import ndimage as ndi, fft

def phase_xcorr(recon_proj,
                exp_proj, 
                sigma, 
                alpha, 
                upsample_factor, 
                return_pcc_2d = False):
    
    n_columns = recon_proj.shape[1]
    n_slices = recon_proj.shape[0]

    recon_proj_filtered = ppu.edge_gauss_filter(recon_proj, sigma, alpha, nx = n_columns, ny = n_slices)
    exp_proj_filtered = ppu.edge_gauss_filter(exp_proj, sigma, alpha, nx = n_columns, ny = n_slices)
    
    if return_pcc_2d:
        recon_proj_fft = fft.fft2(recon_proj_filtered)
        orig_proj_fft = fft.fft2(exp_proj_filtered)
        
        # NOTE: Most likely, there will a slight variation in the following array relative to skimage.registration.phase_cross_correlation();
        # however, for rough estimates of PCC for diagnostics, for instance, that variation can be ignored.
        pcc_2d = np.abs(fft.ifft2(recon_proj_fft*orig_proj_fft.conjugate()/np.abs(recon_proj_fft*orig_proj_fft.conjugate())))

    shift, _, _ = reg.phase_cross_correlation(reference_image = recon_proj_filtered, moving_image = exp_proj_filtered, upsample_factor = upsample_factor)

    y_shift, x_shift = shift[0], shift[1]

    if return_pcc_2d:
        return y_shift, x_shift, pcc_2d
    
    return y_shift, x_shift

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

# def realign_proj(xrt_proj_img_array,
#                  cropped_xrt_proj_img_array,
#                  opt_dens_proj_img_array,
#                  cropped_opt_dens_proj_img_array,
#                  xrf_proj_img_array,
#                  cropped_xrf_proj_img_array,
#                  theta_array,
#                  zero_idx_to_discard,
#                  I0,
#                  n_iterations,
#                  init_x_shift, 
#                  init_y_shift,
#                  sigma,
#                  alpha,
#                  upsample_factor,
#                  eps,
#                  edge_info,
#                  return_aux_data = False):

def realign_proj(xrt_proj_img_array,
                 cropped_xrt_proj_img_array,
                 opt_dens_proj_img_array,
                 cropped_opt_dens_proj_img_array,
                 xrf_proj_img_array,
                 cropped_xrf_proj_img_array,
                 theta_array,
                 sample_flipped_remounted_mid_experiment,
                 n_iterations_cor_correction,
                 eps_cor_correction,
                 I0,
                 n_iterations_iter_reproj,
                 init_x_shift, 
                 init_y_shift,
                 sigma,
                 alpha,
                 upsample_factor,
                 eps_iter_reproj,
                 edge_info,
                 return_aux_data = False):

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

    if upsample_factor is None:
        print('Warning: \'upsample_factor\' not detected. Setting \'sample_factor\' to 100...')

        upsample_factor = 100
    
    if eps is None:
        print('Warning: Nonzero, positive \'eps\' not detected. Setting \'eps\' to 0.3 pixels...')

        eps = 0.3

    if sigma < 0 or alpha < 0 or eps < 0:
        print('Error: \'sigma\', \'alpha\', and \'eps\' must all be positive numbers. Exiting program...')

        sys.exit()

    if not isinstance(sigma, int) or not isinstance(upsample_factor, int):
        print('Error: \'sigma\' and \'upsample_factor\' must be positive integers. Exiting program...')

        sys.exit()

    n_elements_xrf = xrf_proj_img_array.shape[0]
    n_theta = xrt_proj_img_array.shape[0]
    
    net_x_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta))
    net_y_shifts_pcc = np.zeros((n_iterations_iter_reproj, n_theta))
    
    iterations = []
    
    if return_aux_data:
        aligned_exp_proj_array = []
        cropped_aligned_exp_proj_array = []
        synth_proj_array = []
        pcc_2d_array = []
        recon_array = []
    
    n_iter_converge = 3
    
    if cropped_opt_dens_proj_img_array is not None:
        aligned_proj = cropped_opt_dens_proj_img_array.copy()
        
    else:
        cropped_xrt_proj_img_array = xrt_proj_img_array.copy()
        cropped_opt_dens_proj_img_array = opt_dens_proj_img_array.copy()
        cropped_xrf_proj_img_array = xrf_proj_img_array.copy()
        aligned_proj = opt_dens_proj_img_array.copy()

    if np.any(init_x_shift) or np.any(init_y_shift):
        if np.any(init_x_shift) and np.any(init_y_shift):
            print('Executing intial shift(s) in x and y')
            
            net_x_shifts_pcc[0] += init_x_shift
            net_y_shifts_pcc[0] += init_y_shift

            for theta_idx in range(n_theta):
                # aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (init_y_shift[theta_idx], init_x_shift[theta_idx]))
                aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (init_y_shift[theta_idx], init_x_shift[theta_idx]))

        elif np.any(init_x_shift):
            print('Executing initial shift(s) in x')
            
            net_x_shifts_pcc[0] += init_x_shift

            for theta_idx in range(n_theta):
                # aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (0, init_x_shift[theta_idx]))
                aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (0, init_x_shift[theta_idx]))
                
        else:
            print('Executing initial shift(s) in x')
            
            net_y_shifts_pcc[0] += init_y_shift

            for theta_idx in range(n_theta):
                # aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (init_y_shift[theta_idx], 0))
                aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (init_y_shift[theta_idx], 0))

    # if zero_idx_to_discard is not None:
    #     if zero_idx_to_discard != 'first' or zero_idx_to_discard != 'second':
    #         print('Error: \'zero_idx_to_discard\' must be \'first\' or \'second\'. Exiting program...')

    #         sys.exit()

    #     if np.count_nonzero(theta_array == 0) != 2:
    #         print('Error: Must have two 0° angles. Exiting program...')

    #         sys.exit()

        if sample_flipped_remounted_mid_experiment: # This assumes that angles are order from -180° to +180° (360° range) AND that there are two zero degree angles
            if n_iterations_cor_correction is not None:
                print('Error: \'n_iterations_cor_correction\' not detected. Exiting program...')

                sys.exit()

            if n_iterations_cor_correction < 1 or not isinstance(n_iterations_cor_correction, int):
                print('Error: \'n_iterations_cor_correction\' must be a positive integer. Exiting program...')

                sys.exit()

            if np.count_nonzero(theta_array == 0) != 2:
                print('Error: Must have two 0° angles. Exiting program...')

                sys.exit()
            
            zero_deg_idx_array = np.where(theta_array == 0)[0]
            
            theta_idx_pairs_first_part = [0, zero_deg_idx_array[0]]
            theta_idx_pairs_second_part = [zero_deg_idx_array[1], -1]

            theta_array_first_part = theta_array[:zero_deg_idx_array[1]]
            theta_array_second_part = theta_array[zero_deg_idx_array[1]:]

            dx_prev = 0
            
            for i in range(n_iterations_cor_correction):
                print(f'COR correction iteration {i + 1}/{n_iterations_cor_correction}')
                
                center_of_rotation_avg_first_part, center_geom, offset_init_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1]], 
                                                                                                        theta_idx_pairs_first_part, 
                                                                                                        theta_array_first_part)
                
                center_of_rotation_avg_second_part, _, offset_init_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:], 
                                                                                                theta_idx_pairs_second_part, 
                                                                                                theta_array_second_part)

                print(f'Average center of rotation (before flipping sample): {center_of_rotation_avg_first_part}')
                print(f'Average center of rotation (after flipping sample): {center_of_rotation_avg_second_part}\n')
                print(f'Geometric center: {center_geom}\n')
                print(f'Center of rotation error (before flipping sample): {ppu.round_correct(offset_init_first_part, ndec = 3)}')
                print(f'Center of rotation error (after flipping sample): {ppu.round_correct(offset_init_second_part, ndec = 3)}\n')

                if i == 0 and offset_init_first_part == 0 and offset_init_second_part == 0:
                    print('No COR correction needed.')

                    break

                else:
                    print(f'Applying initial COR correction to pre-flipped, pre-remounted sample angles: {ppu.round_correct(-offset_init_first_part, ndec = 3)}')
                    
                    for theta_idx in range(len(theta_array_first_part)):
                        aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (0, -offset_init_first_part))

                    net_x_shifts_pcc[0, :zero_deg_idx_array[1]] -= offset_init_first_part

                    print(f'Applying initial COR correction to post-flipped, post-remounted sample angles: {ppu.round_correct(-offset_init_second_part, ndec = 3)}')
                    
                    for theta_idx in range(len(theta_array_second_part)):
                        aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (0, -offset_init_second_part))

                    center_of_rotation_avg_first_part, center_geom, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1]], 
                                                                                                        theta_idx_pairs_first_part, 
                                                                                                        theta_array_first_part)
                
                    center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:], 
                                                                                                    theta_idx_pairs_second_part, 
                                                                                                    theta_array_second_part)

                    print(f'New center of rotation (before flipping sample): {center_of_rotation_avg_first_part}')
                    print(f'New center of rotation (after flipping sample): {center_of_rotation_avg_second_part}\n')
                    print(f'Geometric center: {center_geom}\n')
                    print(f'New center of rotation error (before flipping sample): {ppu.round_correct(offset_first_part, ndec = 3)}')
                    print(f'New center of rotation error (after flipping sample): {ppu.round_correct(offset_second_part, ndec = 3)}\n')

                    net_x_shifts_pcc[0, zero_deg_idx_array[1]:] -= offset_init_second_part

                    _, dx = phase_xcorr(aligned_proj[zero_deg_idx_array[0]], 
                                         aligned_proj[zero_deg_idx_array[1]], 
                                         sigma, 
                                         alpha, 
                                         upsample_factor)

                    if np.abs(dx - dx_prev) <= eps_cor_correction or (i == 0 and dx == 0):
                        print('No further COR correction needed.')

                        break
                    
                    print(f'Applying additional COR correction to pre-flipped, pre-remounted sample angles: {ppu.round_correct(-dx, ndec = 3)}')

                    net_x_shifts_pcc[0, zero_deg_idx_array[1]:] += dx

                    for theta_idx in range(len(theta_array_first_part)):
                        aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (0, net_x_shifts_pcc[0, zero_deg_idx_array[1]:]))

                    dx_prev = dx
                    
        # theta_idx_pairs_first_part = ppu.find_theta_combos(theta_array_first_part)
        # theta_idx_pairs_second_part = ppu.find_theta_combos(theta_array_second_part)
            
        # center_of_rotation_avg_first_part, center_geom, offset_init_first_part = rot_center_avg(aligned_proj[:second_zero_deg_idx], 
        #                                                                                         theta_idx_pairs_first_part, 
        #                                                                                         theta_array_first_part)
        

        # center_of_rotation_avg_second_part, _, offset_init_second_part = rot_center_avg(aligned_proj[second_zero_deg_idx:], 
        #                                                                                 theta_idx_pairs_second_part, 
        #                                                                                 theta_array_second_part)


        # print(f'Average center of rotation (before flipping sample): {center_of_rotation_avg_first_part}')
        # print(f'Average center of rotation (after flipping sample): {center_of_rotation_avg_second_part}\n')
        # print(f'Geometric center: {center_geom}\n')
        # print(f'Center of rotation error (before flipping sample): {ppu.round_correct(offset_init_first_part, ndec = 3)}')
        # print(f'Center of rotation error (after flipping sample): {ppu.round_correct(offset_init_second_part, ndec = 3)}\n')

        # print(f'Applying initial center of rotation correction for angles before sample flipped: {ppu.round_correct(-offset_init_first_part, ndec = 3)}')
        
        # for theta_idx in range(len(theta_array_first_part)):
        #     # aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (0, -offset_init_first_part))
        #     aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (0, -offset_init_first_part))

        # net_x_shifts_pcc[0, :second_zero_deg_idx] -= offset_init_first_part

        # print(f'Applying initial center of rotation correction for angles after sample flipped: {ppu.round_correct(-offset_init_second_part, ndec = 3)}')

        # for theta_idx in range(len(theta_array_second_part)):
        #     # aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (0, -offset_init_second_part))
        #     aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (0, -offset_init_second_part))

        # net_x_shifts_pcc[0, second_zero_deg_idx:] -= offset_init_second_part
    
        # print(f'Average center of rotation after initial COR correction (before flipping sample): {center_of_rotation_avg_first_part}')
        # print(f'Average center of rotation after initial COR correction (after flipping sample): {center_of_rotation_avg_second_part}\n')
        # print(f'Geometric center: {center_geom}\n')
        # print(f'Center of rotation error (before flipping sample): {ppu.round_correct(offset_init_first_part, ndec = 3)}')
        # print(f'Center of rotation error (after flipping sample): {ppu.round_correct(offset_init_second_part, ndec = 3)}\n')

        # theta_idx_array = np.arange(n_theta)
        
        # if zero_idx_to_discard == 'first':
        #     print(f'Discarding first 0° projection image from XRT, OD, and XRF data...')

        #     mask = theta_idx_array != first_zero_deg_idx

        # elif zero_idx_to_discard == 'second':
        #     print(f'Discarding second 0° projection image from XRT, OD, and XRF data...')
            
        #     mask = theta_idx_array != second_zero_deg_idx
        
        # theta_array_new = theta_array[mask]
        # aligned_proj_new = aligned_proj[mask]
        # # xrt_proj_img_array_new = xrt_proj_img_array[mask]
        # cropped_xrt_proj_img_array_new = cropped_xrt_proj_img_array[mask]
        # opt_dens_proj_img_array_new = opt_dens_proj_img_array[mask]
        # cropped_opt_dens_proj_img_array_new = cropped_opt_dens_proj_img_array[mask]
        # # xrf_proj_img_array_new = xrf_proj_img_array[:, mask]
        # cropped_xrf_proj_img_array_new = cropped_xrf_proj_img_array[:, mask]
        # net_x_shifts_pcc_new = net_x_shifts_pcc[:, mask]
        # net_y_shifts_pcc_new = net_y_shifts_pcc[:, mask]
        
        # theta_idx_pairs_new = ppu.find_theta_combos(theta_array_new)

    else:
        # theta_idx_pairs = ppu.find_theta_combos(theta_array, dtheta = 1)
        theta_idx_pairs = ppu.find_theta_combos(theta_array)

        center_of_rotation_avg, center_geom, offset_init = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)
    
        print(f'Average center of rotation: {center_of_rotation_avg}')
        print(f'Geometric center: {center_geom}')
        print(f'Center of rotation error: {ppu.round_correct(offset_init, ndec = 3)}')
        print(f'Applying initial center of rotation correction: {ppu.round_correct(-offset_init, ndec = 3)}')

        for theta_idx in range(n_theta):
            aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (0, -offset_init))

        center_of_rotation_avg, _, _ = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

        offset = center_of_rotation_avg - center_geom

        print(f'Center of rotation after initial COR correction: {ppu.round_correct(center_of_rotation_avg, ndec = 3)}')
        print(f'Geometric center: {center_geom}')
        print(f'Center of rotation error: {ppu.round_correct(offset, ndec = 3)}')
    
    # add_shift = -0.8
    
    # print(f'Shifting by additional {-add_shift} pixels...')

    # for theta_idx in range(n_theta):
        # aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (0, -(offset_init + add_shift)))

        net_x_shifts_pcc[0] -= offset_init

        # theta_array_new = theta_array
        # aligned_proj_new = aligned_proj
        # xrt_proj_img_array_new = xrt_proj_img_array
        # cropped_xrt_proj_img_array_new = cropped_xrt_proj_img_array
        # opt_dens_proj_img_array_new = opt_dens_proj_img_array
        # cropped_opt_dens_proj_img_array_new = cropped_opt_dens_proj_img_array
        # xrf_proj_img_array_new = xrf_proj_img_array
        # cropped_xrf_proj_img_array_new = cropped_xrf_proj_img_array
        # net_x_shifts_pcc_new = net_x_shifts_pcc
        # net_y_shifts_pcc_new = net_y_shifts_pcc

        # theta_idx_pairs_new = theta_idx_pairs

    _, n_slices_orig, n_columns_orig = xrt_proj_img_array.shape
    # n_theta, n_slices, n_columns = cropped_xrt_proj_img_array_new.shape
    n_theta, n_slices, n_columns = cropped_xrt_proj_img_array.shape

    aligned_proj_total_xrt = np.zeros((n_theta, n_slices, n_columns))
    aligned_proj_total_opt_dens = np.zeros((n_theta, n_slices, n_columns))
    aligned_proj_total_xrf = np.zeros((n_elements_xrf, n_theta, n_slices, n_columns))
    synth_proj = np.zeros((n_theta, n_slices, n_columns))
    dx_array_pcc = np.zeros((n_iterations_iter_reproj, n_theta))
    dy_array_pcc = np.zeros((n_iterations_iter_reproj, n_theta))

    rms_x_shift_prev = 0

    for i in range(n_iterations_iter_reproj):
        iterations.append(i)
        
        print(f'Iteration {i + 1}/{n_iterations_iter_reproj}')

        if i > 0:
            for theta_idx in range(n_theta):
                    # net_x_shift = net_x_shifts_pcc_new[i - 1, theta_idx]
                    # net_y_shift = net_y_shifts_pcc_new[i - 1, theta_idx]

                net_x_shift = net_x_shifts_pcc[i - 1, theta_idx]
                net_y_shift = net_y_shifts_pcc[i - 1, theta_idx]
                
                if (theta_idx % 7) == 0:
                    # print(f'Shifting projection by net x shift = {ppu.round_correct(net_x_shift, ndec = 3)} (theta = {ppu.round_correct(theta_array_new[theta_idx], ndec = 1)})...')
                    print(f'Shifting projection by net x shift = {ppu.round_correct(net_x_shift, ndec = 3)} (theta = {ppu.round_correct(theta_array[theta_idx], ndec = 1)})...')
                    print(f'Shifting projection by net y shift = {ppu.round_correct(net_y_shift, ndec = 3)}...')

                # aligned_proj_new[theta_idx] = ndi.shift(opt_dens_proj_img_array_new[theta_idx], shift = (net_y_shift, net_x_shift))
                # aligned_proj_new[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array_new[theta_idx], shift = (net_y_shift, net_x_shift))
                aligned_proj[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift))

            # center_of_rotation_avg, _, offset = rot_center_avg(aligned_proj_new, theta_idx_pairs_new, theta_array_new)
            center_of_rotation_avg, _, offset = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

            print(f'Center of rotation: {ppu.round_correct(center_of_rotation_avg, ndec = 3)}')
            print(f'Geometric center: {center_geom}')
            print(f'Center of rotation error: {ppu.round_correct(offset, ndec = 3)}')
    
        if return_aux_data:
            if edge_info is None:
                # aligned_exp_proj_array.append(aligned_proj_new.copy())
                aligned_exp_proj_array.append(aligned_proj.copy())
            else:
                # remapped_aligned_exp_proj = opt_dens_proj_img_array_new.copy()
                remapped_aligned_exp_proj = opt_dens_proj_img_array.copy()
                print(remapped_aligned_exp_proj.shape)
                start_slice = edge_info['top']
                end_slice = n_slices_orig - edge_info['bottom']

                start_column = edge_info['left']
                end_column = n_columns_orig - edge_info['right']

                # remapped_aligned_exp_proj[:, start_slice:end_slice, start_column:end_column] = aligned_proj_new.copy()
                remapped_aligned_exp_proj[:, start_slice:end_slice, start_column:end_column] = aligned_proj.copy()

                aligned_exp_proj_array.append(remapped_aligned_exp_proj)
                cropped_aligned_exp_proj_array.append(aligned_exp_proj_array.copy())

        # recon = tomo.recon(aligned_proj_new, theta_array_new*np.pi/180, algorithm = 'gridrec', filter_name = 'ramlak')
        recon = tomo.recon(aligned_proj, theta_array*np.pi/180, algorithm = 'gridrec', filter_name = 'ramlak')

        if return_aux_data:
            recon_array.append(recon)
        
        for slice_idx in range(n_slices):
            print(f'\rReprojecting slice {slice_idx + 1}/{n_slices}', end = '', flush = True)

            sinogram = (xform.radon(recon[slice_idx].copy(), theta_array)).T
            # sinogram = radon_manual(recon[slice_idx].copy(), theta_array)

            synth_proj[:, slice_idx, :] = sinogram
        
        if return_aux_data:
            synth_proj_array.append(synth_proj.copy())
        
        for theta_idx in range(n_theta):
            if not return_aux_data:
                # dy, dx = phase_xcorr(synth_proj[theta_idx], 
                #                      aligned_proj_new[theta_idx], 
                #                      sigma, 
                #                      alpha, 
                #                      upsample_factor)

                dy, dx = phase_xcorr(synth_proj[theta_idx], 
                                     aligned_proj[theta_idx], 
                                     sigma, 
                                     alpha, 
                                     upsample_factor)
            
            else:
                # dy, dx, pcc_2d = phase_xcorr(synth_proj[theta_idx], 
                #                              aligned_proj_new[theta_idx], 
                #                              sigma, 
                #                              alpha, 
                #                              upsample_factor, 
                #                              return_pcc_2d = True)

                dy, dx, pcc_2d = phase_xcorr(synth_proj[theta_idx], 
                                             aligned_proj[theta_idx], 
                                             sigma, 
                                             alpha, 
                                             upsample_factor, 
                                             return_pcc_2d = True)

                pcc_2d_array.append(pcc_2d)

            dx_array_pcc[i, theta_idx] = dx
            dy_array_pcc[i, theta_idx] = dy
            
            if i == 0:                 
                net_x_shifts_pcc_new[i, theta_idx] += dx
                net_y_shifts_pcc_new[i, theta_idx] += dy
            
            else:
                net_x_shifts_pcc_new[i, theta_idx] = net_x_shifts_pcc_new[i - 1, theta_idx] + dx
                net_y_shifts_pcc_new[i, theta_idx] = net_y_shifts_pcc_new[i - 1, theta_idx] + dy
            
            if (theta_idx % 7) == 0:
                if theta_idx == 0:
                    # print(f'\nCurrent x-shift: {ppu.round_correct(dx, ndec = 3)} (theta = {ppu.round_correct(theta_array_new[theta_idx], ndec = 1)})')
                    print(f'\nCurrent x-shift: {ppu.round_correct(dx, ndec = 3)} (theta = {ppu.round_correct(theta_array[theta_idx], ndec = 1)})')
                
                else:
                    #  print(f'Current x-shift: {ppu.round_correct(dx, ndec = 3)} (theta = {ppu.round_correct(theta_array_new[theta_idx], ndec = 1)})')
                    print(f'Current x-shift: {ppu.round_correct(dx, ndec = 3)} (theta = {ppu.round_correct(theta_array[theta_idx], ndec = 1)})')
                
                print(f'Current y-shift: {ppu.round_correct(dy, ndec = 3)}')


        if not sample_flipped_remounted_mid_experiment:
        # center_of_rotation_avg_synth, _, offset_synth = rot_center_avg(synth_proj, theta_idx_pairs_new, theta_array_new)
            center_of_rotation_avg_synth, _, offset_synth = rot_center_avg(synth_proj, theta_idx_pairs, theta_array)

            print(f'Average synthetic center of rotation after jitter, dynamic COR correction attempts: {ppu.round_correct(center_of_rotation_avg_synth, ndec = 3)}')
            print(f'Geometric center: {center_geom}')
            print(f'Center of rotation error: {ppu.round_correct(offset_synth, ndec = 3)}')
        
        else:
            center_of_rotation_avg_first_part, center_geom, offset_first_part = rot_center_avg(aligned_proj[:zero_deg_idx_array[1]], 
                                                                                               theta_idx_pairs_first_part, 
                                                                                               theta_array_first_part)
                
            center_of_rotation_avg_second_part, _, offset_second_part = rot_center_avg(aligned_proj[zero_deg_idx_array[1]:], 
                                                                                       theta_idx_pairs_second_part, 
                                                                                       theta_array_second_part)

            print(f'Synthetic center of rotation (before flipping sample): {center_of_rotation_avg_first_part}')
            print(f'Synthetic center of rotation (after flipping sample): {center_of_rotation_avg_second_part}\n')
            print(f'Geometric center: {center_geom}\n')
            print(f'Center of rotation error (before flipping sample): {ppu.round_correct(offset_first_part, ndec = 3)}')
            print(f'Center of rotation error (after flipping sample): {ppu.round_correct(offset_second_part, ndec = 3)}\n')


            _, _dx = phase_xcorr(synth_proj[zero_deg_idx_array[0]], 
                                 synth_proj[zero_deg_idx_array[1]], 
                                 sigma, 
                                 alpha, 
                                 upsample_factor)
            
            print(f'The two zero degree angles are offset from each other by {ppu.round_correct(np.abs(_dx), ndec = 3)} pixels')
        
        # if i == 1:
            # sys.exit()

        if i > (n_iter_converge - 1):
            rms_x_shift_current = np.sqrt((dx_array_pcc[i]**2).mean())  # Look at RMS shift per iteration for convergence (show that typical shifts between images are settling down)
            
            if np.abs(rms_x_shift_current - rms_x_shift_prev) <= eps_iter_reproj:
                iterations = np.array(iterations)

                dx_array_new = dx_array_pcc[:len(iterations)]
                dy_array_new = dy_array_pcc[:len(iterations)]
           
                # net_x_shifts_pcc_new_1 = net_x_shifts_pcc_new[:len(iterations)]
                # net_y_shifts_pcc_new_1 = net_y_shifts_pcc_new[:len(iterations)]
                
                net_x_shifts_pcc_new = net_x_shifts_pcc[:len(iterations)]
                net_y_shifts_pcc_new = net_y_shifts_pcc[:len(iterations)]

                print(f'Number of iterations taken: {len(iterations)}')
                print('Shifting all elements in cropped XRT, optical density aggregate projection arrays by current net shifts...')

                for theta_idx in range(n_theta):
                    # net_x_shift = net_x_shifts_pcc_new_1[i][theta_idx]
                    # net_y_shift = net_y_shifts_pcc_new_1[i][theta_idx]

                    net_x_shift = net_x_shifts_pcc_new[i][theta_idx]
                    net_y_shift = net_y_shifts_pcc_new[i][theta_idx]

                    # aligned_proj_total_xrt[theta_idx] = ndi.shift(xrt_proj_img_array_new[theta_idx], shift = (net_y_shift, net_x_shift), cval = I0)
                    # aligned_proj_total_opt_dens[theta_idx] = ndi.shift(opt_dens_proj_img_array_new[theta_idx], shift = (net_y_shift, net_x_shift))
                    
                    # aligned_proj_total_xrt[theta_idx] = ndi.shift(cropped_xrt_proj_img_array_new[theta_idx], shift = (net_y_shift, net_x_shift), cval = I0)
                    # aligned_proj_total_opt_dens[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array_new[theta_idx], shift = (net_y_shift, net_x_shift))

                    aligned_proj_total_xrt[theta_idx] = ndi.shift(cropped_xrt_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift), cval = I0)
                    aligned_proj_total_opt_dens[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift))

                print('Shifting all elements in cropped XRF aggregate projection array by current net shifts...')

                for element_idx in range(n_elements_xrf):
                    print(f'\rElement {element_idx + 1}/{n_elements_xrf}', end = '', flush = True)
                    
                    for theta_idx in range(n_theta):
                        # net_x_shift = net_x_shifts_pcc_new_1[i][theta_idx]
                        # net_y_shift = net_y_shifts_pcc_new_1[i][theta_idx]
                        net_x_shift = net_x_shifts_pcc_new[i][theta_idx]
                        net_y_shift = net_y_shifts_pcc_new[i][theta_idx]

                        # aligned_proj_total_xrf[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array_new[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))
                        aligned_proj_total_xrf[element_idx, theta_idx] = ndi.shift(cropped_xrf_proj_img_array[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))

                # net_x_shifts = net_x_shifts_pcc_new[i]
                # net_y_shifts = net_y_shifts_pcc_new_1[i]
                net_y_shifts = net_y_shifts_pcc_new[i]

                # if zero_idx_to_discard is not None:
                #     aligned_proj_xrt_final = xrt_proj_img_array[mask]
                #     aligned_proj_opt_dens_final = opt_dens_proj_img_array[mask]
                #     aligned_proj_xrf_final = xrf_proj_img_array[mask]
                        
                # else:
                #     aligned_proj_xrt_final = xrt_proj_img_array
                #     aligned_proj_opt_dens_final = opt_dens_proj_img_array
                #     aligned_proj_xrf_final = xrf_proj_img_array

                if np.any(net_y_shifts): # Any nonzero y shifts
                    print("\nTruncating cropped XRT, OD, XRF projection images in y so object is in every projection image's field of view...")

                    if edge_info is not None:
                        print("...and remapping cropped projection images back onto original counterparts...")

                    start_array = np.array([int(max(0, -np.ceil(y_shift))) for y_shift in net_y_shifts])
                    end_array = np.array([int(min(n_slices, n_slices - np.floor(y_shift))) for y_shift in net_y_shifts])

                    start_slice = start_array.max()
                    end_slice = end_array.min()
                    
                    if edge_info is not None:
                        start_slice_total = start_slice + edge_info['top']
                        end_slice_total = n_slices_orig - edge_info['bottom'] - end_slice

                        if start_slice_total + end_slice_total >= n_slices:
                            print('Error: Overlapping edge, field-of-view crops in y. Exiting program...')

                            sys.exit()

                        start_column = edge_info['left']
                        end_column = n_columns_orig - edge_info['right']

                        aligned_proj_xrt_final[:, start_slice_total:end_slice_total, start_column:end_column] = aligned_proj_total_xrt[:, start_slice:end_slice]
                        aligned_proj_opt_dens_final[:, start_slice_total:end_slice_total, start_column:end_column] = aligned_proj_total_opt_dens[:, start_slice:end_slice]
                        aligned_proj_xrf_final[:, :, start_slice_total:end_slice_total, start_column:end_column] = aligned_proj_total_xrf[:, :, start_slice:end_slice]

                else:
                    if edge_info is not None:
                        print("\nRemapping cropped XRT, OD, XRF projection images back onto original counterparts...")
                        
                        start_slice = edge_info['top']
                        end_slice = n_slices_orig - edge_info['bottom']

                        start_column = edge_info['left']
                        end_column = n_columns_orig - edge_info['right']

                        aligned_proj_xrt_final[:, start_slice:end_slice, start_column:end_column] = aligned_proj_total_xrt[:, start_slice:end_slice]
                        aligned_proj_opt_dens_final[:, start_slice:end_slice, start_column:end_column] = aligned_proj_total_opt_dens[:, start_slice:end_slice]
                        aligned_proj_xrf_final[:, :, start_slice:end_slice, start_column:end_column] = aligned_proj_total_xrf[:, :, start_slice:end_slice]
                    
                    else:
                        aligned_proj_xrt_final = aligned_proj_total_xrt
                        aligned_proj_opt_dens_final = aligned_proj_total_opt_dens
                        aligned_proj_xrf_final = aligned_proj_total_xrf

                break
            
            rms_x_shift_prev = rms_x_shift_current

        if i == n_iterations_iter_reproj - 1:
            print('Iterative reprojection complete. Shifting XRT intensities and optical densities by current net shifts...')

            iterations = np.array(iterations)

            dx_array_new, dy_array_new = dx_array_pcc, dy_array_pcc
            net_x_shifts_pcc_new_1, net_y_shifts_pcc_new_1 = net_x_shifts_pcc_new, net_y_shifts_pcc_new
            
            for theta_idx in range(n_theta):
                net_x_shift = net_x_shifts_pcc_new_1[i]
                net_y_shift = net_y_shifts_pcc_new_1[i]

                # aligned_proj_total_xrt[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array_new[theta_idx], shift = (net_y_shift, net_x_shift), cval = I0)
                # aligned_proj_total_opt_dens[theta_idx] = ndi.shift(cropped_xrt_proj_img_array_new[theta_idx], shift = (net_y_shift, net_x_shift))

                aligned_proj_total_xrt[theta_idx] = ndi.shift(cropped_xrt_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift), cval = I0)
                aligned_proj_total_opt_dens[theta_idx] = ndi.shift(cropped_opt_dens_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift))
            
            print('Shifting all XRF elements by current net shifts...')

            for element_idx in range(n_elements_xrf):
                print(f'\rElement {element_idx + 1}/{n_elements_xrf}', end = '', flush = True)
                
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pcc_new_1[i, theta_idx]
                    net_y_shift = net_y_shifts_pcc_new_1[i, theta_idx]
                        
                    # aligned_proj_total_xrf[element_idx, theta_idx] = ndi.shift(cropped_xrf_proj_img_array_new[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))
                    aligned_proj_total_xrf[element_idx, theta_idx] = ndi.shift(cropped_xrf_proj_img_array[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))
            
            
            # net_x_shifts = net_x_shifts_pcc_new[i]
            net_y_shifts = net_y_shifts_pcc_new_1[i]

            # if zero_idx_to_discard is not None:
            #     aligned_proj_xrt_final = xrt_proj_img_array[mask]
            #     aligned_proj_opt_dens_final = opt_dens_proj_img_array[mask]
            #     aligned_proj_xrf_final = xrf_proj_img_array[mask]
                        
            # else:
            #     aligned_proj_xrt_final = xrt_proj_img_array
            #     aligned_proj_opt_dens_final = opt_dens_proj_img_array
            #     aligned_proj_xrf_final = xrf_proj_img_array

            if np.any(net_y_shifts): # Any nonzero y shifts
                print("\nTruncating cropped XRT, OD, XRF projection images in y so object is in every projection image's field of view...")

                if edge_info is not None:
                    print("...and remapping cropped projection images back onto original counterparts...")

                start_array = np.array([int(max(0, -np.ceil(y_shift))) for y_shift in net_y_shifts])
                end_array = np.array([int(min(n_slices, n_slices - np.floor(y_shift))) for y_shift in net_y_shifts])

                start_slice = start_array.max()
                end_slice = end_array.min()
                    
                if edge_info is not None:
                    start_slice_total = start_slice + edge_info['top']
                    end_slice_total = n_slices_orig - edge_info['bottom'] - end_slice

                    if start_slice_total + end_slice_total >= n_slices:
                        print('Error: Overlapping edge, field-of-view crops in y. Exiting program...')

                        sys.exit()

                    start_column = edge_info['left']
                    end_column = n_columns_orig - edge_info['right']

                    aligned_proj_xrt_final[:, start_slice_total:end_slice_total, start_column:end_column] = aligned_proj_total_xrt[:, start_slice:end_slice]
                    aligned_proj_opt_dens_final[:, start_slice_total:end_slice_total, start_column:end_column] = aligned_proj_total_opt_dens[:, start_slice:end_slice]
                    aligned_proj_xrf_final[:, :, start_slice_total:end_slice_total, start_column:end_column] = aligned_proj_total_xrf[:, :, start_slice:end_slice]

            else:
                if edge_info is not None:
                    print("\nRemapping cropped XRT, OD, XRF projection images back onto original counterparts...")
                        
                    start_slice = edge_info['top']
                    end_slice = n_slices_orig - edge_info['bottom']

                    start_column = edge_info['left']
                    end_column = n_columns_orig - edge_info['right']

                    aligned_proj_xrt_final[:, start_slice:end_slice, start_column:end_column] = aligned_proj_total_xrt[:, start_slice:end_slice]
                    aligned_proj_opt_dens_final[:, start_slice:end_slice, start_column:end_column] = aligned_proj_total_opt_dens[:, start_slice:end_slice]
                    aligned_proj_xrf_final[:, :, start_slice:end_slice, start_column:end_column] = aligned_proj_total_xrf[:, :, start_slice:end_slice]
                    
                else:
                    aligned_proj_xrt_final = aligned_proj_total_xrt
                    aligned_proj_opt_dens_final = aligned_proj_total_opt_dens
                    aligned_proj_xrf_final = aligned_proj_total_xrf

                    print('\nDone')

    if return_aux_data:
        aligned_exp_proj_array = np.array(aligned_exp_proj_array)
        synth_proj_array = np.array(synth_proj_array)
        pcc_2d_array = np.array(pcc_2d_array)
        recon_array = np.array(recon_array)

        if edge_info is not None:
            # return aligned_proj_xrt_final, \
            #        aligned_proj_opt_dens_final, \
            #        aligned_proj_xrf_final, \
            #        net_x_shifts_pcc_new_1[-1], \
            #        net_y_shifts_pcc_new_1[-1], \
            #        aligned_exp_proj_array, \
            #        cropped_aligned_exp_proj_array, \
            #        synth_proj_array, \
            #        pcc_2d_array, \
            #        recon_array, \
            #        theta_array_new, \
            #        net_x_shifts_pcc_new_1, \
            #        net_y_shifts_pcc_new_1, \
            #        dx_array_new, \
            #        dy_array_new

            return aligned_proj_xrt_final, \
                   aligned_proj_opt_dens_final, \
                   aligned_proj_xrf_final, \
                   net_x_shifts_pcc_new[-1], \
                   net_y_shifts_pcc_new[-1], \
                   aligned_exp_proj_array, \
                   cropped_aligned_exp_proj_array, \
                   synth_proj_array, \
                   pcc_2d_array, \
                   recon_array, \
                   net_x_shifts_pcc_new, \
                   net_y_shifts_pcc_new, \
                   dx_array_new, \
                   dy_array_new
        
        else:
            # return aligned_proj_xrt_final, \
            #        aligned_proj_opt_dens_final, \
            #        aligned_proj_xrf_final, \
            #        net_x_shifts_pcc_new_1[-1], \
            #        net_y_shifts_pcc_new_1[-1], \
            #        aligned_exp_proj_array, \
            #        synth_proj_array, \
            #        pcc_2d_array, \
            #        recon_array, \
            #        theta_array_new, \
            #        net_x_shifts_pcc_new_1, \
            #        net_y_shifts_pcc_new_1, \
            #        dx_array_new, \
            #        dy_array_new

            return aligned_proj_xrt_final, \
                   aligned_proj_opt_dens_final, \
                   aligned_proj_xrf_final, \
                   net_x_shifts_pcc_new[-1], \
                   net_y_shifts_pcc_new[-1], \
                   aligned_exp_proj_array, \
                   synth_proj_array, \
                   pcc_2d_array, \
                   recon_array, \
                   net_x_shifts_pcc_new, \
                   net_y_shifts_pcc_new, \
                   dx_array_new, \
                   dy_array_new
    
    # return aligned_proj_xrt_final, \
    #        aligned_proj_opt_dens_final, \
    #        aligned_proj_xrf_final, \
    #        theta_array_new, \
    #        net_x_shifts_pcc_new_1[-1], \
    #        net_y_shifts_pcc_new_1[-1]

    return aligned_proj_xrt_final, \
           aligned_proj_opt_dens_final, \
           aligned_proj_xrf_final, \
           net_x_shifts_pcc_new_1[-1], \
           net_y_shifts_pcc_new_1[-1]