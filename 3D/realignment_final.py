import numpy as np, \
       tomopy as tomo, \
       xrf_xrt_preprocess_utils as ppu, \
       sys

from skimage import transform as xform, registration as reg
from scipy import ndimage as ndi, fft
from matplotlib import pyplot as plt
from itertools import combinations as combos

# TODO Align other elements while keeping padding (good practice)

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def round_correct(num, ndec): # CORRECTLY round a number (num) to chosen number of decimal places (ndec)
    if ndec == 0:
        return int(num + 0.5)
    
    else:
        digit_value = 10**ndec
        
        if num > 0:
            return int(num*digit_value + 0.5)/digit_value
        
        else:
            return int(num*digit_value - 0.5)/digit_value

def find_theta_combos(theta_array_deg, dtheta):
    '''
    
    Make sure angles are in degrees!

    '''

    theta_array_idx_pairs = list(combos(np.arange(len(theta_array_deg)), 2)) # Generate a list of all pairs of theta_array indices

    valid_theta_idx_pairs = [(theta_idx_1, theta_idx_2) for theta_idx_1, theta_idx_2 in theta_array_idx_pairs 
                             if (180 - dtheta <= np.abs(theta_array_deg[theta_idx_1] - theta_array_deg[theta_idx_2]) <= 180 + dtheta)]
                            # Compound inequality syntax is acceptable in Python in certain cases

    return valid_theta_idx_pairs

def phase_xcorr(recon_proj, exp_proj, upsample_factor, return_pccc_2d = False):
    n_columns = recon_proj.shape[1]
    n_slices = recon_proj.shape[0]

    recon_proj_filtered = ppu.edge_gauss_filter(recon_proj, sigma = 5, alpha = 10, nx = n_columns, ny = n_slices)
    exp_proj_filtered = ppu.edge_gauss_filter(exp_proj, sigma = 5, alpha = 10, nx = n_columns, ny = n_slices)
    
    if return_pccc_2d:
        recon_proj_fft = fft.fft2(recon_proj_filtered)
        orig_proj_fft = fft.fft2(exp_proj_filtered)
        
        # NOTE: Most likely, there will a slight variation in the following array relative to skimage.registration.phase_cross_correlation();
        # however, for rough estimates of PCC for diagnostics, for instance, that variation can be ignored.
        pcc_2d = np.abs(fft.ifft2(recon_proj_fft*orig_proj_fft.conjugate()/np.abs(recon_proj_fft*orig_proj_fft.conjugate())))

    shift, _, _ = reg.phase_cross_correlation(reference_image = recon_proj_filtered, moving_image = exp_proj_filtered, upsample_factor = upsample_factor)

    y_shift, x_shift = shift[0], shift[1]

    if pcc_2d:
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

    real = T[Nz].real
    imag = T[Nz].imag

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

        print(f'Center of rotation ({theta_array[theta_pair[0]]} degrees, {theta_array[theta_pair[1]]} degrees) = {round_correct(center_of_rotation, ndec = 3)}')

        center_of_rotation_sum += center_of_rotation
    
    center_rotation_avg = center_of_rotation_sum/len(theta_pair_array)

    geom_center = n_columns//2

    offset = center_rotation_avg - geom_center

    return center_rotation_avg, geom_center, offset

def iter_reproj(xrt_proj_img_array,
                opt_dens_proj_img_array,
                xrf_proj_img_array,
                theta_array,
                I0,
                n_iterations,
                init_x_shift = None, 
                init_y_shift = None,
                eps = 0.3,
                return_aux_data = False):
    
    '''

    iter_reproj: Perform iterative reprojection for joint realignment of XRF, XRT tomography datasets

    ------
    Inputs
    ------
    
    theta_array: Array of projection angles (array-like; dtype: float)
    
    xrt_proj_img_array: 3D XRT tomography data (elements, projection angles, slices, scan positions) (array-like; dtype: float)

    opt_dens_proj_img_array: 3D optical density data derived from xrt_proj_img_array (projection angles, slices, scan positions)
    
    n_iterations: Maximum number of iterative reprojection iterations (dtype: int)
    
    init_x_shift: Initial x-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)
    
    init_y_shift: Initial y-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)
    
    eps: Desired differential shift for convergence criterion (dtype: float)

    return_aux_data: Flag for returning per-iteration auxiliary data (default: False)


    -------
    Outputs
    -------
    
    aligned_proj_total: 4D XRF tomography data (XRT data will come later) corrected for per-projection jitter, center of rotation misalignment (array-like; dtype: float)

    aligned_exp_proj_array: 1D array of experimental 3D XRF tomography data arrays for each iteration for ref_element (array-like; dtype: float)

    synth_proj_array: 1D array of synthetic 3D XRF tomography data arrays for each iteration for ref_element (array-like; dtype: float)

    recon_array: 1D array of 3D reconstruction slices for each iteration for ref_element (array-like; dtype: float)

    net_x_shifts_pcc_new: Array of net x shifts with dimensions (n_iterations, n_theta) (array-like; dtype: float) (Note: n_iterations can be smaller the more quickly iter_reproj() converges)
    
    net_y_shifts_pcc_new: Array of net y shifts with dimensions (n_iterations, n_theta) (array-like; dtype: float) (Note: n_iterations can be smaller the more quickly iter_reproj() converges)

    '''

    n_elements_xrf = xrf_proj_img_array.shape[0]
    n_theta, n_slices, n_columns = xrt_proj_img_array.shape

    print('XRF Tomography dataset dimensions: ' + str(xrt_proj_img_array.shape))
    
    aligned_proj_total_xrt = np.zeros((n_theta, n_slices, n_columns))
    aligned_proj_total_opt_dens = np.zeros((n_theta, n_slices, n_columns))
    aligned_proj_total_xrf = np.zeros((n_elements_xrf, n_theta, n_slices, n_columns))
    aligned_proj = np.zeros((n_theta, n_slices, n_columns))
    synth_proj = np.zeros((n_theta, n_slices, n_columns))
    dx_array_pcc = np.zeros((n_iterations, n_theta))
    dy_array_pcc = np.zeros((n_iterations, n_theta))
    net_x_shifts_pcc = np.zeros((n_iterations, n_theta))
    net_y_shifts_pcc = np.zeros((n_iterations, n_theta))
    
    iterations = []
    
    if return_aux_data:
        aligned_exp_proj_array = []
        synth_proj_array = []
        pcc_2d_array = []
        recon_array = []
    
    n_iter_converge = 3

    if init_x_shift is None:
        init_x_shift = np.zeros(n_theta)
        
    if init_y_shift is None:
        init_y_shift = np.zeros(n_theta)
    
    if np.isscalar(init_x_shift):
        init_x_shift *= np.ones(n_theta)
        
    if np.isscalar(init_y_shift):
        init_y_shift *= np.ones(n_theta)
    
    aligned_proj = opt_dens_proj_img_array.copy()

    if np.any(init_x_shift) or np.any(init_y_shift):
        if np.any(init_x_shift) and np.any(init_y_shift):
            print('Executing intial shift(s) in x and y')
            
            net_x_shifts_pcc[0] += init_x_shift
            net_y_shifts_pcc[0] += init_y_shift

            for theta_idx in range(n_theta):
                aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (init_y_shift[theta_idx], init_x_shift[theta_idx]))
        
        elif np.any(init_x_shift):
            print('Executing initial shift(s) in x')
            
            net_x_shifts_pcc[0] += init_x_shift

            for theta_idx in range(n_theta):
                aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (0, init_x_shift[theta_idx]))
                
        else:
            print('Executing initial shift(s) in x')
            
            net_y_shifts_pcc[0] += init_y_shift

            for theta_idx in range(n_theta):
                aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (init_y_shift[theta_idx], 0))

    theta_idx_pairs = find_theta_combos(theta_array, dtheta = 1)

    center_of_rotation_avg, center_geom, offset_init = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

    print(f'Geometric center: {center_geom}')
    print(f'Center of rotation error: {round_correct(offset_init, ndec = 3)}')
    print(f'Applying initial center of rotation correction: {round_correct(-offset_init, ndec = 3)}')

    for theta_idx in range(n_theta):
        aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (0, -offset_init))

    center_of_rotation_avg, _, _ = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

    offset = center_of_rotation_avg - center_geom

    print(f'Center of rotation after initial COR correction: {round_correct(center_of_rotation_avg, ndec = 3)}')
    print(f'Geometric center: {center_geom}')
    print(f'Center of rotation error: {round_correct(offset, ndec = 3)}')
    
    # add_shift = -0.8
    
    # print(f'Shifting by additional {-add_shift} pixels...')

    # for theta_idx in range(n_theta):
        # aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (0, -(offset_init + add_shift)))

    net_x_shifts_pcc[0] -= offset_init

    for i in range(n_iterations):
        iterations.append(i)
        
        print(f'Iteration {i + 1}/{n_iterations}')

        if i > 0:
            for theta_idx in range(n_theta):
                net_x_shift = net_x_shifts_pcc[i - 1, theta_idx]
                net_y_shift = net_y_shifts_pcc[i - 1, theta_idx]
                
                if (theta_idx % 7) == 0:
                    print(f'Shifting projection by net x shift = {round_correct(net_x_shift, ndec = 3)} (theta = {round_correct(theta_array[theta_idx], ndec = 1)})...')
                    print(f'Shifting projection by net y shift = {round_correct(net_y_shift, ndec = 3)}...')

                aligned_proj[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift))

            center_of_rotation_avg, _, offset = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

            print(f'Center of rotation after initial COR correction: {round_correct(center_of_rotation_avg, ndec = 3)}')
            print(f'Geometric center: {center_geom}')
            print(f'Center of rotation error: {round_correct(offset, ndec = 3)}')
    
        if return_aux_data:
            aligned_exp_proj_array.append(aligned_proj.copy())

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
                dy, dx = phase_xcorr(synth_proj[theta_idx], aligned_proj[theta_idx], upsample_factor = 100)
            
            else:
                dy, dx, pcc_2d = phase_xcorr(synth_proj[theta_idx], aligned_proj[theta_idx], upsample_factor = 100, return_pccc_2d = True)

                pcc_2d_array.append(pcc_2d)

            dx_array_pcc[i, theta_idx] = dx
            dy_array_pcc[i, theta_idx] = dy
            
            if i == 0:                 
                net_x_shifts_pcc[i, theta_idx] += dx
                net_y_shifts_pcc[i, theta_idx] += dy
            
            else:
                net_x_shifts_pcc[i, theta_idx] = net_x_shifts_pcc[i - 1, theta_idx] + dx
                net_y_shifts_pcc[i, theta_idx] = net_y_shifts_pcc[i - 1, theta_idx] + dy
            
            if (theta_idx % 7) == 0:
                print(f'Current x-shift: {round_correct(dx, ndec = 3)} (theta = {round_correct(theta_array[theta_idx], ndec = 1)})')
                print(f'Current y-shift: {round_correct(dy, ndec = 3)}')

        center_of_rotation_avg_synth, _, offset_synth = rot_center_avg(synth_proj, theta_idx_pairs, theta_array)

        print(f'Average synthetic center of rotation after jitter, dynamic COR correction attempts: {round_correct(center_of_rotation_avg_synth, ndec = 3)}')
        print(f'Geometric center: {center_geom}')
        print(f'Center of rotation error: {round_correct(offset_synth, ndec = 3)}')
        
        if i == 1:
            sys.exit()

        if i > (n_iter_converge - 1):
            rms_x_shift_current = np.sqrt((dx_array_pcc[i]**2).mean())  # Look at RMS shift per iteration for convergence (show that typical shifts between images are settling down)
            
            if np.abs(rms_x_shift_current - rms_x_shift_prev) <= eps:
                iterations = np.array(iterations)

                dx_array_new = dx_array_pcc[:len(iterations)]
                dy_array_new = dy_array_pcc[:len(iterations)]
           
                net_x_shifts_pcc_new = net_x_shifts_pcc[:len(iterations)]
                net_y_shifts_pcc_new = net_y_shifts_pcc[:len(iterations)]

                print(f'Number of iterations taken: {len(iterations)}')
                print('Shifting all elements in XRT aggregate projection array by current net shifts...')

                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pcc_new[i]
                    net_y_shift = net_y_shifts_pcc_new[i]

                    aligned_proj_total_xrt[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift), cval = I0)
                    aligned_proj_total_opt_dens[theta_idx] = ndi.shift(xrt_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift))
                    
                print('Shifting all elements in XRF aggregate projection array by current net shifts...')

                for element_idx in range(n_elements_xrf):
                    for theta_idx in range(n_theta):
                        net_x_shift = net_x_shifts_pcc_new[i]
                        net_y_shift = net_y_shifts_pcc_new[i]

                        aligned_proj_total_xrf[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))

                print('Done')

                break
            
            rms_x_shift_prev = rms_x_shift_current.copy()

        if i == n_iterations - 1:
            print('Iterative reprojection complete. Shifting XRT intensities and optical densities by current net shifts...')

            iterations = np.array(iterations)

            dx_array_new, dy_array_new = dx_array_pcc, dy_array_pcc
            net_x_shifts_pcc_new, net_y_shifts_pcc_new = net_x_shifts_pcc, net_y_shifts_pcc
            
            for theta_idx in range(n_theta):
                net_x_shift = net_x_shifts_pcc_new[i]
                net_y_shift = net_y_shifts_pcc_new[i]

                aligned_proj_total_xrt[theta_idx] = ndi.shift(opt_dens_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift), cval = I0)
                aligned_proj_total_opt_dens[theta_idx] = ndi.shift(xrt_proj_img_array[theta_idx], shift = (net_y_shift, net_x_shift))
            
            print('Shifting all XRF elements by current net shifts...')

            for element_idx in range(n_elements_xrf):
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pcc_new[i, theta_idx]
                    net_y_shift = net_y_shifts_pcc_new[i, theta_idx]
                        
                    aligned_proj_total_xrf[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))
            
            print('Done')

    if return_aux_data:
        aligned_exp_proj_array = np.array(aligned_exp_proj_array)
        synth_proj_array = np.array(synth_proj_array)
        pcc_2d_array = np.array(pcc_2d_array)
        recon_array = np.array(recon_array)

        return aligned_proj_total_xrt, \
               aligned_proj_total_opt_dens, \
               aligned_proj_total_xrf, \
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
    
    return aligned_proj_total_xrt, \
           aligned_proj_total_opt_dens, \
           aligned_proj_total_xrf, \
           net_x_shifts_pcc_new[-1], \
           net_y_shifts_pcc_new[-1]

# file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
# file_path_xrt = '/home/bwr0835/2_ide_aggregate_xrt.h5'

# output_dir_path_base = '/home/bwr0835'

# # output_file_name_base = 'gridrec_5_iter_vacek_cor_and_shift_correction_padding_-22_deg_158_deg'
# # output_file_name_base = 'xrt_mlem_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025'
# # output_file_name_base = 'xrt_mlem_1_iter_manual_shift_-20_no_log_tomopy_default_cor_w_padding_07_09_2025'
# output_file_name_base = 'xrt_gridrec_6_iter_dynamic_ps_cor_correction_log_w_padding_gridrec_cor_299_5_aug_04_2025'
# # output_file_name_base = 'xrt_gridrec_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025'

# if output_file_name_base == '':
#     print('No output base file name chosen. Ending program...')

#     sys.exit()

# # create_aggregate_xrf_h5(file_path_array, file_path_xrf, synchrotron = 'aps')

# try:
#     elements_xrt, counts_xrt, theta_xrt, dataset_type_xrt, filenames = util.extract_h5_aggregate_xrt_data(file_path_xrt)

# except:
#     print('Cannot open XRT HDF5 file')
    
#     sys.exit()

# try:
#     elements_xrf, counts_xrf, _, dataset_type_xrf = util.extract_h5_aggregate_xrt_data(file_path_xrf)

# except:
#     print('Cannot open XRF HDF5 file')

#     sys.exit()

# phi_inc = 8.67768e5
# t_dwell_s = 0.01 

# counts_inc = phi_inc*t_dwell_s

# desired_element = 'ds_ic'
# desired_element_idx = elements_xrt.index(desired_element)

# nonzero_mask = counts_xrt[desired_element_idx] > 0

# counts_xrt[desired_element_idx][nonzero_mask] = -np.log(counts_xrt[desired_element_idx][nonzero_mask]/counts_inc)

# n_theta, n_slices, n_columns = counts_xrt.shape

# init_x_shift = 0

# n_desired_iter = 6 # For the reprojection scheme, NOT for reconstruction by itself

# algorithm = 'gridrec'

# if (n_slices % 2) or (n_columns % 2): # Padding for odd-integer detector positions and/or slices
#     if (n_slices % 2) and (n_columns % 2):
#         xrt_proj_img_array = pad_col_row(counts_xrt)
#         xrf_proj_img_array = pad_col_row(counts_xrf)
            
#         n_slices += 1
#         n_columns += 1
        
#     elif n_slices % 2:
#         xrt_proj_img_array = pad_row(counts_xrt)
#         xrf_proj_img_array = pad_row(counts_xrf)

#         n_slices += 1

#     else:

#         xrt_proj_img_array = pad_col(counts_xrt)
#         xrf_proj_img_array = pad_col(counts_xrf)

#         n_columns += 1

# orig_proj_ref, \
# aligned_proj_total_xrt, \
# aligned_proj_total_opt_dens, \
# aligned_proj_total_xrf, \
# aligned_exp_proj_array, \
# synth_proj_array, \
# recon_array, \
# net_x_shifts, \
# net_y_shifts, \
# dx_array, \
# dy_array = iter_reproj(desired_element, 
#                        elements_xrt,
#                        theta_xrt,
#                        counts_inc, 
#                        counts_xrt, 
#                        counts_xrf,
#                        algorithm, 
#                        n_desired_iter)

# print('Saving XRT aux files...')

# full_output_dir_path = os.path.join(output_dir_path_base, 'iter_reproj', output_file_name_base)

# os.makedirs(full_output_dir_path, exist_ok = True)

# np.save(os.path.join(full_output_dir_path, 'theta_array.npy'), theta_xrt)
# np.save(os.path.join(full_output_dir_path, 'aligned_proj_all_elements.npy'), aligned_proj_total_xrt)
# np.save(os.path.join(full_output_dir_path, 'aligned_proj_array_iter_' + desired_element + '.npy'), aligned_exp_proj_array)
# np.save(os.path.join(full_output_dir_path, 'synth_proj_array_iter_' + desired_element + '.npy'), synth_proj_array)
# np.save(os.path.join(full_output_dir_path, 'recon_array_iter_' + desired_element + '.npy'), recon_array)
# np.save(os.path.join(full_output_dir_path, 'net_x_shifts_' + desired_element + '.npy'), net_x_shifts)
# np.save(os.path.join(full_output_dir_path, 'net_y_shifts_' + desired_element + '.npy'), net_y_shifts)
# np.save(os.path.join(full_output_dir_path, 'orig_exp_proj_' + desired_element + '.npy'), orig_proj_ref)

# print('Exporting aligned XR and optical density data to HDF5 file...')

# with h5py.File(os.path.join(output_dir_path_base, '2_ide_aggregate_xrt_aligned.h5'), 'w') as f:
#     exchange = f.create_group('exchange')
#     file_info = f.create_group('corresponding_file_info')

#     exchange.create_dataset('data', data = aligned_proj_total_xrt, compression = 'gzip', compression_opts = 6)
#     exchange.create_dataset('elements', data = elements_xrt)
#     exchange.create_dataset('theta', data = theta_xrt)
        
#     file_info.create_dataset('filenames', data = filenames)
#     file_info.create_dataset('dataset_type', data = 'xrt')

# print('Exporting aligned XRF data to HDF5 file...')

# with h5py.File(os.path.join(output_dir_path_base, '2_ide_aggregate_xrf_aligned.h5'), 'w') as f:
#     exchange = f.create_group('exchange')
#     file_info = f.create_group('corresponding_file_info')

#     exchange.create_dataset('data', data = aligned_proj_total_xrf, compression = 'gzip', compression_opts = 6)
#     exchange.create_dataset('elements', data = elements_xrf)
#     exchange.create_dataset('theta', data = theta_xrt)
        
#     file_info.create_dataset('filenames', data = filenames)
#     file_info.create_dataset('dataset_type', data = 'xrf')

# print('Done')