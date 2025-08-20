import numpy as np, h5py, os, sys, tkinter as tk, tomopy as tomo, csv, h5_util as util, warnings

from skimage import transform as xform, registration as reg
from scipy import ndimage as ndi, fft
from matplotlib import pyplot as plt
from itertools import combinations as combos

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

def pad_col_row(array):
    array_new = np.zeros((array.shape[0], array.shape[1], array.shape[2] + 1, array.shape[3] + 1))

    for element_idx in range(array.shape[0]):
        for theta_idx in range(array.shape[1]):
            final_column = array[element_idx, theta_idx, :, -1].reshape(-1, 1) # Reshape to column vector (-1 means Python automatically determines missing dimension based on original orray length)
            
            # print(final_column.shape)
            # print(array.shape)

            array_temp = np.hstack((array[element_idx, theta_idx, :, :], final_column))
            
            final_row = array_temp[-1, :]

            array_new[element_idx, theta_idx, :, :] = np.vstack((array_temp, final_row))

    return array_new

def pad_col(array):
    for element_idx in range(array.shape[0]):
        for theta_idx in range(array.shape[1]):
            final_column = array[element_idx, theta_idx, :, -1].T
                
            array[element_idx, theta_idx, :, :] = np.hstack((array[element_idx, theta_idx, :, :], final_column))

    return array

def pad_row(array):
    for element_idx in range(array.shape[0]):
        for theta_idx in range(array.shape[1]):
            final_row = array[element_idx, theta_idx, -1, :]
                
            array[element_idx, theta_idx, :, :] = np.vstack((array[element_idx, theta_idx, :, :], final_row))

    return array

def edge_gauss_filter(image, sigma, alpha, nx, ny):
    n_rolloff = int(0.5 + alpha*sigma)
    
    if n_rolloff > 2:
        exp_arg =  np.arange(n_rolloff)/float(sigma)

        rolloff = 1 - np.exp(-0.5*exp_arg**2)
    
    edge_total = 0
        
    # Bottom edge
 
    y1 = ny - sigma
    y2 = ny

    edge_total = edge_total + np.sum(image[y1:y2, 0:nx])

    # Top edge
        
    y3 = 0
    y4 = sigma
        
        
    edge_total = edge_total + np.sum(image[y3:y4, 0:nx])

    # Left edge

    x1 = 0
    x2 = sigma

    edge_total = edge_total + np.sum(image[y4:y1, x1:x2])

    # Right edge

    x3 = nx - sigma
    x4 = nx

    edge_total = edge_total + np.sum(image[y4:y1, x3:x4])

    n_pixels = 2*sigma*(nx + ny - 2*sigma) # Total number of edge pixels for this "vignetting"
                                           # sigma*nx + sigma*nx + [(ny - sigma) - sigma]*sigma + (ny - sigma) - sigma]*sigma
    
    dc_value = edge_total/n_pixels # Average of edge_total over total number of edge pixels

    image = np.copy(image) - dc_value
    
    # Top edge

    xstart = 0
    xstop = nx - 1
    idy = 0

    for i_rolloff in range(n_rolloff):
        image[idy, xstart:(xstop+1)] = image[idy, xstart:(xstop+1)]*rolloff[idy]
        
        if xstart < (nx/2 - 1):
            xstart += 1
        
        if xstop > (nx/2):
            xstop -= 1
        
        if idy < (ny - 1):
            idy += 1

    # Bottom edge
    
    xstart = 0
    xstop = nx - 1
    idy = ny - 1

    for i_rolloff in range(n_rolloff):
        image[idy, xstart:(xstop+1)] = image[idy, xstart:(xstop+1)]*rolloff[ny - 1 - idy]

        if xstart < (nx/2 - 1):
            xstart += 1
        
        if xstop > nx/2:
            xstop -= 1
        
        if idy > 0:
            idy -= 1

    # Left edge

    ystart = 1
    ystop = ny - 2
    idx = 0

    for i_rolloff in range(n_rolloff):
        image[ystart:(ystop+1), idx] = image[ystart:(ystop+1), idx]*rolloff[idx]

        if ystart < (ny/2 - 1):
            ystart += 1
       
        if ystop > (ny/2):
            ystop -= 1
        
        if idx < (nx - 1):
            idx += 1

    # Right edge

    ystart = 1
    ystop = ny - 2
    idx = nx - 1

    for i_rolloff in range(n_rolloff):
        image[ystart:(ystop+1), idx] = image[ystart:(ystop+1), idx]*rolloff[nx - 1 - idx]

        if ystart < (ny/2 - 1):
            ystart += 1
        
        if ystop > (ny/2):
            ystop -= 1
        
        if idx > 0:
            idx -= 1
    
    return (image + dc_value)

def find_theta_combos(theta_array_deg, dtheta):
    '''
    
    Make sure angles are in degrees!

    '''

    theta_array_idx_pairs = list(combos(np.arange(len(theta_array_deg)), 2)) # Generate a list of all pairs of theta_array indices

    valid_theta_idx_pairs = [(theta_idx_1, theta_idx_2) for theta_idx_1, theta_idx_2 in theta_array_idx_pairs 
                             if (180 - dtheta <= np.abs(theta_array_deg[theta_idx_1] - theta_array_deg[theta_idx_2]) <= 180 + dtheta)]
                            # Compound inequality syntax is acceptable in Python in certain cases

    return valid_theta_idx_pairs

def create_ref_pair_theta_idx_array(ref_pair_theta_array, theta_array):
    ref_pair_theta_idx_1 = np.where(theta_array == ref_pair_theta_array[0])[0][0]
    ref_pair_theta_idx_2 = np.where(theta_array == ref_pair_theta_array[1])[0][0]

    return np.array([ref_pair_theta_idx_1, ref_pair_theta_idx_2])

def radon_manual(image, theta_array, center = None):
    n_cols = image.shape[0]
    n_theta = len(theta_array)
    
    # if image.dtype == np.float16:
        # image = image.astype(np.float32)

    # if circle:
    #     shape_min = min(image.shape)
    #     radius = shape_min // 2
    #     img_shape = np.array(image.shape)
    #     coords = np.array(np.ogrid[: image.shape[0], : image.shape[1]], dtype=object)
    #     dist = ((coords - img_shape // 2) ** 2).sum(0)
    #     outside_reconstruction_circle = dist > radius**2
    #     if np.any(image[outside_reconstruction_circle]):
    #         warnings.warn(
    #             'Radon transform: image must be zero outside the '
    #             'reconstruction circle'
    #         )

    if center is None:
        cx, cy = n_cols//2, n_cols//2
    
    else:
        cx, cy = center

    # compute diagonal length for output
    diagonal = int(np.ceil(np.sqrt(n_cols**2 + n_cols**2)))
    t = np.linspace(-diagonal//2, diagonal//2, diagonal)
    sino = np.zeros(((diagonal, n_theta)), dtype=image.dtype)

    # coordinates relative to center (detector axis along x')
    x = t
    y = np.zeros_like(t)

    for i, angle in enumerate(np.deg2rad(theta_array)):
        # rotation
        xr = x * np.cos(angle) - y * np.sin(angle) + cx
        yr = x * np.sin(angle) + y * np.cos(angle) + cy

        coords = np.vstack([yr, xr])
        # interpolate along rotated line
        sino[:, i] = xform.map_coordinates(image, coords, order=1)

    return sino

def phase_correlate(recon_proj, exp_proj, upsample_factor):
    n_columns = recon_proj.shape[1]
    n_slices = recon_proj.shape[0]

    recon_proj_filtered = edge_gauss_filter(recon_proj, sigma = 5, alpha = 10, nx = n_columns, ny = n_slices)
    exp_proj_filtered = edge_gauss_filter(exp_proj, sigma = 5, alpha = 10, nx = n_columns, ny = n_slices)
    
    shift, _, _ = reg.phase_cross_correlation(reference_image = recon_proj_filtered, moving_image = exp_proj_filtered, upsample_factor = upsample_factor)

    y_shift, x_shift = shift[0], shift[1]

    return y_shift, x_shift

def rot_center(theta_sum):
    """
    Code written by E. Vacek (2021): 
    https://github.com/everettvacek/PhaseSymmetry/blob/master/PhaseSymmetry.py

    Calculates the center of rotation of a sinogram.

    Parameters
    ----------
    thetasum: array-like
        The 2D theta-sum array (z,theta).

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

    geom_center_index = n_columns//2

    offset = center_rotation_avg - geom_center_index

    return center_rotation_avg, geom_center_index, offset

def iter_reproj(ref_element,
                element_array, 
                theta_array, 
                xrf_proj_img_array,
                algorithm, 
                n_iterations,
                init_x_shift = None, 
                init_y_shift = None,
                eps = 0.3):
    
    '''

    iter_reproj: Perform iterative reprojection for joint realignment of XRF, XRT tomography datasets


    ------
    Inputs
    ------

    ref_element: Element to base realignment off of (dtype: str)
    
    element_array: List of elements (list-like; dtype: str)
    
    theta_array: Array of projection angles (array-like; dtype: float)
    
    xrf_proj_img_array: 4D XRF tomography data (elements, projection angles, slices, scan positions) (array-like; dtype: float)

    algorithm: Desired reconstruction algorithm (dtype: str)
    
    n_iterations: Maximum number of iterative reprojection iterations (dtype: int)
    
    init_x_shift: Initial x-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)
    
    init_y_shift: Initial y-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)
    
    eps: Desired differential shift for convergence criterion (dtype: float)


    -------
    Outputs
    -------
    
    aligned_proj_total: 4D XRF tomography data (XRT data will come later) corrected for per-projection jitter, center of rotation misalignment (array-like; dtype: float)

    aligned_exp_proj_array: 1D array of experimental 3D XRF tomography data arrays for each iteration for ref_element (array-like; dtype: float)

    synth_proj_array: 1D array of synthetic 3D XRF tomography data arrays for each iteration for ref_element (array-like; dtype: float)

    recon_array: 1D array of 3D reconstruction slices for each iteration for ref_element (array-like; dtype: float)

    net_x_shifts_pc_new: Array of net x shifts with dimensions (n_iterations, n_theta) (array-like; dtype: float) (Note: n_iterations can be smaller the more quickly iter_reproj() converges)
    
    net_y_shifts_pc_new: Array of net y shifts with dimensions (n_iterations, n_theta) (array-like; dtype: float) (Note: n_iterations can be smaller the more quickly iter_reproj() converges)

    '''
    
    n_elements = xrf_proj_img_array.shape[0]
    n_theta = xrf_proj_img_array.shape[1]
    n_slices = xrf_proj_img_array.shape[2] 
    n_columns = xrf_proj_img_array.shape[3]

    ref_element_idx = element_array.index(ref_element)

    if (n_slices % 2) or (n_columns % 2): # Padding for odd-integer detector positions and/or slices

        if (n_slices % 2) and (n_columns % 2):
            xrf_proj_img_array = pad_col_row(xrf_proj_img_array)
            
            n_slices += 1
            n_columns += 1
        
        elif n_slices % 2:
            xrf_proj_img_array = pad_row(xrf_proj_img_array)

            n_slices += 1

        else:

            xrf_proj_img_array = pad_col(xrf_proj_img_array)

            n_columns += 1

    print('XRF Tomography dataset dimensions: ' + str(xrf_proj_img_array.shape))

    orig_ref_proj = xrf_proj_img_array[ref_element_idx].copy()
    
    aligned_proj_total = np.zeros((n_elements, n_theta, n_slices, n_columns))
    aligned_proj = np.zeros((n_theta, n_slices, n_columns))
    synth_proj = np.zeros((n_theta, n_slices, n_columns))
    dx_array_pc = np.zeros((n_iterations, n_theta))
    dy_array_pc = np.zeros((n_iterations, n_theta))
    net_x_shifts_pc = np.zeros((n_iterations, n_theta))
    net_y_shifts_pc = np.zeros((n_iterations, n_theta))
    
    iterations = []
    recon_array = []
    aligned_exp_proj_array = []
    synth_proj_array = []
    center_of_rotation_array = []
# If no initial shift(s) given or scalar shifts given
    if init_x_shift is None:
        init_x_shift = np.zeros(n_theta)
        
    if init_y_shift is None:
        init_y_shift = np.zeros(n_theta)
    
    if np.isscalar(init_x_shift):
        init_x_shift *= np.ones(n_theta)
        
    if np.isscalar(init_y_shift):
        init_y_shift *= np.ones(n_theta)
    
    if np.any(init_x_shift) or np.any(init_y_shift):
        if np.any(init_x_shift) and np.any(init_y_shift):
            print('Executing intial shift(s) in x and y')
            
            # net_x_shifts_pc[0] = init_x_shift
            # net_y_shifts_pc[0] = init_y_shift

            for element_idx in range(n_elements):
                for theta_idx in range(n_theta):
                    xrf_proj_img_array[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (init_y_shift[theta_idx], init_x_shift[theta_idx]))
        
        elif np.any(init_x_shift):
            print('Executing initial shift(s) in x')
            
            # net_x_shifts_pc[0] = init_x_shift

            for element_idx in range(n_elements):
                for theta_idx in range(n_theta):
                    xrf_proj_img_array[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (0, init_x_shift[theta_idx]))
                
        else:
            print('Executing initial shift(s) in x')
            # net_y_shifts_pc[0] = init_y_shift

            for element_idx in range(n_elements):
                for theta_idx in range(n_theta):
                    xrf_proj_img_array[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (init_y_shift[theta_idx], 0))

    theta_idx_pairs = find_theta_combos(theta_array, dtheta = 1)

    # center_of_rotation_avg, geom_center, offset = rot_center_avg(xrf_proj_img_array[ref_element_idx], theta_idx_pairs, theta_array)
    # # offset_copy = offset.copy()

    # print(f'Average COR: {(center_of_rotation_avg)}')

    
    # print(f'Center of rotation error: {round_correct(offset, ndec = 3)}')
    # print(f'Incorporating x-shift = {round_correct(-offset, ndec = 3)} to all projection images...')

    # for element_idx in range(n_elements):
    #     for theta_idx in range(n_theta):
    #         xrf_proj_img_array[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx], shift = (0, -offset))
    
    # Iterative center of rotation correction for reference element
    # print("Starting iterative center of rotation correction...")
    
    # max_cor_iterations = 20  # Maximum iterations for center of rotation correction
    # eps_cor = 0.001     # Tolerance for center of rotation convergence

    aligned_proj = xrf_proj_img_array[ref_element_idx].copy()
    
    center_of_rotation_avg, center_geom, offset_init = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

    # for cor_iter in range(max_cor_iterations):
        # Calculate current center of rotation
        # center_of_rotation_avg, center_geom, offset = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

        # if cor_iter == 0:
            # net_offset = offset.copy()
        
        # else:
            # net_offset_copy = net_offset.copy()
            # net_offset += offset
        
        # print(f'COR iteration {cor_iter + 1}: Center of rotation = {round_correct(center_of_rotation_avg, ndec = 3)}')
        # print(f'Geometric center: {center_geom}')
        # print(f'Center of rotation error: {round_correct(offset, ndec = 3)}')
        
        # Check if we've converged
        # if abs(offset) <= eps_cor:
            # print(f'Center of rotation converged after {cor_iter + 1} iterations')
            
            # for theta_idx in range(n_theta):
            #     aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (0, -(net_offset_copy - 1.2)))

            # break

        # Apply correction to reference element
        # print(f'Applying center of rotation correction: {round_correct(-net_offset, ndec = 3)}')
        
        # for theta_idx in range(n_theta):
            # aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (0, -net_offset))
    
    print(f'Geometric center: {center_geom}')
    print(f'Center of rotation error: {round_correct(offset_init, ndec = 3)}')
    print(f'Applying center of rotation correction: {round_correct(-offset_init, ndec = 3)}')

    for theta_idx in range(n_theta):
        aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (0, -offset_init))
    
    if offset_init != 0:
        offset_crop_idx = int(np.ceil(np.abs(offset_init))) 
        print(offset_crop_idx)
        theta_idx_pairs_nparray = np.array(theta_idx_pairs).ravel()

        aligned_proj_temp = np.zeros((n_theta, n_slices, n_columns - offset_crop_idx))
        print(aligned_proj_temp.shape)
        if offset_init > 0:
            aligned_proj_temp[theta_idx_pairs_nparray] = aligned_proj[theta_idx_pairs_nparray, :, :-offset_crop_idx]
        
        else:
            aligned_proj_temp[theta_idx_pairs_nparray] = aligned_proj[theta_idx_pairs_nparray, :, offset_crop_idx:]
    
    else:
        aligned_proj_temp = aligned_proj
    
    center_of_rotation_avg, _, _ = rot_center_avg(aligned_proj_temp, theta_idx_pairs, theta_array)

    offset = center_of_rotation_avg - center_geom

    if offset_init < 0:
        offset += offset_crop_idx

    # print(f'Final center of rotation after iterative correction: {round_correct(center_of_rotation_avg, ndec = 3)}')
    print(f'Final center of rotation after initial COR correction: {round_correct(center_of_rotation_avg, ndec = 3)}')
    print(f'Geometric center: {center_geom}')
    print(f'Final center of rotation error: {round_correct(offset, ndec = 3)}')
    
    # sys.exit()
    # Calculate the total shift needed as the difference between final and initial COR
    # total_cor_shift_needed = final_center_of_rotation_avg - init_cor_avg
    
    # print(f'Total COR shift needed: {round_correct(-net_offset, ndec = 3)}')
    
    # net_x_shifts_pc[0] -= net_offset
    net_x_shifts_pc[0] -= offset_init
    # net_x_shifts_pc[0] -= (net_offset - 1.2)
    # for theta_idx in range(n_theta):
    #     aligned_proj[theta_idx] = xrf_proj_img_array[ref_element_idx, theta_idx].copy()

    # plt.imshow(aligned_proj[0])
    # plt.show()

    for i in range(n_iterations):
        iterations.append(i)
        
        print(f'Iteration {i + 1}/{n_iterations}')

        # if i == 0:
        #     for theta_idx in range(n_theta):
        #         aligned_proj[theta_idx] = xrf_proj_img_array[ref_element_idx, theta_idx]
            
        # else:
        if i > 0:
            for theta_idx in range(n_theta):
                net_x_shift = net_x_shifts_pc[i - 1, theta_idx]
                net_y_shift = net_y_shifts_pc[i - 1, theta_idx]
                
                if (theta_idx % 7) == 0:
                    print(f'Shifting projection by net x shift = {round_correct(net_x_shift, ndec = 3)} (theta = {round_correct(theta_array[theta_idx], ndec = 1)})...')
                    print(f'Shifting projection by net y shift = {round_correct(net_y_shift, ndec = 3)}...')

                aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (net_y_shift, net_x_shift))
        
            # center_of_rotation_avg, center_geom, offset = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

            # print(f'New average center of rotation after jitter correction: {round_correct(center_of_rotation_avg, ndec = 3)}')
            # print(f'Geometric center: {center_geom}')
            # print(f'Center of rotation error: {round_correct(offset, ndec = 3)}')
            
            # if offset != 0:
                # net_x_shifts_pc[i - 1, :] -= (offset - 1.2)
                # net_x_shifts_pc[i - 1, :] -= offset
                
                # print(f'Incorporating x shift = {round_correct(-offset, ndec = 3)} + 1.2 pixels to all projection images for reference element {element_array[ref_element_idx]}...')
                # print(f'Incorporating x shift = {round_correct(-offset, ndec = 3)} pixels to all projection images for reference element {element_array[ref_element_idx]}...')

                # for theta_idx in range(n_theta):
                    # net_x_shift = net_x_shifts_pc[i - 1, theta_idx]
                    # net_y_shift = net_y_shifts_pc[i - 1, theta_idx]

                    # if (theta_idx % 7) == 0:
                        # print(f'Shifting projection by net x shift = {round_correct(net_x_shift - 1, ndec = 3)} (theta = {round_correct(theta_array[theta_idx], ndec = 1)})...')
                        # print(f'Shifting projection by net y shift = {round_correct(net_y_shift - 1, ndec = 3)}...')
                    
                    # aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (net_y_shift, net_x_shift))
                    # aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (net_y_shift, net_x_shift + 1.2))

                # net_x_shifts_pc[i, :] += 1.2

                # center_of_rotation_avg, _, offset = rot_center_avg(aligned_proj, theta_idx_pairs, theta_array)

                # print(f'New average center of rotation after dynamic COR correction: {round_correct(center_of_rotation_avg, ndec = 3)}')
                # print(f'Geometric center: {center_geom}')
                # print(f'Center of rotation error: {round_correct(offset, ndec = 3)}')

        aligned_exp_proj_array.append(aligned_proj.copy())

        if algorithm == 'gridrec':
            recon = tomo.recon(aligned_proj, theta_array*np.pi/180, algorithm = algorithm, center = center_of_rotation_avg, filter_name = 'ramlak')
            print(recon.shape)
            # recon = tomo.recon(aligned_proj, theta_array*np.pi/180, center = (n_columns - 1)/2, algorithm = algorithm, filter_name = 'ramlak')
        
        elif algorithm == 'mlem':
            recon = tomo.recon(aligned_proj, theta_array*np.pi/180, algorithm = algorithm, num_iter = 60)

        else:
            print('Error: Algorithm not available. Exiting...')
            
            sys.exit()
        
        recon_array.append(recon)

        for slice_idx in range(n_slices):
            print(f'Slice {slice_idx + 1}/{n_slices}')
            
            # sinogram = (xform.radon(recon[slice_idx].copy(), theta_array)).T
            sinogram = (radon_manual(recon[slice_idx].copy(), theta_array, center = (n_slices//2, center_of_rotation_avg))).T

            synth_proj[:, slice_idx, :] = sinogram
        
        synth_proj_array.append(synth_proj.copy())
        
        for theta_idx in range(n_theta):            
            dy, dx = phase_correlate(synth_proj[theta_idx], aligned_proj[theta_idx], upsample_factor = 100)

            dx_array_pc[i, theta_idx] = dx
            dy_array_pc[i, theta_idx] = dy
            
            if i == 0: 
                # net_x_shifts_pc[i, theta_idx] = init_x_shift[theta_idx] + dx
                # net_y_shifts_pc[i, theta_idx] = init_y_shift[theta_idx] + dy
                
                net_x_shifts_pc[i, theta_idx] += dx
                net_y_shifts_pc[i, theta_idx] += dy
            
            else:
                net_x_shifts_pc[i, theta_idx] = net_x_shifts_pc[i - 1, theta_idx] + dx
                net_y_shifts_pc[i, theta_idx] = net_y_shifts_pc[i - 1, theta_idx] + dy
            
            if (theta_idx % 7) == 0:
                print(f'Current x-shift: {round_correct(dx, ndec = 3)} (theta = {round_correct(theta_array[theta_idx], ndec = 1)})')
                print(f'Current y-shift: {round_correct(dy, ndec = 3)}')

        center_of_rotation_avg_synth, _, offset_synth = rot_center_avg(synth_proj, theta_idx_pairs, theta_array)

        print(f'Average synthetic center of rotation after jitter, dynamic COR correction attempts: {round_correct(center_of_rotation_avg_synth, ndec = 3)}')
        print(f'Geometric center: {center_geom}')
        print(f'Center of rotation error: {round_correct(offset_synth, ndec = 3)}')
        
        # if i == 1:
            # sys.exit()

        if np.max(np.abs(dx_array_pc[i])) < eps and np.max(np.abs(dy_array_pc[i])) < eps:
            iterations = np.array(iterations)
           
            net_x_shifts_pc_new = net_x_shifts_pc[:len(iterations)]
            net_y_shifts_pc_new = net_y_shifts_pc[:len(iterations)]

            dx_array_new = dx_array_pc[:len(iterations)]
            dy_array_new = dy_array_pc[:len(iterations)]

            print(f'Number of iterations taken: {len(iterations)}')
            print('Shifting all elements in aggregate aligned projection array by current net shifts...')

            for element_idx in range(n_elements):
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pc_new[i]
                    net_y_shift = net_y_shifts_pc_new[i]

                    aligned_proj_total[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))
            
            # net_x_shifts_pc_new -= offset_copy

            print('Done')

            break

        if i == n_iterations - 1:
            print('Iterative reprojection complete. Shifting other elements...')

            iterations = np.array(iterations)

            net_x_shifts_pc_new, net_y_shifts_pc_new = net_x_shifts_pc, net_y_shifts_pc

            dx_array_new = dx_array_pc
            dy_array_new = dy_array_pc
            
            for element_idx in range(n_elements):
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pc_new[i, theta_idx]
                    net_y_shift = net_y_shifts_pc_new[i, theta_idx]
                        
                    aligned_proj_total[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))

            # net_x_shifts_pc_new -= offset_copy
            
            print('Done')

    aligned_exp_proj_array = np.array(aligned_exp_proj_array)
    synth_proj_array = np.array(synth_proj_array)
    recon_array = np.array(recon_array)
    
    return orig_ref_proj, aligned_proj_total, aligned_exp_proj_array, synth_proj_array, recon_array, net_x_shifts_pc_new, net_y_shifts_pc_new, dx_array_new, dy_array_new

# file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
file_path_xrt = '/home/bwr0835/2_ide_aggregate_xrt.h5'
# file_path_xrt = '/Users/bwr0835/Documents/2_ide_aggregate_xrt.h5'
output_dir_path_base = '/home/bwr0835'

# output_file_name_base = input('Choose a base file name: ')
# output_file_name_base = 'gridrec_5_iter_vacek_cor_and_shift_correction_padding_-22_deg_158_deg'
# output_file_name_base = 'xrt_mlem_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025'
# output_file_name_base = 'xrt_mlem_1_iter_manual_shift_-20_no_log_tomopy_default_cor_w_padding_07_09_2025'
output_file_name_base = 'xrt_gridrec_6_iter_initial_ps_cor_correction_updated_log_w_padding_aug_20_2025'
# output_file_name_base = 'xrt_gridrec_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025'

if output_file_name_base == '':
    print('No output base file name chosen. Ending program...')

    sys.exit()

# create_aggregate_xrf_h5(file_path_array, file_path_xrf, synchrotron = 'aps')

# file_path_xrt = ''

# try:
    # elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = util.extract_h5_aggregate_xrt_data(file_path_xrf)
elements_xrt, counts_xrt, theta_xrt, dataset_type_xrt, _ = util.extract_h5_aggregate_xrt_data(file_path_xrt)

# except:
    # print('Cannot upload HDF5 file. Check file structure. Ending...')

    # sys.exit()

desired_element = 'ds_ic'
# desired_element = 'Fe'
# desired_element_idx = elements_xrf.index(desired_element)
desired_element_idx = elements_xrt.index(desired_element)

nonzero_mask = counts_xrt[desired_element_idx] > 0

phi_inc = 8.67768e5
t_dwell_s = 0.01 

counts_inc = phi_inc*t_dwell_s

counts_xrt[desired_element_idx][nonzero_mask] = -np.log(counts_xrt[desired_element_idx][nonzero_mask]/counts_inc)

# output_dir_path = filedialog.askdirectory(parent = root, title = "Choose directory to output NPY files to.")
n_theta = counts_xrt.shape[1]
n_slices = counts_xrt.shape[2]
# n_theta = counts_xrf.shape[1]
# n_slices = counts_xrf.shape[2]

# init_x_shift = -20*np.ones(n_theta)
init_x_shift = 0

n_desired_iter = 6 # For the reprojection scheme, NOT for reconstruction by itself

algorithm = 'gridrec'

# orig_proj_ref, \
# aligned_proj_total, \
# aligned_exp_proj_array, \
# synth_proj_array, \
# recon_array, \
# net_x_shifts, \
# net_y_shifts = iter_reproj(desired_element, 
#                            elements_xrf, 
#                            theta_xrf, 
#                            counts_xrf, 
#                            algorithm, 
#                            n_desired_iter,
#                            init_x_shift = init_x_shift)

orig_proj_ref, \
aligned_proj_total, \
aligned_exp_proj_array, \
synth_proj_array, \
recon_array, \
net_x_shifts, \
net_y_shifts, \
dx_array, \
dy_array = iter_reproj(desired_element, 
                       elements_xrt, 
                       theta_xrt, 
                       counts_xrt, 
                       algorithm, 
                       n_desired_iter,
                       init_x_shift = init_x_shift)

print('Saving files...')

full_output_dir_path = os.path.join(output_dir_path_base, 'iter_reproj', output_file_name_base)

os.makedirs(full_output_dir_path, exist_ok = True)

np.save(os.path.join(full_output_dir_path, 'theta_array.npy'), theta_xrt)
# np.save(os.path.join(full_output_dir_path, 'theta_array.npy'), theta_xrf)
np.save(os.path.join(full_output_dir_path, 'aligned_proj_all_elements.npy'), aligned_proj_total)
np.save(os.path.join(full_output_dir_path, 'aligned_proj_array_iter_' + desired_element + '.npy'), aligned_exp_proj_array)
np.save(os.path.join(full_output_dir_path, 'synth_proj_array_iter_' + desired_element + '.npy'), synth_proj_array)
np.save(os.path.join(full_output_dir_path, 'recon_array_iter_' + desired_element + '.npy'), recon_array)
np.save(os.path.join(full_output_dir_path, 'net_x_shifts_' + desired_element + '.npy'), net_x_shifts)
np.save(os.path.join(full_output_dir_path, 'net_y_shifts_' + desired_element + '.npy'), net_y_shifts)
np.save(os.path.join(full_output_dir_path, 'dx_array_iter_' + desired_element + '.npy'), dx_array)
np.save(os.path.join(full_output_dir_path, 'dy_array_iter_' + desired_element + '.npy'), dy_array)
np.save(os.path.join(full_output_dir_path, 'orig_exp_proj_' + desired_element + '.npy'), orig_proj_ref)