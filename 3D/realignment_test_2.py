import numpy as np, h5py, os, sys, tkinter as tk, tomopy as tomo, csv

from skimage import transform as xform, registration as reg
from scipy import ndimage as ndi, fft
from h5_util import extract_h5_aggregate_xrf_data, create_aggregate_xrf_h5
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
    
    import numpy as np

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

def find_theta_combos(theta_array, dtheta):
    theta_array_idx_pairs = list(combos(np.arange(len(theta_array)), 2)) # Generate a list of all pairs of theta_array indices

    valid_theta_idx_pairs = [(theta_idx_1, theta_idx_2) for theta_idx_1, theta_idx_2 in theta_array_idx_pairs 
                             if (180 - dtheta <= np.abs(theta_array[theta_idx_1] - theta_array[theta_idx_2]) <= 180 + dtheta)]
                            # Compound inequality syntax is acceptable in Python

    return valid_theta_idx_pairs

def create_ref_pair_theta_idx_array(ref_pair_theta_array, theta_array):
    ref_pair_theta_idx_1 = np.where(theta_array == ref_pair_theta_array[0])[0][0]
    ref_pair_theta_idx_2 = np.where(theta_array == ref_pair_theta_array[1])[0][0]

    return np.array([ref_pair_theta_idx_1, ref_pair_theta_idx_2])

def phase_correlate(recon_proj, exp_proj, upsample_factor):
    shift, _, _ = reg.phase_cross_correlation(reference_image = recon_proj, moving_image = exp_proj, upsample_factor = upsample_factor)

    y_shift, x_shift = shift[0], shift[1]

    return y_shift, x_shift

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

    aligned_proj_total = np.zeros((n_elements, n_theta, n_slices, n_columns))
    aligned_proj = np.zeros((n_theta, n_slices, n_columns))
    synth_proj = np.zeros((n_theta, n_slices, n_columns))
    dx_array_pc = np.zeros(n_theta)
    dy_array_pc = np.zeros(n_theta)
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

    theta_idx_pairs = find_theta_combos(theta_array, dtheta = 1)

    for theta_idx_pair in theta_idx_pairs:
        print(np.array([theta_array[theta_idx_pair[0]], theta_array[theta_idx_pair[1]]]))

    center_of_rotation_array = np.array([tomo.find_center_pc(xrf_proj_img_array[ref_element_idx, theta_idx_pair[1]], 
                                                             xrf_proj_img_array[ref_element_idx, theta_idx_pair[0]], 
                                                             tol = 0.01) for theta_idx_pair in theta_idx_pairs]) 
                                                          # The second image is flipped about the vertical axis within the TomoPy function
    
    center_of_rotation = np.mean(center_of_rotation_array)

    plt.plot(np.arange(len(center_of_rotation_array)), center_of_rotation_array, 'o', markersize = 3)
    plt.plot(np.arange(len(center_of_rotation_array)), center_of_rotation*np.ones(len(center_of_rotation_array)))

    plt.show()
        
    center_geom = (n_columns - 1)/2
    
    offset = center_of_rotation - center_geom

    print(f'Center of rotation via phase cross-correlation: {round_correct(center_of_rotation, ndec = 3)}')
    print(f'Geometric center: {center_geom}')
    print(f'Center of rotation error: {offset}')
    print(f'Incorporating x-shift = {-offset} to all projection images...')

    for element_idx in range(n_elements):
        for theta_idx in range(n_theta):
            xrf_proj_img_array[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx], shift = (0, -offset))

    center_of_rotation_array = np.array([tomo.find_center_pc(xrf_proj_img_array[ref_element_idx, theta_idx_pair[0]], 
                                                    xrf_proj_img_array[ref_element_idx, theta_idx_pair[1]], 
                                                    tol = 0.01) for theta_idx_pair in theta_idx_pairs])
    
    center_of_rotation = np.mean(center_of_rotation_array)
    
    print(f'New center of rotation: {center_of_rotation}')
    print('Performing iterative reprojection...')

    for i in range(n_iterations):
        print(f'Iteration {i + 1}/{n_iterations}')

        if i == 0:
            for theta_idx in range(n_theta):
                aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (init_y_shift, init_x_shift))
            
        else:
            for theta_idx in range(n_theta):
                net_x_shift = net_x_shifts_pc[i - 1, theta_idx]
                net_y_shift = net_y_shifts_pc[i - 1, theta_idx]
                
                if (theta_idx % 7) == 0:
                    print(f'Shifting projection by net x shift = {round_correct(net_x_shift, ndec = 3)} (theta = {round_correct(theta_array[theta_idx], ndec = 1)})...')
                    print(f'Shifting projection by net y shift = {round_correct(net_y_shift, ndec = 3)}...')

                aligned_proj[theta_idx] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx], shift = (net_y_shift, net_x_shift))

        aligned_exp_proj_array.append(aligned_proj)
        print(aligned_proj.shape)
        if algorithm == 'gridrec':
            recon = tomo.recon(aligned_proj, theta_array, center_of_rotation, algorithm = algorithm)
        
        elif algorithm == 'mlem':
            recon = tomo.recon(aligned_proj, theta_array, center_of_rotation, algorithm, num_iter = 60)

        else:
            print('Error: Algorithm not available. Exiting...')
            
            sys.exit()
        
        recon_array.append(recon)

        for slice_idx in range(n_slices):
            sinogram = (xform.radon(recon[slice_idx].copy(), theta_array)).T

            synth_proj[:, slice_idx, :] = sinogram
        
        synth_proj_array.append(synth_proj)

        for theta_idx in range(n_theta):
            dy, dx = phase_correlate(synth_proj[theta_idx], aligned_proj[theta_idx])

            dx_array_pc[theta_idx] = dx
            dy_array_pc[theta_idx] = dy
            
            if i == 0: 
                net_x_shifts_pc[i, theta_idx] = init_x_shift[theta_idx] + dx
                net_y_shifts_pc[i, theta_idx] = init_y_shift[theta_idx] + dy
            
            else:
                net_x_shifts_pc[i, theta_idx] = net_x_shifts_pc[i - 1, theta_idx] + dx
                net_y_shifts_pc[i, theta_idx] = net_y_shifts_pc[i - 1, theta_idx] + dy
            
            if (theta_idx % 7) == 0:
                print(f'Current x-shift: {round_correct(dx, ndec = 3)} (theta = {round_correct(theta_array[theta_idx], ndec = 1)})')
                print(f'Current y-shift: {round_correct(dy, ndec = 3)}')
        
        if np.max(np.abs(dx_array_pc)) < eps and np.max(np.abs(dy_array_pc)) < eps:
            iterations = np.array(iterations)
           
            net_x_shifts_pc_new = net_x_shifts_pc[:len(iterations)]
            net_y_shifts_pc_new = net_y_shifts_pc[:len(iterations)]

            print(f'Number of iterations taken: {len(iterations)}')
            print('Shifting all elements in aggregate aligned projection array by current net shifts...')

            for element_idx in range(n_elements):
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pc_new[i]
                    net_y_shift = net_y_shifts_pc_new[i]

                    aligned_proj_total[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))
            
            print('Done')

            break

        if i == n_iterations - 1:
            print('Iterative reprojection complete. Shifting other elements...')

            iterations = np.array(iterations)

            net_x_shifts_pc_new, net_y_shifts_pc_new = net_x_shifts_pc, net_y_shifts_pc
            
            for element_idx in range(n_elements):
                for theta_idx in range(n_theta):
                    net_x_shift = net_x_shifts_pc_new[i, theta_idx]
                    net_y_shift = net_y_shifts_pc_new[i, theta_idx]
                        
                    aligned_proj_total[element_idx, theta_idx] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx], shift = (net_y_shift, net_x_shift))

            print('Done')

    aligned_exp_proj_array = np.array(aligned_exp_proj_array)
    synth_proj_array = np.array(synth_proj_array)
    recon_array = np.array(recon_array)
    
    return aligned_proj_total, aligned_exp_proj_array, synth_proj_array, recon_array, net_x_shifts_pc_new, net_y_shifts_pc_new

file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
output_dir_path_base = '/home/bwr0835'

# output_file_name_base = input('Choose a base file name: ')
# output_file_name_base = 'gridrec_5_iter_vacek_cor_and_shift_correction_padding_-22_deg_158_deg'
output_file_name_base = 'gridrec_5_iter_tomopy_cor_alg_no_cor_correction_padding_04_17_2025'

if output_file_name_base == '':
    print('No output base file name chosen. Ending program...')

    sys.exit()

# create_aggregate_xrf_h5(file_path_array, file_path_xrf, synchrotron = 'aps')

# file_path_xrt = ''

try:
    elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

except:
    print('Cannot upload HDF5 file. Check file structure. Ending...')

    sys.exit()

desired_element = 'Fe'
desired_element_idx = elements_xrf.index(desired_element)
# output_dir_path = filedialog.askdirectory(parent = root, title = "Choose directory to output NPY files to.")
n_theta = counts_xrf.shape[1]
n_slices = counts_xrf.shape[2]

init_x_shift = np.zeros(n_theta)
# init_x_shift[0] = -50

n_desired_iter = 5 # For the reprojection scheme, NOT for reconstruction by itself

algorithm = 'gridrec'

aligned_proj_total, \
aligned_exp_proj_array, \
synth_proj_array, \
recon_array, \
net_x_shifts, \
net_y_shifts = iter_reproj(desired_element, 
                                  elements_xrf, 
                                  theta_xrf, 
                                  counts_xrf, 
                                  algorithm, 
                                  n_desired_iter,
                                  init_x_shift = init_x_shift)

print('Saving files...')

full_output_dir_path = os.path.join(output_dir_path_base, 'iter_reproj', output_file_name_base)

os.makedirs(full_output_dir_path, exist_ok = True)

np.save(os.path.join(full_output_dir_path, 'theta_array.npy'), theta_xrf)
np.save(os.path.join(full_output_dir_path, 'aligned_proj_all_elements.npy'), aligned_proj_total)
np.save(os.path.join(full_output_dir_path, 'aligned_proj_array_iter_' + desired_element + '.npy'), aligned_exp_proj_array)
np.save(os.path.join(full_output_dir_path, 'synth_proj_array_iter_' + desired_element + '.npy'), synth_proj_array)
np.save(os.path.join(full_output_dir_path, 'recon_array_iter_' + desired_element + '.npy'), recon_array)
np.save(os.path.join(full_output_dir_path, 'net_x_shifts_' + desired_element + '.npy'), net_x_shifts)
np.save(os.path.join(full_output_dir_path, 'net_y_shifts_' + desired_element + '.npy'), net_y_shifts)