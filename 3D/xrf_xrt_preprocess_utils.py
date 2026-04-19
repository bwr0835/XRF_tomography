import numpy as np, tomopy as tomo, xraylib as xrl, sys

from scipy import ndimage as ndi
from itertools import combinations as combos

def round_correct(num, ndec):
    if ndec == 0:
        return int(num + 0.5)
    
    else:
        digit_value = 10**ndec
        
        if num > 0:
            return int(num*digit_value + 0.5)/digit_value
        
        else:
            return int(num*digit_value - 0.5)/digit_value

def pad_col_row(array, dataset_type):
    if dataset_type == '' or array.ndim != 4:
        print('Error: No dataset type specified or number of array dimesions ≠ 4. Exiting program...')

        sys.exit()

    n_elements, n_theta, n_slices, n_columns = array.shape
    
    padded_array = np.zeros((n_elements, n_theta, n_slices + 1, n_columns + 1))

    if dataset_type == 'xrt':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()

                padded_row = array_avg*np.ones(n_columns)
                
                temp_array = np.vstack((array[element_idx, theta_idx], padded_row))

                padded_column = array_avg*np.ones((temp_array.shape[0], 1))

                padded_array[element_idx, theta_idx] = np.hstack((temp_array, padded_column))
    
    elif dataset_type == 'xrf':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()

                padded_row = array_avg*np.zeros(n_columns)
                
                temp_array = np.vstack((array[element_idx, theta_idx], padded_row))

                padded_column = array_avg*np.zeros((temp_array.shape[0], 1))

                padded_array[element_idx, theta_idx] = np.hstack((temp_array, padded_column))

    else:
        print('Error: Invalid dataset type. Exiting program...')

        sys.exit()

    return padded_array

def pad_col(array, dataset_type):
    if dataset_type == '' or array.ndim != 4:
        print('Error: No dataset type specified or number of array dimesions ≠ 4. Exiting program...')

        sys.exit()

    n_elements, n_theta, n_slices, _ = array.shape

    if dataset_type == 'xrt':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()
                # final_column = array_avg*array[element_idx, theta_idx, :, -1].T
                final_column = array_avg*np.ones((n_slices, 1))
                
                array[element_idx, theta_idx] = np.hstack((array[element_idx, theta_idx], final_column))

    elif dataset_type == 'xrf':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                # final_column = array[element_idx, theta_idx, :, -1].T
                final_column = np.zeros((n_slices, 1))
                
                array[element_idx, theta_idx, :, :] = np.hstack((array[element_idx, theta_idx, :, :], final_column))

    else:
        print('Error: Invalid dataset type. Exiting program...')

        sys.exit()
    
    return array

def pad_row(array, dataset_type):
    if dataset_type == '' or array.ndim != 4:
        print('Error: No dataset type specified or number of array dimesions ≠ 4. Exiting...')

        sys.exit()
    
    n_elements, n_theta, _, n_columns = array.shape
    
    if dataset_type == 'xrt':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()
                
                final_row = array_avg*np.ones(n_columns)

                array[element_idx, theta_idx] = np.vstack((array[element_idx, theta_idx], final_row))

    elif dataset_type == 'xrf':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()
                
                final_row = array_avg*np.zeros(n_columns)

                array[element_idx, theta_idx] = np.vstack((array[element_idx, theta_idx], final_row))

    else:
        print('Error: Invalid dataset. Exiting program...')

        sys.exit()

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
    
    return (image + dc_value), dc_value

def joint_fluct_norm(xrt_array,
                     xrf_array,
                     data_percentile,
                     xrt_photon_counting,
                     incident_photodiode_flux_photons_per_s,
                     t_dwell_s,
                     return_aux_data,
                     sigma_1 = 5,
                     alpha = 10,
                     sigma_2 = 10):

    if xrt_array.ndim != 3 or xrf_array.ndim != 4:
        print('Error: Number of XRT dimensions ≠ 3 and/or number of XRF array dimesions ≠ 4. Exiting program...')

        sys.exit()

    if data_percentile is None or data_percentile < 0 or data_percentile > 100:
        print('Error: \'data_percentile\' must be between 0 and 100. Exiting program...')

        sys.exit() 
    
    n_theta, n_slices, n_columns = xrt_array.shape

    print(xrt_array.shape)

    convolution_mag_array = np.zeros((n_theta, n_slices, n_columns))
    
    norm_array_xrt = np.zeros(n_theta)
    norm_array_xrf = np.zeros(n_theta)
    
    xrt_mask_avg_sum = 0
    
    for theta_idx in range(n_theta):
        xrt_vignetted = edge_gauss_filter(xrt_array[theta_idx], sigma = sigma_1, alpha = alpha, nx = n_columns, ny = n_slices)

        convolution_mag = ndi.gaussian_filter(xrt_vignetted, sigma = sigma_2) # Blur the entire image using Gaussian filter/convolution

        threshold = np.percentile(convolution_mag, data_percentile) # EX: Take top 20% of data (data_percentile = 80)

        mask = convolution_mag >= threshold

        xrt_mask_avg = xrt_array[theta_idx, mask].mean()

        norm_array_xrt[theta_idx] = 1/xrt_mask_avg
        norm_array_xrf[theta_idx] = 1/xrt_mask_avg

        xrt_array[theta_idx] /= xrt_mask_avg # First part of I0' = I0(<I_theta,mask,avg>/I_theta,mask,avg)
        xrf_array[:, theta_idx] /= xrt_mask_avg

        xrt_mask_avg_sum += xrt_mask_avg

        convolution_mag_array[theta_idx] = convolution_mag
    
    global_xrt_mask_avg = xrt_mask_avg_sum/n_theta
    
    if xrt_photon_counting:
        inc_intensity = global_xrt_mask_avg
    
    else:
        inc_intensity = incident_photodiode_flux_photons_per_s*t_dwell_s # Incident intensity in photons (instead of, for instance, ion chamber units)
    
    xrt_array *= inc_intensity
    xrf_array *= global_xrt_mask_avg

    norm_array_xrt *= inc_intensity
    norm_array_xrf *= global_xrt_mask_avg
    
    if return_aux_data:
        return xrt_array, xrf_array, norm_array_xrt, norm_array_xrf, inc_intensity, np.array(convolution_mag_array)
    
    return xrt_array, xrf_array, norm_array_xrt, norm_array_xrf, inc_intensity, None

def calculate_abs_incident_intensity_photons(xrt_array,
                                             data_percentile, 
                                             sigma_1 = 5, 
                                             alpha = 10, 
                                             sigma_2 = 10):
    
    if xrt_array.ndim != 3:
        print('Error: XRT array must be 3D. Exiting program...')

        sys.exit()
    
    if data_percentile is None or data_percentile < 0 or data_percentile > 100:
        print('Error: \'data_percentile\' must be between 0 and 100. Exiting program...')

        sys.exit()
    
    n_theta, n_slices, n_columns = xrt_array.shape

    xrt_mask_avg_sum = 0
    
    for theta_idx in range(n_theta):
        xrt_vignetted = edge_gauss_filter(xrt_array[theta_idx], sigma = sigma_1, alpha = alpha, nx = n_columns, ny = n_slices)
        convolution_mag = ndi.gaussian_filter(xrt_vignetted, sigma = sigma_2) # Blur the entire image using Gaussian filter/convolution

        threshold = np.percentile(convolution_mag, data_percentile) # EX: Take top 20% of data (data_percentile = 80)

        mask = convolution_mag >= threshold

        xrt_mask_avg = xrt_array[theta_idx, mask].mean()

        xrt_mask_avg_sum += xrt_mask_avg
    
    incident_intensity_photons = xrt_mask_avg_sum/n_theta

    return incident_intensity_photons

def find_theta_combos(theta_array_deg, dtheta = 0): # Output type: List of tuples
    '''
    
    Make sure angles are in degrees!

    '''

    theta_array_idx_pairs = combos(np.arange(len(theta_array_deg)), 2) # Generate a list of all pairs of theta_array indices

    valid_theta_idx_pairs = [(theta_idx_1, theta_idx_2) for theta_idx_1, theta_idx_2 in theta_array_idx_pairs 
                             if (180 - dtheta <= np.abs(theta_array_deg[theta_idx_1] - theta_array_deg[theta_idx_2]) <= 180 + dtheta)]
                            # Compound inequality syntax is acceptable in Python in certain cases

    return valid_theta_idx_pairs

def crop_array(array, net_y_shift_array, edge_dict):
    if array.ndim != 3:
        print('Error: Array must be 3D. Exiting program...')

        sys.exit()

    n_slices, n_columns = array.shape[1], array.shape[2]
    
    if np.any(net_y_shift_array != 0):
        start_slice = int(np.clip(np.ceil(np.max(net_y_shift_array)), 0, n_slices)) # Crop top: exclude rows where positive shifts pushed content out
        end_slice = int(np.clip(n_slices + np.floor(np.min(net_y_shift_array)), 0, n_slices)) # Crop bottom: exclude rows where negative shifts pushed content out. This index is exclusive.

        if edge_dict is not None:
            start_slice += edge_dict['top']
            end_slice -= edge_dict['bottom']

    else:
        start_slice = 0
        end_slice = n_slices

    if edge_dict is not None:
        start_column = edge_dict['left']
        end_column = n_columns - edge_dict['right']

    else:
        start_column = 0
        end_column = n_columns

    if end_slice <= start_slice or end_column <= start_column:
        print('Error: Empty field of view detected - net shifts exceed the number of slices. Exiting program...')

        sys.exit()
    
    if (start_slice, start_column) != (0, 0) or (end_slice, end_column) != (n_slices, n_columns):
        array_final = array[:, start_slice:end_slice, start_column:end_column]
    
    else:
        print('Warning: No cropping performed - common field of view is the original field of view. Returning original projection images...')

        array_final = array

    return array_final, start_slice, end_slice, start_column, end_column

def joint_array_crop(xrf_array, xrt_array, opt_dens_array, net_y_shift_array, edge_dict):
    if xrf_array.ndim != 4 or opt_dens_array.ndim != 3:
        print('Error: XRF, XRT, and optical density arrays must be 4D and 3D, respectively. Exiting program...')

        sys.exit()

    n_slices, n_columns = xrt_array.shape[1], xrt_array.shape[2]

    if np.any(net_y_shift_array != 0):
        start_slice = int(np.clip(np.ceil(np.max(net_y_shift_array)), 0, n_slices)) # Crop top: exclude rows where positive shifts pushed content out
        end_slice = int(np.clip(n_slices + np.floor(np.min(net_y_shift_array)), 0, n_slices)) # Crop bottom: exclude rows where negative shifts pushed content out. This index is exclusive.

        if edge_dict is not None:
            start_slice += edge_dict['top']
            end_slice -= edge_dict['bottom']

    else:
        start_slice = 0
        end_slice = n_slices

    if edge_dict is not None:
        start_column = edge_dict['left']
        end_column = n_columns - edge_dict['right']

    else:
        start_column = 0
        end_column = n_columns

    if end_slice <= start_slice or end_column <= start_column:
        print('Error: Empty field of view detected - net shifts exceed the number of slices. Exiting program...')

        sys.exit()
    
    if (start_slice, start_column) != (0, 0) or (end_slice, end_column) != (n_slices, n_columns):
        xrf_array_final = xrf_array[:, :, start_slice:end_slice, start_column:end_column]
        xrt_array_final = xrt_array[:, start_slice:end_slice, start_column:end_column]
        opt_dens_array_final = opt_dens_array[:, start_slice:end_slice, start_column:end_column]
    
    else:
        print('Warning: No cropping performed - common field of view is the original field of view. Returning original projection images...')

        xrf_array_final = xrf_array
        xrt_array_final = xrt_array
        opt_dens_array_final = opt_dens_array

    return xrf_array_final, xrt_array_final, opt_dens_array_final, start_slice, end_slice, start_column, end_column

def create_gridrec_density_maps(xrf_proj_array, elements_xrf, theta_array):
    if xrf_proj_array.ndim != 4:
        print('Error: XRF projection array must be 4D. Exiting program...')

        sys.exit()

    if elements_xrf is None:
        print('Error: Elements XRF must be provided. Exiting program...')

        sys.exit()
    
    n_elements_xrf, _, n_slices, n_columns = xrf_proj_array.shape

    rho = np.zeros((n_elements_xrf, n_slices, n_columns, n_columns))

    for element_idx, element in enumerate(elements_xrf):
        if '_' in element:
            element = element.split('_')[0]
        
        xrf_recon = tomo.recon(xrf_proj_array[element_idx], theta_array*np.pi/180, algorithm = 'gridrec', filter_name = 'ramlak')

        vmin = xrf_recon.min()
        vmax = xrf_recon.max()
        
        rho[element_idx] = xrl.ElementDensity(xrl.SymbolToAtomicNumber(element))*(xrf_recon - vmin)/(vmax - vmin) # Initial guess of density based on 0-100% concentration of each element

    return rho

def normalize_array_for_rgb(array):
    return (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))

def normalize_array_for_gif(array):
    array_nonzero = array[array != 0]

    return np.clip((array - np.nanmin(array_nonzero))/(np.nanmax(array_nonzero) - np.nanmin(array_nonzero)), 0, 1)