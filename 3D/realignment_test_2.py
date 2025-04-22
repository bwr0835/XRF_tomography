import numpy as np, h5py, os, sys, tkinter as tk, tomopy as tomo, csv

from skimage import transform as xform, registration as reg
from scipy import ndimage as ndi, fft
from h5_util import extract_h5_aggregate_xrf_data, create_aggregate_xrf_h5
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

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
    element_array: List of elements (list-like)
    theta_array: Array of projection angles (array-like; dtype: float)
    xrf_proj_img_array: 4D XRF tomography data (elements, projection angles, slices, scan positions) (array-like; dtype: float)
    algorithm: Desired reconstruction algorithm (dtype: str)
    n_iterations: Maximum number of iterative reprojection iterations
    init_x_shift: Initial x-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)
    init_y_shift: Initial y-shifts for all projections (for helping speed up convergence) (array-like; dtype: float)
    eps: Desired differential shift for convergence criterion (dtype: float)

    -------
    Outputs
    -------
    
    '''
    
    n_elements = xrf_proj_img_array.shape[0] # Number of elements
    n_theta = xrf_proj_img_array.shape[1] # Number of projection angles (projection images)
    n_slices = xrf_proj_img_array.shape[2] # Number of rows in a projection image
    n_columns = xrf_proj_img_array.shape[3] # Number of columns in a projection image

    ref_element_idx = element_array.index(ref_element)

    if (n_slices % 2) or (n_columns % 2):

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

    