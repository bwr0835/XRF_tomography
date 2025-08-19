import numpy as np, h5_util

from matplotlib import pyplot as plt
from numpy import fft
from scipy import ndimage as ndi

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

def pad_col_row(array):
    array_new = np.zeros((array.shape[0], array.shape[1] + 1, array.shape[2] + 1))

    for theta_idx in range(array.shape[0]):
        array_avg = np.mean(array[theta_idx])

        final_column = array_avg*np.ones((array.shape[1], 1)) # Create final column whose pixel values are all the average value of the original 2D array
        
        # final_column = array[theta_idx, :, -1].reshape(-1, 1) # Reshape to column vector (-1 means Python automatically determines missing dimension based on original orray length)
            
        # print(final_column.shape)
        # print(array.shape)

        array_temp = np.hstack((array[theta_idx, :, :], final_column))
            
        # final_row = array_temp[-1, :]
        final_row = array_avg*np.ones(array_temp.shape[1])

        array_new[theta_idx, :, :] = np.vstack((array_temp, final_row))

    return array_new

def pad_col(array):
    for theta_idx in range(array.shape[1]):
        final_column = array[theta_idx, :, -1].T
                
        array[theta_idx, :, :] = np.hstack((array[theta_idx, :, :], final_column))

    return array

def pad_row(array):
    for theta_idx in range(array.shape[1]):
        final_row = array[theta_idx, -1, :]
                
        array[theta_idx, :, :] = np.vstack((array[theta_idx, :, :], final_row))

    return array

def gaussian_2d(nx, ny, center = None, sigma = 10):
    x = np.zeros(nx)
    y = np.zeros(ny)
    rsq = np.zeros((ny, nx))

    if center is None:
        center_x = nx//2
        center_y = ny//2
    
    else:
        center_x = center[1]
        center_y = center[0]
    
    j = 0

    for i in range(nx):
        x[i] = -center_x + j

        j += 1
    
    j = 0

    for i in range(ny):
        y[i] = -center_y + j

        j += 1
    
    for idy in range(ny):
        rsq[idy, 0:nx] = x**2 + y[idy]**2
    
    _gaussian_2d = (1/(2*np.pi*sigma**2))*np.exp(-0.5*rsq/(sigma**2))

    return _gaussian_2d

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

def convolve_2d(img1, img2, filter_enabled = True):
    if filter_enabled == True:
        nx = img1.shape[1]
        ny = img1.shape[0]

        sigma = 10
        alpha = 5

        img1_filtered = edge_gauss_filter(img1, sigma, alpha, nx, ny)
        img2_filtered = edge_gauss_filter(img2, sigma, alpha, nx, ny)
    
    else:
        img1_filtered = img1
        img2_filtered = img2

    img1_fft = fft.fftshift(fft.fft2(fft.fftshift(img1_filtered), norm = 'ortho'))
    img2_fft = fft.fftshift(fft.fft2(fft.fftshift(img2_filtered), norm = 'ortho'))

    convolution_fft = img1_fft*img2_fft
    convolution_ifft = fft.ifft2(fft.ifftshift(convolution_fft), norm = 'ortho')

    return convolution_ifft

file_path_xrt = '/Users/bwr0835/Documents/2_ide_aggregate_xrt.h5'

elements_xrt, counts_xrt, theta_xrt, dataset_type_xrt, _ = h5_util.extract_h5_aggregate_xrt_data(file_path_xrt)

ref_element = 'ds_ic'

ref_element_idx = elements_xrt.index(ref_element)
counts = counts_xrt[ref_element_idx]

n_theta = counts.shape[0] # Number of projection angles (projection images)
n_slices = counts.shape[1] # Number of rows in a projection image
n_columns = counts.shape[2] # Number of columns in a projection image

n_bins = 100


if (n_slices % 2) or (n_columns % 2):
    if (n_slices % 2) and (n_columns % 2):
        counts = pad_col_row(counts)
            
        n_slices += 1
        n_columns += 1
        
    elif n_slices % 2:
        counts = pad_row(counts)

        n_slices += 1

    else:
        counts = pad_col(counts)

        n_columns += 1

_gaussian_2d = gaussian_2d(n_columns, n_slices)

# counts[mask] = -np.log(counts[mask]/counts_inc)

vmin = np.min(counts)
vmax = np.max(counts)

counts_copy = counts.copy()

for theta_idx in range(n_theta):
    # counts[theta_idx] = ndi.shift(counts[theta_idx], shift = (0, -7.9))
    convolution_mag = fft.fftshift(np.abs(convolve_2d(counts[theta_idx], _gaussian_2d)))

    # threshold = np.quantile(convolution_mag, [0.88, 0.9])
    threshold = np.percentile(convolution_mag, 80)

    mask = convolution_mag >= threshold

    counts_avg = np.mean(counts[theta_idx, mask])

    counts_copy[theta_idx, mask] /= counts_avg

print(counts_copy.dtype)

vmin_conv = np.min(convolution_mag)
vmax_conv = np.max(convolution_mag)

for theta_idx in range(n_theta):
    if theta_idx % 7 == 0:  
        fig, axs = plt.subplots(2, 2)
        
        a = convolution_mag.copy()
        a[~mask] = 0
        
        axs[0, 0].imshow(convolution_mag, vmin = vmin_conv, vmax = vmax_conv)
        axs[0, 1].imshow(a, vmin = vmin_conv, vmax = vmax_conv)
        axs[1, 0].imshow(counts[theta_idx], vmin = vmin, vmax = vmax)
        axs[1, 1].imshow(counts_copy[theta_idx], vmin = vmin, vmax = vmax)
        axs[0, 0].axis('off')
        axs[0, 1].axis('off')
        axs[1, 0].axis('off')
        axs[1, 1].axis('off')
        axs[0, 0].set_title(r'Convolution $\rightarrow$'.format(theta_xrt[theta_idx]), fontsize = 16)
        axs[1, 0].set_title(r'Raw data $\rightarrow$'.format(theta_xrt[theta_idx]), fontsize = 16)
        fig.suptitle(r'$\theta = {0}$\textdegree'.format(theta_xrt[theta_idx]), fontsize = 18)
        fig.tight_layout()
        plt.show()

# counts[mask] = -np.log10(counts[mask]/counts_inc)


