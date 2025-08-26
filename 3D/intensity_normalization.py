import numpy as np, h5_util, os

from matplotlib import pyplot as plt
from numpy import fft
from scipy import ndimage as ndi
from imageio import v2 as iio2

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

def fluct_norm(img_array, sigma_1 = 5, alpha = 10, sigma_2 = 10):
    n_theta, n_slices, n_columns = img_array.shape

    img_array_avg = np.mean(img_array)

    intensity_avg_array = []
    intensity_norm_avg_array = []
    convolution_mag_array = []

    for theta_idx in range(n_theta):
        intensity_avg_array.append(img_array[theta_idx].copy().mean())

        img_vignetted = edge_gauss_filter(img_array[theta_idx], sigma = sigma_1, alpha = alpha, nx = n_columns, ny = n_slices)

        convolution_mag = ndi.gaussian_filter(img_vignetted, sigma = sigma_2) # Blur pretty much the entire image using Gaussian filter/convolution

        threshold = np.percentile(convolution_mag, 80) # Take top 20% of intensities for masking

        mask = convolution_mag >= threshold

        img_avg = np.mean(img_array[theta_idx, mask])

        img_array[theta_idx] *= (img_array_avg/img_avg) # I0' = I0[avg(I0)/mask]

        convolution_mag_array.append(convolution_mag)
        intensity_norm_avg_array.append(img_array[theta_idx, mask].mean())

    return img_array, np.array(intensity_avg_array), np.array(convolution_mag_array), np.array(intensity_norm_avg_array)

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
output_dir = '/Users/bwr0835/Documents'

elements_xrt, counts_xrt, theta_xrt, dataset_type_xrt, _ = h5_util.extract_h5_aggregate_xrt_data(file_path_xrt)

ref_element = 'ds_ic'

fps = 1

ref_element_idx = elements_xrt.index(ref_element)
counts = counts_xrt[ref_element_idx]

n_theta = counts.shape[0] # Number of projection angles (projection images)
n_slices = counts.shape[1] # Number of rows in a projection image
n_columns = counts.shape[2] # Number of columns in a projection image

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

counts_copy = counts.copy()

I0_avg = np.mean(counts_copy)

opt_dens_copy = -np.log(counts_copy/I0_avg)

counts_norm, cts_avg_array, convolution_mag_array, cts_norm_avg_array = fluct_norm(counts)

opt_dens = -np.log(counts_norm/I0_avg)

vmin = counts_copy.min()
vmax = counts_copy.max()

vmin_norm = counts_norm.min()
vmax_norm = counts_norm.max()

vmin_conv = convolution_mag_array.min()
vmax_conv = convolution_mag_array.max()

vmin_od = opt_dens_copy.min()
vmax_od = opt_dens_copy.max()

vmin_od_norm = opt_dens.min()
vmax_od_norm = opt_dens.max()

fig1, axs1 = plt.subplots(3, 2)
fig2, axs2 = plt.subplots()

threshold = np.percentile(convolution_mag_array[0], 80)

conv_mask = np.where(convolution_mag_array[0] < threshold, convolution_mag_array[0], 0)

im1_1 = axs1[0, 0].imshow(convolution_mag_array[0], vmin = vmin_conv, vmax = vmax_conv)
im1_2 = axs1[0, 1].imshow(conv_mask, vmin = vmin_conv, vmax = vmax_conv)
im1_3 = axs1[1, 0].imshow(counts_copy[0], vmin = vmin, vmax = vmax)
im1_4 = axs1[1, 1].imshow(counts_norm[0], vmin = vmin_norm, vmax = vmax_norm)
im1_5 = axs1[2, 0].imshow(opt_dens_copy[0], vmin = vmin_od, vmax = vmax_od)
im1_6 = axs1[2, 1].imshow(opt_dens[0], vmin = vmin_od_norm, vmax = vmax_od_norm)

curve1, = axs2.plot(theta_xrt, cts_avg_array, 'ko-', linewidth = 2, label = r'Raw XRT Avg.')
curve2, = axs2.plot(theta_xrt, cts_norm_avg_array, 'ro-', linewidth = 2, label = r'Norm. XRT Avg.')

y_min = np.min([cts_avg_array.min(), cts_norm_avg_array.min()])
y_max = np.max([cts_avg_array.max(), cts_norm_avg_array.max()])

axs2.tick_params(axis = 'both', which = 'major', labelsize = 14)
axs2.tick_params(axis = 'both', which = 'minor', labelsize = 14)
axs2.set_xlabel(r'$\theta$ (\textdegree)', fontsize = 16)
axs2.set_xlabel(r'Intensity', fontsize = 16)
axs2.set_xlim(theta_xrt.min(), theta_xrt.max())
axs2.set_ylim(y_min, y_max)
axs2.legend(frameon = False, fontsize = 14)

text_1 = axs1[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_xrt[0]), transform = axs1[0, 0].transAxes, color = 'white')

for axs in fig1.axes:
    axs.axis('off')
    axs.axvline(x = 300, color = 'red', linewidth = 2)

axs1[0, 0].set_title(r'Convolution $\rightarrow$', fontsize = 14)
axs1[1, 0].set_title(r'Raw XRT data $\rightarrow$', fontsize = 14)
axs1[2, 0].set_title(r'Raw OD data $\rightarrow$', fontsize = 14)

theta_frames = []

for theta_idx in range(n_theta):
    threshold = np.percentile(convolution_mag_array[theta_idx], 80)
        
    conv_mask = np.where(convolution_mag_array[theta_idx] >= threshold, convolution_mag_array[theta_idx], 0)

    im1_1.set_data(convolution_mag_array[theta_idx])
    im1_2.set_data(conv_mask)
    im1_3.set_data(counts_copy[theta_idx])
    im1_4.set_data(counts_norm[theta_idx])
    im1_5.set_data(opt_dens_copy[theta_idx])
    im1_6.set_data(opt_dens[theta_idx])

    text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_xrt[theta_idx]))
        
    # fig1.suptitle(r'$\theta = {0}$\textdegree'.format(theta_xrt[theta_idx]), fontsize = 18)
    # fig1.tight_layout()

    fig1.canvas.draw() # Rasterize and store Matplotlib figure contents in special buffer

    frame = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3] # Rasterize the contents in the stored buffer, access 

    theta_frames.append(frame)

plt.close(fig1)

iio2.mimsave(os.path.join(output_dir, 'xrt_norm.gif'), theta_frames, fps = 10)

plt.show()
        # plt.show()

# counts[mask] = -np.log10(counts[mask]/counts_inc)


