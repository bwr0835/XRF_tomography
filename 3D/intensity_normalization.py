import numpy as np, h5_util, os, sys

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

def pad_col_row(array, dataset):
    if dataset == '' or array.ndim != 4:
        print('Error: No dataset specified or number of array dimesions ≠ 4. Exiting program...')

        sys.exit()

    n_elements, n_theta, n_slices, n_columns = array.shape
    
    # Create a new array with the expected final shape
    new_shape = (n_elements, n_theta, n_slices + 1, n_columns + 1)
    result_array = np.zeros(new_shape)

    if dataset == 'xrt':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()

                padded_row = array_avg*np.ones(n_columns)
                
                # Add row first
                temp_array = np.vstack((array[element_idx, theta_idx], padded_row))

                # After adding row, the shape is now (n_slices + 1, n_columns)
                # Get the current shape after row addition
                current_shape = temp_array.shape
                padded_column = array_avg*np.ones((current_shape[0], 1))

                result_array[element_idx, theta_idx] = np.hstack((temp_array, padded_column))
    
    elif dataset == 'xrf':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()

                padded_row = array_avg*np.zeros(n_columns)
                
                # Add row first
                temp_array = np.vstack((array[element_idx, theta_idx], padded_row))

                # After adding row, the shape is now (n_slices + 1, n_columns)
                # Get the current shape after row addition
                current_shape = temp_array.shape
                padded_column = array_avg*np.zeros((current_shape[0], 1))

                result_array[element_idx, theta_idx] = np.hstack((temp_array, padded_column))

    else:
        print('Error: Invalid dataset. Exiting program...')

        sys.exit()

    return result_array

def pad_col(array, dataset):
    if dataset == '' or array.ndim != 4:
        print('Error: No dataset specified or number of array dimesions ≠ 4. Exiting program...')

        sys.exit()

    n_elements, n_theta, n_slices, _ = array.shape

    if dataset == 'xrt':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()
                # final_column = array_avg*array[element_idx, theta_idx, :, -1].T
                final_column = array_avg*np.ones((n_slices, 1))
                
                array[element_idx, theta_idx] = np.hstack((array[element_idx, theta_idx], final_column))

    elif dataset == 'xrf':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                # final_column = array[element_idx, theta_idx, :, -1].T
                final_column = np.zeros((n_slices, 1))
                
                array[element_idx, theta_idx, :, :] = np.hstack((array[element_idx, theta_idx, :, :], final_column))

    else:
        print('Error: Invalid dataset. Exiting program...')

        sys.exit()
    
    return array

def pad_row(array, dataset):
    if dataset == '' or array.ndim != 4:
        print('Error: No dataset specified or number of array dimesions ≠ 4. Exiting...')

        sys.exit()
    
    n_elements, n_theta, _, n_columns = array.shape
    
    if dataset == 'xrt':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()
                
                # final_row = array[element_idx, theta_idx, -1, :]
                final_row = array_avg*np.ones(n_columns)

                array[element_idx, theta_idx] = np.vstack((array[element_idx, theta_idx], final_row))

    elif dataset == 'xrf':
        for element_idx in range(n_elements):
            for theta_idx in range(n_theta):
                array_avg = array[element_idx, theta_idx].mean()
                
                # final_row = array[element_idx, theta_idx, -1, :]
                final_row = array_avg*np.zeros(n_columns)

                array[element_idx, theta_idx] = np.vstack((array[element_idx, theta_idx], final_row))

    else:
        print('Error: Invalid dataset. Exiting program...')

        sys.exit()

    return array

def joint_fluct_norm(xrt_array, xrf_array, sigma_1 = 5, alpha = 10, sigma_2 = 10):
    if xrt_array.ndim != 3 and xrf_array != 3:
        print('Error: Number of XRT and/or XRF array dimesions ≠ 4. Exiting program...')

        sys.exit()

    n_theta, n_slices, n_columns = xrt_array.shape

    convolution_mag_array = []

    xrt_mask_avg_sum = 0
    
    for theta_idx in range(n_theta):
        xrt_vignetted = edge_gauss_filter(xrt_array[theta_idx], sigma = sigma_1, alpha = alpha, nx = n_columns, ny = n_slices)

        convolution_mag = ndi.gaussian_filter(xrt_vignetted, sigma = sigma_2) # Blur the entire image using Gaussian filter/convolution

        threshold = np.percentile(convolution_mag, 80) # Take top 20% of intensities for masking

        mask = convolution_mag >= threshold

        xrt_mask_avg = xrt_array[theta_idx, mask].mean()

        xrt_array[theta_idx] /= xrt_mask_avg # First part of I0' = I0(<I_theta,mask,avg>/I_theta,mask,avg)
        xrf_array[theta_idx] /= xrt_mask_avg

        xrt_mask_avg_sum += xrt_mask_avg

        convolution_mag_array.append(convolution_mag)
    
    global_xrt_mask_avg = xrt_mask_avg_sum/n_theta

    xrt_array *= global_xrt_mask_avg # Second part of I0' = I0(<I_theta,mask,avg>/I_theta,mask,avg)
    xrf_array *= global_xrt_mask_avg
    
    return xrt_array, xrf_array, global_xrt_mask_avg, np.array(convolution_mag_array)

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

file_path_xrt = '/Users/bwr0835/Documents/2_ide_aggregate_xrt.h5'
file_path_xrf = '/Users/bwr0835/Documents/2_ide_aggregate_xrf.h5'
# file_path_xrf = '/Users/bwr0835/Documents/hxn_aggregate_xrf.h5'

output_dir = '/Users/bwr0835/Documents'

elements_xrt, counts_xrt, _, _, _ = h5_util.extract_h5_aggregate_xrt_data(file_path_xrt)
elements_xrf, counts_xrf, theta_xrf, _ = h5_util.extract_h5_aggregate_xrf_data(file_path_xrf)

ref_element_xrt = 'ds_ic'
ref_element_xrf = 'Fe'
# ref_element_xrf_array = ['Ni_K', 'Cu_K', 'Zn_K', 'Ce_L']

# ref_element_idx_array = np.zeros(len(ref_element_xrf_array), dtype = int)

fps = 10

n_theta = counts_xrf.shape[1] # Number of projection angles (projection images)
n_slices = counts_xrf.shape[2] # Number of rows in a projection image
n_columns = counts_xrf.shape[3] # Number of columns in a projection image

if (n_slices % 2) or (n_columns % 2):
    if (n_slices % 2) and (n_columns % 2):
        counts_xrt = pad_col_row(counts_xrt, 'xrt')
        counts_xrf = pad_col_row(counts_xrf, 'xrf')
            
        n_slices += 1
        n_columns += 1
        
    elif n_slices % 2:
        counts_xrt = pad_row(counts_xrt, 'xrt')
        counts_xrf = pad_row(counts_xrf, 'xrf')

        n_slices += 1

    else:
        counts_xrt = pad_col(counts_xrt, 'xrt')
        counts_xrf = pad_col(counts_xrf, 'xrf')

        n_columns += 1

# for idx, desired_element in enumerate(ref_element_xrf_array):
    # ref_element_idx_array[idx] = elements_xrf.index(desired_element)

# print(ref_element_idx_array)

# ref_element_idx_array = []

# for ref_element in ref_element_xrf_array:
#     ref_element_idx = elements_xrf.index(ref_element)

#     ref_element_idx_array.append(ref_element_idx)

ref_element_idx_xrt = elements_xrt.index(ref_element_xrt)
ref_element_idx_xrf = elements_xrf.index(ref_element_xrf)

vmin = counts_xrf.min()
vmax = counts_xrf.max()

cts = counts_xrt[ref_element_idx_xrt]
counts = counts_xrf[ref_element_idx_xrf]

# cts_copy = cts.copy()
# counts_copy = counts.copy()

# I0_avg = np.mean(cts_copy)

# opt_dens_copy = -np.log(cts_copy/I0_avg)

counts_xrt_norm, counts_xrf_norm, counts_xrt_mask_global_avg, convolution_mag_array = joint_fluct_norm(cts, counts)

opt_dens = -np.log(counts_xrt_norm/counts_xrt_mask_global_avg)

# vmin_xrt = cts_copy.min()
# vmax_xrt = cts_copy.max()

# vmin_xrf = counts_copy.min()
# vmax_xrf = counts_copy.max()

vmin_xrt_norm = counts_xrt_norm.min()
vmax_xrt_norm = counts_xrt_norm.max()

vmin_xrf_norm = counts_xrf_norm.min()
vmax_xrf_norm = counts_xrf_norm.max()

# vmin_conv = convolution_mag_array.min()
# vmax_conv = convolution_mag_array.max()

# vmin_od = opt_dens_copy.min()
# vmax_od = opt_dens_copy.max()

vmin_od_norm = opt_dens.min()
vmax_od_norm = opt_dens.max()

# fig1, axs1 = plt.subplots(3, 2)
# fig2, axs2 = plt.subplots(2, 2)
# fig3, axs3 = plt.subplots(3, 2)
# fig4, axs4 = plt.subplots()
# fig5, axs5 = plt.subplots(2, 2) # BNL XRF DATA
fig6, axs6 = plt.subplots(3, 1) # Normalized XRT, OD, XRF only

# im_fig_array = [fig1, fig2, fig3]

# threshold = np.percentile(convolution_mag_array[0], 80)

# conv_mask = np.where(convolution_mag_array[0] >= threshold, convolution_mag_array[0], 0)

# im1_1 = axs1[0, 0].imshow(convolution_mag_array[0], vmin = vmin_conv, vmax = vmax_conv)
# im1_2 = axs1[0, 1].imshow(conv_mask, vmin = vmin_conv, vmax = vmax_conv)
# im1_3 = axs1[1, 0].imshow(cts_copy[0], vmin = vmin_xrt, vmax = vmax_xrt)
# im1_4 = axs1[1, 1].imshow(counts_xrt_norm[0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm)
# im1_5 = axs1[2, 0].imshow(opt_dens_copy[0], vmin = vmin_od, vmax = vmax_od)
# im1_6 = axs1[2, 1].imshow(opt_dens[0], vmin = vmin_od_norm, vmax = vmax_od_norm)

# im2_1 = axs2[0, 0].imshow(convolution_mag_array[0], vmin = vmin_conv, vmax = vmax_conv)
# im2_2 = axs2[0, 1].imshow(conv_mask, vmin = vmin_conv, vmax = vmax_conv)
# im2_3 = axs2[1, 0].imshow(counts_copy[0], vmin = vmin_xrf, vmax = vmax_xrf)
# im2_4 = axs2[1, 1].imshow(counts_xrf_norm[0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)

# im3_1 = axs3[0, 0].imshow(cts_copy[0], vmin = vmin_xrt, vmax = vmax_xrt)
# im3_2 = axs3[0, 1].imshow(counts_xrt_norm[0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm)
# im3_3 = axs3[1, 0].imshow(opt_dens_copy[0], vmin = vmin_od, vmax = vmax_od)
# im3_4 = axs3[1, 1].imshow(opt_dens[0], vmin = vmin_od_norm, vmax = vmax_od_norm)
# im3_5 = axs3[2, 0].imshow(counts_copy[0], vmin = vmin_xrf, vmax = vmax_xrf)
# im3_6 = axs3[2, 1].imshow(counts_xrf_norm[0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)

# im5_1 = axs5[0, 0].imshow(counts_xrf[ref_element_idx_array[0], 0], vmin = vmin, vmax = vmax)
# im5_2 = axs5[0, 1].imshow(counts_xrf[ref_element_idx_array[1], 0], vmin = vmin, vmax = vmax)
# im5_3 = axs5[1, 0].imshow(counts_xrf[ref_element_idx_array[2], 0], vmin = vmin, vmax = vmax)
# im5_4 = axs5[1, 1].imshow(counts_xrf[ref_element_idx_array[3], 0], vmin = vmin, vmax = vmax)

# neg_84_idx = np.where(theta_xrf == -84)[0][0]

# im5_1 = axs5[0, 0].imshow(counts_xrf[ref_element_idx_array[0], neg_84_idx], vmin = vmin, vmax = vmax)
# im5_2 = axs5[0, 1].imshow(counts_xrf[ref_element_idx_array[1], neg_84_idx], vmin = vmin, vmax = vmax)
# im5_3 = axs5[1, 0].imshow(counts_xrf[ref_element_idx_array[2], neg_84_idx], vmin = vmin, vmax = vmax)
# im5_4 = axs5[1, 1].imshow(counts_xrf[ref_element_idx_array[3], neg_84_idx], vmin = vmin, vmax = vmax)

im6_1 = axs6[0].imshow(counts_xrt_norm[0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm)
im6_2 = axs6[1].imshow(opt_dens[0], vmin = vmin_od_norm, vmax = vmax_od_norm)
im6_3 = axs6[2].imshow(counts_xrf_norm[0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)

axs6[0].set_title(r'XRT')
axs6[1].set_title(r'Opt. Dens.')
axs6[2].set_title(r'XRF')

# text_1 = axs1[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_xrf[0]), transform = axs1[0, 0].transAxes, color = 'white')
# text_2 = axs2[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_xrf[0]), transform = axs2[0, 0].transAxes, color = 'white')
# text_3 = axs3[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_xrf[0]), transform = axs3[2, 0].transAxes, color = 'white')
# text_5 = axs5[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_xrf[0]), transform = axs5[0, 0].transAxes, color = 'white')
text_6 = axs6[2].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_xrf[0]), transform = axs6[2].transAxes, color = 'white')

# plt.show()

# theta_frames5 = []
theta_frames6 = []

# idx = 0

# for axs in fig5.axes:
#     axs.axis('off')
#     axs.axvline(x = n_columns//2, color = 'red', linewidth = 2)
#     axs.set_title(r'{0}'.format(ref_element_xrf_array[idx]))

#     idx += 1

# fig5.suptitle(r'HXN XRF')

for axs in fig6.axes:
    axs.axis('off')
    axs.axvline(x = n_columns//2, color = 'red', linewidth = 2)

# for theta_idx in range(n_theta):
#     im5_1.set_data(counts_xrf[ref_element_idx_array[0], theta_idx])
#     im5_2.set_data(counts_xrf[ref_element_idx_array[1], theta_idx])
#     im5_3.set_data(counts_xrf[ref_element_idx_array[2], theta_idx])
#     im5_4.set_data(counts_xrf[ref_element_idx_array[3], theta_idx])

    



#     text_5.set_text(r'$\theta = {0}$\textdegree'.format(theta_xrf[theta_idx]))

#     fig5.canvas.draw()

#     frame5 = np.array(fig5.canvas.renderer.buffer_rgba())[:, :, :3]

#     theta_frames5.append(frame5)


# cts_xrt_mask_avg = np.zeros(n_theta)
# cts_xrt_mask_avg_norm = np.zeros(n_theta)

# for theta_idx in range(n_theta):
#     threshold = np.percentile(convolution_mag_array[theta_idx], 80)

#     mask = convolution_mag_array[theta_idx] >= threshold

#     cts_xrt_mask_avg[theta_idx] = cts_copy[theta_idx, mask].mean()
#     cts_xrt_mask_avg_norm[theta_idx] = counts_xrt_norm[theta_idx, mask].mean()

# curve1, = axs4.plot(theta_xrf, cts_xrt_mask_avg, 'ko-', linewidth = 2, label = r'Raw XRT Avg.')
# curve2, = axs4.plot(theta_xrf, cts_xrt_mask_avg_norm, 'ro-', linewidth = 2, label = r'Norm. XRT Avg.')

# y_min = np.min([cts_xrt_mask_avg.min(), cts_xrt_mask_avg_norm.min()])
# y_max = np.max([cts_xrt_mask_avg.max(), cts_xrt_mask_avg_norm.max()])

# axs4.tick_params(axis = 'both', which = 'major', labelsize = 14)
# axs4.tick_params(axis = 'both', which = 'minor', labelsize = 14)
# axs4.set_xlabel(r'$\theta$ (\textdegree)', fontsize = 16)
# axs4.set_xlabel(r'Intensity', fontsize = 16)
# axs4.set_xlim(theta_xrf.min(), theta_xrf.max())
# axs4.set_ylim(y_min, y_max)
# axs4.legend(frameon = False, fontsize = 14)

# for fig_idx in im_fig_array:
#     for axs in fig_idx.axes:
#         axs.axis('off')
#         axs.axvline(x = 300, color = 'red', linewidth = 2)

# axs1[0, 0].set_title(r'XRT Conv. $\rightarrow$', fontsize = 14)
# axs1[1, 0].set_title(r'Raw XRT data $\rightarrow$', fontsize = 14)
# axs1[2, 0].set_title(r'Raw OD data $\rightarrow$', fontsize = 14)

# axs2[0, 0].set_title(r'XRT Conv. $\rightarrow$', fontsize = 14)
# axs2[1, 0].set_title(r'Raw XRF data $\rightarrow$', fontsize = 14)

# axs3[0, 0].set_title(r'Raw XRT data $\rightarrow$', fontsize = 14)
# axs3[1, 0].set_title(r'Raw OD data $\rightarrow$', fontsize = 14)
# axs3[2, 0].set_title(r'Raw XRF data $\rightarrow$', fontsize = 14)

# theta_frames1 = []
# theta_frames2 = []
# theta_frames3 = []

fig6.tight_layout()

for theta_idx in range(n_theta):
#     threshold = np.percentile(convolution_mag_array[theta_idx], 80)
        
#     conv_mask = np.where(convolution_mag_array[theta_idx] >= threshold, convolution_mag_array[theta_idx], 0)

#     im1_1.set_data(convolution_mag_array[theta_idx])
#     im1_2.set_data(conv_mask)
#     im1_3.set_data(cts_copy[theta_idx])
#     im1_4.set_data(counts_xrt_norm[theta_idx])
#     im1_5.set_data(opt_dens_copy[theta_idx])
#     im1_6.set_data(opt_dens[theta_idx])

#     im2_1.set_data(convolution_mag_array[theta_idx])
#     im2_2.set_data(conv_mask)
#     im2_3.set_data(counts_copy[theta_idx])
#     im2_4.set_data(counts_xrf_norm[theta_idx])

#     im3_1.set_data(cts_copy[theta_idx])
#     im3_2.set_data(counts_xrt_norm[theta_idx])
#     im3_3.set_data(opt_dens_copy[theta_idx])
#     im3_4.set_data(opt_dens[theta_idx])
#     im3_5.set_data(counts_copy[theta_idx])
#     im3_6.set_data(counts_xrf_norm[theta_idx])

    im6_1.set_data(counts_xrt_norm[theta_idx])
    im6_2.set_data(opt_dens[theta_idx])
    im6_3.set_data(counts_xrf_norm[theta_idx])

#     text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_xrf[theta_idx]))
#     text_2.set_text(r'$\theta = {0}$\textdegree'.format(theta_xrf[theta_idx]))
#     text_3.set_text(r'$\theta = {0}$\textdegree'.format(theta_xrf[theta_idx]))
    text_6.set_text(r'$\theta = {0}$\textdegree'.format(theta_xrf[theta_idx]))

#     fig1.canvas.draw() # Rasterize and store Matplotlib figure contents in special buffer
#     fig2.canvas.draw()
#     fig3.canvas.draw()
    fig6.canvas.draw()

#     frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3] # Rasterize the contents in the stored buffer, access 
#     frame2 = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3]
#     frame3 = np.array(fig3.canvas.renderer.buffer_rgba())[:, :, :3]
    frame6 = np.array(fig6.canvas.renderer.buffer_rgba())[:, :, :3]

#     theta_frames1.append(frame1)
#     theta_frames2.append(frame2)
#     theta_frames3.append(frame3)
    theta_frames6.append(frame6)

# plt.close(fig1)
# plt.close(fig2)
# plt.close(fig3)
# plt.close(fig5)
plt.close(fig6)

# iio2.mimsave(os.path.join(output_dir, 'xrt_norm.gif'), theta_frames1, fps = fps)
# iio2.mimsave(os.path.join(output_dir, 'xrf_norm.gif'), theta_frames2, fps = fps)
# iio2.mimsave(os.path.join(output_dir, 'xrt_xrf_norm.gif'), theta_frames3, fps = fps)
# iio2.mimsave(os.path.join(output_dir, 'init_hxn_xrf.gif'), theta_frames5, fps = fps)
iio2.mimsave(os.path.join(output_dir, '2_ide_init_xrt_opt_dens_xrf.gif'), theta_frames6, fps = fps)

# plt.show()