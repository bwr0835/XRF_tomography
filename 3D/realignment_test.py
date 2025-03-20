import numpy as np, h5py, os, skimage, tkinter as tk, tomopy as tomo, matplotlib as mpl, scipy.ndimage as spndi, os

from skimage import transform as xform, registration as reg
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq, fftn, ifftn, fft2, ifft2
from h5_util import extract_h5_aggregate_xrf_data, create_aggregate_xrf_h5
from matplotlib import pyplot as plt
from tkinter import filedialog
from pystackreg import StackReg as sr

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def ramp_filter(sinogram):
    n_theta, n_rows, n_columns = sinogram.shape
    
    fft_sinogram = fft(sinogram, axis = 2) # Fourier transform along columns/horizontal scan dimension
    frequency_array = fftfreq(n_columns) # Create NORMALIZED (w.r.t. Nyquist frequency) frequency array

    ramp_filt = np.abs(frequency_array)

    filtered_sinogram = np.real(ifft(fft_sinogram*ramp_filt, axis = 2)) # Only want the real component, or else artifacts will show up

    return filtered_sinogram

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

def round_correct(num, ndec): # CORRECTLY round a number (num) to chosen number of decimal places (ndec)

    if ndec == 0:
        return int(num + 0.5)
    
    else:
        digit_value = 10**ndec
        
        if num > 0:
            return int(num*digit_value + 0.5)/digit_value
        
        else:
            return int(num*digit_value - 0.5)/digit_value

def cross_correlate(recon_proj, orig_proj):
    # Credit goes to Fabricio Marin and the XRFTomo GUI

    recon_proj_fft = fft2(recon_proj)
    orig_proj_fft = fft2(orig_proj)

    img_dims = orig_proj.shape
    
    y_dim = img_dims[0]
    x_dim = img_dims[1]

    cross_corr = np.abs(ifft2(recon_proj_fft*orig_proj_fft.conjugate()))
    
    y_shift, x_shift = np.unravel_index(np.argmax(cross_corr), img_dims) # Locate maximum peak in cross-correlation matrix and output the 
    
    if y_shift > y_dim//2:
        y_shift -= y_dim
    
    if x_shift > x_dim//2:
        x_shift -= x_dim

    return y_shift, x_shift, cross_corr

def phase_correlate(recon_proj, orig_proj, upsample_factor):
    shift, _, _ = reg.phase_cross_correlation(reference_image = recon_proj, moving_image = orig_proj, upsample_factor = upsample_factor)

    y_shift, x_shift = shift[0], shift[1]

    return y_shift, x_shift

def subpixel_cross_correlation(recon_proj, orig_proj):
    y_shift, x_shift, cross_corr = cross_correlate(recon_proj, orig_proj)

    cross_corr = fftshift(cross_corr)

    y_center, x_center = recon_proj.shape[0]//2, recon_proj.shape[1]//2

    x_shift_array = np.array([x_shift - 1, x_shift, x_shift + 1]) + x_center
    y_shift_array = np.array([y_shift - 1, y_shift, y_shift + 1]) + y_center

    x_values = cross_corr[y_shift_array[1], x_shift_array]
    y_values = cross_corr[y_shift_array, x_shift_array[1]]

    p2x, p1x, _ = np.polyfit(x_shift_array, x_values, deg = 2)
    p2y, p1y, _ = np.polyfit(y_shift_array, y_values, deg = 2)

    x_shift_subpixel = -p1x/(2*p2x)
    y_shift_subpixel = -p1y/(2*p2y)

    return y_shift_subpixel, x_shift_subpixel

def save_proj_img_npy(array, iter_idx, theta, proj_type, recon_mode, output_file_path):
    if output_file_path == "":
        return
    
    iter_idx += 1

    full_output_dir_path = os.path.join(output_file_path, recon_mode, 'iteration_' + str(iter_idx), proj_type)

    os.makedirs(full_output_dir_path, exist_ok = True)

    full_output_file_path = os.path.join(full_output_dir_path, 'proj_img_' + str(int(theta)) + '_deg.npy')

    np.save(full_output_file_path, array)

    return

def save_recon_slice_npy(array, iter_idx, slice_idx, recon_mode, output_file_path):
    if output_file_path == "":
        return
    
    iter_idx += 1
    slice_idx += 1

    full_output_dir_path = os.path.join(output_file_path, recon_mode, 'iteration_' + str(iter_idx), 'recon')

    os.makedirs(full_output_dir_path, exist_ok = True)

    full_output_file_path = os.path.join(full_output_dir_path, 'recon_slice_' + str(slice_idx) + '.npy')

    np.save(full_output_file_path, array)

    return

def save_net_shift_data_npy(shift_array, shift_dxn, recon_mode, output_file_path):
    if output_file_path == "":
        return

    full_output_dir_path = os.path.join(output_file_path, recon_mode, 'net_shifts')

    os.makedirs(full_output_dir_path, exist_ok = True)

    full_output_file_path = os.path.join(full_output_dir_path, shift_dxn + '_shift_array.npy')

    np.save(full_output_file_path, shift_array)

    return

def save_theta_array(theta_array, recon_mode, output_file_path):
    if output_file_path == "":
        return
    
    full_output_dir_path = os.path.join(output_dir_path, recon_mode)

    os.makedirs(full_output_dir_path, exist_ok = True)

    full_output_file_path = os.path.join(full_output_dir_path, 'theta_array.npy')

    np.save(full_output_file_path, theta_array)

    return

def iter_reproj(ref_element, element_array, theta_array, xrf_proj_img_array, n_iterations, output_dir_path, eps = 0.3): # Assuming no initial shift in
    n_elements = xrf_proj_img_array.shape[0] # Number of elements
    n_theta = xrf_proj_img_array.shape[1] # Number of projection angles (projection images)
    n_slices = xrf_proj_img_array.shape[2] # Number of rows in a projection image
    n_columns = xrf_proj_img_array.shape[3] # Number of columns in a projection image
    
    ref_element_idx = element_array.index(ref_element)
    reference_projection_imgs = xrf_proj_img_array[ref_element_idx] # These are effectively sinograms for the element of interest (highest contrast -> for realignment purposes)
                                                                    # (n_theta, n_slices -> n_rows, n_columns)

    center_of_rotation = tomo.find_center(reference_projection_imgs, theta_array*np.pi/180)

    iterations = []

    recon = np.zeros((n_slices, n_columns, n_columns))
    
    aligned_proj = np.zeros_like(xrf_proj_img_array)

    proj_imgs_from_3d_recon = np.zeros((n_theta, n_slices, n_columns))

    x_shifts_pc = np.zeros((n_iterations, n_theta))
    y_shifts_pc = np.zeros((n_iterations, n_theta))

    x_shift_pc_array = np.zeros(n_theta)
    y_shift_pc_array = np.zeros(n_theta)

    init_x_shift = 0
    init_y_shift = 0
    
    for iteration_idx in range(n_iterations):
        print('Iteration ' + str(iteration_idx + 1) + '/' + str(n_iterations))

        iterations.append(iteration_idx)
        
        if iteration_idx > 0:
            for theta_idx in range(n_theta):
                cum_x_shift = x_shifts_pc[iteration_idx - 1, theta_idx] # Cumulative shift
                cum_y_shift = y_shifts_pc[iteration_idx - 1, theta_idx]

                x_shift = (x_shift_pc_array[theta_idx]).copy()
                y_shift = (y_shift_pc_array[theta_idx]).copy()

                if theta_idx % 7 == 0:
                    print('Cumulative x shift = ' + str(cum_x_shift))
                    print('Cumulative y shift = ' + str(cum_y_shift))
                    
                # aligned_proj[ref_element_idx, theta_idx, :, :] = spndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx, :, :], shift = (y_shift, x_shift))
                aligned_proj[ref_element_idx, theta_idx, :, :] = spndi.shift(aligned_proj[ref_element_idx, theta_idx, :, :], shift = (y_shift, x_shift))

        elif init_x_shift != 0 or init_y_shift != 0:
            print('Initial x shift: ' + str(init_x_shift))
            print('Initial y shift: ' + str(init_y_shift))
            
            for theta_idx in range(n_theta):
                aligned_proj[ref_element_idx, theta_idx, :, :] = spndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx, :, :], shift = (init_y_shift, init_x_shift))

                # if theta_idx == n_theta//2:
                #     diff = aligned_proj[ref_element_idx, theta_idx, :, :] - xrf_proj_img_array[ref_element_idx, theta_idx, :, :]
                    
                #     y, x = phase_correlate(xrf_proj_img_array[ref_element_idx, theta_idx, :, :], aligned_proj[ref_element_idx, theta_idx, :, :], upsample_factor = 50)

                #     print(str(y) + ', ' + str(x))

                #     plt.imshow(diff)
                    
                #     shift = spndi.shift(aligned_proj[ref_element_idx, theta_idx, :, :], shift = (y, x))

                #     y, x = phase_correlate(xrf_proj_img_array[ref_element_idx, theta_idx, :, :], shift, upsample_factor = 50)

                #     print(str(y) + ', ' + str(x))

                #     plt.show()
        else:
            aligned_proj = xrf_proj_img_array

        algorithm = 'gridrec'

        print('Performing ' + algorithm)

        if algorithm == 'gridrec':
            recon = tomo.recon(aligned_proj[ref_element_idx], theta = theta_array*np.pi/180, center = center_of_rotation, algorithm = algorithm, filter_name = 'ramlak')
        
        elif algorithm == 'mlem':
            recon = tomo.recon(aligned_proj[ref_element_idx], theta = theta_array*np.pi/180, center = center_of_rotation, algorithm = 'mlem', num_iter = 60)
        
        print(recon.shape)
                                    
        for slice_idx in range(n_slices):
            print('Slice ' + str(slice_idx + 1) + '/' + str(n_slices))
            
            proj_slice = recon[slice_idx, :, :]

            proj_imgs_from_3d_recon[:, slice_idx, :] = (skimage.transform.radon(proj_slice, theta = theta_array)).T # This radon transform assumes slices are defined by columns and not rows
                    
            # save_recon_slice_npy(proj_slice, iteration_idx, slice_idx, 'gridrec', output_dir_path)

        for theta_idx in range(n_theta):
            # y_shift_pc, x_shift_cc, corr_mat_cc = cross_correlate(proj_imgs_from_3d_recon[theta_idx, :, :], aligned_proj[ref_element_idx, theta_idx, :, :]) # Cross-correlation
            y_shift_pc, x_shift_pc = phase_correlate(proj_imgs_from_3d_recon[theta_idx, :, :], aligned_proj[ref_element_idx, theta_idx, :, :], upsample_factor = 50)
            
            x_shift_pc_array[theta_idx] = x_shift_pc
            y_shift_pc_array[theta_idx] = y_shift_pc

            # save_proj_img_npy(proj_imgs_from_3d_recon[theta_idx, :, :], iteration_idx, theta_array[theta_idx], 'synthesized', 'gridrec', output_dir_path)
            # save_proj_img_npy(aligned_proj[ref_element_idx, theta_idx, :, :], iteration_idx, theta_array[theta_idx], 'experimental', 'gridrec', output_dir_path)
            # save_proj_img_npy(corr_mat_cc, iteration_idx, theta_array[theta_idx], 'xcorr', 'gridrec', output_dir_path)

            if theta_idx % 7 == 0:
                print('x-shift: ' + str(x_shift_pc) + ' (Theta = ' + str(theta_array[theta_idx]) + ' degrees')
                print('y-shift: ' + str(y_shift_pc))
            
            if iteration_idx == 0:
                x_shifts_pc[iteration_idx, theta_idx] = x_shift_pc + init_x_shift
                y_shifts_pc[iteration_idx, theta_idx] = y_shift_pc + init_y_shift
                
            else:
                x_shifts_pc[iteration_idx, theta_idx] = x_shifts_pc[iteration_idx - 1, theta_idx] + x_shift_pc
                y_shifts_pc[iteration_idx, theta_idx] = y_shifts_pc[iteration_idx - 1, theta_idx] + y_shift_pc

        if np.max(np.abs(x_shift_pc_array)) <= eps and np.max(np.abs(y_shift_pc_array)) <= eps:
            print('Number of iterations taken: ' + str(iteration_idx + 1))
            
            iterations = np.array(iterations)
           
            x_shifts_pc_new = x_shifts_pc[:len(iterations)]
            y_shifts_pc_new = y_shifts_pc[:len(iterations)]

            # save_net_shift_data_npy(x_shifts_pc_new, 'x', algorithm, output_dir_path)
            # save_net_shift_data_npy(y_shifts_pc_new, 'y', algorithm, output_dir_path)
            # save_theta_array(theta_array, algorithm, output_dir_path)
        
            break
        
        if iteration_idx == n_iterations - 1:
            print('Iterative reprojection complete')

            iterations = np.array(iterations)
            
            # save_net_shift_data_npy(x_shifts_pc, 'x', 'gridrec', output_dir_path)
            # save_net_shift_data_npy(y_shifts_pc, 'y', 'gridrec', output_dir_path)
            # save_theta_array(theta_array, 'gridrec', output_dir_path)

            break

# root = tk.Tk()
    
# root.withdraw()

# output_h5_file_path = filedialog.asksaveasfilename(parent = root, title = "Select save path", filetypes = (('HDF5 Files', '*.h5')))

# elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'

# create_aggregate_xrf_h5(file_path_array, file_path_xrf, synchrotron = 'aps')

# # file_path_xrt = ''

elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

# output_dir_path = filedialog.askdirectory(parent = root, title = "Choose directory to output NPY files to.")

output_dir_path = '/raid/users/roter'

iter_reproj('Fe', elements_xrf, theta_xrf, counts_xrf, 10, output_dir_path)




