import numpy as np, h5py, os, sys, skimage, tkinter as tk, tomopy as tomo, csv

from skimage import transform as xform, registration as reg
from scipy import ndimage as ndi
from numpy.fft import fft, ifft, fftshift, fftfreq, fft2, ifft2
from scipy.fft import rfft
from h5_util import extract_h5_aggregate_xrf_data, create_aggregate_xrf_h5
from matplotlib import pyplot as plt, animation as anim
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

def create_ref_pair_theta_idx_array(ref_pair_theta_array, theta_array):
    ref_pair_theta_idx_1 = np.where(theta_array == ref_pair_theta_array[0])[0]
    ref_pair_theta_idx_2 = np.where(theta_array == ref_pair_theta_array[1])[0]

    return np.array([ref_pair_theta_idx_1, ref_pair_theta_idx_2])

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
    T = rfft(theta_sum.ravel())
    
    # Get components of the AC spatial frequency for axis perpendicular to rotation axis.
    
    imag = T[theta_sum.shape[0]].imag
    real = T[theta_sum.shape[0]].real
    
    # Get phase of thetasum and return center of rotation.
    
    phase = np.arctan2(imag*np.sign(real), real*np.sign(real)) 
    
    COR = theta_sum.shape[-1]*(1 - phase/np.pi)/2

    return COR

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

def iter_reproj(ref_element,
                element_array, 
                theta_array, 
                xrf_proj_img_array,
                algorithm, 
                n_iterations,
                cor_desired_angles, # A numpy array
                init_x_shift = None, 
                init_y_shift = None,
                eps = 0.3, 
                xrt_proj_img_array = None):

    global cannot_reconstruct_flag

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
    
    print(xrf_proj_img_array.shape)

    reference_projection_imgs = xrf_proj_img_array[ref_element_idx].copy() # These are effectively sinograms for the element of interest (highest contrast -> for realignment purposes)
    
    # theta_sum = np.zeros((n_slices, n_columns))

    # proj_list = [reference_projection_imgs[theta_idx, :, :] for theta_idx in range(n_theta)]

    # for proj in proj_list:
    #     theta_sum += proj

    # theta_sum = np.sum(reference_projection_imgs, axis = 0)

    # center_of_rotation = rot_center(theta_sum)

    # center_of_rotation = tomo.find_center(reference_projection_imgs, theta_array*np.pi/180, init = n_columns/2, tol = 0.1)[0]
    # print('Center of rotation = ' + str(round_correct(center_of_rotation, ndec = 2)) + ' (Projection image geometric center: ' + str(n_columns/2) + ')')

    # cor_diff = center_of_rotation - n_columns/2

    # # print('Center of rotation error = ' + str(round_correct(cor_diff, ndec = 2)))
    # print('Correcting for center of rotation error...')

    # for element_idx in range(n_elements):
    #     for theta_idx in range(n_theta):
    #         xrf_proj_img_array[element_idx, theta_idx, :, :] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx, :, :], shift = (0, cor_diff))

    iterations = []
    recon_iter_array = []
    aligned_exp_proj_iter_array = []
    synth_proj_iter_array = []

    recon = np.zeros((n_slices, n_columns, n_columns))
    
    aligned_proj = np.zeros_like(xrf_proj_img_array)
    # aligned_proj_test = np.zeros_like(reference_projection_imgs)

    proj_imgs_from_3d_recon = np.zeros((n_theta, n_slices, n_columns))
    
    # synth_test = np.zeros((n_theta, n_slices, n_columns))

    x_shifts_pc = np.zeros((n_iterations, n_theta))
    y_shifts_pc = np.zeros((n_iterations, n_theta))

    x_shift_pc_array = np.zeros(n_theta)
    y_shift_pc_array = np.zeros(n_theta)

    if init_x_shift is None or init_y_shift is None:
        if init_x_shift is None:
            init_x_shift = np.zeros(n_theta)
        
        if init_y_shift is None:
            init_y_shift = np.zeros(n_theta)
    
    elif np.isscalar(init_x_shift) or np.isscalar(init_y_shift):
        if np.isscalar(init_x_shift):
            init_x_shift *= np.ones(n_theta)
        
        if np.isscalar(init_y_shift):
            init_y_shift *= np.ones(n_theta)

    if np.any(~np.isin(cor_desired_angles, theta_array)): # If there is at least one angle not in the list of projection angles provided:
        print('Error: At least one angle is not in the provided list of projection angles. Exiting...')

        cannot_reconstruct_flag = 1

        sys.exit()

    if (np.abs(cor_desired_angles[0] - cor_desired_angles[1]) > 183) or (np.abs(cor_desired_angles[0] - cor_desired_angles[1]) < 177):
        print('Error: Angles cannot be more than 3 degrees apart. Exiting...')

        cannot_reconstruct_flag = 1

        sys.exit()
    
    # center_of_rotation = tomo.find_center(aligned_proj[ref_element_idx], theta_array*np.pi/180, init = n_columns/2, tol = 0.1)[0]
        
    # print('Center of rotation = ' + str(round_correct(center_of_rotation, ndec = 2)) + ' (Projection image geometric center: ' + str(n_columns/2) + ')')

    # cor_diff = center_of_rotation - n_columns/2

    theta_sum = np.zeros((n_slices, n_columns))

    reflection_pair_idx_array = create_ref_pair_theta_idx_array(cor_desired_angles, theta_xrf)

    for slice_idx in range(n_slices):
        sino = reference_projection_imgs[:, slice_idx, :]

        slice_proj_angle_1 = sino[reflection_pair_idx_array[0], :]
        slice_proj_angle_2 = sino[reflection_pair_idx_array[1], :]

        theta_sum[slice_idx, :] = slice_proj_angle_1 + slice_proj_angle_2
        
    center_of_rotation = rot_center(theta_sum) 

    cor_diff = center_of_rotation - n_columns/2
    
    print('Center of rotation: ' + str(round_correct(center_of_rotation, ndec = 2)))
    print('Center of rotation error = ' + str(round_correct(cor_diff, ndec = 2)))
    print('Incorporating an x-shift of ' + str(round_correct(cor_diff, ndec = 2)) + ' to all projections to correct for COR offset...') 

    for element_idx in range(n_elements):
        for theta_idx in range(n_theta):
            xrf_proj_img_array[element_idx, theta_idx, :, :] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx, :, :], shift = (0, -cor_diff))

    center_of_rotation -= cor_diff

    print('Performing iterative projection...')

    for iteration_idx in range(n_iterations):
        print('Iteration ' + str(iteration_idx + 1) + '/' + str(n_iterations))

        iterations.append(iteration_idx)
        
        if iteration_idx > 0:
            for theta_idx in range(n_theta):
                net_x_shift = x_shifts_pc[iteration_idx - 1, theta_idx] # Cumulative shift
                net_y_shift = y_shifts_pc[iteration_idx - 1, theta_idx]

                if theta_idx % 7 == 0:
                    print('Cumulative x shift = ' + str(net_x_shift))
                    print('Cumulative y shift = ' + str(net_y_shift))
                    
                aligned_proj[ref_element_idx, theta_idx, :, :] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx, :, :], shift = (net_y_shift, net_x_shift))
        
        else:
            print('Initial x shift: ' + str(init_x_shift))
            print('Initial y shift: ' + str(init_y_shift))
            
            for theta_idx in range(n_theta):
               aligned_proj[ref_element_idx, theta_idx, :, :] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx, :, :], shift = (init_y_shift[theta_idx], init_x_shift[theta_idx]))  

                # if theta_idx == n_theta//2:
                #     diff = aligned_proj[ref_element_idx, theta_idx, :, :] - xrf_proj_img_array[ref_element_idx, theta_idx, :, :]
                    
                #     y, x = phase_correlate(xrf_proj_img_array[ref_element_idx, theta_idx, :, :], aligned_proj[ref_element_idx, theta_idx, :, :], upsample_factor = 50)

                #     print(str(y) + ', ' + str(x))

                #     plt.imshow(diff)
                    
                #     shift = ndi.shift(aligned_proj[ref_element_idx, theta_idx, :, :], shift = (y, x))

                #     y, x = phase_correlate(xrf_proj_img_array[ref_element_idx, theta_idx, :, :], shift, upsample_factor = 50)

                #     print(str(y) + ', ' + str(x))
        
        # plt.imshow(aligned_proj[ref_element_idx, 0, :, :])
        # plt.show()

        aligned_exp_proj_iter_array.append(np.copy(aligned_proj[ref_element_idx]))
    
        print('Performing ' + algorithm)

        if algorithm == 'gridrec':
            recon = tomo.recon(aligned_proj[ref_element_idx], theta = theta_array*np.pi/180, center = center_of_rotation, algorithm = algorithm, filter_name = 'ramlak')
        
        elif algorithm == 'mlem':
            recon = tomo.recon(aligned_proj[ref_element_idx], theta = theta_array*np.pi/180, center = center_of_rotation, algorithm = 'mlem', num_iter = 60)
        
        else:
            print('Error: Algorithm not available.')
            
            cannot_reconstruct_flag = 1

            break
        
        recon_iter_array.append(recon)

        print(recon.shape)
                                    
        for slice_idx in range(n_slices):
            print('Slice ' + str(slice_idx + 1) + '/' + str(n_slices))
            
            proj_slice = recon[slice_idx, :, :]

            proj_imgs_from_3d_recon[:, slice_idx, :] = (skimage.transform.radon(proj_slice, theta = theta_array)).T # This radon transform assumes slices are defined by columns and not rows

        synth_proj_iter_array.append(np.copy(proj_imgs_from_3d_recon))

        for theta_idx in range(n_theta):
            
            # y_shift_pc, x_shift_cc, corr_mat_cc = cross_correlate(proj_imgs_from_3d_recon[theta_idx, :, :], aligned_proj[ref_element_idx, theta_idx, :, :]) # Cross-correlation
            y_shift_pc, x_shift_pc = phase_correlate(proj_imgs_from_3d_recon[theta_idx, :, :], aligned_proj[ref_element_idx, theta_idx, :, :], upsample_factor = 100)

            x_shift_pc_array[theta_idx] = x_shift_pc
            y_shift_pc_array[theta_idx] = y_shift_pc

            if theta_idx % 7 == 0:
                print('x-shift: ' + str(x_shift_pc) + ' (Theta = ' + str(theta_array[theta_idx]) + ' degrees')
                print('y-shift: ' + str(y_shift_pc))

                # if theta_idx == 0:
                #     fig1, axs1 = plt.subplots(2, 1)
                #     axs1[0].imshow(proj_imgs_from_3d_recon[theta_idx])
                #     axs1[1].imshow(aligned_proj[ref_element_idx, theta_idx, :, :])
                #     plt.show()
            
            if iteration_idx == 0:
                x_shifts_pc[iteration_idx, theta_idx] = x_shift_pc + init_x_shift[theta_idx]
                y_shifts_pc[iteration_idx, theta_idx] = y_shift_pc + init_y_shift[theta_idx]
     
                x_shifts_pc[iteration_idx, theta_idx] = x_shift_pc + init_x_shift[theta_idx]
                y_shifts_pc[iteration_idx, theta_idx] = y_shift_pc + init_y_shift[theta_idx]
    
                # aligned_proj_test[theta_idx, :, :] = ndi.shift(xrf_proj_img_array[ref_element_idx, theta_idx, :, :], shift = (y_shift_pc + init_y_shift, x_shift_pc + init_x_shift))

                # yshift, xshift = phase_correlate(proj_imgs_from_3d_recon[theta_idx, :, :], aligned_proj_test[theta_idx, :, :], upsample_factor = 100)
                
                # if theta_idx == 0:
                #     print('dx: ' + str(xshift))
                #     print('dy: ' + str(yshift))
                # yshift, xshift = phase_correlate(proj_imgs_from_3d_recon[theta_idx], aligned_proj_test, upsample_factor = 100)

                # print('x-shift = ' + str(xshift))
                # print('y-shift = ' + str(yshift))
                    
                        
                    
                # fig1, axs1 = plt.subplots(2, 1)
                # axs1[0].imshow(proj_imgs_from_3d_recon[theta_idx])
                # axs1[1].imshow(aligned_proj[ref_element_idx, theta_idx, :, :])
                # plt.show()

            else:
                x_shifts_pc[iteration_idx, theta_idx] = x_shifts_pc[iteration_idx - 1, theta_idx] + x_shift_pc
                y_shifts_pc[iteration_idx, theta_idx] = y_shifts_pc[iteration_idx - 1, theta_idx] + y_shift_pc

        # recon_test = tomo.recon(aligned_proj_test, theta = theta_array*np.pi/180, center = center_of_rotation, algorithm = algorithm, filter_name = 'ramlak')

        # for slice_idx in range(n_slices):
        #     print('Test slice ' + str(slice_idx + 1) + '/' + str(n_slices))
        #     synth_test[:, slice_idx, :] = (skimage.transform.radon(recon_test[slice_idx, :, :], theta = theta_array)).T # This radon transform assumes slices are defined by columns and not rows

        # yshift_test, xshift_test = phase_correlate(synth_test[0], aligned_proj_test[0], upsample_factor = 100)

        # print('Test x-shift: ' + str(xshift_test))
        # print('Test y-shift: ' + str(yshift_test))

        # fig1, axs1 = plt.subplots(2, 1)
        # axs1[0].imshow(synth_test[0])
        # axs1[1].imshow(aligned_proj_test[0])
        # plt.show()

        if np.max(np.abs(x_shift_pc_array)) <= eps and np.max(np.abs(y_shift_pc_array)) <= eps:
            print('Number of iterations taken: ' + str(iteration_idx + 1))
            print('Shifting other elements...')
            iterations = np.array(iterations)
           
            x_shifts_pc_new = x_shifts_pc[:len(iterations)]
            y_shifts_pc_new = y_shifts_pc[:len(iterations)]

            for element_idx in range(n_elements):
                if element_idx != ref_element_idx:
                    for theta_idx in range(n_theta):
                        x_shift = x_shifts_pc_new[-1, theta_idx]
                        y_shift = y_shifts_pc_new[-1, theta_idx]
                        
                        aligned_proj[element_idx, theta_idx, :, :] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx, :, :], shift = (y_shift, x_shift))
            
            print('Done')
        
            break
        
        if iteration_idx == n_iterations - 1:
            print('Iterative reprojection complete. Shifting other elements...')

            iterations = np.array(iterations)

            x_shifts_pc_new, y_shifts_pc_new = x_shifts_pc, y_shifts_pc
            
            for element_idx in range(n_elements):
                if element_idx != ref_element_idx:
                    for theta_idx in range(n_theta):
                        x_shift = x_shifts_pc_new[-1, theta_idx]
                        y_shift = y_shifts_pc_new[-1, theta_idx]
                        
                        aligned_proj[element_idx, theta_idx, :, :] = ndi.shift(xrf_proj_img_array[element_idx, theta_idx, :, :], shift = (y_shift, x_shift))
            
            print('Done')

    if cannot_reconstruct_flag:
        return

    aligned_exp_proj_iter_array = np.array(aligned_exp_proj_iter_array)
    synth_proj_iter_array = np.array(synth_proj_iter_array)
    recon_iter_array = np.array(recon_iter_array)
    
    return aligned_proj, x_shifts_pc_new, y_shifts_pc_new, aligned_exp_proj_iter_array, recon_iter_array, synth_proj_iter_array

# root = tk.Tk()
    
# root.withdraw()

# output_h5_file_path = filedialog.asksaveasfilename(parent = root, title = "Select save path", filetypes = (('HDF5 Files', '*.h5')))

# elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
output_dir_path_base = '/home/bwr0835'

# output_file_name_base = input('Choose a base file name: ')
output_file_name_base = 'gridrec_5_iter_vacek_cor_no_correction_padding_-22_deg_158_deg'

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
cor_desired_angles = np.array([-22, 158])

algorithm = 'gridrec'

cannot_reconstruct_flag = 0

aligned_proj, \
net_x_shifts, \
net_y_shifts, \
aligned_proj_iter_array, \
recon_iter_array, \
synth_proj_iter_array = iter_reproj(desired_element, 
                                    elements_xrf, 
                                    theta_xrf, 
                                    counts_xrf, 
                                    algorithm, 
                                    n_desired_iter,
                                    cor_desired_angles, 
                                    init_x_shift = init_x_shift)

if cannot_reconstruct_flag:
    print('Cannot reconstruct. Exiting...')

    sys.exit()

print('Saving files...')

full_output_dir_path = os.path.join(output_dir_path_base, 'iter_reproj', output_file_name_base)

os.makedirs(full_output_dir_path, exist_ok = True)

np.save(os.path.join(full_output_dir_path, 'theta_array.npy'), theta_xrf)
np.save(os.path.join(full_output_dir_path, 'aligned_proj_all_elements.npy'), aligned_proj)
np.save(os.path.join(full_output_dir_path, 'aligned_proj_array_iter_' + desired_element + '.npy'), aligned_proj_iter_array)
np.save(os.path.join(full_output_dir_path, 'synth_proj_array_iter_' + desired_element + '.npy'), synth_proj_iter_array)
np.save(os.path.join(full_output_dir_path, 'recon_array_iter_' + desired_element + '.npy'), recon_iter_array)
np.save(os.path.join(full_output_dir_path, 'net_x_shifts_' + desired_element + '.npy'), net_x_shifts)
np.save(os.path.join(full_output_dir_path, 'net_y_shifts_' + desired_element + '.npy'), net_y_shifts)

    # with open(os.path.join(full_output_dir_path, 'net_x_shifts.csv'), 'w') as f:
    #     writer = csv.writer(f)
        
    #     writer.writerows(net_x_shifts)
    
    # with open(os.path.join(full_output_dir_path, 'net_y_shifts.csv'), 'w') as f:
    #     writer = csv.writer(f)

    #     writer.writerows(net_y_shifts)