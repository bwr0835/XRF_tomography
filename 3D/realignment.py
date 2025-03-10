import numpy as np, h5py, os, skimage, tkinter as tk, tomopy as tomo, matplotlib as mpl, scipy.ndimage as spndi, pystackreg as psr

from skimage import transform as xform
from scipy import signal as sig
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq, fftn, ifftn, fft2, ifft2
from h5_util import extract_h5_aggregate_xrf_data, create_aggregate_xrf_h5
from matplotlib import pyplot as plt
from tkinter import filedialog

def ramp_filter(sinogram):
    n_theta, n_rows, n_columns = sinogram.shape
    
    fft_sinogram = fft(sinogram, axis = 2) # Fourier transform along columns/horizontal scan dimension
    frequency_array = fftfreq(n_columns) # Create NORMALIZED (w.r.t. Nyquist frequency) frequency array

    ramp_filt = np.abs(frequency_array)

    filtered_sinogram = np.real(ifft(fft_sinogram*ramp_filt, axis = 2)) # Only want the real component, or else artifacts will show up

    return filtered_sinogram

def round_correct(num, ndec): # CORRECTLY round a number (num) to chosen number of decimal places (ndec)

    if ndec == 0:
        return int(num + 0.5)
    
    else:
        digit_value = 10**ndec
        
        if num > 0:
            return int(num*digit_value + 0.5)/digit_value
        
        else:
            return int(num*digit_value - 0.5)/digit_value

def iter_reproj(ref_element, element_array, theta_array, xrf_proj_img_array, n_iterations, eps = 1):
    n_elements = xrf_proj_img_array.shape[0] # Number of elements
    n_theta = xrf_proj_img_array.shape[1] # Number of projection angles (projection images)
    n_slices = xrf_proj_img_array.shape[2] # Number of rows in a projection image
    n_columns = xrf_proj_img_array.shape[3] # Number of columns in a projection image
    
    ref_element_idx = element_array.index(ref_element)
    reference_projection_imgs = xrf_proj_img_array[ref_element_idx] # These are effectively sinograms for the element of interest (highest contrast -> for realignment purposes)
                                                                    # (n_theta, n_slices -> n_rows, n_columns)

    center_of_rotation = tomo.find_center(reference_projection_imgs, theta_array*np.pi/180)

    iterations = []
    
    # filtered_projection_imgs = ramp_filter(reference_projection_imgs)
        
    # recon = tomo.recon(filtered_projection_imgs, theta = theta_array*np.pi/180, center = center_of_rotation, algorithm = 'fbp', sinogram_order = False)

    # current_proj_imgs = reference_projection_imgs

    sr_trans = psr.StackReg(psr.StackReg.TRANSLATION)

    recon = np.zeros((n_elements, n_slices, n_columns, n_columns))
    
    aligned_proj_from_3d_recon = np.zeros_like(xrf_proj_img_array)

    current_xrf_proj_img_array = xrf_proj_img_array.copy()
    proj_imgs_from_3d_recon = np.zeros_like(xrf_proj_img_array)

    x_shifts = np.zeros(n_theta)
    y_shifts = np.zeros(n_theta)
    prev_x_shifts = np.zeros(n_theta)
    prev_y_shifts = np.zeros(n_theta)

    max_x_shift = []
    max_y_shift = []
    
    for iteration_idx in range(n_iterations):
        print('Iteration ' + str(iteration_idx + 1) + '/' + str(n_iterations))

        iterations.append(iteration_idx + 1)
        
        # Perform FBP for each element and create 2D projection images using the same available angles

        for element_idx in range(current_xrf_proj_img_array.shape[0]):
            if element_idx == ref_element_idx:
                # center_of_rotation = tomo.find_center(current_xrf_proj_img_array[element_idx], theta_array*np.pi/180)
                
                # filtered_proj = ramp_filter(current_xrf_proj_img_array[element_idx])
                proj = current_xrf_proj_img_array[element_idx]

                # recon[element_idx] = tomo.recon(filtered_proj, theta = theta_array*np.pi/180, center = center_of_rotation, algorithm = 'fbp')
                recon[element_idx] = tomo.recon(proj, theta = theta_array*np.pi/180, center = center_of_rotation, algorithm = 'mlem', num_iter = 60)
                print(recon.shape)

                # plt.imshow(recon[element_idx, 0, :, :])
                # plt.show()

                # proj_imgs_from_3d_recon[element_idx] = tomo.project(recon[element_idx], theta = theta_array*np.pi/180, pad = False)
                for slice_idx in range(n_slices):
                    print('Slice ' + str(slice_idx + 1) + '/' + str(n_slices))
                    proj_slice = recon[element_idx, slice_idx, :, :]
                    proj_imgs_from_3d_recon[element_idx, :, slice_idx, :] = (skimage.transform.radon(proj_slice, theta = theta_array)).T # This radon transform assumes slices are defined by columns and not rows
                    # plt.imshow(proj_imgs_from_3d_recon[element_idx, :, slice_idx, :], aspect = 'auto')
                    # plt.show()
        # plt.imshow(proj_imgs_from_3d_recon[ref_element_idx, :, n_slices//2, :])
        # plt.show()

        for theta_idx in range(n_theta):
            y_shift, x_shift, corr_mat = cross_correlate(reference_projection_imgs[theta_idx], proj_imgs_from_3d_recon[ref_element_idx, theta_idx, :, :]) # Cross-correlation
            
            x_shifts[theta_idx] = x_shift
            y_shifts[theta_idx] = y_shift
            
            if (theta_idx % 7 == 0):
                print('x-shift: ' + str(x_shift) + ' (Theta = ' + str(theta_array[theta_idx]) + ' degrees')
                print('y-shift: ' + str(y_shift))

                if iteration_idx > 0:
                    fig1 = plt.figure(1)
                    plt.imshow(recon[ref_element_idx, 0, :, :])
                    plt.title('Slice 0')
                    fig2 = plt.figure(2)
                    plt.imshow(recon[ref_element_idx, n_slices//2, :, :])
                    plt.title('Slice ' + str(n_slices//2))
                    fig3 = plt.figure(3)
                    plt.imshow(reference_projection_imgs[theta_idx])
                    fig4 = plt.figure(4)
                    plt.imshow(proj_imgs_from_3d_recon[ref_element_idx, theta_idx, :, :])
                    plt.title('Synthesized Projection from Reconstruction')
                    fig5 = plt.figure(5)
                    plt.imshow(fftshift(corr_mat))
                    fig6 = plt.figure(6)
                    plt.imshow(current_xrf_proj_img_array[ref_element_idx, theta_idx, :, :])
                    plt.title('Aligned Projection from Previous Iteration')
                    plt.show()
        
        print('Maximum x-shift (magnitude): ' + str(np.max(np.abs(x_shifts))))
        print('Maximum y-shift (magnitude): ' + str(np.max(np.abs(y_shifts))))

        if np.max(np.abs(x_shifts - prev_x_shifts)) <= eps and np.max(np.abs(y_shifts - prev_y_shifts)) <= eps:
            print('Number of iterations taken: ' + str(iteration_idx + 1))
        
            break
        
        max_x_shift.append(np.max(np.abs(x_shifts)))
        max_y_shift.append(np.max(np.abs(y_shifts)))

        for element_idx in range(n_elements):
            if element_idx == ref_element_idx:
                for theta_idx in range(n_theta):
                    y_shift = y_shifts[theta_idx]
                    x_shift = x_shifts[theta_idx]
                    
                    aligned_proj_from_3d_recon[element_idx, theta_idx, :, :] = spndi.shift(proj_imgs_from_3d_recon[ref_element_idx, theta_idx, :, :], shift = (-y_shift, -x_shift), order = 5, cval = 0) # Undo the translational shifts by the cross-correlation peak

        current_xrf_proj_img_array = aligned_proj_from_3d_recon.copy()
        prev_x_shifts = x_shifts.copy()
        prev_y_shifts = y_shifts.copy()

        if iteration_idx == n_iterations - 1:
            fig7 = plt.figure(7)
            
            max_x_shift = np.array(max_x_shift)
            max_y_shift = np.array(max_y_shift)
            iterations = np.array(iterations)
            
            plt.plot(iterations, max_x_shift, 'k-o')
            plt.plot(iterations, max_y_shift, 'b-o')
            plt.show()
            
        # mse_exponent = np.floor(np.log10(mse))  # Calculate exponent
        # mse_mantissa = round_correct(mse / (10 ** mse_exponent), ndec = 3)
        
        # print('MSE = ' + str(mse_mantissa) + 'E' + str(int(mse_exponent)))

        # if mse <= eps:
        #     print('Number of iterations taken: ' + str(iteration_idx + 1))
            
        #     break

    #     for theta_idx in range(n_theta):
    #         tmat = sr_trans.register(ref = reference_projection_imgs[theta_idx], mov = proj_imgs_from_3d_recon[ref_element_idx, theta_idx, :, :]) # Transformation matrix for a particular angle relative to the experimental projection image for that angle

    #         for element_idx in range(n_elements):
    #             if element_idx == ref_element_idx:
    #                 aligned_proj_from_3d_recon[element_idx, theta_idx, :, :] = sr_trans.transform(mov = proj_imgs_from_3d_recon[element_idx, theta_idx, :, :], tmat = tmat)

    #     current_xrf_proj_img_array = aligned_proj_from_3d_recon.copy()
    #     print(current_xrf_proj_img_array.shape)
         
    # tmat = sr_trans.register(ref = reference_projection_imgs[n_theta//2, :, :], mov = current_xrf_proj_img_array[ref_element_idx, n_theta//2, :, :])

    # fig1 = plt.figure(1)
    # plt.imshow(current_xrf_proj_img_array[ref_element_idx, n_theta//2, :, :])
    # fig2 = plt.figure(2)
    # plt.imshow(reference_projection_imgs[n_theta//2, :, :])
    # fig3 = plt.figure(3)
    # # plt.show(tmat)
            

                

    # proj_3d = np.rot90(skimage.transform.radon(recon[recon.shape[0]//2, :, :], theta = theta_array), k = 1)
    # tomo.project()
    
    # mse = skimage.metrics.mean_squared_error(proj_3d, reference_projection_imgs[:, reference_projection_imgs.shape[1]//2, :])/np.size(proj_3d)

    
    # if mse > eps:
    #     for iteration in range(n_iterations):
    #         if mse > eps:
    #         # proj_3d = realign_translate(reference_projection_imgs[:, reference_projection_imgs.shape[1]//2, :], proj_3d)
    #             sr = psr.StackReg(psr.StackReg.TRANSLATION)

    #             proj_3d = sr.register_transform( proj_3d)

        # else:
            # break

    # recon = tomo.circ_mask(recon, axis = 0, ratio = 0.95)

    # fig1 = plt.figure(1)
    # plt.imshow(np.rot90(reference_projection_imgs[:, xrf_proj_img_array.shape[2]//2, :], k = 1), aspect = 'auto', extent = [0, reference_projection_imgs.shape[2], np.min(theta_array), np.max(theta_array)])
    # plt.imshow(np.rot90(reference_projection_imgs[:, xrf_proj_img_array.shape[2]//2, :], k = 1), aspect = 'auto', extent = [np.min(theta_array), np.max(theta_array), 0, reference_projection_imgs.shape[2]], cmap = 'Grays')
    # fig2 = plt.figure(2)
    # plt.imshow(filtered_projection_imgs[:, xrf_proj_img_array.shape[2]//2, :], extent = [0, reference_projection_imgs.shape[2], np.min(theta_array), np.max(theta_array)], aspect = 'auto')
    # fig3 = plt.figure(3)
    # plt.imshow(proj_3d, aspect = 'auto')
    # plt.show()

def cross_correlate(orig_proj, recon_proj):
    # Credit goes to Fabricio Marin and the XRFTomo GUI

    orig_proj_fft = fft2(orig_proj)
    recon_proj_fft = fft2(recon_proj)

    img_dims = orig_proj.shape
    
    y_dim = img_dims[0]
    x_dim = img_dims[1]

   
    cross_corr = np.abs(ifft2(orig_proj_fft*recon_proj_fft.conjugate()))
    
    y_shift, x_shift = np.unravel_index(np.argmax(cross_corr), img_dims) # Locate maximum peak in cross-correlation matrix and output the 
    
    if y_shift > y_dim//2:
        y_shift -= y_dim
    
    if x_shift > x_dim//2:
        x_shift -= x_dim

    return y_shift, x_shift, cross_corr

#     shift_y, shift_x = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)

#     if shift_y > array1.shape[0]//2:
#         shift_y -= array1.shape[0]
    
#     if shift_x > array1.shape[1]//2:
#         shift_x -= array1.shape[1]
    
#     return shift_x, shift_y

    
    
    # plt.imshow(sinogram[:, xrf_proj_img_array.shape[2]//15, :])
    

# root = tk.Tk()
    
# root.withdraw()

# file_path_array = filedialog.askopenfilenames(parent = root, title = "Upload HDF5 files", filetypes = [('HDF5 Files', '*.h5')])
# output_h5_file_path = filedialog.asksaveasfilename(parent = root, title = "Select save path", filetypes = (('HDF5 Files', '*.h5')))

# elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'

# create_aggregate_xrf_h5(file_path_array, file_path_xrf, synchrotron = 'aps')

# # file_path_xrt = ''

elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

iter_reproj('Fe', elements_xrf, theta_xrf, counts_xrf, 3)




