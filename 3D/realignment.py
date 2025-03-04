import numpy as np, h5py, os, skimage, tkinter as tk, tomopy as tomo, matplotlib as mpl, scipy.ndimage as spndi, pystackreg as psr

from skimage import transform as xform
from scipy import signal as sig
from numpy.fft import fft, ifft, fftfreq, fftn, ifftn, fft2, ifft2
from h5_util import extract_h5_aggregate_xrf_data, create_aggregate_xrf_h5
from matplotlib import pyplot as plt
from tkinter import filedialog

def ramp_filter(sinogram):
    n_theta, n_rows, n_columns = sinogram.shape
    
    fft_sinogram = fft(sinogram, axis = 2) # Fourier transform along columns/horizontal scan dimension
    frequency_array = fftfreq(n_columns) # Create

    ramp_filt = np.abs(frequency_array)

    filtered_sinogram = np.real(ifft(fft_sinogram*ramp_filt, axis = 2)) # Only want the real component, or else artifacts will show up

    return filtered_sinogram

def iter_reproj(ref_element, element_array, theta_array, xrf_proj_img_array, n_iterations, eps = 1e-8): 
    # theta_array *= np.pi/180
    
    ref_element_idx = element_array.index(ref_element)
    reference_projection_imgs = xrf_proj_img_array[ref_element_idx] # These are effectively sinograms for the element of interest
                                                                    # (n_theta, n_slices -> n_rows, n_columns)
    # tomo.normalize_roi(reference_projection_imgs)
    # reference_projection_imgs = tomo.minus_log(reference_projection_imgs)

    center_of_rotation = tomo.find_center(reference_projection_imgs, theta_array)
    filtered_projection_imgs = ramp_filter(reference_projection_imgs)
    recon = tomo.recon(filtered_projection_imgs, theta = theta_array, center = center_of_rotation, algorithm = 'fbp', sinogram_order = False)
    # recon = tomo.recon(reference_projection_imgs, theta = theta_array, center = center_of_rotation, algorithm = 'mlem', sinogram_order = False)
    print(xrf_proj_img_array.shape[2]//2)
    print(recon.shape)

    # proj_3d = np.rot90(skimage.transform.radon(recon[recon.shape[0]//2, :, :], theta = theta_array), k = 1)
    # tomo.project()
    
    mse = skimage.metrics.mean_squared_error(proj_3d, reference_projection_imgs[:, reference_projection_imgs.shape[1]//2, :])/np.size(proj_3d)

    
    if mse > eps:
        for iteration in range(n_iterations):
            if mse > eps:
            # proj_3d = realign_translate(reference_projection_imgs[:, reference_projection_imgs.shape[1]//2, :], proj_3d)
                sr = psr.StackReg(psr.StackReg.TRANSLATION)

                proj_3d = sr.register_transform( proj_3d)

        # else:
            # break

    # recon = tomo.circ_mask(recon, axis = 0, ratio = 0.95)

    # fig1 = plt.figure(1)
    # plt.imshow(np.rot90(reference_projection_imgs[:, xrf_proj_img_array.shape[2]//2, :], k = 1), aspect = 'auto', extent = [0, reference_projection_imgs.shape[2], np.min(theta_array), np.max(theta_array)])
    # plt.imshow(np.rot90(reference_projection_imgs[:, xrf_proj_img_array.shape[2]//2, :], k = 1), aspect = 'auto', extent = [np.min(theta_array), np.max(theta_array), 0, reference_projection_imgs.shape[2]], cmap = 'Grays')
    # fig2 = plt.figure(2)
    # plt.imshow(filtered_projection_imgs[:, xrf_proj_img_array.shape[2]//2, :], extent = [0, reference_projection_imgs.shape[2], np.min(theta_array), np.max(theta_array)], aspect = 'auto')
    # fig3 = plt.figure(3)
    plt.imshow(proj_3d, aspect = 'auto')
    plt.show()

# def cross_correlate(orig_proj, recon_proj):
#     orig_proj_fft = fft2(orig_proj)
#     recon_proj_fft = fft2(recon_proj)

#     cross_corr = (ifft2(orig_proj_fft*recon_proj_fft.conj())).real # Imaginary component will only yield artifacts

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

iter_reproj('Fe', elements_xrf, theta_xrf, counts_xrf, 1000)




