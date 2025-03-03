import numpy as np, h5py, os, skimage, tkinter as tk, tomopy as tomo, matplotlib as mpl

from skimage import transform as xform
from scipy import signal as sig
from numpy.fft import fft, ifft, fftfreq
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

def iter_reproj(ref_element, element_array, theta_array, xrf_proj_img_array): 
    sinogram = np.zeros((xrf_proj_img_array.shape[1], xrf_proj_img_array.shape[2], xrf_proj_img_array.shape[3])) # (num_projections, num_rows, num_columns)
    # reconstructed_slice_array = []

    theta_array *= np.pi/180
    
    ref_element_idx = element_array.index(ref_element)
    reference_projection_imgs = xrf_proj_img_array[ref_element_idx]

    for slice_idx in range(xrf_proj_img_array.shape[2]):
        sinogram[:, slice_idx, :] = reference_projection_imgs[:, slice_idx, :]

    filtered_sinogram = ramp_filter(sinogram)

    

    fig1 = plt.figure(1)
    plt.imshow(sinogram[:, xrf_proj_img_array.shape[2]//2, :], aspect = 'auto')
    fig2 = plt.figure(2)
    plt.imshow(filtered_sinogram[:, xrf_proj_img_array.shape[2]//2, :], aspect = 'auto')
    plt.show()




    
    
    # plt.imshow(sinogram[:, xrf_proj_img_array.shape[2]//15, :])
    

root = tk.Tk()
    
root.withdraw()


# elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
# # file_path_xrt = ''

elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

iter_reproj('Fe', elements_xrf, theta_xrf, counts_xrf)




