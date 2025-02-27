import numpy as np, h5py, os, skimage, tkinter as tk

from skimage import transform as xform
from scipy import signal as sig
from h5_util import extract_h5_aggregate_xrf_data, create_aggregate_xrf_h5
from matplotlib import pyplot as plt
from tkinter import filedialog
        
def iter_reproj(ref_element, element_array, xrf_proj_img_array):
    sinogram = np.zeros((xrf_proj_img_array.shape[1], xrf_proj_img_array.shape[2], xrf_proj_img_array.shape[3]))
    reconstructed_slice_array = []

    ref_element_idx = element_array.index(ref_element)
    
    reference_projection_imgs = xrf_proj_img_array[ref_element_idx]

    for slice_idx in range(xrf_proj_img_array.shape[2]):
        sinogram[:, slice_idx, :] = reference_projection_imgs[:, slice_idx, :]
        
        reconstructed_slice = xform.iradon(sinogram[:, slice_idx, :], filter_name='ramp', circle=True)

        reconstructed_slice_array.append(reconstructed_slice)
    
    reconstructed_object = np.stack(reconstructed_slice_array, axis = 1)

    plt.imshow(reconstructed_object[:, 40, :])
    plt.show()




    
    
    # plt.imshow(sinogram[:, xrf_proj_img_array.shape[2]//15, :])
    
    

root = tk.Tk()
    
root.withdraw()

input_xrf_file_path_array = filedialog.askopenfilenames(parent = root, title = "Select XRF HDF5 files", filetypes = [("HDF5 files", "*.h5")])
# file_path_xrt = ''

create_aggregate_xrf_h5(input_xrf_file_path_array)

file_path_xrf = filedialog.askopenfilenames(parent = root, title = "Select aggregate XRF HDF5 file", filetypes = [("HDF5 files", "*.h5")])

elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

iter_reproj('Fe', elements_xrf, counts_xrf)




