import numpy as np, tomopy as tomo, tkinter as tk, matplotlib as mpl

from skimage import transform as xform
from scipy import ndimage as ndi
from matplotlib import pyplot as plt, animation as anim
from tkinter import filedialog as fd
from h5_util import extract_h5_aggregate_xrf_data
from scipy import rfft

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def pad_col_row(array):
    array_new = np.zeros((array.shape[0], array.shape[1] + 1, array.shape[2] + 1))

    for theta_idx in range(array.shape[0]):
        final_column = array[theta_idx, :, -1].reshape(-1, 1) # Reshape to column vector (-1 means Python automatically determines missing dimension based on original orray length)
            
        # print(final_column.shape)
        # print(array.shape)

        array_temp = np.hstack((array[theta_idx, :, :], final_column))
            
        final_row = array_temp[-1, :]

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
        1D or 2D theta-sum array (z,t).

    Returns
    -------
    COR: float
        The center of rotation.
    """
    if theta_sum.ndim == 1:
        theta_sum = theta_sum[None, :]
    
    T = rfft(theta_sum, axis = 1)

    n_slices = theta_sum.shape[0]
    n_columns = theta_sum.shape[1]

    real, imag = T[:, 1].real, T[:, 1].imag
    
    # In a sinogram, a feature may be more positive or less positive than the background (i.e. X-ray fluorescence vs. X-ray absorption contrast).
    # This can mess with T_phase --> multiply real, imag. components by sign function
    # T = rfft(theta_sum.ravel())
    
    # # Get components of the AC spatial frequency for axis perpendicular to rotation axis.
    
    # imag = T[theta_sum.shape[0]].imag
    # real = T[theta_sum.shape[0]].real
    
    # Get phase of thetasum and return center of rotation.
    
    phase = (np.arctan2(imag*np.sign(real), real*np.sign(real)) )[n_slices//2]
    
    COR = n_columns*(1 - phase/np.pi)/2
    # COR = theta_sum.shape[-1]*(1 - phase/np.pi)/2

    return COR

file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
# output_path = '/home/bwr0835/cor_correction_proj_shift_array.npy'
ref_element = 'Fe'

elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

ref_element_idx = elements_xrf.index(ref_element)
counts = counts_xrf[ref_element_idx]

n_theta = counts.shape[0] # Number of projection angles (projection images)
n_slices = counts.shape[1] # Number of rows in a projection image
n_columns = counts.shape[2] # Number of columns in a projection image

# if (n_slices % 2) or (n_columns % 2):
#     if (n_slices % 2) and (n_columns % 2):
#         counts = pad_col_row(counts)
            
#         n_slices += 1
#         n_columns += 1
        
#     elif n_slices % 2:
#         counts = pad_row(counts)

#         n_slices += 1

#     else:
#         counts = pad_col(counts)

#         n_columns += 1

print(counts.shape)
# theta_sum = np.zeros((n_slices, n_columns))

# proj_list = [counts[theta_idx, :, :] for theta_idx in range(n_theta)]

# for proj in proj_list:
#     theta_sum += proj

reflection_pair_idx_array_1 = create_ref_pair_theta_idx_array(np.array([-22, 158]), theta_xrf)

# for slice_idx in range(n_slices):
    # theta_sum[slice_idx, :] = counts[reflection_pair_idx_array_1[0], slice_idx, :] + counts[reflection_pair_idx_array_1[1], slice_idx, :]

# theta_sum = (counts[reflection_pair_idx_array_1[0], :, :] + counts[reflection_pair_idx_array_1[1], :, :]).T
# theta_sum = np.tile(theta_sum, (n_slices, n_columns))
# theta_sum = counts[:, 0, :]

theta_sum = np.sum(counts, axis = 1)

center_of_rotation = rot_center(theta_sum)

# center_of_rotation = tomo.find_center(counts, theta_xrf*np.pi/180, tol = 0.1)

print(center_of_rotation)

# slice_desired_idx = 61

# fps_images = 25