import numpy as np, tomopy as tomo, tkinter as tk, matplotlib as mpl
import h5_util

from skimage import transform as xform
from scipy import ndimage as ndi
from matplotlib import pyplot as plt, animation as anim
from tkinter import filedialog as fd
# from h5_util import extract_h5_aggregate_xrt_data
from scipy import fft
from itertools import combinations as combos

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

def find_theta_combos(theta_array_deg, dtheta):
    '''
    
    Make sure angles are in degrees!

    '''

    theta_array_idx_pairs = list(combos(np.arange(len(theta_array_deg)), 2)) # Generate a list of all pairs of theta_array indices

    valid_theta_idx_pairs = [(theta_idx_1, theta_idx_2) for theta_idx_1, theta_idx_2 in theta_array_idx_pairs 
                             if (180 - dtheta <= np.abs(theta_array_deg[theta_idx_1] - theta_array_deg[theta_idx_2]) <= 180 + dtheta)]
                            # Compound inequality syntax is acceptable in Python in certain cases

    return valid_theta_idx_pairs

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

    T = fft.rfft(theta_sum.ravel())
    
    # Get components of the AC spatial frequency for axis perpendicular to rotation axis.
    
    imag = T[theta_sum.shape[1]].imag
    real = T[theta_sum.shape[1]].real
    
    # Get phase of thetasum and return center of rotation.
    
    phase = np.arctan2(imag*np.sign(real), real*np.sign(real)) 
    
    COR = theta_sum.shape[-1]*(1 - phase/np.pi)/2

    return COR

file_path_xrt = '/Users/bwr0835/Documents/2_ide_aggregate_xrt.h5'
# output_path = '/home/bwr0835/cor_correction_proj_shift_array.npy'
ref_element = 'ds_ic'

elements_xrt, counts_xrt, theta_xrt, dataset_type_xrt = h5_util.extract_h5_aggregate_xrt_data(file_path_xrt)

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

print(counts.shape)
# for theta_idx in range(n_theta):
    # counts[theta_idx] = ndi.shift(counts[theta_idx], shift = (0, -22.6328223508))


cor_array = []
# theta_sum = np.zeros((n_slices, n_columns))

# proj_list = [counts[theta_idx, :, :] for theta_idx in range(n_theta)]

# for proj in proj_list:
    # theta_sum += proj

# angle_pair = np.array([-22, 158])
# angle_pair = np.array([-22, 158])

reflection_pair_idx_array = find_theta_combos(theta_xrt, dtheta = 1)
# print(theta_xrt[reflection_pair_idx_array])

sino = counts

for theta_idx in range(len(reflection_pair_idx_array)):
# 
    # slice_proj_neg_22 = np.flip(sino, axis = 1)[reflection_pair_idx_array[theta_idx][1]]
    slice_proj_neg_22 = sino[reflection_pair_idx_array[theta_idx][0]]
    slice_proj_158 = np.flip(sino, axis = 1)[reflection_pair_idx_array[theta_idx][1]]
    # slice_proj_158 = (sino[reflection_pair_idx_array[theta_idx][0]])

    theta_sum = slice_proj_158 + slice_proj_neg_22

    center_of_rotation = rot_center(theta_sum)

    cor_array.append(center_of_rotation)

    print(center_of_rotation)

print(f'Mean COR = {np.mean(np.array(cor_array))}')

# for slice_idx in range(n_slices):
    # theta_sum[slice_idx, :] = counts[reflection_pair_idx_array_1[0], slice_idx, :] + counts[reflection_pair_idx_array_1[1], slice_idx, :]

# for slice_idx in range(n_slices):

#     sino = counts[:, slice_idx, :]

#     slice_proj_neg_22 = sino[reflection_pair_idx_array_1[0], :]
#     # slice_proj_158 = np.flip(sino[reflection_pair_idx_array_1[0], :], axis = 1)
#     slice_proj_158 = (sino[reflection_pair_idx_array_1[1], :])

    # theta_sum[slice_idx, :] = slice_proj_neg_22 + slice_proj_158

# theta_sum = proj_neg_22 + proj_158

# theta_sum = (counts[reflection_pair_idx_array_1[0], :, :] + counts[reflection_pair_idx_array_1[1], :, :]).T
# theta_sum = np.tile(theta_sum, (n_slices, n_columns))
# theta_sum = counts[:, 0, :]

# theta_sum = np.sum(counts, axis = 0)

# center_of_rotation = rot_center(theta_sum)

# center_of_rotation = tomo.find_center(counts, theta_xrf*np.pi/180, tol = 0.1)

# print(center_of_rotation)

# slice_desired_idx = 61

# fps_images = 25