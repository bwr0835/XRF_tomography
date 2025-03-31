import numpy as np, tomopy as tomo, tkinter as tk, matplotlib as mpl

from skimage import transform as xform
from scipy import ndimage as ndi
from matplotlib import pyplot as plt, animation as anim
from tkinter import filedialog as fd
from h5_util import extract_h5_aggregate_xrf_data

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def pad_col_row(array):
    array_new = np.zeros((array.shape[0], array.shape[1] + 1, array.shape[2] + 1))

    for theta_idx in range(array.shape[1]):
        final_column = array[theta_idx, :, -1] # Reshape to column vector (-1 means Python automatically determines missing dimension based on original orray length)
            
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

# def update(frame):
#     im.set_array(recon_array[:, frame, :])
#     im.set_text(r'')

file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
ref_element = 'Fe'

elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

ref_element_idx = elements_xrf.index(ref_element)
counts_fe = counts_xrf[ref_element_idx]

n_theta = counts_fe.shape[0] # Number of projection angles (projection images)
n_slices = counts_fe.shape[1] # Number of rows in a projection image
n_columns = counts_fe.shape[2] # Number of columns in a projection image

if (n_slices % 2) or (n_columns % 2):
    if (n_slices % 2) and (n_columns % 2):
        xrf_proj_img_array = pad_col_row(counts_fe)
            
        n_slices += 1
        n_columns += 1
        
    elif n_slices % 2:
        xrf_proj_img_array = pad_row(counts_fe)

        n_slices += 1

    else:
        xrf_proj_img_array = pad_col(counts_fe)

        n_columns += 1

cor_x_shift = np.linspace(-20, 20, 81)
center_of_rotation = tomo.find_center(counts_fe)

# print(center_of_rotation)

# recon_array = []

# for x_shift in range(len(cor_x_shift)):
#     for theta_idx in range(n_theta):
#         counts_fe[theta_idx] = ndi.shift(counts_fe[theta_idx], shift = (0, cor_x_shift[x_shift]))
    
#     print('Performing gridrec for shift = ' + str(cor_x_shift[x_shift]))

#     recon = tomo.recon(counts_fe, theta = theta_xrf*np.pi/180, center = center_of_rotation, algorithm = 'gridrec', filter_name = 'ramlak')

#     recon_array.append(recon)

# recon_array = np.array(recon_array)

# fig, axs = plt.subplot()

# slice_desired_idx = n_slices//2

# im = axs.imshow(recon_array[:, slice_desired_idx, :], animated = True)
# text = axs.text(0.02, 0.02, r'COR Shift = {0}'.format(cor_x_shift[0]), transform = axs.transAxes, color = 'white')

