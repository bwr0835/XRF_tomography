import numpy as np, tomopy as tomo, tkinter as tk, matplotlib as mpl

from skimage import transform as xform
from scipy import ndimage as ndi, fft
from matplotlib import pyplot as plt, animation as anim
from tkinter import filedialog as fd
from h5_util import extract_h5_aggregate_xrf_data

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
    
    imag = T[theta_sum.shape[0]].imag
    real = T[theta_sum.shape[0]].real
    
    # Get phase of thetasum and return center of rotation.
    
    phase = np.arctan2(imag*np.sign(real), real*np.sign(real)) 
    
    COR = theta_sum.shape[-1]*(1 - phase/np.pi)/2

    return COR

def create_ref_pair_theta_idx_array(ref_pair_theta_array, theta_array):
    ref_pair_theta_idx_1 = np.where(theta_array == ref_pair_theta_array[0])[0]
    ref_pair_theta_idx_2 = np.where(theta_array == ref_pair_theta_array[1])[0]

    return np.array([ref_pair_theta_idx_1, ref_pair_theta_idx_2])

def create_save_recon_shifts(elements_xrf, counts_xrf, theta_xrf, ref_element, cor_x_shift, output_path):
    ref_element_idx = elements_xrf.index(ref_element)
    counts = counts_xrf[ref_element_idx]

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

    center_of_rotation = tomo.find_center(counts, theta_xrf*np.pi/180, tol = 0.05) # COR given with tolerance of ±0.05 pixels

    recon_array = []

    for x_shift in range(len(cor_x_shift)):
        for theta_idx in range(n_theta):
            counts[theta_idx] = ndi.shift(counts[theta_idx], shift = (0, cor_x_shift[x_shift]))
    
        print('Performing gridrec for shift = ' + str(cor_x_shift[x_shift]))

        # recon = tomo.recon(counts, theta = theta_xrf*np.pi/180, center = center_of_rotation + cor_x_shift[x_shift], algorithm = 'gridrec', filter_name = 'ramlak')
        recon = tomo.recon(counts, theta = theta_xrf*np.pi/180, center = center_of_rotation, algorithm = 'gridrec', filter_name = 'ramlak')
        
        recon_array.append(recon)

    recon_array = np.array(recon_array)

    np.save(output_path, recon_array)

    return

def create_save_proj_shifts(elements_xrf, counts_xrf, theta_xrf, ref_element, cor_x_shift, output_path):
    ref_element_idx = elements_xrf.index(ref_element)
    counts = counts_xrf[ref_element_idx]

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

    counts_new = np.zeros_like(counts)
    theta_sum = np.zeros((n_slices, n_columns))

    recon_array = []

    reflection_pair_idx_array = create_ref_pair_theta_idx_array(np.array([-22, 158]), theta_xrf)

    for slice_idx in range(n_slices):
        sino = counts[:, slice_idx, :]

        slice_proj_angle_1 = sino[reflection_pair_idx_array[0], :]
        slice_proj_angle_2 = sino[reflection_pair_idx_array[1], :]

        theta_sum[slice_idx, :] = slice_proj_angle_1 + slice_proj_angle_2
        
    center_of_rotation = rot_center(theta_sum)
    # center_of_rotation = tomo.find_center(counts_new, theta_xrf*np.pi/180, tol = 0.05) # COR given with tolerance of ±0.05 pixels

    for x_shift in range(len(cor_x_shift)):    
        for theta_idx in range(n_theta):
            counts_new[theta_idx] = ndi.shift(counts[theta_idx], shift = (0, cor_x_shift[x_shift]))

        # center_of_rotation_new = center_of_rotation + cor_x_shift[x_shift]

        print('Performing gridrec for projection x-shift = ' + str(cor_x_shift[x_shift]))
        # print('Performing gridrec for projection x-shift = ' + str(cor_x_shift[x_shift]) + ' (COR = ' + str(center_of_rotation_new) + ')')

        recon = tomo.recon(counts_new, theta = theta_xrf*np.pi/180, center = center_of_rotation, algorithm = 'gridrec', filter_name = 'ramlak')
        
        if x_shift % 20 == 0:
            plt.clf()
            plt.imshow(recon[64])
            plt.show()

        recon_array.append(recon)

    recon_array = np.array(recon_array)

    np.save(output_path, recon_array)

    return

def update(frame):
    im.set_array(recon_array[frame][slice_desired_idx])
    text.set_text(r'COR shift = {0} pixels'.format(cor_x_shift[frame]))

    return im, text

file_path_xrf = '/home/bwr0835/2_ide_aggregate_xrf.h5'
output_path = '/home/bwr0835/vacek_alg_cor_correction_proj_shift_array.npy'
# output_path = '/home/bwr0835/cor_correction_cor_shift_array.npy'
ref_element = 'Fe'

cor_x_shift = np.linspace(-40, 40, 161)

elements_xrf, counts_xrf, theta_xrf, dataset_type_xrf = extract_h5_aggregate_xrf_data(file_path_xrf)

# create_save_recon_shifts(elements_xrf, counts_xrf, theta_xrf, 'Fe', cor_x_shift, output_path)
create_save_proj_shifts(elements_xrf, counts_xrf, theta_xrf, 'Fe', cor_x_shift, output_path)
print('Loading...')

recon_array = np.load(output_path)

print('Loading finished')

slice_desired_idx = 61

fps_images = 25 # Frames per second

fig, axs = plt.subplots()

axs.set_title(r'Slice {0}'.format(slice_desired_idx))

im = axs.imshow(recon_array[0][slice_desired_idx], animated = True)
text = axs.text(0.02, 0.02, r'COR shift = {0} pixels'.format(cor_x_shift[0]), transform = axs.transAxes, color = 'white')

animation = anim.FuncAnimation(fig, update, frames = len(cor_x_shift), interval = 1000/fps_images, blit = True) # Interval is ms/frame (NOT frames per second, or fps)

plt.show()

# output_path1 = '/home/bwr0835/vacek_alg_recon_gridrec_cor_correction_proj_shift.mp4'
# writer1 = anim.FFMpegWriter(fps = fps_images, metadata = {'title': 'recon'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264'])
    
# print('Saving')
    
# animation.save(output_path1, writer1, dpi = 600)

# plt.show()