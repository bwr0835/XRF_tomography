import numpy as np, tomopy as tomo, tkinter as tk, matplotlib as mpl, sys
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

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

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

# def rot_center(theta_sum):
def rot_center(theta_sum, slice_idx_desired): # Use only with F. Marin's code
    """
    Code written by E. Vacek (2021): 
    https://github.com/everettvacek/PhaseSymmetry/blob/master/PhaseSymmetry.py

    Calculates the center of rotation of a sinogram.

    Parameters
    ----------
    thetasum: array-like
        2D theta-sum array (Nz,Nt).

        Nz = number of slices
        Nt = number of scan positions

    Returns
    -------
    COR: float
        The center of rotation.
    """
    
    # Nz = theta_sum.shape[0]
    # Nt = theta_sum.shape[1]

    # T = fft.rfft(theta_sum.ravel()) # Real FFT (no negative frequencies) of flattened 2D array of length Nt*Nz ('C'/row-major order)

    # Get real, imaginary components of the first AC spatial frequency for axis perpendicular to rotation axis.
    # Nt is the spatial period (there are Nt columns per row); Nz is the (fundamental) spatial frequency (thus, the first AC frequency)

    # real = T[Nz].real
    # imag = T[Nz].imag

    # Get phase of thetasum and return center of rotation.
    
    # In a sinogram the feature may be more positive or less positive than the background (i.e. fluorescence vs
    # absorption contrast). This can mess with the T_phase value so we multiply by the sign of the even function
    # to account for this. (Comment from F. Marin's XRFTomo code)

    # phase = np.arctan2(imag*np.sign(real), real*np.sign(real)) 
    
    # COR = Nt*(1 - phase/np.pi)/2 - 1/2 # Extra -1/2 since Python starts indexing at zero

    # return COR

    # # The bottom chunk of code is Fabricio's XRFTomo code if looking for phase ramp contributions for individual slices (????)
    # # For total phase ramp (????), use second options when applicable (like for T and cor)
    if theta_sum.ndim == 1:
        theta_sum = theta_sum[None, :]
    
    # T = fft.fft(theta_sum, axis = 1)
    T = np.sum(fft.fft(theta_sum, axis = 1), axis = 0)
        
    # # Collect real and imaginary coefficients.
    
    # real, imag = T[:, 1].real, T[:, 1].imag
    real, imag = T[1].real, T[1].imag
    
    Nz = theta_sum.shape[0]
    Nt = theta_sum.shape[1]
   
    T_phase = np.arctan2(imag*np.sign(real), real*np.sign(real))
    print(Nt)
    # cor = Nt*(1 - T_phase[slice_idx_desired]/(np.pi))/2 - 1/2
    cor = Nt//2 - Nt*T_phase/(2*np.pi)

    # print(cor)

    return cor


file_path_xrt = '/Users/bwr0835/Documents/2_ide_aggregate_xrt.h5'
# output_path = '/home/bwr0835/cor_correction_proj_shift_array.npy'
ref_element = 'ds_ic'
# ref_element = 'Ca'

elements_xrt, counts_xrt, theta_xrt, dataset_type_xrt, _ = h5_util.extract_h5_aggregate_xrt_data(file_path_xrt)

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

phi_inc = 8.67768e5
t_dwell_s = 0.01 

counts_inc = phi_inc*t_dwell_s

counts[counts > 0] = -np.log(counts[counts > 0]/counts_inc)
# print(counts.shape)

cor_array = []

reflection_pair_idx_array = find_theta_combos(theta_xrt, dtheta = 1)

print(np.array(reflection_pair_idx_array))

color_array = ['r', 'orange', 'gold', 'g', 'c', 'b', 'indigo', 'darkviolet', 'm', 'saddlebrown', 'gray', 'k']

sino = counts

# fig1, axs1 = plt.subplots(2, 1, figsize = (11, 8))

slice_idx_desired = 151

# axs1[0].set_title(r'Slice {0}'.format(slice_idx_desired))
# axs1[0].set_xlabel(r'Scan position index')
# axs1[0].set_ylabel(r'$\theta$-sum')

# axs1[1].set_title(r'Total'.format(slice_idx_desired))
# axs1[1].set_xlabel(r'Scan position index')
# axs1[1].set_ylabel(r'$\theta$-sum')

# for theta_idx in range(len(reflection_pair_idx_array)):
#     slice_proj_neg_22 = sino[reflection_pair_idx_array[theta_idx][0]]
#     slice_proj_158 = (sino[reflection_pair_idx_array[theta_idx][1]])

#     theta_sum = slice_proj_158 + slice_proj_neg_22

#     center_of_rotation = rot_center(theta_sum, slice_idx_desired)

#     print(center_of_rotation)

#     cor_array.append(center_of_rotation)

#     plt.plot(np.arange(n_columns), np.sum(theta_sum, axis = 0))
    
#     axs1[0].plot(np.arange(n_columns), theta_sum[slice_idx_desired], color = color_array[theta_idx], label = r'({0}\textdegree, {1}\textdegree)'.format(theta_xrt[reflection_pair_idx_array[theta_idx][0]], theta_xrt[reflection_pair_idx_array[theta_idx][1]]))
#     axs1[1].plot(np.arange(n_columns), np.sum(theta_sum, axis = 0), color = color_array[theta_idx])


# plt.show()

# print(f'Mean COR = {cor_total/12}')



# print(f'Mean COR = {np.mean(np.array(cor_array))}')

# offset = np.mean(np.array(cor_array)) - (counts.shape[2]/2 - 1/2)


# offset = 6.745766492238204
# offset = 7.963059334690836
# offset = 7.929977403864523
# offset = 7.963059334690836
# offset = 7.975633256200336
# offset = 7.925353803699164
# offset = 7.944292884322675
# offset = 7.937159098256455
# offset = 7.939846200150093
# offset = 7.938834044186073
# offset = 7.939 + 1.2

# offset = 6.745766492238204 + 0.8

# print(offset)

# for theta_idx in range(n_theta):
#     counts[theta_idx] = ndi.shift(counts[theta_idx], shift = (0, -offset))

#     if theta_idx == 0:
#         plt.imshow(counts[0])
#         plt.show()
#         sys.exit()

cor_array = []

for theta_idx in range(len(reflection_pair_idx_array)):
    slice_proj_neg_22 = sino[reflection_pair_idx_array[theta_idx][0]]
    slice_proj_158 = (sino[reflection_pair_idx_array[theta_idx][1]])

    theta_sum = slice_proj_158 + slice_proj_neg_22

    center_of_rotation = rot_center(theta_sum, slice_idx_desired)
    # center_of_rotation = tomo.find_center_pc(slice_proj_neg_22, slice_proj_158, tol = 0.01)

    print(f'{center_of_rotation} ({theta_xrt[reflection_pair_idx_array[theta_idx][0]]}, {theta_xrt[reflection_pair_idx_array[theta_idx][1]]})')

    cor_array.append(center_of_rotation)

geom_center = 300

offset = np.mean(np.array(cor_array)) - geom_center

print(f'Mean COR = {np.mean(np.array(cor_array))}')
print(f'Offset = {offset}')

add = 0
# add = 0.1545368896 - 0.005924987

offset_crop = int(np.ceil(np.abs(-(offset + add))))

# counts_new = np.zeros((n_theta, n_slices, n_columns - int(round_correct(np.abs(np.mean(np.array(cor_array)) - geom_center), ndec = 0))))
counts_new = np.zeros((n_theta, n_slices, n_columns - offset_crop))
# counts_new = np.zeros_like(counts)
cts = counts.copy()

# print(int(np.ceil(np.abs(offset))))

for theta_idx in range(n_theta):
    counts[theta_idx] = ndi.shift(counts[theta_idx], shift = (0, -(offset + add)))

    # if theta_idx == 0:
    #     plt.imshow(counts[theta_idx])
    #     plt.show()
    #     sys.exit()

    # if theta_idx == 1:
    #     plt.plot(np.arange(cts.shape[-1]), cts[theta_idx, 151])
    #     plt.plot(np.arange(counts.shape[-1]), counts[theta_idx, 151])
    #     plt.show()
    # counts_new[theta_idx] = counts[theta_idx]
    # counts_new[theta_idx] = counts[theta_idx, :, :-int(round_correct(np.abs(np.mean(np.array(cor_array)) - geom_center), ndec = 0))]
    # counts_new[theta_idx] = counts[theta_idx, :, :-int(np.ceil(np.abs(np.mean(np.array(cor_array)) - geom_center)))]
    # counts_new[theta_idx] = counts[theta_idx, :, :-int(np.ceil(np.abs(offset)))]
counts_new[np.array(reflection_pair_idx_array).ravel()] = counts[np.array(reflection_pair_idx_array).ravel(), :, :(-offset_crop)]
# plt.imshow(counts[0])
# plt.show()

# for i in np.array(reflection_pair_idx_array).ravel():
#     print(i)

cor_array = []
# sino = counts

for theta_idx in range(len(reflection_pair_idx_array)):
    slice_proj_neg_22 = counts_new[reflection_pair_idx_array[theta_idx][0]]
    slice_proj_158 = (counts_new[reflection_pair_idx_array[theta_idx][1]])

    theta_sum = slice_proj_158 + slice_proj_neg_22

    # summed_proj = np.sum(counts_new[theta_idx], axis = 0)

    # x_coords = np.arange(len(summed_proj))
    
    # center_of_rotation = np.sum(x_coords*summed_proj)/np.sum(summed_proj)

    center_of_rotation = rot_center(theta_sum, slice_idx_desired)
    # center_of_rotation = tomo.find_center_pc(slice_proj_neg_22, slice_proj_158, tol = 0.01)

    print(f'{center_of_rotation} ({theta_xrt[reflection_pair_idx_array[theta_idx][0]]}, {theta_xrt[reflection_pair_idx_array[theta_idx][1]]})')

    cor_array.append(center_of_rotation)

offset = np.mean(np.array(cor_array)) - geom_center

print(f'Mean COR = {np.mean(np.array(cor_array))}')
print(f'Offset = {offset}')

