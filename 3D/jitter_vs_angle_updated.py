import numpy as np, os, sys

from matplotlib import pyplot as plt
from imageio import v2 as iio2
from numpy import fft
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

def find_theta_combos(theta_array_deg, dtheta):
    '''
    
    Make sure angles are in degrees!

    '''

    theta_array_idx_pairs = list(combos(np.arange(len(theta_array_deg)), 2)) # Generate a list of all pairs of theta_array indices

    valid_theta_idx_pairs = [(theta_idx_1, theta_idx_2) for theta_idx_1, theta_idx_2 in theta_array_idx_pairs 
                             if (180 - dtheta <= np.abs(theta_array_deg[theta_idx_1] - theta_array_deg[theta_idx_2]) <= 180 + dtheta)]
                            # Compound inequality syntax is acceptable in Python in certain cases

    return valid_theta_idx_pairs

def create_ref_pair_theta_idx_array(ref_pair_theta_array, theta_array):
    ref_pair_theta_idx_1 = np.where(theta_array == ref_pair_theta_array[0])[0][0]
    ref_pair_theta_idx_2 = np.where(theta_array == ref_pair_theta_array[1])[0][0]

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
    Nz = theta_sum.shape[0] # Number of slices
    Nt = theta_sum.shape[1] # Number of scan positions

    T = fft.rfft(theta_sum.ravel()) # Real FFT (no negative frequencies) of flattened 2D array of length Nt*Nz ('C'/row-major order)

    # Get real, imaginary components of the first AC spatial frequency for axis perpendicular to rotation axis.
    # Nt is the spatial period (there are Nt columns per row); Nz is the (fundamental) spatial frequency (thus, the first AC frequency)

    real = T[Nz].real
    imag = T[Nz].imag

    # Get phase of thetasum and return center of rotation.
    
    # In a sinogram the feature may be more positive or less positive than the background (i.e. fluorescence vs
    # absorption contrast). This can mess with the T_phase value so we multiply by the sign of the even function
    # to account for this. (Comment from F. Marin's XRFTomo code)

    phase = np.arctan2(imag*np.sign(real), real*np.sign(real)) 
    
    COR = Nt//2 - Nt*phase/(2*np.pi)

    return COR

def rot_center_avg(proj_img_array, theta_pair_array, theta_array):
    n_columns = proj_img_array.shape[2]
    
    center_of_rotation_sum = 0
    
    for theta_pair in theta_pair_array:
        theta_sum = proj_img_array[theta_pair[0]] + proj_img_array[theta_pair[1]]

        center_of_rotation = rot_center(theta_sum)

        # print(f'Center of rotation ({theta_array[theta_pair[0]]} degrees, {theta_array[theta_pair[1]]} degrees) = {round_correct(center_of_rotation, ndec = 3)}')

        center_of_rotation_sum += center_of_rotation
    
    center_rotation_avg = center_of_rotation_sum/len(theta_pair_array)

    geom_center_index = n_columns//2

    offset = center_rotation_avg - geom_center_index

    return center_rotation_avg, geom_center_index, offset

dir_path = 'xrt_gridrec_6_iter_initial_ps_cor_correction_norm_opt_dens_w_padding_08_28_2025'

aligned_proj_file = os.path.join(dir_path, 'aligned_proj_array_iter_ds_ic.npy')
synth_proj_file = os.path.join(dir_path, 'synth_proj_array_iter_ds_ic.npy')
dx_file = os.path.join(dir_path, 'dx_array_iter_ds_ic.npy')
theta_file = os.path.join(dir_path, 'theta_array.npy')

aligned_proj_array = np.load(aligned_proj_file)
synth_proj_array = np.load(synth_proj_file)
dx_iter_array = np.load(dx_file)
theta_array = np.load(theta_file)


n_columns = aligned_proj_array[0].shape[2]
n_theta = len(theta_array)

iter_idx_desired = 0
slice_idx_desired = 151

iteration_idx_array = np.arange(dx_iter_array.shape[0])

theta_idx_pairs = find_theta_combos(theta_array, dtheta = 1)

for iter_idx in iteration_idx_array:
    center_of_rotation_avg_exp, center_geom_exp, offset_exp = rot_center_avg(aligned_proj_array[iter_idx], theta_idx_pairs, theta_array)
    center_of_rotation_avg_synth, _, offset_synth = rot_center_avg(synth_proj_array[iter_idx], theta_idx_pairs, theta_array)

    print(f'Iteration {iter_idx} - COR (exp.): {center_of_rotation_avg_exp}; Offset: {offset_exp}')
    print(f'Iteration {iter_idx} - COR (synth.): {center_of_rotation_avg_synth}; Offset: {offset_synth}\n')

# sys.exit()

# plt.imshow(aligned_proj_array[0][0])
# plt.show()

fig1, axs1 = plt.subplots()
fig2, axs2 = plt.subplots(1, 2)
fig3, axs3 = plt.subplots()

theta_frames = []

theta_pair_idx_desired = 0

theta_idx_pair = theta_idx_pairs[theta_pair_idx_desired]

curve1, = axs1.plot(theta_array, dx_iter_array[iter_idx_desired], 'k-o', markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[iter_idx_desired]))
curve2, = axs1.plot(theta_array, dx_iter_array[iter_idx_desired + 2], 'b-o',  markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[iter_idx_desired + 2]))
curve3, = axs1.plot(theta_array, dx_iter_array[iter_idx_desired + 3], 'g-o', markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[iter_idx_desired + 3]))
curve4, = axs1.plot(theta_array, dx_iter_array[-1], 'r-o', markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[-1]))

axs1.tick_params(axis = 'both', which = 'major', labelsize = 14)
axs1.tick_params(axis = 'both', which = 'minor', labelsize = 14)
# axs1.set_title(r'Iteration index {0}'.format(iter_idx_desired), fontsize = 18)
axs1.set_xlabel(r'$\theta$ (\textdegree{})', fontsize = 16)
axs1.set_ylabel(r'$\delta x$', fontsize = 16)
axs1.legend(frameon = False, fontsize = 14)

curve5, = axs3.plot(np.arange(n_columns), aligned_proj_array[iter_idx_desired][theta_idx_pair[0], slice_idx_desired], 'k', linewidth = 2, label = r'$\theta = {0}^{{\circ}}$ (exp.)'.format(theta_array[theta_idx_pair[0]]))
curve6, = axs3.plot(np.arange(n_columns), aligned_proj_array[iter_idx_desired][theta_idx_pair[1], slice_idx_desired], 'k--', linewidth = 2, label = r'$\theta = {0}^{{\circ}}$'.format(theta_array[theta_idx_pair[1]]))
curve7, = axs3.plot(np.arange(n_columns), synth_proj_array[iter_idx_desired][theta_idx_pair[0], slice_idx_desired], 'r', linewidth = 2, label = r'$\theta = {0}^{{\circ}}$ (synth.)'.format(theta_array[theta_idx_pair[0]]))
curve8, = axs3.plot(np.arange(n_columns), synth_proj_array[iter_idx_desired][theta_idx_pair[1], slice_idx_desired], 'r--', linewidth = 2, label = r'$\theta = {0}^{{\circ}}$'.format(theta_array[theta_idx_pair[1]]))

global_min = np.min([np.min(aligned_proj_array[iter_idx_desired][theta_idx_pair[0], slice_idx_desired]), 
                     np.min(aligned_proj_array[iter_idx_desired][theta_idx_pair[1], slice_idx_desired]),
                     np.min(synth_proj_array[iter_idx_desired][theta_idx_pair[0], slice_idx_desired]),
                     np.min(synth_proj_array[iter_idx_desired][theta_idx_pair[1], slice_idx_desired])])

global_max = np.max([np.max(aligned_proj_array[iter_idx_desired][theta_idx_pair[0], slice_idx_desired]), 
                     np.max(aligned_proj_array[iter_idx_desired][theta_idx_pair[1], slice_idx_desired]),
                     np.max(synth_proj_array[iter_idx_desired][theta_idx_pair[0], slice_idx_desired]),
                     np.max(synth_proj_array[iter_idx_desired][theta_idx_pair[1], slice_idx_desired])])

axs3.set_xlim(0, n_columns - 1)
axs3.set_ylim(global_min, global_max)
axs3.tick_params(axis = 'both', which = 'major', labelsize = 14)
axs3.tick_params(axis = 'both', which = 'minor', labelsize = 14)
axs3.set_title(r'Iteration {0}'.format(iter_idx_desired), fontsize = 18)
axs3.set_xlabel(r'Scan position index', fontsize = 16)
axs3.set_ylabel(r'Optical density', fontsize = 16)
axs3.legend(frameon = False, fontsize = 14)

fig1.tight_layout()
fig3.tight_layout()

# nonzero_mask = aligned_proj_array[iter_idx_desired] > 0

# aligned_proj_array[iter_idx_desired][nonzero_mask] = -np.log(aligned_proj_array[iter_idx_desired][nonzero_mask]/counts_inc) 

vmin = np.min(aligned_proj_array[0])
vmax = np.max(aligned_proj_array[0])

im2_1 = axs2[0].imshow(aligned_proj_array[iter_idx_desired][0], vmin = vmin, vmax = vmax)
im2_2 = axs2[1].imshow(aligned_proj_array[iter_idx + 2][0], vmin = vmin, vmax = vmax)

text2 = axs2[0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs2[0].transAxes, color = 'white')

for axs in fig2.axes:
    axs.axvline(x = 300, color = 'red')
    axs.axis('off')

# axs2.set_title(r'Iteration index {0}'.format(iter_idx_desired))

# fig2.tight_layout()

# fps = 25

# for theta_idx, theta in enumerate(theta_array):
#     im2.set_data(aligned_proj_array[iter_idx_desired][theta_idx])
#     text2.set_text(r'$\theta = {0}$\textdegree'.format(theta))

#     # if theta_idx == 18:
#         # plt.show()

#     fig2.canvas.draw() # Rasterize and store Matplotlib figure contents in special buffer

#     frame = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3] # Rasterize the contents in the stored buffer, access 

#     theta_frames.append(frame)


# plt.close(fig2)

# iio2.mimsave(os.path.join(dir_path, f'cor_aligned_object_iter_idx_{iter_idx_desired}_opt_dens.gif'), theta_frames, duration = 1/fps)

plt.show()

# create_gif(filename2_array, os.path.join(dir_path, 'cor_aligned_object_iter_idx_0_opt_dens.gif'), fps = 25)

# gif_to_animated_svg_write(os.path.join(dir_path, 'cor_aligned_object_iter_idx_0_opt_dens.gif'), os.path.join(dir_path, 'cor_aligned_object_iter_idx_0_opt_dens.svg'), fps = 25)

