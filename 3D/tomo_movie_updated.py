import numpy as np, xrf_xrt_preprocess_file_util as futil, xrf_xrt_preprocess_utils as ppu, os

from matplotlib import pyplot as plt
from itertools import combinations as combos
from scipy import ndimage as ndi
from numpy import fft
from imageio import v2 as iio2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

def normalize_array(array):
    return (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))

def create_multielement_proj_movie(dir_path, counts_xrf, desired_elements, element_array, theta_array, fps, vmin, vmax, figsize = None):

    desired_elements_idx = [element_array.index(element) for element in desired_elements]

    if figsize is not None:
        fig, axs = plt.subplots(2, 2, figsize = figsize)
    
    else:
        fig, axs = plt.subplots(2, 2)

    imgs = []

    for idx, ax in enumerate(fig.axes):
        img = ax.imshow(counts_xrf[desired_elements_idx[idx], 0], vmin = vmin[desired_elements_idx[idx]], vmax = vmax[desired_elements_idx[idx]])
        ax.axis('off')
        ax.set_title(r'{0}'.format(element_array[desired_elements_idx[idx]]))

        imgs.append(img)
    
    text = axs[1, 1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs[1, 1].transAxes, color = 'white')
    
    frames = []

    for theta_idx in range(counts_xrf.shape[1]):
        for idx, img in enumerate(imgs):
            img.set_data(counts_xrf[desired_elements_idx[idx], theta_idx])

        text.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

        fig.canvas.draw()
        
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
        frames.append(frame)
    
    gif_filename = os.path.join(dir_path, f'multielement_proj_movie.gif')  

    iio2.mimsave(gif_filename, frames, fps = fps)

    plt.close(fig)

    return

def create_multielement_recon_movie(dir_path, counts_xrf, desired_elements, element_array, theta_array, fps, vmin, vmax, figsize = None):

    desired_elements_idx = [element_array.index(element) for element in desired_elements]
    
    n_slices = counts_xrf.shape[0]
    
    if figsize is not None:
        fig, axs = plt.subplots(2, 2, figsize = figsize)
    
    else:
        fig, axs = plt.subplots(2, 2)

    imgs = []

    for idx, ax in enumerate(fig.axes):
        img = ax.imshow(counts_xrf[desired_elements_idx[idx], 0], vmin = vmin[desired_elements_idx[idx]], vmax = vmax[desired_elements_idx[idx]])
        ax.axis('off')
        ax.set_title(r'{0}'.format(element_array[desired_elements_idx[idx]]))

        imgs.append(img)
    
    text = axs[1, 1].text(0.02, 0.02, r'Slice index {0}/{1}'.format(theta_array[0], n_slices - 1), transform = axs[1, 1].transAxes, color = 'white')
    
    frames = []

    for theta_idx in range(counts_xrf.shape[1]):
        for idx, img in enumerate(imgs):
            img.set_data(counts_xrf[desired_elements_idx[idx], theta_idx])

        text.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

        fig.canvas.draw()
        
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
        frames.append(frame)
    
    gif_filename = os.path.join(dir_path, f'multielement_proj_movie.gif')  

    iio2.mimsave(gif_filename, frames, fps = fps)

    plt.close(fig)

    return

# aps_xrf_file_path = '/Users/bwr0835/Documents/2_ide_aggregate_xrf.h5'
# hxn_xrf_file_path = '/Users/bwr0835/Documents/3_id_aggregate_xrf.h5'

# aps_xrt_file_path = '/Users/bwr0835/Documents/2_ide_aggregate_xrt.h5'
# hxn_xrt_file_path = '/Users/bwr0835/Documents/3_id_aggregate_xrt.h5'

aps_dir_path = '/Users/bwr0835/Documents/2_ide_realigned_data_03_27_2026_iter_reproj_cor_correction_only_final'
# aps_dir_path = '/Users/bwr0835/Documents/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only'
# hxn_dir_path = '/Users/bwr0835/Documents/3_id_realigned_data_02_10_2026'

# aps_elements = ['Si', 'Ti','Fe', 'Ba_L']
# aps_elements = ['Si', 'Ti','Fe', 'Ba']
# hxn_elements = ['Ni', 'Cu','Zn', 'Ce_L']

# aps_elements_xrf, aps_counts_xrf, aps_theta_xrf, _, _ = futil.extract_h5_aggregate_xrf_data(aps_xrf_file_path)
# hxn_elements_xrf, hxn_counts_xrf, hxn_theta_xrf, _, _  = futil.extract_h5_aggregate_xrf_data(hxn_xrf_file_path)

# aps_elements_xrt, aps_counts_xrt, _, _, _ = futil.extract_h5_aggregate_xrt_data(aps_xrt_file_path)
# hxn_elements_xrt, hxn_counts_xrt, _, _, _ = futil.extract_h5_aggregate_xrt_data(hxn_xrt_file_path)

# aps_counts_xrt_sig = aps_counts_xrt[aps_elements_xrt.index('xrt_sig')]
# hxn_counts_xrt_sig = hxn_counts_xrt[hxn_elements_xrt.index('xrt_sig')]

# _, aps_counts_xrf_norm, _, _ = ppu.joint_fluct_norm(aps_counts_xrt_sig,
#                                                     aps_counts_xrf, 
#                                                     xrt_data_percentile = 80)

# _, hxn_counts_xrf_norm, _, _ = ppu.joint_fluct_norm(hxn_counts_xrt_sig,
#                                                     hxn_counts_xrf, 
#                                                     xrt_data_percentile = 80)

# vmin_xrf_aps = [np.quantile(aps_counts_xrf_norm[idx], [0.01, 0.99][0]) for idx in range(len(aps_elements_xrf))]
# vmax_xrf_aps = [np.quantile(aps_counts_xrf_norm[idx], [0.01, 0.99][1]) for idx in range(len(aps_elements_xrf))]
# vmin_xrf_hxn = [np.quantile(hxn_counts_xrf_norm[idx], [0.01, 0.99][0]) for idx in range(len(hxn_elements_xrf))]
# vmax_xrf_hxn = [np.quantile(hxn_counts_xrf_norm[idx], [0.01, 0.99][1]) for idx in range(len(hxn_elements_xrf))]

# vmin_xrf_aps = [aps_counts_xrf_norm[idx].min() for idx in range(len(aps_elements_xrf))]
# vmax_xrf_aps = [aps_counts_xrf_norm[idx].max() for idx in range(len(aps_elements_xrf))]
# vmin_xrf_hxn = [hxn_counts_xrf_norm[idx].min() for idx in range(len(hxn_elements_xrf))]
# vmax_xrf_hxn = [hxn_counts_xrf_norm[idx].max() for idx in range(len(hxn_elements_xrf))]

# create_multielement_proj_movie(aps_dir_path, aps_counts_xrf_norm, aps_elements, aps_elements_xrf, aps_theta_xrf, fps = 10, vmin = vmin_xrf_aps, vmax = vmax_xrf_aps)
# create_multielement_proj_movie(hxn_dir_path, hxn_counts_xrf_norm, hxn_elements, hxn_elements_xrf, hxn_theta_xrf, fps = 10, vmin = vmin_xrf_hxn, vmax = vmax_xrf_hxn)