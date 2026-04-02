import numpy as np, os, sys, xrf_xrt_preprocess_file_util as futil, xrf_xrt_preprocess_utils as ppu

from matplotlib import pyplot as plt
from scipy import ndimage as ndi, fft
from imageio import v2 as iio2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def create_final_aligned_proj_data_gif(dir_path,
                                       aligning_element,
                                       raw_proj,
                                       aligned_proj,
                                       theta_array,
                                       fps):
    
    n_theta, n_slices, n_columns = aligned_proj.shape
    n_slices_raw, n_columns_raw = raw_proj.shape[0], raw_proj.shape[2]
    
    vmin = aligned_proj.min()
    vmax = aligned_proj.max()

    fig1, axs1 = plt.subplots(1, 2)

    img1_1 = axs1[0].imshow(raw_proj[0], vmin = vmin, vmax = vmax)
    img1_2 = axs1[1].imshow(aligned_proj[0], vmin = vmin, vmax = vmax)

    for axs in fig1.axes:
        axs.axis('off')
        axs.axvline(x = n_columns_raw//2, color = 'red', linewidth = 2)
        axs.axhline(y = n_slices_raw//2, color = 'red', linewidth = 2)
    
    axs1[0].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    axs1[1].set_title(r'{0} (aligned)'.format(aligning_element), fontsize = 14)
    
    text1 = axs1[0].text(0.02, 0.02, r'$\theta ={0}$\textdegree'.format(theta_array[0]), transform = axs1[0].transAxes, color = 'white')

    theta_frames = []

    for theta_idx in range(n_theta):
        img1_1.set_data(raw_proj[theta_idx])
        img1_2.set_data(aligned_proj[theta_idx])
        text1.set_text(r'$\theta ={0}$\textdegree'.format(theta_array[theta_idx]))

        fig1.canvas.draw()

        frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]

        theta_frames.append(frame1)

    plt.close(fig1)

    gif_filename = os.path.join(dir_path, f'final_aligned_proj_data_comp_aligning_element_{aligning_element}.gif')

    iio2.mimsave(gif_filename, theta_frames, fps = fps)

    print('Creating final aligned projection sinogram GIF...')

    fig2, axs2 = plt.subplots(1, 2)

    im2_1 = axs2[0].imshow(raw_proj[:, 0], vmin = vmin, vmax = vmax, origin = 'lower', extent = [0, n_slices_raw - 1, -180, 180], aspect = 10)
    im2_2 = axs2[1].imshow(aligned_proj[:, 0], vmin = vmin, vmax = vmax, origin = 'lower', extent = [0, n_slices_raw - 1, -180, 180], aspect = 10)

    text2 = axs2[0].text(0.02, 0.02, r'Slice index 0/{0}'.format(n_slices_raw - 1), transform = axs2[0].transAxes, color = 'white')
    
    axs2[0].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    axs2[1].set_title(r'{0} (aligned)'.format(aligning_element), fontsize = 14)

    slice_frame_list = []

    for axs in fig2.axes:
        axs.set_xlabel(r'Pixel index', fontsize = 14)
        axs.set_ylabel(r'$\theta$ (\textdegree)', fontsize = 14)

    for slice_idx in range(n_slices_raw):
        im2_1.set_data(raw_proj[:, slice_idx])
        im2_2.set_data(aligned_proj[:, slice_idx])
        text2.set_text(r'Slice index {0}/{1}'.format(slice_idx, n_slices_raw - 1))

        fig2.canvas.draw()

        slice_frame = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3]

        slice_frame_list.append(slice_frame)

    plt.close(fig2)

    gif_filename = os.path.join(dir_path, f'final_aligned_sinogram_data_comp_aligning_element_{aligning_element}.gif')

    iio2.mimsave(gif_filename, slice_frame_list, fps = fps)

    return

dir_path = '/Users/bwr0835/Documents'

init_aps_xrt_file_path = f'{dir_path}/2_ide_aggregate_xrt.h5'
init_hxn_xrt_file_path = f'{dir_path}/3_id_aggregate_xrt.h5'

init_aps_xrf_file_path = f'{dir_path}/2_ide_aggregate_xrf.h5'
init_hxn_xrf_file_path = f'{dir_path}/3_id_aggregate_xrf.h5'





