import numpy as np, h5py, pandas as pd, pandas as pd, xrf_xrt_preprocess_utils as ppu, sys, os, xrf_xrt_preprocess_file_util as futil
# import xrf_xrt_preprocess_file_util as futil_pp, xraylib_np as xrl_np, xraylib as xrl
# import  xrf_xrt_jxrft_file_util as futil_jxrft
from matplotlib import pyplot as plt
from itertools import combinations as combos
from scipy import ndimage as ndi
from imageio import v2 as iio2
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_xrf_intensity_norm_plots_hxn(intensity_xrf_norm, elements_xrf, theta_xrf, dir_path, fps):
    fig, axs = plt.subplots(2, 2)

    vmin_xrf_norm = intensity_xrf_norm.min()
    vmax_xrf_norm = intensity_xrf_norm.max()

    element_list = ['Ni', 'Cu', 'Zn', 'Ce_L']

    element_idx_array = [elements_xrf.index(element) for element in element_list]

    im1_1 = axs[0, 0].imshow(intensity_xrf_norm[element_idx_array[0], 0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)
    im1_2 = axs[0, 1].imshow(intensity_xrf_norm[element_idx_array[1], 0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)
    im1_3 = axs[1, 0].imshow(intensity_xrf_norm[element_idx_array[2], 0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)
    im1_4 = axs[1, 1].imshow(intensity_xrf_norm[element_idx_array[3], 0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)

    axs[0, 0].set_title(r'{0}'.format(element_list[0]))
    axs[0, 1].set_title(r'{0}'.format(element_list[1]))
    axs[1, 0].set_title(r'{0}'.format(element_list[2]))
    axs[1, 1].set_title(r'{0}'.format(element_list[3]))

    text = axs[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_xrf[0]), transform = axs[0, 0].transAxes, color = 'white')

    for ax in fig.axes:
        ax.axis('off')
    
    theta_frames = []

    for theta_idx in range(len(theta_xrf)):
        im1_1.set_data(intensity_xrf_norm[element_idx_array[0], theta_idx])
        im1_2.set_data(intensity_xrf_norm[element_idx_array[1], theta_idx])
        im1_3.set_data(intensity_xrf_norm[element_idx_array[2], theta_idx])
        im1_4.set_data(intensity_xrf_norm[element_idx_array[3], theta_idx])

        text.set_text(r'$\theta = {0}$\textdegree'.format(theta_xrf[theta_idx]))

        fig.canvas.draw()

        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        theta_frames.append(frame)

    plt.close(fig)

    filename = os.path.join(dir_path, 'xrf_intensity_norm_plots_hxn.gif')
    
    iio2.mimsave(filename, theta_frames, fps = fps)

    return  

def create_xrf_intensity_norm_plots_aps_hxn(intensity_xrf_aps, intensity_xrf_hxn, element_aps, element_hxn, element_array_aps, element_array_hxn, theta_aps, theta_hxn, theta_array_aps, theta_array_hxn):
    fig, axs = plt.subplots(1, 2)
    
    element_idx_aps = element_array_aps.index(element_aps)
    element_idx_hxn = element_array_hxn.index(element_hxn)

    theta_idx_aps = np.argmin(np.abs(theta_array_aps - theta_aps))
    theta_idx_hxn = np.argmin(np.abs(theta_array_hxn - theta_hxn))

    vmin_xrf_aps = intensity_xrf_aps[element_idx_aps, theta_idx_aps].min()
    vmax_xrf_aps = intensity_xrf_aps[element_idx_aps, theta_idx_aps].max()

    vmin_xrf_hxn = intensity_xrf_hxn[element_idx_hxn, theta_idx_hxn].min()
    vmax_xrf_hxn = intensity_xrf_hxn[element_idx_hxn, theta_idx_hxn].max()

    im1_1 = axs[0].imshow(intensity_xrf_aps[element_idx_aps, theta_idx_aps], interpolation = 'none')
    im1_2 = axs[1].imshow(intensity_xrf_hxn[element_idx_hxn, theta_idx_hxn], interpolation = 'none')

    axs[0].set_title(r'{0} (APS)'.format(element_aps), fontsize = 16)
    axs[1].set_title(r'{0} (NSLS-II)'.format(element_hxn), fontsize = 16)

    # cb_1 = fig.colorbar(im1_1, ax = axs[0], fraction = 0.047)
    # cb_2 = fig.colorbar(im1_2, ax = axs[1], fraction = 0.047)

    # cb_1.ax.set_title(r'Intensity (counts)', fontsize = 16)
    # cb_2.ax.set_title(r'Intensity (counts)', fontsize = 16)
    
    # cb_1.ax.tick_params(labelsize = 16)
    # cb_2.ax.tick_params(labelsize = 16)

    text_1 = axs[0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_aps), transform = axs[0].transAxes, color = 'white', fontsize = 14)
    text_2 = axs[1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_hxn), transform = axs[1].transAxes, color = 'white', fontsize = 14)

    for ax, im in zip(axs, [im1_1, im1_2]):    
        divider = make_axes_locatable(ax)    
        cax = divider.append_axes("right", size="5%", pad=0.05)    
        cb = fig.colorbar(im, cax=cax)
        cb.ax.set_title(r'photons/' + '\n' + r'pixel', fontsize = 16)
        cb.ax.tick_params(labelsize = 16)
        ax.axis('off')
        
        if ax == axs[0]:
            cb.ax.set_ylim(vmin_xrf_aps, vmax_xrf_aps)
            ax.text(0.02, 0.9, r'(a)', transform = ax.transAxes, color = 'white', fontsize = 14)
        
        else:
            cb.ax.set_ylim(vmin_xrf_hxn, vmax_xrf_hxn)
            ax.text(0.02, 0.9, r'(b)', transform = ax.transAxes, color = 'white', fontsize = 14)

    fig.tight_layout()
    plt.show()

    return

dir_path = '/Users/bwr0835/Documents'

aggregate_xrf_h5_file_path_aps = f'{dir_path}/2_ide_aggregate_xrf.h5'
aggregate_xrt_h5_file_path_aps = f'{dir_path}/2_ide_aggregate_xrt.h5'

aggregate_xrf_h5_file_path_hxn = f'{dir_path}/3_id_aggregate_xrf.h5'
aggregate_xrt_h5_file_path_hxn = f'{dir_path}/3_id_aggregate_xrt.h5'

theta_aps = 77
theta_hxn = 24

fps = 10

elements_xrf_aps, intensity_xrf_aps, theta_xrf_aps, incident_energy_keV_aps, _, dataset_type_aps = futil.extract_h5_aggregate_xrf_data(aggregate_xrf_h5_file_path_aps)
elements_xrt_aps, intensity_xrt_aps, theta_xrt_aps, _, _, dataset_type_aps, xrt_photon_counting_aps = futil.extract_h5_aggregate_xrt_data(aggregate_xrt_h5_file_path_aps)

elements_xrf_hxn, intensity_xrf_hxn, theta_xrf_hxn, incident_energy_keV_hxn, _, dataset_type_hxn = futil.extract_h5_aggregate_xrf_data(aggregate_xrf_h5_file_path_hxn)
elements_xrt_hxn, intensity_xrt_hxn, theta_xrt_hxn, _, _, dataset_type_hxn, xrt_photon_counting_hxn = futil.extract_h5_aggregate_xrt_data(aggregate_xrt_h5_file_path_hxn)

intensity_xrt_aps = ppu.pad_col_row(intensity_xrt_aps, 'xrt')
intensity_xrf_aps = ppu.pad_col_row(intensity_xrf_aps, 'xrf')

xrt_sig_hxn = intensity_xrt_hxn[elements_xrt_hxn.index('xrt_sig')]
xrt_sig_aps = intensity_xrt_aps[elements_xrt_aps.index('xrt_sig')]

intensity_xrt_norm_aps, intensity_xrf_norm_aps, norm_array_xrt_aps, norm_array_xrf_aps, I0_photons_aps, _ = ppu.joint_fluct_norm(xrt_sig_aps,
                                                                                                                              intensity_xrf_aps,
                                                                                                                              93,
                                                                                                                              xrt_photon_counting_aps,
                                                                                                                              8.6776e8,
                                                                                                                              0.01)

intensity_xrt_norm_hxn, intensity_xrf_norm_hxn, norm_array_xrt_hxn, norm_array_xrf_hxn, I0_photons_hxn, _ = ppu.joint_fluct_norm(xrt_sig_hxn,
                                                                                                                              intensity_xrf_hxn,
                                                                                                                              93,
                                                                                                                              xrt_photon_counting_hxn,
                                                                                                                              None,
                                                                                                                              None)

element_aps = 'Fe'
element_hxn = 'Ni'

fe = intensity_xrf_norm_aps[elements_xrf_aps.index(element_aps)]

idx_77 = np.argmin(np.abs(theta_xrf_aps - theta_aps))

# plt.imshow(fe[idx_77])
# plt.show()
# create_xrf_intensity_norm_plots_hxn(intensity_xrf_norm_hxn, elements_xrf_hxn, theta_hxn, dir_path, fps)
create_xrf_intensity_norm_plots_aps_hxn(intensity_xrf_norm_aps, intensity_xrf_norm_hxn, element_aps, element_hxn, elements_xrf_aps, elements_xrf_hxn, theta_aps, theta_hxn, theta_xrf_aps, theta_xrf_hxn)