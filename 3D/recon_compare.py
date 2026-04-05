import numpy as np, os, matplotlib as mpl

from matplotlib import pyplot as plt, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from xrf_xrt_jxrft_file_util import extract_h5_post_recon_data_non_mpi as eh5
from imageio import v2 as iio2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

input_dir_path = '/Users/bwr0835/Documents/2_ide_realigned_data_03_27_2026_iter_reproj_cor_correction_only_final/xrt_od_xrf_realignment'

recon_file_path = os.path.join(input_dir_path, 'gridrec_density_maps.h5')

densities, elements = eh5(recon_file_path)

n_elements = len(elements)

element_1 = 'Si'
element_2 = 'Fe'
element_3 = 'Ba'

dens_1 = densities[elements.index(element_1)]
dens_2 = densities[elements.index(element_2)]
dens_3 = densities[elements.index(element_3)]

dens_array = [dens_1, dens_2, dens_3]
element_array = [element_1, element_2, element_3]

fig, axs = plt.subplots(1, 3, figsize = (15, 4.5))


im1_1 = axs[0].imshow(dens_1[0], vmin = dens_1.min(), vmax = dens_1.max())
im1_2 = axs[1].imshow(dens_2[0], vmin = dens_2.min(), vmax = dens_2.max())
im1_3 = axs[2].imshow(dens_3[0], vmin = dens_3.min(), vmax = dens_3.max())


divider1 = make_axes_locatable(axs[0])
divider2 = make_axes_locatable(axs[1])
divider3 = make_axes_locatable(axs[2])

cax1 = divider1.append_axes('right', size = '5%', pad = 0.05)
cax2 = divider2.append_axes('right', size = '5%', pad = 0.05)
cax3 = divider3.append_axes('right', size = '5%', pad = 0.05)

cbar1 = fig.colorbar(im1_1, cax = cax1, extend = 'both')
cbar2 = fig.colorbar(im1_2, cax = cax2, extend = 'both')
cbar3 = fig.colorbar(im1_3, cax = cax3, extend = 'both')

cbar1.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)
cbar2.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)
cbar3.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)

text1 = axs[0].text(0.02, 0.02, r'Slice index 0/{0}'.format(dens_1.shape[0] - 1), transform = axs[0].transAxes, color = 'white')

for idx, ax in enumerate(axs):
    ax.set_title(r'{0}'.format(element_array[idx]))
    ax.axis('off')

frames = []

for slice_idx in range(dens_1.shape[0]):
    im1_1.set_data(dens_1[slice_idx])
    im1_2.set_data(dens_2[slice_idx])
    im1_3.set_data(dens_3[slice_idx])
    
    text1.set_text(r'Slice index {0}/{1}'.format(slice_idx, dens_1.shape[0] - 1))
    
    fig.canvas.draw()

    frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

    frames.append(frame)
    # plt.show()

plt.close(fig)

iio2.mimsave(os.path.join(input_dir_path, 'recons.gif'), frames, fps = 10)
# # plt.close(fig)