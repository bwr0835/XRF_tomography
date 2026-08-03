import numpy as np, os, matplotlib as mpl, h5py, sys

from matplotlib import pyplot as plt, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from imageio import v2 as iio2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

def eh5(file_path):
    if not os.path.isfile(file_path):
        print('Error: Cannot locate post-reconstruction HDF5 file. Exiting program...', flush = True)

        sys.exit()
    
    if not file_path.endswith('.h5'):
        print('Error: Post-reconstruction file extension must be \'.h5\'. Exiting program...', flush = True)

        sys.exit()
    
    try:
        with h5py.File(file_path, 'r') as h5:
            sample = h5['sample']
        
            densities = sample['densities'][()]
            elements = list(sample['elements'].asstr()[:])
    
    except KeyboardInterrupt:
        print('Keyboard interrupt. Exiting program...')

        sys.exit()
    
    except:
        print('Error: Incorrect post-reconstruction HDF5 file structure. Exiting program...')

        sys.exit()

    return densities, elements

input_dir_path = '/Users/bwr0835/Documents/2_ide_realigned_data_cor_manual_07_25_2026'

recon_file_path = os.path.join(input_dir_path, 'gridrec_density_maps.h5')

densities, elements = eh5(recon_file_path)

n_elements = len(elements)

element_1 = 'Si'
element_2 = 'Ti'
element_3 = 'Fe'
element_4 = 'Ba'

dens_1 = densities[elements.index(element_1)]
dens_2 = densities[elements.index(element_2)]
dens_3 = densities[elements.index(element_3)]
dens_4 = densities[elements.index(element_4)]

dens_array = [dens_1, dens_2, dens_3, dens_4]
element_array = [element_1, element_2, element_3, element_4]

fig, axs = plt.subplots(2, 2)

vmin1 = 0
vmax1 = 1.75

# vmin2 = dens_2.min()
# vmax2 = dens_2.max()

vmin2 = 0
vmax2 = 3.6

vmin3 = 0
vmax3 = dens_3.max()

# vmin4 = dens_4.min()
# vmax4 = dens_4.max()

vmin4 = 0
vmax4 = 3.5

im1_1 = axs[0, 0].imshow(dens_1[0], vmin = vmin1, vmax = vmax1)
im1_2 = axs[0, 1].imshow(dens_2[0], vmin = vmin2, vmax = vmax2)
im1_3 = axs[1, 0].imshow(dens_3[0], vmin = vmin3, vmax = vmax3)
im1_4 = axs[1, 1].imshow(dens_4[0], vmin = vmin4, vmax = vmax4)

for idx, ax in enumerate(fig.axes):
    ax.set_title(r'{0}'.format(element_array[idx]))
    ax.axis('off')

# plt.show()

divider1 = make_axes_locatable(axs[0, 0])
divider2 = make_axes_locatable(axs[0, 1])
divider3 = make_axes_locatable(axs[1, 0])
divider4 = make_axes_locatable(axs[1, 1])

cax1 = divider1.append_axes('right', size = '5%', pad = 0.05)
cax2 = divider2.append_axes('right', size = '5%', pad = 0.05)
cax3 = divider3.append_axes('right', size = '5%', pad = 0.05)
cax4 = divider4.append_axes('right', size = '5%', pad = 0.05)

cbar1 = fig.colorbar(im1_1, cax = cax1, extend = 'both')
cbar2 = fig.colorbar(im1_2, cax = cax2, extend = 'both')
cbar3 = fig.colorbar(im1_3, cax = cax3, extend = 'both')
cbar4 = fig.colorbar(im1_4, cax = cax4, extend = 'both')

cbar1.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)
cbar2.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)
cbar3.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)
cbar4.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)

text1 = axs[0, 0].text(0.02, 0.02, r'Slice index 0/{0}'.format(dens_1.shape[0] - 1), transform = axs[0, 0].transAxes, color = 'white')

frames = []

fig.tight_layout()

for slice_idx in range(dens_1.shape[0]):
    im1_1.set_data(dens_1[slice_idx])
    im1_2.set_data(dens_2[slice_idx])
    im1_3.set_data(dens_3[slice_idx])
    im1_4.set_data(dens_4[slice_idx])
    
    text1.set_text(r'Slice index {0}/{1}'.format(slice_idx, dens_1.shape[0] - 1))
    
    fig.canvas.draw()

    frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

    frames.append(frame)
    # plt.show()

plt.close(fig)

iio2.mimsave(os.path.join(input_dir_path, 'recons.gif'), frames, fps = 10)
# # plt.close(fig)