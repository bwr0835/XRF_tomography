import numpy as np, h5py, os

from matplotlib import pyplot as plt
from imageio import v2 as iio2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

def eh5(file_path):
    with h5py.File(file_path, 'r') as f:
        exchange = f['exchange']
        
        elements = exchange['elements']
        data = exchange['data']

        elements_xrt = list(elements['xrt'].asstr()[:])
        elements_xrf = list(elements['xrf'].asstr()[:])

        data_xrt = data['xrt'][()]
        data_xrf = data['xrf'][()]
        
        theta = exchange['theta'][()]

    return elements_xrt, elements_xrf, data_xrt, data_xrf, theta

def create_sinogram_gif(elements_xrt, elements_xrf, data_xrt, data_xrf, fps):
    fig, axs = plt.subplots(1, 3)

    desired_xrf_element = 'Fe'

    vmin1 = data_xrt[0].min()
    vmax1 = data_xrt[0].max()

    vmin2 = data_xrt[1].min()
    vmax2 = data_xrt[1].max()

    vmin3 = data_xrf[elements_xrf.index(desired_xrf_element)].min()
    vmax3 = data_xrf[elements_xrf.index(desired_xrf_element)].max()

    im1 = axs[0].imshow(data_xrt[0, :, 0], origin = 'lower', vmin = vmin1, vmax = vmax1, aspect = 'auto', extent = [0, data_xrt.shape[-1] - 1, -180, 180])
    axs[0].set_title(r'{0}'.format(elements_xrt[0]), fontsize = 16)

    im2 = axs[1].imshow(data_xrt[0, :, 0], origin = 'lower', vmin = vmin2, vmax = vmax2, aspect = 'auto', extent = [0, data_xrt.shape[-1] - 1, -180, 180])
    axs[1].set_title(r'{0}'.format(elements_xrt[1]), fontsize = 16)

    im3 = axs[2].imshow(data_xrf[elements_xrf.index(desired_xrf_element), :, 0], origin = 'lower', vmin = vmin3, vmax = vmax3, aspect = 'auto', extent = [0, data_xrf.shape[-1] - 1, -180, 180])
    axs[2].set_title(r'{0}'.format(desired_xrf_element), fontsize = 16)

    for ax in axs:
        # ax.axis('off')
        ax.axvline(x = 300, color = 'red', linewidth = 2)
        ax.axhline(y = 0, color = 'red', linewidth = 2)
    
    text = axs[1].text(0.02, 0.02, r'Slice index 0/{0}'.format(data_xrf.shape[2] - 1), transform = axs[1].transAxes, color = 'white', fontsize = 14)
    
    fig.tight_layout()
    
    slice_frame = []

    for slice_idx in range(data_xrf.shape[2]):
        im1.set_data(data_xrt[0, :, slice_idx])
        im2.set_data(data_xrt[1, :, slice_idx])
        im3.set_data(data_xrf[elements_xrf.index(desired_xrf_element), :, slice_idx])
        
        text.set_text(r'Slice index {0}/{1}'.format(slice_idx, data_xrf.shape[2] - 1))
        
        fig.canvas.draw()
        
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
        slice_frame.append(frame)
    
    plt.close(fig)

    gif_file_path = os.path.join(dir_path, 'aligned_sinogram.gif')
    
    iio2.mimsave(gif_file_path, slice_frame, fps = fps)

    return

dir_path = '/Users/bwr0835/Documents/2_ide_realigned_data_cor_manual_07_25_2026'

input_file_path = os.path.join(dir_path, 'aligned_data/aligned_aggregate_xrf_xrt.h5')

elements_xrt, elements_xrf, data_xrt, data_xrf, theta = eh5(input_file_path)

create_sinogram_gif(elements_xrt, elements_xrf, data_xrt, data_xrf, fps = 10)

