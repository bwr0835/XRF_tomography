import numpy as np, \
       xrf_xrt_jxrft_file_util as futil, \
       os

from matplotlib import pyplot as plt
from imageio import v2 as iio2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

def create_mmse_plot(mmse):
    fig, axs = plt.subplots()
    
    axs.semilogy(np.arange(len(mmse)) + 1, mmse, 'k')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('MMSE')
    axs.minorticks_on()

    return fig, axs

def create_recon_gif(dir_path, recon_array, desired_elements, element_array, fps):
    fig, axs = plt.subplots(3, 1)
    
    n_slices = recon_array.shape[0]

    desired_elements_idx = [element_array.index(element) for element in desired_elements]

    for idx, ax in enumerate(fig.axes):
        vmin = recon_array[desired_elements_idx[idx], :, 0].min()
        vmax = recon_array[desired_elements_idx[idx], :, 0].max()

        ax.imshow(recon_array[desired_elements_idx[idx], :, 0], vmin = vmin, vmax = vmax)
        ax.axis('off')
        ax.set_title(r'{0}'.format(element_array[desired_elements_idx[idx]]))

    text = axs[0].text(0.02, 0.02, r'Slice index 0/{0}'.format(n_slices - 1), transform = axs[0].transAxes, color = 'white')

    slice_frames = []
    
    for slice_idx in range(n_slices):
        for idx, ax in enumerate(fig.axes):
            ax.set_data(recon_array[desired_elements_idx[idx], :, slice_idx])

        text.set_text(r'Slice index {0}/{1}'.format(slice_idx, n_slices - 1))
        
        fig.canvas.draw()
        
        slice_frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
        slice_frames.append(slice_frame)
    
    gif_filename = os.path.join(dir_path, f'recon_movie.gif')
    
    iio2.mimsave(gif_filename, slice_frames, fps = fps)
    
    plt.close(fig)
    
    return fig, axs

file_name = '/home/bwr0835/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only/model_change_mse_epoch.csv'
recon_file_name = '/home/bwr0835/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only/grid_concentration.h5'

mmse_array = np.loadtxt(file_name, delimiter = ',')
densities, elements = futil.extract_h5_post_recon_data_non_mpi(recon_file_name)



plt.plot(mmse_array[:, 1])
plt.xlabel('Epoch')
plt.ylabel('MMSE')

plt.show()