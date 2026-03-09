import numpy as np, \
       xrf_xrt_jxrft_file_util as futil, \
       xrf_xrt_preprocess_file_util as futil2, \
       os

from matplotlib import pyplot as plt
from imageio import v2 as iio2
from scipy import ndimage as ndi

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

def warp_rowwise_dx(img, dx_vec, dy, cval = 0):
    """
    Warp image: shift each row horizontally by dx_vec[i], and shift all rows vertically by dy.
    dx_vec: length ny, horizontal shift per row.
    dy: scalar, same vertical shift for all rows.
    """
    ny, nx = img.shape
    dy = np.asarray(dy).flat[0]  # ensure scalar

    # rows_new: same row offset for all columns; cols_new: per-row horizontal offset
    rows_new = (np.arange(ny, dtype=float) - dy)[:, None] + np.zeros((1, nx))  # (ny, nx)
    cols_new = np.arange(nx, dtype=float)[None, :] - dx_vec[:, None]           # (ny, nx)

    coords = np.stack([rows_new, cols_new], axis=0)  # shape: (2, ny, nx)

    return ndi.map_coordinates(img, coords, cval = cval)

def create_mmse_plot(mmse_arrays):
    fig, axs = plt.subplots()
    
    colors = ['k', 'r', 'b', 'g', 'y', 'c', 'm']
    lambdas = [0.01, 1, 100, 10000]

    for idx, mmse_array in enumerate(mmse_arrays):
        axs.semilogy(np.arange(len(mmse_array)) + 1, mmse_array, colors[idx], label = r'$\lambda = {0}$'.format(lambdas[idx]))
        axs.set_xlabel('Epoch')
        axs.set_ylabel('MMSE')
        axs.minorticks_on()
    
    axs.set_xlim(0, len(mmse_arrays[-1]))
    axs.legend(frameon = False)
    
    return fig, axs

def create_recon_gif(dir_path, recon_array, desired_elements, element_array, fps):
    """
    Create a GIF of reconstruction slices. recon_array shape: (n_element, n_slices, n_y, n_x).
    """
    n_elements = len(desired_elements)
    fig, axs = plt.subplots(n_elements, 1)
    if n_elements == 1:
        axs = np.array([axs])
    
    # Densities layout: (n_element, sample_height_n, sample_size_n, sample_size_n)
    # Slice dimension is axis 1 (sample_height_n)
    n_slices = recon_array.shape[1]

    desired_elements_idx = [element_array.index(element) for element in desired_elements]
    imgs = []
    
    for idx in range(n_elements):
        ax = axs[idx]
        elem_idx = desired_elements_idx[idx]
        vmin = recon_array[elem_idx].min()
        vmax = recon_array[elem_idx].max()

        img = ax.imshow(recon_array[elem_idx, 0, :, :], vmin = vmin, vmax = vmax)
        ax.axis('off')
        ax.set_title(r'{0}'.format(element_array[elem_idx]))
        
        imgs.append(img)

    text = axs[0].text(0.02, 0.02, r'Slice index 0/{0}'.format(n_slices - 1), transform = axs[0].transAxes, color = 'white')

    frames = []
    
    for slice_idx in range(n_slices):
        for idx, img in enumerate(imgs):
            img.set_data(recon_array[desired_elements_idx[idx], slice_idx, :, :])

        text.set_text(r'Slice index {0}/{1}'.format(slice_idx, n_slices - 1))
        
        fig.canvas.draw()
        
        frame = np.ascontiguousarray(np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3].astype(np.uint8))
        
        frames.append(frame)
    
    gif_filename = os.path.join(dir_path, f'recon_movie.gif')
    
    # Use duration (ms per frame) for imageio compatibility; fps=10 -> 100ms per frame
    
    iio2.mimsave(gif_filename, frames, fps = fps)
    
    plt.close(fig)
    
    return

# file_name1 = '/Users/bwr0835/Documents/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_100/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_100/model_change_mse_epoch.csv'
# file_name1 = '/home/bwr0835/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_0_01/model_change_mse_epoch.csv'
# file_name2 = '/home/bwr0835/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_1/model_change_mse_epoch.csv'
# file_name3 = '/home/bwr0835/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_100/model_change_mse_epoch.csv'
# file_name1 = '/Users/bwr0835/Documents/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_0_01/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_0_01/model_change_mse_epoch.csv'
# file_name2 = '/Users/bwr0835/Documents/2_ide_realigned_data_02_12_2026_iter_reproj_cor_only_reg_0_01_mmse_selfab.csv/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_0_01_selfab/model_change_mse_epoch.csv'


# recon_file_name = '/Users/bwr0835/Documents/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only/grid_concentration.h5'
recon_file_name = '/Users/bwr0835/Documents/2_ide_aggregate_xrf.h5'
# file_name = '/Users/bwr0835/Documents/2_ide_realigned_data_02_12_2026_iter_reproj_cor_only_reg_1_mmse.csv'
# file_name2 = '/Users/bwr0835/Documents/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_100/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only_reg_100/model_change_mse_epoch.csv'

dir_path = '/Users/bwr0835/Documents/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only'

# mmse_array1 = np.loadtxt(file_name1, delimiter = ',')
# mmse_array2 = np.loadtxt(file_name2, delimiter = ',')
# mmse_array3 = np.loadtxt(file_name3, delimiter = ',')
# mmse_array4 = np.loadtxt(file_name4, delimiter = ',')

# mmse_arrays = [mmse_array1[:, 1], mmse_array2[:, 1], mmse_array3[:, 1], mmse_array4[:, 1]]
# mmse_arrays = [mmse_array1[:, 1], mmse_array2[:, 1]]
# densities, elements = futil.extract_h5_post_recon_data_non_mpi(recon_file_name)

elements, densities, theta, _, _ = futil2.extract_h5_aggregate_xrf_data(recon_file_name)

# a = np.random.randint(20, size = densities.shape[2])
a = np.zeros(densities.shape[2])
a[densities.shape[2]//2] = 20
# b = np.full(densities.shape[2], 25, dtype = float)
b = 0

densities[list(elements).index('Fe'), 26] = warp_rowwise_dx(densities[list(elements).index('Fe'), 26], a, b)

plt.imshow(densities[elements.index('Fe'), 26], vmin = densities[elements.index('Fe')].min(), vmax = densities[elements.index('Fe')].max())
plt.show()
# desired_elements = ['Si', 'Fe', 'Ba']

# create_recon_gif(dir_path, densities, desired_elements, list(elements), fps = 10)

# fig, axs = create_mmse_plot(mmse_arrays)

# plt.show()