import numpy as np, os, tifffile as tf

from matplotlib import pyplot as plt
from matplotlib import backends
from imageio import v2 as iio2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

#TODO Directly export 2D frames to TIFF files (WHILE BYPASSING MATPLOTLIB DUE TO POTENTIAL REMAPPING OF PIXELS); make sure all images are VIEWED with same minimum, maximum intensities

# def create_gif(tiff_filename_array, output_filepath, fps):
#     writer = iio.get_writer(output_filepath, mode = 'I', duration = 1/fps)
    
#     for filename in tiff_filename_array:
#         img = iio.imread(filename)

#         writer.append_data(img)
    
#     writer.close()

    # for filename in tiff_filename_array:
    #     os.remove(filename)

dir_path = '/Users/bwr0835/Documents/xrt_gridrec_6_iter_dynamic_ps_cor_correction_log_w_padding_gridrec_cor_idx_300_skimage_rado_aug_14_2025'

aligned_proj_file = '/Users/bwr0835/Documents/xrt_gridrec_6_iter_dynamic_ps_cor_correction_log_w_padding_gridrec_cor_idx_300_skimage_radon_aug_14_2025/aligned_proj_array_iter_ds_ic.npy'
dx_file = '/Users/bwr0835/Documents/xrt_gridrec_6_iter_dynamic_ps_cor_correction_log_w_padding_gridrec_cor_idx_300_skimage_radon_aug_14_2025/dx_array_iter_ds_ic.npy'
theta_file = '/Users/bwr0835/Documents/xrt_gridrec_6_iter_dynamic_ps_cor_correction_log_w_padding_gridrec_cor_idx_300_skimage_radon_aug_14_2025/theta_array.npy'

aligned_proj_array = np.load(aligned_proj_file)
dx_iter_array = np.load(dx_file)
theta_array = np.load(theta_file)

n_columns = aligned_proj_array[0].shape[2]
n_theta = len(theta_array)

iter_idx_desired = 0

iteration_idx_array = np.arange(dx_iter_array.shape[0])

# plt.imshow(aligned_proj_array[0][0])
# plt.show()

fig1, axs1 = plt.subplots()
fig2, axs2 = plt.subplots(dpi = 200)

theta_frames = []

curve1, = axs1.plot(theta_array, dx_iter_array[iter_idx_desired], 'k-o', markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[iter_idx_desired]))
curve2, = axs1.plot(theta_array, dx_iter_array[iter_idx_desired + 3], 'r-o', markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[iter_idx_desired + 3]))
curve3, = axs1.plot(theta_array, dx_iter_array[-1], 'b-o', markersize = 3, linewidth = 2, label = r'Iteration {0}'.format(iteration_idx_array[-1]))

axs1.tick_params(axis = 'both', which = 'major', labelsize = 14)
axs1.tick_params(axis = 'both', which = 'minor', labelsize = 14)
# axs1.set_title(r'Iteration index {0}'.format(iter_idx_desired), fontsize = 18)
axs1.set_xlabel(r'$\theta$ (\textdegree{})', fontsize = 16)
axs1.set_ylabel(r'$\delta x$', fontsize = 16)
axs1.legend(frameon = False, fontsize = 14)

fig1.tight_layout()

phi_inc = 8.67768e5
t_dwell_s = 0.01 

counts_inc = phi_inc*t_dwell_s

nonzero_mask = aligned_proj_array[iter_idx_desired] > 0

# aligned_proj_array[iter_idx_desired][nonzero_mask] = -np.log(aligned_proj_array[iter_idx_desired][nonzero_mask]/counts_inc) 

vmin = np.min(aligned_proj_array[0])
vmax = np.max(aligned_proj_array[0])

im2 = axs2.imshow(aligned_proj_array[iter_idx_desired][0], vmin = vmin, vmax = vmax)
text2 = axs2.text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs2.transAxes, color = 'white')

axs2.axvline(x = )
axs2.axis('off')
axs2.set_title(r'Iteration index {0}'.format(iter_idx_desired))

fig2.tight_layout()

for theta_idx, theta in enumerate(theta_array):
    im2.set_data(aligned_proj_array[iter_idx_desired][theta_idx])
    text2.set_text(r'$\theta = {0}$\textdegree'.format(theta))

    # if theta_idx == 18:
        # plt.show()

    fig2.canvas.draw() # Rasterize and store Matplotlib figure contents in special buffer

    frame = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3] # Rasterize the contents in the stored buffer, access 

    theta_frames.append(frame)


plt.close(fig2)

iio2.mimsave(os.path.join(dir_path, 'cor_aligned_object_iter_idx_0_opt_dens.gif'), theta_frames, duration = 1/25)

# create_gif(filename2_array, os.path.join(dir_path, 'cor_aligned_object_iter_idx_0_opt_dens.gif'), fps = 25)

# gif_to_animated_svg_write(os.path.join(dir_path, 'cor_aligned_object_iter_idx_0_opt_dens.gif'), os.path.join(dir_path, 'cor_aligned_object_iter_idx_0_opt_dens.svg'), fps = 25)

