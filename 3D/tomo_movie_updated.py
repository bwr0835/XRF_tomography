import numpy as np, tkinter as tk, os, sys

from tkinter import filedialog

from matplotlib import pyplot as plt, animation as anim

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def normalize_array(array):
    return (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))

# def update_proj_theta(frame):
#     im1_1.set_data(aligned_proj_theta_array_aux[frame])
#     im1_2.set_data(synth_proj_theta_array_aux[frame])
#     im1_3.set_data(rgb_proj_theta_array[frame])

#     text_1.set_text(r'$\theta = {0}$'.format(theta_array[frame]))

#     return im1_1, im1_2, im1_3, text_1

# def update_proj_iter(frame):
#     im2_1.set_data(aligned_proj_iter_array_aux[frame])
#     im2_2.set_data(synth_proj_iter_array_aux[frame])
#     im2_3.set_data(rgb_proj_iter_array[frame])

#     text_2.set_text(r'Iter. {0}'.format(frame))

#     return im2_1, im2_2, im2_3, text_2

# def update_recon_slice(frame):
#     im3.set_data(recon_slice_array_aux[frame])

#     text_3.set_text(r'Slice {0}'.format(frame))

#     return im3, text_3

# def update_recon_iter(frame):
#     im4.set_data(recon_iter_array_aux[frame])

#     text_4.set_text(r'Iter. {0}'.format(frame))

#     return im4, text_4

# def update_shifts(frame):
#     net_shift_x = net_x_shifts[:, frame]
#     net_shift_y = net_y_shifts[:, frame]

#     curve1.set_ydata(net_shift_x)
#     curve2.set_ydata(net_shift_y)

#     min_shift = np.min([np.min(net_shift_x), np.min(net_shift_y)])
#     max_shift = np.max([np.max(net_shift_x), np.max(net_shift_y)])

#     axs5.set_ylim(min_shift, max_shift + 0.1)
#     axs5.set_title(r'$\theta = {0}$\textdegree'.format(theta_array[frame]))

#     return curve1, curve2

# root = tk.Tk()

# root.withdraw()

# dir_path = filedialog.askdirectory(parent = root, title = 'Select directory containing alignment NPY files')

dir_path = '/raid/users/roter/iter_reproj/gridrec_10_iter'

if dir_path == "":
    print('No directory chosen. Exiting...')

    sys.exit()

print('Loading data...')

file_array = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))] # Get all contents of directory path and output only files

for f in file_array:
    if f == 'aligned_proj_all_elements.npy':
        aligned_proj = np.load(os.path.join(dir_path, f))
    
    elif 'aligned_proj_array_iter' in f and '.npy' in f:
        full_path = os.path.join(dir_path, f)

        aligned_proj_iter_array = np.load(full_path)

        # element_idx_desired = full_path.split('_')[-1].split('.')[0] # <directory path>_aligned_proj_array_iter_<desired element>_idx_<desired element index>.npy

    elif 'synth_proj_array_iter' in f and '.npy' in f:
        synth_proj_iter_array = np.load(os.path.join(dir_path, f))

    elif 'recon_array_iter' in f and '.npy' in f:
        recon_iter_array = np.load(os.path.join(dir_path, f))
    
    elif 'net_x_shifts' in f and '.npy' in f:
        net_x_shifts = np.load(os.path.join(dir_path, f))
    
    elif 'net_y_shifts' in f and '.npy' in f:
        net_y_shifts = np.load(os.path.join(dir_path, f))
    
    elif f == 'theta_array.npy':
        theta_array = np.load(os.path.join(dir_path, f))
    
    else:
        print('Error: One or more files not found. Exiting...')

        sys.exit()

n_theta = aligned_proj.shape[1]
n_slices = aligned_proj.shape[2]

n_iter = len(aligned_proj_iter_array)

aligned_proj_theta_array_aux = []
aligned_proj_iter_array_aux = []
synth_proj_theta_array_aux = []
synth_proj_iter_array_aux = []
rgb_proj_theta_array = []
rgb_proj_iter_array = []
recon_slice_array_aux = []
recon_iter_array_aux = []

theta_idx_desired = 0
iter_idx_desired = 0
slice_idx_desired = 64
element_idx_desired = 11 # Fe for this directory

for theta_idx in range(n_theta):
    aligned_proj_norm = normalize_array(aligned_proj_iter_array[iter_idx_desired][element_idx_desired, theta_idx, :, :])
    synth_proj_norm = normalize_array(synth_proj_iter_array[iter_idx_desired][theta_idx])
    rgb = np.dstack((aligned_proj_norm, np.zeros_like(aligned_proj_norm), synth_proj_norm))

    aligned_proj_theta_array_aux.append(aligned_proj_iter_array[iter_idx_desired][element_idx_desired, theta_idx, :, :])
    synth_proj_theta_array_aux.append(synth_proj_iter_array[iter_idx_desired][theta_idx])
    rgb_proj_theta_array.append(rgb)
    
for iter_idx in range(n_iter):
    aligned_proj_norm = normalize_array(aligned_proj_iter_array[iter_idx][element_idx_desired, theta_idx_desired])
    synth_proj_norm = normalize_array(synth_proj_iter_array[iter_idx][theta_idx_desired])

    rgb = np.dstack((aligned_proj_norm, np.zeros_like(aligned_proj_norm), synth_proj_norm))

    aligned_proj_iter_array_aux.append(aligned_proj_iter_array[iter_idx][element_idx_desired, theta_idx_desired])
    synth_proj_iter_array_aux.append(synth_proj_iter_array[iter_idx][theta_idx_desired])
    rgb_proj_iter_array.append(rgb)
    recon_iter_array_aux.append(recon_iter_array[iter_idx][:, slice_idx_desired, :])

for slice_idx in range(n_slices):
    recon_slice_array_aux.append(recon_iter_array[iter_idx_desired][:, slice_idx, :])

aligned_proj_theta_array_aux = np.array(aligned_proj_iter_array_aux)
aligned_proj_iter_array_aux = np.array(aligned_proj_iter_array_aux)
synth_proj_theta_array_aux = np.array(synth_proj_theta_array_aux)
synth_proj_iter_array_aux = np.array(synth_proj_iter_array_aux)
rgb_proj_theta_array = np.array(rgb_proj_theta_array)
rgb_proj_iter_array = np.array(rgb_proj_iter_array)
recon_slice_array_aux = np.array(recon_slice_array_aux)
recon_iter_array_aux = np.array(recon_iter_array_aux)

print('Generating figures')

iter_array = np.arange(n_iter)

fps_imgs = 25 # Frames per second (fps)
fps_plots = 15

plt.imshow(aligned_proj_iter_array[0][:, 64, :])
plt.show()

# fig1, axs1 = plt.subplots(1, 3) # Aligned experimental projection, synthetic experimental projection, overlays at different angles
# fig2, axs2 = plt.subplots(1, 3) # Same as above, but for different iterations - use first projection angle
# fig3, axs3 = plt.subplots() # Reconstructed object for different slices (use first iteration?)
# fig4, axs4 = plt.subplots() # Reconstructed object for different iteration (use slice index 68?)
# fig5, axs5 = plt.subplots()

# im1_1 = axs1[0].imshow(aligned_proj_iter_array_aux[0])
# im1_2 = axs1[1].imshow(synth_proj_iter_array_aux[0])
# im1_3 = axs1[2].imshow(rgb_proj_iter_array[0])

# im2_1 = axs2[0].imshow(aligned_proj_iter_array_aux[0])
# im2_2 = axs2[1].imshow(synth_proj_iter_array_aux[0])
# im2_3 = axs2[2].imshow(rgb_proj_theta_array[0])

# im3 = axs3.imshow(recon_slice_array_aux[0])

# im4 = axs4.imshow(recon_iter_array_aux[0])

# curve1, = axs3.plot(iter_array, net_x_shifts[:, 0], 'k-o', markersize = 3, label = r'$\Delta x$')
# curve2, = axs3.plot(iter_array, net_y_shifts[:, 0], 'r-o', markersize = 3, label = r'$\Delta y$')

# text_1 = axs1[0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[0].transAxes, color = 'white')
# text_2 = axs2[0].text(0.02, 0.02, r'Iter. 0', transform = axs2[0].transAxes, color = 'white')
# text_3 = axs3.text(0.02, 0.02, r'Slice 0', transform = axs3.transAxes, color = 'white')
# text_4 = axs4.text(0.02, 0.02, r'Iter. 0', transform = axs4.transAxes, color = 'white')

# axs1[0].set_title(r'Exp. Proj. (Iter. {0})'.format(iter_idx_desired), color = 'red')
# axs1[1].set_title(r'Synth. Proj.', color = 'blue')
# axs1[2].set_title(r'Overlay')

# axs2[0].set_title(r'Exp. Proj. ($\theta = {0}$\textdegree)'.format(theta_array[theta_idx_desired]), color = 'red')
# axs2[1].set_title(r'Synth. Proj.', color = 'blue')
# axs2[2].set_title(r'Overlay')

# axs3.set_title(r'Recon')

# axs5.set_title(r'\theta = {0}'.format(theta_array[0]))
# axs5.set_xlabel(r'Iteration index $i$')
# axs5.set_ylabel(r'Net shift')
# axs5.legend(frameon = False)
# axs5.set_xlim(0, n_iter - 1)

# axs4.set_title(r'Slice {0}'.format(slice_idx_desired))

# axs5.set_title('Iteration 0')

# anim1 = anim.FuncAnimation(fig1, update_proj_theta, frames = n_theta, interval = 1000/fps_imgs, blit = True)
# anim2 = anim.FuncAnimation(fig2, update_proj_iter, frames = n_iter, interval = 1000/fps_imgs, blit = True)
# anim3 = anim.FuncAnimation(fig3, update_recon_slice, frames = n_slices, interval = 1000/fps_imgs, blit = True)
# anim4 = anim.FuncAnimation(fig4, update_recon_iter, frames = n_iter, interval = 1000/fps_imgs, blit = True)
# anim5 = anim.FuncAnimation(fig5, update_shifts, frames = n_theta, interval = 1000/fps_imgs, blit = False)
plt.show()
# print('Exporting projections (changing thetas) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_imgs, metadata = {'title': 'proj_theta'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264'])

# anim1.save(os.path.join(dir_path, 'proj_theta'), writer, dpi = 400)

# print('Exporting projections (changing iterations) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_imgs, metadata = {'title': 'proj_iter'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264'])

# anim2.save(os.path.join(dir_path, 'proj_iter'), writer, dpi = 400)

# print('Exporting reconstructions (changing slices) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_imgs, metadata = {'title': 'recon_slice'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264'])

# anim3.save(os.path.join(dir_path, 'recon_slice'), writer, dpi = 400)

# print('Exporting reconstructions (changing iterations) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_imgs, metadata = {'title': 'recon_iter'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264'])

# anim4.save(os.path.join(dir_path, 'recon_iter'), writer, dpi = 400)

# print('Exporting net shifts (changing thetas) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_plots, metadata = {'title': 'recon_slice'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264'])

# anim5.save(os.path.join(dir_path, 'recon_slice'), writer, dpi = 400)

# print('Finished')

