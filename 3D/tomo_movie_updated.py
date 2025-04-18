import numpy as np, tkinter as tk, os, sys, imageio as iio

from tkinter import filedialog

from matplotlib import pyplot as plt, animation as anim

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def normalize_array(array):
    return (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))

def create_gif(tiff_filename_array, output_filepath, fps):
    writer = iio.get_writer(output_filepath, mode = 'I', duration = 1/fps)
    
    for filename in tiff_filename_array:
        img = iio.imread(filename)

        writer.append_data(img)
    
    writer.close()

    for filename in tiff_filename_array:
        os.remove(filename)
        
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

#     # return im2_1, im2_2, im2_3, text_2

# def update_recon_slice(frame):
#     im3.set_data(recon_slice_array_aux[frame])

#     text_3.set_text(r'Slice {0}'.format(frame))

#     # return im3, text_3

# def update_recon_iter(frame):
#     im4.set_data(recon_iter_array_aux[frame])

#     text_4.set_text(r'Iter. {0}'.format(frame))

#     # return im4, text_4

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

dir_path = '/home/bwr0835/iter_reproj/gridrec_5_iter_tomopy_cor_alg_no_cor_correction_padding_04_17_2025'

if dir_path == "":
    print('No directory chosen. Exiting...')

    sys.exit()

print('Loading data...')

file_array = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))] # Get all contents of directory path and output only files

for f in file_array:
    if f == 'aligned_proj_all_elements.npy':
        aligned_proj = np.load(os.path.join(dir_path, f))
    
    elif 'aligned_proj_array_iter' in f and f.endswith('.npy'):
        aligned_proj_iter_array = np.load(os.path.join(dir_path, f))

    elif 'synth_proj_array_iter' in f and f.endswith('.npy'):
        synth_proj_iter_array = np.load(os.path.join(dir_path, f))

    elif 'recon_array_iter' in f and f.endswith('npy'):
        recon_iter_array = np.load(os.path.join(dir_path, f))
    
    elif 'net_x_shifts' in f and f.endswith('.npy'):
        net_x_shifts = np.load(os.path.join(dir_path, f))
    
    elif 'net_y_shifts' in f and f.endswith('.npy'):
        net_y_shifts = np.load(os.path.join(dir_path, f))
    
    elif f == 'theta_array.npy':
        theta_array = np.load(os.path.join(dir_path, f))
    
    elif f == 'cor_shifts.npy':
        cor_shifts = np.load(os.path.join(dir_path, f))

    elif f.endswith('.mp4') or f.endswith('.gif'):
        continue
    
    elif f.endswith('.tiff'):
        os.remove(os.path.join(dir_path, f))

    else:
        print('Error: Unable to load one or more files. Exiting...')

        sys.exit()

n_theta = aligned_proj.shape[1]
n_slices = aligned_proj.shape[2]

n_iter = len(aligned_proj_iter_array)

aligned_proj_theta_array_aux = []
aligned_proj_theta_array_aux_2 = []
aligned_proj_iter_array_aux = []
synth_proj_theta_array_aux = []
synth_proj_theta_array_aux_2 = []
synth_proj_iter_array_aux = []
rgb_proj_theta_array = []
rgb_proj_theta_array_2 = []
rgb_proj_iter_array = []
recon_slice_array_aux = []
recon_slice_array_aux_2 = []
recon_iter_array_aux = []
tiff_array_1 = []
tiff_array_2 = []
tiff_array_3 = []
tiff_array_4 = []
tiff_array_5 = []
tiff_array_7 = []

iter_array = np.arange(n_iter)

theta_idx_desired = 0
iter_idx_desired = 0
iter_idx_final = iter_array[-1]
slice_idx_desired = 64
element_idx_desired = 11 # Fe for this directory

for theta_idx in range(n_theta):
    aligned_proj_theta_array_aux.append(aligned_proj_iter_array[iter_idx_desired][theta_idx])
    aligned_proj_theta_array_aux_2.append(aligned_proj_iter_array[-1][theta_idx])
    synth_proj_theta_array_aux.append(synth_proj_iter_array[iter_idx_desired][theta_idx])
    synth_proj_theta_array_aux_2.append(synth_proj_iter_array[-1][theta_idx])

    aligned_proj_norm = normalize_array(aligned_proj_iter_array[iter_idx_desired][theta_idx])
    synth_proj_norm = normalize_array(synth_proj_iter_array[iter_idx_desired][theta_idx])

    rgb = np.dstack((aligned_proj_norm, np.zeros_like(aligned_proj_norm), synth_proj_norm))

    rgb_proj_theta_array.append(rgb)

    aligned_proj_norm = normalize_array(aligned_proj_iter_array[-1][theta_idx])
    synth_proj_norm = normalize_array(synth_proj_iter_array[-1][theta_idx])

    rgb = np.dstack((aligned_proj_norm, np.zeros_like(aligned_proj_norm), synth_proj_norm))
    
    rgb_proj_theta_array_2.append(rgb)

for iter_idx in range(n_iter):
    aligned_proj_norm = normalize_array(aligned_proj_iter_array[iter_idx][theta_idx_desired])
    synth_proj_norm = normalize_array(synth_proj_iter_array[iter_idx][theta_idx_desired])

    rgb = np.dstack((aligned_proj_norm, np.zeros_like(aligned_proj_norm), synth_proj_norm))

    aligned_proj_iter_array_aux.append(aligned_proj_iter_array[iter_idx][theta_idx_desired])
    synth_proj_iter_array_aux.append(synth_proj_iter_array[iter_idx][theta_idx_desired])
    rgb_proj_iter_array.append(rgb)
    recon_iter_array_aux.append(recon_iter_array[iter_idx][slice_idx_desired])

for slice_idx in range(n_slices):
    recon_slice_array_aux.append(recon_iter_array[iter_idx_desired][slice_idx])
    recon_slice_array_aux_2.append(recon_iter_array[-1][slice_idx])

aligned_proj_theta_array_aux = np.array(aligned_proj_theta_array_aux)
aligned_proj_iter_array_aux = np.array(aligned_proj_iter_array_aux)
synth_proj_theta_array_aux = np.array(synth_proj_theta_array_aux)
synth_proj_iter_array_aux = np.array(synth_proj_iter_array_aux)
rgb_proj_theta_array = np.array(rgb_proj_theta_array)
rgb_proj_iter_array = np.array(rgb_proj_iter_array)
recon_slice_array_aux = np.array(recon_slice_array_aux)
recon_iter_array_aux = np.array(recon_iter_array_aux)

print('Generating figures...')

fps_imgs = 25 # Frames per second (fps)
fps_plots = 15

fig1, axs1 = plt.subplots(2, 3) # Aligned experimental projection, synthetic experimental projection, overlays at different angles
fig2, axs2 = plt.subplots(1, 3) # Same as above, but for different iterations - use first projection angle
fig3, axs3 = plt.subplots(1, 2) # Reconstructed object for different slices (use first and final iteration)
fig4, axs4 = plt.subplots() # Reconstructed object for different iteration (use slice index 64?)
fig5, axs5 = plt.subplots() # x- and y-shifts as function of iteration index
fig6, axs6 = plt.subplots() # Center of rotation as function of iteration index
fig7, axs7 = plt.subplots() # Net shifts as function of angle for each slice

im1_1 = axs1[0, 0].imshow(aligned_proj_theta_array_aux[0])
im1_2 = axs1[0, 1].imshow(synth_proj_theta_array_aux[0])
im1_3 = axs1[0, 2].imshow(rgb_proj_theta_array[0])
im1_4 = axs1[1, 0].imshow(aligned_proj_theta_array_aux_2[0])
im1_5 = axs1[1, 1].imshow(synth_proj_theta_array_aux_2[0])
im1_6 = axs1[1, 2].imshow(rgb_proj_theta_array_2[0])

im2_1 = axs2[0].imshow(aligned_proj_iter_array_aux[0])
im2_2 = axs2[1].imshow(synth_proj_iter_array_aux[0])
im2_3 = axs2[2].imshow(rgb_proj_iter_array[0])

im3_1 = axs3[0].imshow(recon_slice_array_aux[0])
im3_2 = axs3[1].imshow(recon_slice_array_aux_2[0])

im4 = axs4.imshow(recon_iter_array_aux[0])

curve1, = axs5.plot(iter_array, net_x_shifts[:, 0], 'k-o', markersize = 3, label = r'$\Delta x$')
curve2, = axs5.plot(iter_array, net_y_shifts[:, 0], 'r-o', markersize = 3, label = r'$\Delta y$')
curve3, = axs6.plot(iter_array, cor_shifts, 'k-o', markersize = 3)
curve4, = axs7.plot(theta_array, net_x_shifts[0, :], 'k-o', markersize = 3, label = r'$\Delta x$')
curve5, = axs7.plot(theta_array, net_y_shifts[0, :], 'k-o', markersize = 3, label = r'$\Delta y$')

text_1 = axs1[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[0, 0].transAxes, color = 'white')
text_2 = axs2[0].text(0.02, 0.02, r'Iter. 0', transform = axs2[0].transAxes, color = 'white')
text_3 = axs3[0].text(0.02, 0.02, r'Slice 0', transform = axs3[0].transAxes, color = 'white')
text_4 = axs4.text(0.02, 0.02, r'Iter. 0', transform = axs4.transAxes, color = 'white')

axs1[0, 0].set_title(r'Exp. Proj. (Iter. {0})'.format(iter_idx_desired), color = 'red')
axs1[0, 1].set_title(r'Synth. Proj.', color = 'blue')
axs1[0, 2].set_title(r'Overlay')

axs1[1, 0].set_title(r'Exp. Proj. (Iter. {0})'.format(iter_idx_final), color = 'red')
axs1[1, 1].set_title(r'Synth. Proj.', color = 'blue')
axs1[1, 2].set_title(r'Overlay')

axs2[0].set_title(r'Exp. Proj. ($\theta = {0}$\textdegree)'.format(theta_array[theta_idx_desired]), color = 'red')
axs2[1].set_title(r'Synth. Proj.', color = 'blue')
axs2[2].set_title(r'Overlay')

axs3[0].set_title(r'Reconstruction (Iter. {0})'.format(iter_idx_desired))
axs3[1].set_title(r'Reconstruction (Iter. {0})'.format(iter_idx_final))

axs4.set_title(r'Reconstruction (Slice {0})'.format(slice_idx_desired))

axs5.set_xlim(0, n_iter - 1)
axs5.set_title(r'\theta = {0}'.format(theta_array[0]))
axs5.set_xlabel(r'Iteration index $i$')
axs5.set_ylabel(r'Net shift')
axs5.legend(frameon = False)

axs6.set_xlim(0, n_iter - 1)
axs6.set_ylim(np.min(cor_shifts), np.max(cor_shifts))
axs6.set_xlabel(r'Iteration index $i$')
axs6.set_ylabel(r'Center of rotation')

axs7.set_xlim(np.min(theta_array), np.max(theta_array))
axs7.set_xlabel(r'$\theta$')
axs7.set_ylabel(r'Net shift')
axs7.legend(frameon = False)

for theta_idx in range(n_theta):
    im1_1.set_data(aligned_proj_theta_array_aux[theta_idx])
    im1_2.set_data(synth_proj_theta_array_aux[theta_idx])
    im1_3.set_data(rgb_proj_theta_array[theta_idx])
    im1_4.set_data(aligned_proj_theta_array_aux_2[theta_idx])
    im1_5.set_data(synth_proj_theta_array_aux_2[theta_idx])
    im1_6.set_data(rgb_proj_theta_array_2[theta_idx])

    text_1.set_text(r'$\theta = {0}$'.format(theta_array[theta_idx]))

    net_shift_x = net_x_shifts[:, theta_idx]
    net_shift_y = net_y_shifts[:, theta_idx]

    curve1.set_ydata(net_shift_x)
    curve2.set_ydata(net_shift_y)

    min_shift = np.min([np.min(net_shift_x), np.min(net_shift_y)])
    max_shift = np.max([np.max(net_shift_x), np.max(net_shift_y)])

    axs5.set_ylim(min_shift, max_shift + 0.1)
    axs5.set_title(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

    filename_1 = os.path.join(dir_path, f'proj_theta{theta_idx:03d}.tiff')
    filename_5 = os.path.join(dir_path, f'net_net_shifts_theta_{theta_idx:03d}.tiff')

    fig1.tight_layout()
    fig5.tight_layout()

    fig1.savefig(filename_1, dpi = 400)
    fig5.savefig(filename_5, dpi = 400)

    tiff_array_1.append(filename_1)
    tiff_array_5.append(filename_5)

plt.close(fig1)
plt.close(fig5)

for slice_idx in range(n_slices):
    im3_1.set_data(recon_slice_array_aux[slice_idx])
    im3_2.set_data(recon_slice_array_aux_2[slice_idx])

    text_3.set_text(r'Slice {0}'.format(slice_idx))

    filename_3 = os.path.join(dir_path, f'recon_slice_{slice_idx:03d}.tiff')

    fig3.tight_layout()
    
    fig3.savefig(filename_3, dpi = 400)

    tiff_array_3.append(filename_3)

plt.close(fig3)

for iter_idx in range(n_iter):
    net_shift_x = net_x_shifts[iter_idx, :]
    net_shift_y = net_y_shifts[iter_idx, :]

    min_shift = np.min([np.min(net_shift_x), np.min(net_shift_y)])
    max_shift = np.max([np.max(net_shift_x), np.max(net_shift_y)])

    im2_1.set_data(aligned_proj_iter_array_aux[iter_idx])
    im2_2.set_data(synth_proj_iter_array_aux[iter_idx])
    im2_3.set_data(rgb_proj_iter_array[iter_idx])

    im4.set_data(recon_iter_array_aux[iter_idx])
    
    curve4.set_ydata(net_shift_x)
    curve5.set_ydata(net_shift_y)

    text_2.set_text(r'Iter. {0}'.format(iter_idx))
    text_4.set_text(r'Iter. {0}'.format(iter_idx))

    axs7.set_ylim(min_shift, max_shift + 0.1)
    axs7.set_title(r'Iteration {0}'.format(iter_idx))

    filename_2 = os.path.join(dir_path, f'proj_iter_{iter_idx:03d}.tiff')
    filename_4 = os.path.join(dir_path, f'recon_iter_{iter_idx:03d}.tiff')
    filename_7 = os.path.join(dir_path, f'net_shifts_iter_{iter_idx:03d}.tiff')

    fig2.tight_layout()
    fig4.tight_layout()
    fig7.tight_layout()
    
    fig2.savefig(filename_2, dpi = 400)
    fig4.savefig(filename_4, dpi = 400)
    fig7.savefig(filename_7, dpi = 400)

    tiff_array_2.append(filename_2)
    tiff_array_4.append(filename_4)
    tiff_array_7.append(filename_7)

plt.close(fig2)
plt.close(fig4)
plt.close(fig7)

print('Saving COR plot...')

filename_6 = os.path.join(dir_path, 'cor_shifts.svg')

fig6.tight_layout()
fig6.savefig(filename_6)

plt.close(fig6)

print('Creating projection GIF (changing thetas)...')

create_gif(tiff_array_1, os.path.join(dir_path, 'proj_theta.gif'), fps = 25)

print('Creating projection GIF (changing iteration)...')

create_gif(tiff_array_2, os.path.join(dir_path, 'proj_iter.gif'), fps = 25)

print('Creating reconstruction GIF (changing slice)...')

create_gif(tiff_array_3, os.path.join(dir_path, 'recon_slice.gif'), fps = 25)

print('Creating reconstruction GIF (changing iteration)...')

create_gif(tiff_array_4, os.path.join(dir_path, 'recon_iter.gif'), fps = 25)

print('Creating net shift GIF (changing theta)...')

create_gif(tiff_array_5, os.path.join(dir_path, 'net_shifts_theta.gif'), fps = 15)

print('Creating net shift GIF (changing iteration)...')

create_gif(tiff_array_7, os.path.join(dir_path, 'net_shifts_iter.gif'), fps = 15)

print('Done')

# anim1 = anim.FuncAnimation(fig1, update_proj_theta, frames = n_theta, interval = 1000/fps_imgs, blit = True) # Interval is in ms --> interval = (1/fps)*1000
# anim2 = anim.FuncAnimation(fig2, update_proj_iter, frames = n_iter, interval = 1000/fps_imgs, blit = True)
# anim3 = anim.FuncAnimation(fig3, update_recon_slice, frames = n_slices, interval = 1000/fps_imgs, blit = True)
# anim4 = anim.FuncAnimation(fig4, update_recon_iter, frames = n_iter, interval = 1000/fps_imgs, blit = True)
# anim5 = anim.FuncAnimation(fig5, update_shifts, frames = n_theta, interval = 1000/fps_plots, blit = False)

# print('Exporting projections (changing thetas) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_imgs, metadata = {'title': 'proj_theta'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264', '-loglevel', 'debug'])

# anim1.save(os.path.join(dir_path, 'proj_theta.mp4'), writer, dpi = 400)

# print('Exporting projections (changing iterations) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_imgs, metadata = {'title': 'proj_iter'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264', '-loglevel', 'debug'])

# anim2.save(os.path.join(dir_path, 'proj_iter.mp4'), writer, dpi = 400)

# print('Exporting reconstructions (changing slices) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_imgs, metadata = {'title': 'recon_slice'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264', '-loglevel', 'debug'])

# anim3.save(os.path.join(dir_path, 'recon_slice.mp4'), writer, dpi = 400)

# print('Exporting reconstructions (changing iterations) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_imgs, metadata = {'title': 'recon_iter'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264', '-loglevel', 'debug'])

# anim4.save(os.path.join(dir_path, 'recon_iter.mp4'), writer, dpi = 400)

# print('Exporting net shifts (changing thetas) to .mp4 file...')

# writer = anim.FFMpegWriter(fps = fps_plots, metadata = {'title': 'recon_slice'}, bitrate = 3500, extra_args = ['-vcodec', 'libx264', '-loglevel', 'debug'])

# anim5.save(os.path.join(dir_path, 'net_shifts.mp4'), writer, dpi = 400)

# print('Finished')

# plt.show()