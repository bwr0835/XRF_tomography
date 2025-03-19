import numpy as np, tkinter as tk, os, re

from matplotlib import pyplot as plt, animation as anim
from tkinter import filedialog
from numpy.fft import fftshift

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

# def load_dir(dir_path, filter_function = None):
#     subdir_array = [subdir for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subdir))]

#     if filter_function:
#         subdir_array_new = [subdir for subdir in subdir_array if filter_function(subdir)]

#         return subdir_array_new
    
#     else:
#         return subdir_array

def load_dir(dir_path):
    subdir_array = [subdir for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subdir))]
    
    theta_array_file_path = os.path.join(dir_path, 'theta_array.npy')

    theta_array = np.load(theta_array_file_path)
    theta_array = np.sort(theta_array)

    return subdir_array, theta_array

def get_theta(file_name):
    return int(file_name.split('_')[2].strip())

def get_slice(file_name):
    return int(file_name.split('_')[2].split('.')[0])

def get_iteration(subdir_name):
    return int(subdir_name.split('_')[1])

def normalize_array(array):
    return (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))
# def get_num_status(file_name):
#     return bool(re.search(r'\d', file_name))

root = tk.Tk()
    
root.withdraw()

# directory_path = filedialog.askdirectory(parent = root, title = "Select directory", filetypes = (('NPY Files', '*.npy')))

directory_path = '/raid/users/roter/gridrec'

if directory_path == '':
    print('Program aborted.')

else:
    n_theta = 49
    n_slices = 301
    n_columns = 599

    recon = np.zeros((n_slices, n_columns, n_columns))
    synthetic_proj = np.zeros((n_theta, n_slices, n_columns))
    actual_proj = np.zeros((n_theta, n_slices, n_columns))
    xcorr_proj = np.zeros((n_theta, n_slices, n_columns))

    file_path_idx = 0

    # subdir_array = load_dir(directory_path, lambda subdir: subdir.startswith('iteration_'))
    subdir_array, theta_array = load_dir(directory_path)
    
    iteration_subdir_array = []
    
    synthetic_proj_data_dict = {}
    actual_proj_data_dict = {}
    xcorr_proj_data_dict = {}
    recon_data_dict = {}
    shift_change_dict = {}
    
    for subdir in subdir_array:
        if subdir != 'net_shifts':
            iteration_subdir_array.append(subdir)
        
        else:
            net_shift_subdir = subdir
    
    iteration_subdir_array = sorted(iteration_subdir_array, key = get_iteration)

    iteration_subdir_array_truncated = [iteration_subdir_array[0], iteration_subdir_array[-1]]

    n_iterations = len(iteration_subdir_array)    

    for idx, subdir in enumerate(iteration_subdir_array_truncated):
        synthetic_proj_data = []
        actual_proj_data = []
        xcorr_proj_data = []
        recon_data = []
        shift_rgb_data = []

        synthetic_proj_dir_path = os.path.join(directory_path, subdir, 'synthesized')
        actual_proj_dir_path = os.path.join(directory_path, subdir, 'experimental')
        xcorr_dir_path = os.path.join(directory_path, subdir, 'xcorr')
        recon_dir_path = os.path.join(directory_path, subdir, 'recon')

        proj_file_name = [file_name for file_name in os.listdir(synthetic_proj_dir_path)]
        # actual_proj_file_path = [file_name for file_name in os.listdir(actual_proj_dir_path)]
        # xcorr_proj_file_path = [file_name for file_name in os.listdir(xcorr_dir_path)]
        recon_file_name = [file_name for file_name in os.listdir(recon_dir_path)]

        proj_file_name = sorted(proj_file_name, key = get_theta)
        recon_file_name = sorted(recon_file_name, key = get_slice)

        synthetic_proj_file_path = [os.path.join(synthetic_proj_dir_path, file_name) for file_name in proj_file_name]
        actual_proj_file_path = [os.path.join(actual_proj_dir_path, file_name) for file_name in proj_file_name]
        xcorr_proj_file_path = [os.path.join(xcorr_dir_path, file_name) for file_name in proj_file_name]
        recon_file_path = [os.path.join(recon_dir_path, file_name) for file_name in recon_file_name]

        print('Loading projection images...')

        for theta_idx in range(n_theta):
            synthetic_proj[theta_idx] = np.load(synthetic_proj_file_path[theta_idx])
            actual_proj[theta_idx] = np.load(actual_proj_file_path[theta_idx])
            xcorr_proj[theta_idx] = fftshift(np.load(xcorr_proj_file_path[theta_idx]))

            synthetic_proj_scaled = normalize_array(synthetic_proj[theta_idx])
            actual_proj_scaled = normalize_array(actual_proj[theta_idx])

            shift_rgb = np.dstack((actual_proj_scaled, np.zeros_like(synthetic_proj_scaled), synthetic_proj_scaled))

            synthetic_proj_data.append(synthetic_proj[theta_idx])
            actual_proj_data.append(actual_proj[theta_idx])
            xcorr_proj_data.append(xcorr_proj[theta_idx])
            shift_rgb_data.append(shift_rgb)

        print('Loading reconstructions...')

        for slice_idx in range(n_slices):
            recon[slice_idx] = np.load(recon_file_path[slice_idx])

            recon_data.append(recon[slice_idx])

        synthetic_proj_data = np.array(synthetic_proj_data)
        actual_proj_data = np.array(actual_proj_data)
        xcorr_proj_data = np.array(xcorr_proj_data)
        recon_data = np.array(recon_data)
        shift_rgb_data = np.array(shift_rgb_data)
        
        synthetic_proj_data_dict[subdir] = synthetic_proj_data
        actual_proj_data_dict[subdir] = actual_proj_data
        xcorr_proj_data_dict[subdir] = xcorr_proj_data
        recon_data_dict[subdir] = recon_data
        shift_change_dict[subdir] = shift_rgb_data

    x_shifts_file_path = os.path.join(directory_path, net_shift_subdir, 'x_shift_array.npy')
    y_shifts_file_path = os.path.join(directory_path, net_shift_subdir, 'y_shift_array.npy')

    x_shifts_data = np.load(x_shifts_file_path)
    y_shifts_data = np.load(y_shifts_file_path)

    fig1, axs1 = plt.subplots(2, 1)
    fig2, axs2 = plt.subplots(2, 4)
    fig3, axs3 = plt.subplots()

    iter_array = 1 + np.arange(n_iterations)

    curve1 = axs3.plot(iter_array, x_shifts_data[:, 0], 'k-o', markersize = 3, label = r'$\Delta x$')[0]
    curve2 = axs3.plot(iter_array, y_shifts_data[:, 0], 'r-o', markersize = 3, label = r'$\Delta y$')[0]

    axs1[0].set_title(r'Recon. Slice (It. 1)')
    axs1[1].set_title(r'Recon. Slice (It. {0})'.format(n_iterations))
    axs2[0, 0].set_title(r'(Al.) Exp. Proj. (It. 1)')
    axs2[1, 0].set_title(r'(Al.) Exp. Proj. (It. {0})'.format(n_iterations))
    axs2[0, 1].set_title(r'Synth. Proj. (It. 1)')
    axs2[1, 1].set_title(r'Synth. Proj. (It. {0})'.format(n_iterations))
    axs2[0, 2].set_title(r'Int.-Pix. CC (It. 1)')
    axs2[1, 2].set_title(r'Int.-Pix. CC (It. {0})'.format(n_iterations))
    axs2[0, 3].set_title(r'Curr. shift (It. 1)')
    axs2[1, 3].set_title(r'Curr. shift (It. {0})'.format(n_iterations))

    axs3.set_xlabel(r'Iteration')
    axs3.set_ylabel(r'Net shift (pixels)')
    axs3.set_xlim(1, n_iterations)
    axs3.legend(frameon = False)

    recon_imgs = []
    exp_proj_imgs = []
    synthetic_proj_imgs = []
    xcorr_imgs = []
    shift_rgb_imgs = []
    
    recon_text = []
    proj_text = []

    for idx, subdir in enumerate(iteration_subdir_array_truncated):
        recons = recon_data_dict[subdir]
        exp_projs = actual_proj_data_dict[subdir]
        synth_projs = synthetic_proj_data_dict[subdir]
        xcorrs = xcorr_proj_data_dict[subdir]
        shift_rgbs = shift_change_dict[subdir]

        im1 = axs1[idx].imshow(recons[0])
        im2_0 = axs2[idx, 0].imshow(exp_projs[0])
        im2_1 = axs2[idx, 1].imshow(synth_projs[0])
        im2_2 = axs2[idx, 2].imshow(xcorrs[0])
        im2_3 = axs2[idx, 3].imshow(shift_rgbs[0])

        recon_imgs.append(im1)
        exp_proj_imgs.append(im2_0)
        synthetic_proj_imgs.append(im2_1)
        xcorr_imgs.append(im2_2)
        shift_rgb_imgs.append(im2_3)

        text_recon = axs1[0].text(0.02, 0.02, r'Slice 0', transform = axs1[0].transAxes, color = 'white')
        text_proj = axs2[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs2[0, 0].transAxes, color = 'white')    
        
        if idx == 0:
            recon_text.append(text_recon)
            proj_text.append(text_proj)

    def animate_recon(frame):
        artists = []

        for idx, subdir in enumerate(iteration_subdir_array_truncated):
            recons = recon_data_dict[subdir]

            recon_imgs[idx].set_array(recons[frame])
            recon_text[0].set_text(r'Slice {0}'.format(frame))
            
            artists.append(recon_imgs[idx])
        
        artists.append(recon_text[0])

        return artists
    
    def animate_proj(frame):
        artists = []

        for idx, subdir in enumerate(iteration_subdir_array_truncated):
            exp_projs = actual_proj_data_dict[subdir]
            synth_projs = synthetic_proj_data_dict[subdir]
            xcorrs = xcorr_proj_data_dict[subdir]
            shift_rgbs = shift_change_dict[subdir]

            exp_proj_imgs[idx].set_array(exp_projs[frame])
            synthetic_proj_imgs[idx].set_array(synth_projs[frame])
            xcorr_imgs[idx].set_array(xcorrs[frame])
            shift_rgb_imgs[idx].set_array(shift_rgbs[frame])

            artists.append(exp_proj_imgs[idx])
            artists.append(synthetic_proj_imgs[idx])
            artists.append(xcorr_imgs[idx])
            artists.append(shift_rgb_imgs[idx])
        
        proj_text[0].set_text(r'$\theta = {0}$\textdegree'.format(theta_array[frame]))

        artists.append(proj_text[0])

        return artists
    
    def animate_shifts(frame):
        artists = []
        
        net_shift_x = x_shifts_data[:, frame]
        net_shift_y = y_shifts_data[:, frame]

        curve1.set_ydata(net_shift_x)
        curve2.set_ydata(net_shift_y)

        min_shift = np.min([np.min(net_shift_x), np.min(net_shift_y)])
        max_shift = np.max([np.max(net_shift_x), np.max(net_shift_y)])

        axs3.set_ylim(min_shift, max_shift)
        axs3.set_title(r'$\theta = {0}$\textdegree'.format(theta_array[frame]))

        artists.append(curve1)
        artists.append(curve2)

        return artists

    anim1 = anim.FuncAnimation(fig1, animate_recon, frames = n_slices, interval = 150, blit = True)
    anim2 = anim.FuncAnimation(fig2, animate_proj, frames = n_theta, interval = 150, blit = True)
    anim3 = anim.FuncAnimation(fig3, animate_shifts, frames = n_theta, interval = 75, blit = True)

    plt.show()



        


