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

    # plt.plot(theta_array, x_shifts_data[0, :], 'k-o', label = r'Iteration Index 0')
    # plt.plot(theta_array, x_shifts_data[-1, :], 'bo-', label = r'Iteration Index 9')
    # plt.legend()
    
    # fig1, axs1 = plt.subplots(2, 1)
    fig1, axs1 = plt.subplots()

    axs1.set_title(r'Recon. Slice (It. 0, No COR Fix)')
   
    recon_array_zero_iter = recon_data_dict[iteration_subdir_array_truncated[0]]

    im1 = axs1.imshow(recon_array_zero_iter[0])
    text1 = axs1.text(0.02, 0.02, r'Slice 0', transform = axs1.transAxes, color = 'white')

    def animate_recon(frame):
        im1.set_array(recon_array_zero_iter[frame])
        text1.set_text(r'Slice {0}'.format(frame))

        return im1, text1
    
    fps_shifts = 10 # Frames per second (fps)
    fps_images = 25

    anim1 = anim.FuncAnimation(fig1, animate_recon, frames = n_slices, interval = 1000/fps_images, blit = True) # Interval is ms/frame (NOT frames per second, or fps)
 
    output_path1 = '/home/bwr0835/recon_gridrec_no_cor_correction.mp4'
    writer1 = anim.FFMpegWriter(fps = fps_images, metadata = {'title': 'recon'}, bitrate = 1800, extra_args = ['-vcodec', 'libx264'])
    
    print('Saving')
    
    anim1.save(output_path1, writer1)

    plt.show()