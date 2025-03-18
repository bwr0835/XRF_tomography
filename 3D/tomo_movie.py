import numpy as np, tkinter as tk, os, re

from matplotlib import pyplot as plt, animation as anim
from tkinter import filedialog

# def load_dir(dir_path, filter_function = None):
#     subdir_array = [subdir for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subdir))]

#     if filter_function:
#         subdir_array_new = [subdir for subdir in subdir_array if filter_function(subdir)]

#         return subdir_array_new
    
#     else:
#         return subdir_array

def load_dir(dir_path):
    subdir_array = [subdir for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subdir))]

    return subdir_array

def get_theta(file_name):
    return int(file_name.split('_')[2].strip())

def get_slice(file_name):
    return int(file_name.split('_')[2].split('.')[0])

def get_iteration(subdir_name):
    return int(subdir_name.split('_')[1])

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
    subdir_array = load_dir(directory_path)
    
    iteration_subdir_array = []
    synthetic_proj_data = []
    actual_proj_data = []
    xcorr_proj_data = []
    recon_data = []
    x_shifts_data = []
    
    
    
    for subdir in subdir_array:
        if subdir != 'net_shifts':
            iteration_subdir_array.append(subdir)
        
        else:
            net_shift_subdir = subdir
    
    iteration_subdir_array = sorted(iteration_subdir_array, key = get_iteration)

    n_iterations = len(iteration_subdir_array)    

    for idx, subdir in enumerate(iteration_subdir_array):
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

        for theta_idx in range(n_theta):
            synthetic_proj[theta_idx] = np.load(synthetic_proj_file_path[theta_idx])
            actual_proj[theta_idx] = np.load(actual_proj_file_path[theta_idx])
            xcorr_proj[theta_idx] = np.load(xcorr_proj_file_path[theta_idx])
        
        for slice_idx in range(n_slices):
            recon[slice_idx] = np.load(recon_file_path[theta_idx])

        synthetic_proj_data = np.append(synthetic_proj_data, synthetic_proj)
        actual_proj_data = np.append(actual_proj_data, actual_proj)
        xcorr_proj_data = np.append(xcorr_proj_data, xcorr_proj)
        recon_data = np.append(recon_data, recon)

    x_shifts = np.zeros((n_iterations, n_theta))
    y_shifts = np.zeros((n_iterations, n_theta))

    x_shifts_dir_path = os.path.join(net_shift_subdir, 'x_shifts')
    y_shifts_dir_path = os.path.join(net_shift_subdir, 'y_shifts')

    x_shifts_file_name = [file_name for file_name in os.listdir(x_shifts_dir_path)]
    y_shifts_file_name = [file_name for file_name in os.listdir(y_shifts_dir_path)]




    
