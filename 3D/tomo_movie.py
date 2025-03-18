import numpy as np, tkinter as tk, os, re

from matplotlib import pyplot as plt, animation as anim
from tkinter import filedialog

def load_dir(dir_path, filter_function = None):
    subdir_array = [subdir for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subdir))]

    if filter_function:
        subdir_array_new = [subdir for subdir in subdir_array if filter_function(subdir)]

        return subdir_array_new
    
    else:
        return subdir_array

def get_theta(file_name):
    return int(file_name.split('_')[2].strip())

def get_slice(file_name):
    return int(file_name.split('_')[2].split('.')[0])

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

    iteration_subdir_array = load_dir(directory_path, lambda subdir: subdir.startswith('iteration_'))

    print(iteration_subdir_array)

    n_iterations = len(iteration_subdir_array)
    
    synthetic_proj_data = []
    actual_proj_data = []
    xcorr_proj_data = []
    recon_data = []

    for idx, subdir in enumerate(iteration_subdir_array):
        synthetic_proj_dir_path = os.path.join(directory_path, subdir, 'synthesized')
        actual_proj_dir_path = os.path.join(directory_path, subdir, 'experimental')
        xcorr_dir_path = os.path.join(directory_path, subdir, 'xcorr')
        recon_dir_path = os.path.join(directory_path, subdir, 'recon')

        synthetic_proj_file_path = [file_name for file_name in os.listdir(synthetic_proj_dir_path)]
        actual_proj_file_path = [file_name for file_name in os.listdir(actual_proj_dir_path)]
        xcorr_proj_file_path = [file_name for file_name in os.listdir(xcorr_dir_path)]
        recon_file_path = [file_name for file_name in os.listdir(recon_dir_path)]

        synthetic_proj_file_path = sorted(synthetic_proj_file_path, key = get_theta)
        if idx == 0:
            print(synthetic_proj_file_path)

        # for theta_idx in range(n_theta):
        #     synthetic_proj[theta_idx] = np.load(synthetic_proj_file_path[theta_idx])
        #     actual_proj[theta_idx] = np.load(actual_proj_file_path[theta_idx])
        #     xcorr_proj[theta_idx] = np.load(xcorr_proj_file_path[theta_idx])
        
        # for slice_idx in range(n_slices):
        #     recon[slice_idx] = np.load(recon_file_path[theta_idx])

        # synthetic_proj_data = np.append(synthetic_proj_data, synthetic_proj)
        # actual_proj_data = np.append(actual_proj_data, actual_proj)
        # xcorr_proj_data = np.append(xcorr_proj_data, xcorr_proj)
        # recon_data = np.append(recon_data, recon)


    
