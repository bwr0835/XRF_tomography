import numpy as np, tkinter as tk, os

from matplotlib import pyplot as plt, animation as anim
from tkinter import filedialog

def load_dir(dir_path, filter_function = None):
    subdir_array = [subdir for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subdir))]

    if filter_function:
        subdir_array = [subdir for subdir in subdir_array if filter_function]

    return subdir_array

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

    n_iterations = len(iteration_subdir_array)
    
    synthetic_proj_data = []
    actual_proj_data = []
    xcorr_proj_data = []
    recon_data = []

    for subdir in iteration_subdir_array:
        synthetic_proj_dir_path = os.path.join(directory_path, subdir, 'synthesized')
        actual_proj_dir_path = os.path.join(directory_path, subdir, 'experimental')
        xcorr_dir_path = os.path.join(directory_path, subdir, 'xcorr')
        recon_dir_path = os.path.join(directory_path, subdir, 'recon')

        synthetic_proj_file_path = [os.path.join(synthetic_proj_dir_path, file_name) for file_name in os.listdir(synthetic_proj_dir_path) if os.path.isfile(os.path.join(synthetic_proj_dir_path, file_name))]
        actual_proj_file_path = [os.path.join(actual_proj_dir_path, file_name) for file_name in os.listdir(actual_proj_dir_path) if os.path.isfile(os.path.join(actual_proj_dir_path, file_name))]
        xcorr_proj_file_path = [os.path.join(xcorr_dir_path, file_name) for file_name in os.listdir(xcorr_dir_path) if os.path.isfile(os.path.join(xcorr_dir_path, file_name))]
        recon_file_path = [os.path.join(recon_dir_path, file_name) for file_name in os.listdir(recon_dir_path) if os.path.isfile(os.path.join(recon_dir_path, file_name))]

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


    
