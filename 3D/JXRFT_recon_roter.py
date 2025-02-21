#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO Create code for selecting transmission and fluorescence HDF5 files and creating separate HDF5 files containing
# information about all projection angles (all values of theta, elements, 2d projection images)
import tkinter as tk
import os
import sys
import numpy as np
import torch as tc
import xraylib as xlib
import warnings

from tkinter import filedialog
from XRF_tomography import reconstruct_jXRFT_tomography
from mpi4py import MPI
from misc import create_summary
from h5_util import extract_h5_aggregate_xrf_data, create_aggregate_xrf_h5

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()
warnings.filterwarnings("ignore")

#========================================================
# Set the device
#========================================================
# stdout_options = {'output_folder': recon_path, 'save_stdout': False, 'print_terminal': True}
gpu_index = rank % 2
# gpu_index = 1
if tc.cuda.is_available():  
    dev = tc.device('cuda:{}'.format(gpu_index))
    print("Process ", rank, "running on", dev)
    sys.stdout.flush()
else:  
    dev = "cpu"
    print("Process", rank, "running on CPU")
    sys.stdout.flush()


fl = {"K": np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE]),
      "L": np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE]),              
      "M": np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])               
     }



params_124_124_32_cabead = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': './data/Cabead/axo_std',
                              'f_std': 'axo_std.mda.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6, 
                              'fitting_method':'XRF_roi_plus',
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': True,
                              'ini_kind': 'const', 
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Cabead_adjusted1_ds4_recon/Ab_T_nEl_6_nDpts_4_b1_1e4_b2_1e0_lr_1.0e-3',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Cabead_adjusted1_ds4',
                              'f_XRF_data': 'cabead_xrf-fits',                 
                              'f_XRT_data': 'cabead_scalers',
                              'scaler_counts_us_ic_dataset_idx':18,
                              'scaler_counts_ds_ic_dataset_idx':11,
                              'XRT_ratio_dataset_idx':21,   
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements',  
                              'this_aN_dic': {"Si": 14, "Ti": 22, "Cr": 24, "Fe": 26, "Ni":28, "Ba": 56},
                              'element_lines_roi': np.array([['Si', 'K'], ['Ti', 'K'], ['Cr', 'K'],
                                                             ['Fe', 'K'], ['Ni', 'K'], ['Ba', 'L']]),  # np.array([["Si, K"], ["Ca, K"]])
                              'n_line_group_each_element': np.array([1, 1, 1, 1, 1, 1]),
                              'sample_size_n': 124, 
                              'sample_height_n': 32,
                              'sample_size_cm': 0.0248,                                    
                              'probe_energy': np.array([10.0]),  
                              'n_epochs': 100,
                              'save_every_n_epochs': 5,
                              'minibatch_size': 124,
                              'b1': 1.0e4, 
                              'b2': 0.0,
                              'lr': 1.0E-3,                          
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, 1.69, 0.70], [0.70, 1.69, -0.70], [-0.70, 1.69, 0.70], [-0.70, 1.69, -0.70]]),
                              'det_on_which_side': "positive",
                              'det_from_sample_cm': None,
                              'det_ds_spacing_cm': None,
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,
                              'det_dia_cm': None,
                              'P_folder': 'data/P_array/sample_124_124_32/Dis_1.69_manual_dpts_4',              
                              'f_P': 'Intersecting_Length_124_124_32',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params_124_124_32_cabead_2 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': './data/Cabead/axo_std',
                              'f_std': 'axo_std.mda.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]), # This is for MAPS only!
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6, 
                              'fitting_method':'XRF_roi_plus',
                              'selfAb': False,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const', 
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Cabead_adjusted1_ds3_recon/Ab_F_nEl_6_nDpts_3_b1_1e4_b2_1e0_lr_1.0e-3',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Cabead_adjusted1_ds4',
                              'f_XRF_data': 'cabead_xrf-fits',                 
                              'f_XRT_data': 'cabead_scalers',
                              'scaler_counts_us_ic_dataset_idx':18,
                              'scaler_counts_ds_ic_dataset_idx':11,
                              'XRT_ratio_dataset_idx':21,   
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements',  
                              'this_aN_dic': {"Si": 14, "Ti": 22, "Cr": 24, "Fe": 26, "Ni":28, "Ba": 56},
                              'element_lines_roi': np.array([['Si', 'K'], ['Ti', 'K'], ['Cr', 'K'],
                                                             ['Fe', 'K'], ['Ni', 'K'], ['Ba', 'L']]),  # np.array([["Si, K"], ["Ca, K"]])
                              'n_line_group_each_element': np.array([1, 1, 1, 1, 1, 1]),
                              'sample_size_n': 124, 
                              'sample_height_n': 32,
                              'sample_size_cm': 0.0248,                                    
                              'probe_energy': np.array([10.0]),  
                              'n_epochs': 200,
                              'save_every_n_epochs': 5,
                              'minibatch_size': 124,
                              'b1': 1.0E4, 
                              'b2': 1.0E0,
                              'lr': 1.0E-3,                          
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, 1.69, 0.70], [-0.70, 1.69, 0.70], [-0.70, 1.69, -0.70]]),
                              'det_on_which_side': "positive",
                              'det_from_sample_cm': None,
                              'det_ds_spacing_cm': None,
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,
                              'det_dia_cm': None,
                              'P_folder': 'data/P_array/sample_124_124_32/Dis_1.69_manual_dpts_3',              
                              'f_P': 'Intersecting_Length_124_124_32',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params_roter = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': None, # Directory path for standard reference material
                              'f_std': None, # Name of standard reference material file
                              'std_element_lines_roi': None, 
                              'density_std_elements': None, # Areal mass density (g/cm^2)
                              'fitting_method':'XRF_roi_plus',
                              'selfAb': None,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const', 
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': None, # Directory for where reconstructions should go
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': None, # Directory for where aggregate XRF and XRT HDF5 files are stored
                              'f_XRF_data': None, # Aggregate XRF HDF5 file name                
                              'f_XRT_data': None, # Aggregate XRT HDF5 file name
                              'scaler_counts_us_ic_dataset_idx': None,
                              'scaler_counts_ds_ic_dataset_idx': None,
                              'XRT_ratio_dataset_idx': None,   
                            #   'theta_ls_dataset': 'exchange/theta', 
                            #   'channel_names': 'exchange/elements',  
                              'this_aN_dic': {},
                              'element_lines_roi': None,
                              'n_line_group_each_element': None,
                              'sample_size_n': None, 
                              'sample_height_n': None,
                              'sample_size_cm': None,                                    
                              'probe_energy': None, # Incident photon energy (keV)
                              'n_epochs': None,
                              'save_every_n_epochs': None,
                              'minibatch_size': None,
                              'b1': None, 
                              'b2': None,
                              'lr': None,                          
                              'manual_det_coord': True,
                              'set_det_coord_cm': None,
                              'det_on_which_side': None,
                              'det_from_sample_cm': None, # Don't change when using experimental data
                              'det_ds_spacing_cm': None, # Don't change when using experimental data
                              'manual_det_area': True, # Don't change when using experimental data
                              'det_area_cm2': None,
                              'det_dia_cm': None, # Don't change when using experimental data
                              'P_folder': 'data/P_array/sample_124_124_32/Dis_1.69_manual_dpts_3',              
                              'f_P': 'Intersecting_Length_124_124_32',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params = params_124_124_32_cabead_2

params_test = params_roter

synchrotron_list = ['Advanced Photon Source (APS)', 
                    'National Synchrotron Light Source II (NSLS-II)']

if __name__ == "__main__":
    root = tk.Tk()
    
    root.withdraw()
    
    print('\nSelect a synchrotron light source from the list below:')
    
    for idx, synchro in enumerate(synchrotron_list):
        if idx == 0:
            print('\n' + synchrotron_list[idx])
        
        else:
            print(synchrotron_list[idx])
    
    synchrotron = input('\nSynchrotron light source: ')

    print('\nDo you have premade, complementary aggregate ' + synchrotron + ' XRF and XRT HDF5 file consisting of the below file structure?')
    print('\nexchange')
    print(' \u2022 data [a 4D array of either XRF or XRT data with all values of theta, where the data is sorted by theta in ascending order: (n_elements, n_theta, ny, nx)]')
    print(' \u2022 elements [a string array of all available elements (XRF) or relevant channels (XRT) (array length = n_elements)]')
    print(' \u2022 theta [a float array of all projection angles (in degrees) in ascending order (array length = n_theta)]\n\n')
    
    premade_file_status = input('(y/N): ')

    if premade_file_status == 'y' or premade_file_status == 'Y':
        select_premade_aggregate_xrf_file_enabled = True
        select_premade_aggregate_xrt_file_enabled = True

        while select_premade_aggregate_xrf_file_enabled:
            input_xrf_file_path = filedialog.askopenfilename(parent = root, title = "Select aggregate XRF HDF5 file", filetypes = [("HDF5 files", "*.h5")])
            
            if input_xrf_file_path == '':
                print("\nNo file selected. Exiting...\n")
                
                exit()
            
            try:
                elements, counts, theta = extract_h5_aggregate_xrf_data(input_xrf_file_path)

                select_premade_file_enabled = False
        
            except:
                print('\nError: Invalid aggregate XRF HDF5 file. Please try again.\n')

                continue
        
        while select_premade_aggregate_xrt_file_enabled:
            input_xrt_file_path = filedialog.askopenfilename(parent = root, title = "Select aggregate XRT HDF5 file", filetypes = [("HDF5 files", "*.h5")])

            if input_xrt_file_path == '':
                print("\nNo file selected. Exiting...\n")
                
                exit()
            
            try:
                elements, counts, theta = extract_h5_aggregate_xrt_data(input_xrt_file_path)

                select_premade_file_enabled = False
            
            except:
                print('\nError: Invalid aggregate XRT HDF5 file. Please try again.\n')

                continue

        select_elements_enabled = True
        
        while select_elements_enabled:
            element_not_found = False
            multiple_elements_selected = False
            
            print('\nSelect which element(s) to reconstruct for XRF.')
            print('\nAvailable element(s): ' + ', '.join(elements))

            elements_to_reconstruct = input('\nElement(s) (use commas to separate elements): ').split(',')
            print(elements_to_reconstruct)

            if len(elements_to_reconstruct) == 0:
                print('\nNo elements selected. Please try again.')
                print('\nAvailable element(s): ' + ', '.join(elements))
                
                elements_to_reconstruct = input('\nElement(s) (use commas to separate elements): ').split(',')

                continue

            elements_to_reconstruct = [element.strip() for element in elements_to_reconstruct] # Remove whitespace from before and after each element

            for element_idx in range(len(elements_to_reconstruct)):
                if elements_to_reconstruct[element_idx] not in elements:
                    element_not_found = True
    
                    break
                
                if elements_to_reconstruct.count(elements_to_reconstruct[element_idx]) > 1:
                    multiple_elements_selected = True

                    break

            if element_not_found:
                print('\nOne or more elements not found in the aggregate XRF HDF5 file. Please try again.')
                print('\nAvailable element(s): ' + ', '.join(elements))
                    
                elements_to_reconstruct = input('\nElement(s) (use commas to separate elements): ').split(',')

                continue

            if multiple_elements_selected:
                print('One or more elements was selected more than once. Please try again.')
                print('\nAvailable element(s): ' + ', '.join(elements))

                elements_to_reconstruct = input('\nElement(s) (use commas to separate elements): ').split(',')

                continue

            select_elements_enabled = False
    
    elif premade_file_status == 'n' or premade_file_status == 'N':
        print('\nDo you want to create aggregate XRF and XRT HDF5 files using ' + synchrotron + ' data?')

        create_aggregate_file_status = input('\n(y/N): ')

        if create_aggregate_file_status == 'y' or create_aggregate_file_status == 'Y':
            select_individual_xrf_files_enabled = True
            select_individual_xrt_files_enabled = True
            
            while select_individual_xrf_files_enabled:
                input_xrf_file_paths = filedialog.askopenfilenames(parent = root, title = "Select individual XRF HDF5 files", filetypes = [("HDF5 files", "*.h5")])

                if input_xrf_file_paths == '':
                    print("\nNo files selected. Exiting...\n")
                    
                    exit()

                output_file_path = filedialog.asksaveasfilename(parent = root, title = "Save aggregate XRF HDF5 file", filetypes = [("HDF5 files", "*.h5")])
                
                if output_file_path == '':
                    print("\nNo output file location selected. Exiting...\n")
                    
                    exit()
                    
                try:
                    elements, counts, theta = create_aggregate_xrf_h5(input_xrf_file_paths, output_file_path, synchrotron)
                
                except:
                    print('\nError: Invalid input file(s). Please try again.\n')
                
                    continue
                
                select_individual_xrf_files_enabled = False

            while select_individual_xrt_files_enabled:
                input_xrt_file_paths = filedialog.askopenfilenames(parent = root, title = "Select individual XRT HDF5 files", filetypes = [("HDF5 files", "*.h5")])

                if input_xrt_file_paths == '':
                    print("\nNo files selected. Exiting...\n")

                    exit()

                try:
                    elements, counts, theta = create_aggregate_xrt_h5(input_xrt_file_paths, output_file_path, synchrotron)
                
                except:
                    print('\nError: Invalid input file(s). Please try again.\n')

                    continue
                
                select_individual_xrt_files_enabled = False

        select_elements_enabled = True
        
        while select_elements_enabled:
            element_not_found = False
            multiple_elements_selected = False
            
            print('\nType in which element(s) to reconstruct')
            print('\nAvailable element(s): ' + ', '.join(elements))

            elements_to_reconstruct = input('\nElement(s) (use commas to separate elements): ').split(',')
            print(elements_to_reconstruct)

            if len(elements_to_reconstruct) == 0:
                print('\nNo elements selected. Please try again.')
                print('\nAvailable element(s): ' + ', '.join(elements))
                
                elements_to_reconstruct = input('\nElement(s) (use commas to separate elements): ').split(',')

                continue

            elements_to_reconstruct = [element.strip() for element in elements_to_reconstruct] # Remove whitespace from before and after each element

            for element_idx in range(len(elements_to_reconstruct)):
                if elements_to_reconstruct[element_idx] not in elements:
                    element_not_found = True
    
                    break
                
                if elements_to_reconstruct.count(elements_to_reconstruct[element_idx]) > 1:
                    multiple_elements_selected = True

                    break

            if element_not_found:
                print('\nOne or more elements not found in the aggregate XRF HDF5 file. Please try again.')
                print('\nAvailable element(s): ' + ', '.join(elements))
                    
                elements_to_reconstruct = input('\nElement(s) (use commas to separate elements): ').split(',')

                continue

            if multiple_elements_selected:
                print('One or more elements was selected more than once. Please try again.')
                print('\nAvailable element(s): ' + ', '.join(elements))

                elements_to_reconstruct = input('\nElement(s) (use commas to separate elements): ').split(',')

                continue

            select_elements_enabled = False

        else:
            print("\nExiting...\n")
            exit()
    
    else:
        print("\nExiting...\n")
        exit()
    
    print("OK")
    exit() 

    reconstruct_jXRFT_tomography(**params)
    
    if rank == 0:
        output_folder = params["recon_path"]
        
        create_summary(output_folder, params)
