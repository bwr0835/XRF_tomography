#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, \
       torch as tc, \
       xraylib as xlib, \
       file_util as futil, \
       sys, \
       warnings

from XRF_tomography import reconstruct_jXRFT_tomography
from mpi4py import MPI
from misc import create_summary

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


params_2_ide_samp = {'f_recon_parameters': 'recon_parameters.txt', # Text file name for list of reconstruction parameters
                     'dev': dev,
                     'synchrotron_enabled': True, # EXPERIMENTAL ONLY
                     'synchrotron': 'aps', # EXPERIMENTAL ONLY
                     'use_std_calibation': True, # Whether to use mass calibration standard during reconstruction
                     'probe_intensity': None, # ONLY FOR SIMULATIONS OR LACK OF CALIBRATION DATA
                     'std_path': './data/Cabead/axo_std', # File path for mass calibration standard
                     'f_std': 'axo_std.mda.h5', # File name for mass calibration standard
                     'std_element_lines_roi': np.array([['Ca', 'K'], 
                                                        ['Fe', 'K'], 
                                                        ['Cu', 'K']]), # Element flurescence lines used for mass calibration
                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1e-6, # Densities of desired elements in calibration standard (g/cm^2) (obtained from HDF$ file)
                     'fitting_method':'XRF_roi_plus', # Raw fluorescence spectrum fitting algorithm # TODO
                     'selfAb': False, # Self-absorption enabled
                     'cont_from_check_point': False,
                     'use_saved_initial_guess': False, # Use a saved initial guess of the object (may help with convergence)
                     'ini_kind': 'const', # Set ini_kind to 'const', 'rand' or 'randn' # TODO
                     'init_const': 0.0, # TODO
                     'ini_rand_amp': 0.1, # TODO
                     'recon_path': './data/Cabead_adjusted1_ds3_recon/Ab_F_nEl_6_nDpts_3_b1_1e4_b2_1e0_lr_1.0e-3', # Directory to reconstruction
                     'f_initial_guess': 'initialized_grid_concentration',
                     'f_recon_grid': 'grid_concentration', # Name of file that saves the most recent reconstructed result 
                     'data_path': './data/Cabead_adjusted1_ds4', # Directory to X-ray data
                     'f_XRF_data': 'cabead_xrf-fits', # File name for XRF data # TODO             
                     'f_XRT_data': 'cabead_scalers', # File name for XRT data (I think)
                     'scaler_counts_us_ic_dataset_idx':18, # TODO
                     'scaler_counts_ds_ic_dataset_idx':11, # TODO
                     'XRT_ratio_dataset_idx':21, # TODO
                     'theta_ls_dataset': 'exchange/theta', # TODO
                     'channel_names': 'exchange/elements', # TODO
                     'this_aN_dic': {"Si": 14, 
                                     "Ti": 22, 
                                     "Cr": 24, 
                                     "Fe": 26, 
                                     "Ni": 28, 
                                     "Ba": 56},
                     'element_lines_roi': np.array([['Si', 'K'], 
                                                    ['Ti', 'K'], 
                                                    ['Cr', 'K'],
                                                    ['Fe', 'K'], 
                                                    ['Ni', 'K'], 
                                                    ['Ba', 'L']]),  # np.array([["Si, K"], ["Ca, K"]])
                     'n_line_group_each_element': np.array([1, 1, 1, 1, 1, 1]),
                     'sample_size_n': 124, # Set to number of pixels along width of projection images when rotation axis is up-down TODO
                     'sample_height_n': 32, # Set to number of pixels along height of projection images when rotation axis is up-down (e.g. # of slices) TODO
                     'sample_size_cm': 5e-5, # Size of sample_size_n (in cm) along direction perpendicular to sample axis of rotation
                     'probe_energy_keV': np.array([13.0]), # Excitation energy (For 2-ID-E, we used 13 keV; for the HXN at BNL, we used 9.7 keV)
                     'n_epochs': 200, # Number of epochs
                     'save_every_n_epochs': 5,
                     'minibatch_size': 124,
                     'b1': 1.0e4, # Regularization prefactor of XRT cost term
                     'b2': 1.0, # Second prefactor inside of XRT cost term (from W. Di's MLEM cost function (2017)) # TODO No idea why she (Di) included that
                     'lr': 1.0e-3, # Learning rate TODO                         
                     'manual_det_coord': True,
                    #  'set_det_coord_cm': np.array([[0.70, 1.69, 0.70], # 
                                                #    [-0.70, 1.69, 0.70], 
                                                #    [-0.70, 1.69, -0.70]]), # np array with dimension (# of detecing points, 3) 
                                                                                    # 3 for (z, x, y) coordinates; probe propagates along +y axis; sample rotates about +z axis;
                                                                                    # The sign of x determines which side the detector locates relative to the sample TODO
                     'set_det_coord_cm': np.array([[0.70, 17.03513, 0.70], # x-coord. calculated when accounting for detector element obliquity
                                                   [-0.70, 17.03513, 0.70], 
                                                   [-0.70, 17.03513, -0.70]]),
                     'det_on_which_side': "positive", # Which side of each projection image the XRF detector is on TODO
                     'det_from_sample_cm': None, # SIMULATION ONLY
                     'det_ds_spacing_cm': None, # SIMULATION ONLY
                     'manual_det_area': True, # Experimental detector active area enable 
                    #  'det_area_eff_cm2': 1.68, # TOTAL experimental detector active area (cm^2)
                     'det_area_eff_cm2': 1.26,
                     'det_dia_cm': None, # SIMULATION ONLY
                     'P_folder': 'data/P_array/sample_124_124_32/Dis_1.69_manual_dpts_3', #       
                     'f_P': 'Intersecting_Length_124_124_32',
                     'fl_K': fl["K"], # doesn't need to change 
                     'fl_L': fl["L"], # doesn't need to change                    
                     'fl_M': fl["M"]}  # doesn't need to change

params = params_2_ide_samp

if __name__ == "__main__": 
    if rank == 0:
        if len(sys.argv) < 3 or len(sys.argv) > 4:
            print('Error: Must have exactly two program input arguments. Exiting program...')

            sys.exit()
        
        recon_param_file_path = sys.argv[1]

        params = futil.extract_csv_input_jxrft_recon_params(recon_param_file_path, fl, dev)
    
    reconstruct_jXRFT_tomography(**params)
    
    if rank == 0:
        output_folder = params["recon_path"]
        
        create_summary(output_folder, params)
