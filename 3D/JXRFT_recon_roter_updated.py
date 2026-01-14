#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, \
       torch as tc, \
       xraylib as xlib, \
       xrl_fluorline_macros, \
       xrf_xrt_jxrft_file_util as futil, \
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

fl = xrl_fluorline_macros.fl

if __name__ == "__main__": 
    if rank == 0:
        n_input_arg = len(sys.argv) - 1 # Number of command line input arguments
        
        if n_input_arg != 1:
            print('Error: Must have exactly one program input argument. Exiting program...', flush = True)

            comm.Abort()
        
        recon_param_file_path = sys.argv[1]

        params = futil.extract_csv_input_jxrft_recon_params(recon_param_file_path, fl, dev)
    
    else:
        params = None
    
    comm.bcast(params) # Broadcast parameter dictionary onto nonzero ranks so each rank has that information

    reconstruct_jXRFT_tomography(**params)
    
    if rank == 0:
        output_folder = params["recon_path"]
        
        create_summary(output_folder, params)
