import numpy as np, \
       torch as tc, \
       util, \
       warnings, \
       sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()
warnings.filterwarnings("ignore")

#========================================================
# Set the device
#========================================================
# stdout_options = {'output_folder': recon_path, 'save_stdout': False, 'print_terminal': True}
gpu_index = rank % 2

sys.stdout.flush()
# gpu_index = 1
if tc.cuda.is_available():
    dev = tc.device('cuda:{}'.format(gpu_index))
    
    print("Process", rank, "running on", dev)
    
    sys.stdout.flush()

else:
    dev = "cpu"
    
    print("Process", rank, "running on CPU", sys.executable)
    
    sys.stdout.flush()

params_64_64_64_cabead_xrt = {'src_path': './data/sample8_size_64_pad/nElements_2/grid_concentration.npy',
                              'theta_st': 0,
                              'theta_end': 360,
                              'n_theta': 200,
                              'sample_height_n': 64, 
                              'sample_size_n': 64,
                              'sample_size_cm': 0.01,
                              'this_aN_dic': {'Ca': 20, 'Sc': 21},
                              'probe_energy_keV': np.array([20.0]),
                              'probe_cts': 1000000.0, # photons
                              'save_path': '/home/bwr0835',
                              'save_fname': 'simulated_proj_data_xrt_64_64_64',
                              'theta_sep': False,
                              'Poisson_noise': True,
                              'dev': 'cuda:0'}

params_64_64_64_cabead_xrf = {'n_ranks': n_ranks, 
                              'rank': rank, 
                              'P_folder': './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',
                              'f_P': 'Intersecting_Length_64_64_64', 
                              'theta_st': 0, 
                              'theta_end': 360, 
                              'n_theta': 200, 
                              'src_path': './data/sample8_size_64_pad/nElements_2/grid_concentration.npy', 
                              'det_size_cm': 0.9, 
                              'det_from_sample_cm': 1.6, 
                              'det_ds_spacing_cm': 0.4, 
                              'sample_size_n': 64,
                              'sample_size_cm': 0.01, 
                              'sample_height_n': 64, 
                              'this_aN_dic': {'Ca': 20, 'Sc': 21},
                              'probe_cts': 1000000.0, # photons
                              'probe_energy_keV': np.array([20.0]),
                              'save_path': '/home/bwr0835', 
                              'save_fname': 'simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64', 
                              'Poisson_noise': True, 
                              'dev': dev,
                              'probe_att': False,
                              'selfAb': False}

if __name__ == "__main__":
    if sys.argv[1] == 'xrt':
        util.create_XRT_data_3d(**params_64_64_64_cabead_xrt)
    
    elif sys.argv[1] == 'xrf':
        util.create_XRF_data_3d(**params_64_64_64_cabead_xrf)
    
    else:
        comm.Abort(1)