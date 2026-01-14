from mpi4py import MPI
from util import intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual

import numpy as np, sys

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

if len(sys.argv) == 1:
        print('Error: Input file argument required. Exiting program...')

        sys.exit()
    
if len(sys.argv) > 2:
        print('Error: More than one input argument detected. Exiting program...')

        sys.exit()


params_cabead_ds4 = {"n_ranks": n_ranks,
          "minibatch_size": 124,
          "rank": rank,
          "manual_det_coord": True, 
          "set_det_coord_cm": np.array([[0.70, 1.69, 0.70], [0.70, 1.69, -0.70], [-0.70, 1.69, 0.70], [-0.70, 1.69, -0.70]]),      
          "det_on_which_side": "positive",
          "manual_det_area": True,
          "det_size_cm": None,
          "det_from_sample_cm": None,
          "det_ds_spacing_cm": None, 
          "sample_size_n": 124,
          "sample_size_cm": 0.0248,
          'sample_height_n': 32,
          "P_folder": f'data/P_array/sample_124_124_32/Dis_1.69_manual_dpts_4',
          'f_P': 'Intersecting_Length_124_124_32'
}

params_cabead_ds4_2 = {"n_ranks": n_ranks,
                       "minibatch_size": 124,
                       "rank": rank,
                       "manual_det_coord": True, 
                       "set_det_coord_cm": np.array([[0.70, 1.69, 0.70], 
                                                     [-0.70, 1.69, 0.70], 
                                                     [-0.70, 1.69, -0.70]]),      
                       "det_on_which_side": "positive",
                       "manual_det_area": True,
                       "det_size_cm": None,
                       "det_from_sample_cm": None,
                       "det_ds_spacing_cm": None, 
                       "sample_size_n": 124,
                       "sample_size_cm": 0.0248,
                       'sample_height_n': 32,
                       "P_folder": f'data/P_array/sample_124_124_32/Dis_1.69_manual_dpts_3',
                       'f_P': 'Intersecting_Length_124_124_32'}



intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual(**params_cabead_ds4_2)