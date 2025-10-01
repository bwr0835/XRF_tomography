import numpy as np, \
       tkinter as tk, \
       file_util as futil, \
       xrf_xrt_preprocess_utils as ppu, \
       sys, \
       os

from tkinter import filedialog as fd
from realignment_final import iter_reproj as irprj

def preprocess_xrf_xrt_data(synchrotron,
                            synchrotron_beamline,
                            create_aggregate_xrf_xrt_files_enabled,
                            aggregate_xrf_file_path,
                            aggregate_xrt_file_path,
                            pre_existing_align_norm_mass_calib_file_enabled,
                            pre_existing_align_norm_mass_calib_file_path,
                            norm_enabled,
                            norm_method,
                            I0_cts_per_s,
                            t_dwell_s,
                            mass_calibration_enabled,
                            mass_calib_dir,
                            mass_calib_filepath,
                            mass_calib_elements,
                            areal_mass_dens_mass_calib_elements_g_cm2,
                            iterative_reproj_enabled,
                            n_iter_iter_reproj,
                            return_aux_data):

    available_synchrotrons = ['aps', 'nsls-ii']
    
    if synchrotron == '' or synchrotron_beamline == '':
        print('Error: Synchrotron and/or synchrotron beamline fields empty. Exiting program...')

        sys.exit()
    
    synchrotron = synchrotron.lower()

    if synchrotron not in available_synchrotrons:
        print('Error: Synchrotron unavailable. Exiting program...')

        sys.exit()

    if create_aggregate_xrf_xrt_files_enabled:
        # xrf_file_array = ['/raid/users/roter/Jacobsen/img.dat/2xfm_0030.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0031.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0032.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0033.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0034.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0117.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0118.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0037.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0038.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0039.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0040.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0047.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0042.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0048.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0049.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0050.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0051.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0052.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0053.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0054.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0055.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0056.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0057.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0058.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0059.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0060.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0094.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0095.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0096.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0097.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0098.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0099.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0100.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0101.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0102.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0103.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0104.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0105.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0106.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0107.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0108.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0109.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0110.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0111.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0112.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0113.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0114.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0115.mda.h5',
        #                   '/raid/users/roter/Jacobsen/img.dat/2xfm_0116.mda.h5']
        
        xrf_file_array = ['/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235324.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235327.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235330.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235333.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235336.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235339.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235342.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235345.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235348.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235351.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235354.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235358.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235364.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235374.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235377.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235386.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235389.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235392.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235395.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235398.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235401.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235404.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235407.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235410.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235413.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235416.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235419.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235422.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235425.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235428.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235431.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235437.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235440.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235443.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235446.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235449.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235452.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235455.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235458.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235461.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235464.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235467.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235470.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235473.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235476.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235479.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235482.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235485.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235488.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235491.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235494.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235497.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235500.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235503.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235506.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235509.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235515.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235518.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235546.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235549.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235552.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235555.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235558.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235561.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235564.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235567.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235570.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235573.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235576.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235579.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235582.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235585.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235588.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235592.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235598.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235601.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235604.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235607.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235610.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235613.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235616.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235619.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235622.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235625.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235628.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235631.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235637.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235640.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235643.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235646.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235649.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235652.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235655.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235658.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235664.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235667.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235670.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235673.h5']
        xrt_file_array = ['/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235324.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235327.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235330.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235333.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235336.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235339.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235342.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235345.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235348.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235351.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235354.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235358.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235364.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235374.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235377.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235386.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235389.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235392.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235395.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235398.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235401.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235404.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235407.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235410.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235413.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235416.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235419.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235422.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235425.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235428.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235431.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235437.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235440.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235443.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235446.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235449.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235452.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235455.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235458.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235461.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235464.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235467.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235470.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235473.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235476.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235479.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235482.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235485.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235488.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235491.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235494.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235497.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235500.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235503.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235506.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235509.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235515.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235518.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235546.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235549.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235552.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235555.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235558.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235561.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235564.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235567.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235570.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235573.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235576.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235579.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235582.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235585.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235588.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235592.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235598.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235601.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235604.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235607.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235610.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235613.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235616.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235619.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235622.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235625.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235628.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235631.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235637.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235640.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235643.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235646.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235649.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235652.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235655.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235658.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235664.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235667.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235670.h5',
                          '/raid/users/roter/Jacobsen-nslsii/data/ptycho/h5_data/scan_235673.h5']

        # xrt_file_array = xrf_file_array.copy()
        # root = tk.Tk()

        # xrf_file_array = fd.askopenfilenames(parent = root, title = "Choose XRF files to aggregate.", filetypes = [('HDF5 files', '*.h5')])
        # xrt_file_array = fd.askopenfilenames(parent = root, title = "Choose XRT files to aggregate.", filetypes = [('HDF5 files', '*.h5')])

        # if xrf_file_array == '' or xrt_file_array == '':
        #     print('Error: XRF and/or XRT filename array empty. Exiting program...')
            
        #     sys.exit()

        xrf_array_dir = os.path.dirname(xrf_file_array[0])
        xrt_array_dir = os.path.dirname(xrt_file_array[0])

        output_xrf_filepath = os.path.join(xrf_array_dir, f'{synchrotron_beamline}_aggregate_xrf.h5')
        output_xrt_filepath = os.path.join(xrt_array_dir, f'{synchrotron_beamline}_aggregate_xrt.h5')

        print('Creating aggregate XRF data file...')

        if synchrotron == 'aps':
            futil.create_aggregate_xrf_h5(xrf_file_array, 
                                        output_xrf_filepath, 
                                        synchrotron)

            print('Creating aggregate XRT data file...')

            futil.create_aggregate_xrt_h5(xrt_file_array, 
                                        output_xrt_filepath, 
                                        synchrotron)
        
        elif synchrotron == 'nsls-ii':
            us_ic = futil.create_aggregate_xrf_h5(xrf_file_array, 
                                                  output_xrf_filepath, 
                                                  synchrotron, 
                                                  us_ic_enabled = True) # us_ic_array only returned since that data is not present in NSLS-II ptychography files

            print('Creating aggregate XRT data file...')

            futil.create_aggregate_xrt_h5(xrt_file_array, 
                                          output_xrt_filepath, 
                                          synchrotron,
                                          us_ic = us_ic)
            
        sys.exit()

    else:
        elements_xrf, counts_xrf, theta, raw_spectrum_fitting_method, dataset_type = futil.extract_h5_aggregate_xrf_data(aggregate_xrf_file_path)
        elements_xrt, counts_xrt, _, _, dataset_type = futil.extract_h5_aggregate_xrt_data(aggregate_xrt_file_path)

        aggregate_xrf_file_dir = os.path.dirname(aggregate_xrf_file_path)
        aggregate_xrt_file_dir = os.path.dirname(aggregate_xrt_file_path)

        n_elements, n_theta, n_slices, n_columns = counts_xrf.shape

        if (n_slices % 2) or (n_columns % 2):
            if (n_slices % 2) and (n_columns % 2):
                print('Odd number of slices (rows) and scan positions (columns) detected. Padding one additional slice and scan position column to XRF and XRT data...')

                counts_xrt = ppu.pad_col_row(counts_xrt, 'xrt')
                counts_xrf = ppu.pad_col_row(counts_xrf, 'xrf')
            
                n_slices += 1
                n_columns += 1
        
            elif n_slices % 2:
                print('Odd number of slices (rows) detected. Padding one additional slice to XRF and XRT data...')
                
                counts_xrt = ppu.pad_row(counts_xrt, 'xrt')
                counts_xrf = ppu.pad_row(counts_xrf, 'xrf')

                n_slices += 1

            else:
                print('Odd number of scan positions (columns) detected. Padding one additional scan position column to XRF and XRT data...')
                
                counts_xrt = ppu.pad_col(counts_xrt, 'xrt')
                counts_xrf = ppu.pad_col(counts_xrf, 'xrf')

                n_columns += 1

        if pre_existing_align_norm_mass_calib_file_enabled:
            norm_array, \
            net_x_shift_array, \
            net_y_shift_array, \
            I0_norm_cts, \
            I0_calibrated_cts = futil.extract_norm_mass_calibration_net_shift_data(pre_existing_align_norm_mass_calib_file_path, theta)

            print('Applying pre-existing per-projection normalizations to XRF, XRT arrays...')

            counts_xrf *= norm_array
            counts_xrt *= norm_array

            print('Calculating optical densities...')

            if np.array_equiv(norm_array, np.ones(n_theta)):            
                if I0_cts_per_s is not None and I0_cts_per_s > 0 and t_dwell_s is not None and t_dwell_s > 0:
                    I0_cts = I0_cts_per_s*t_dwell_s
                
                else:
                    print('Error: \'I0_cts_per_s\' and \'t_dwell_s\' must be positive values. Exiting program...')

                    sys.exit()
            
            else:
                I0_cts = I0_norm_cts
            
            opt_dens = -np.log(counts_xrt/I0_cts)
            
            if iterative_reproj_enabled:
                if return_aux_data:
                    aligned_proj_final_xrt, \
                    aligned_proj_final_opt_dens, \
                    aligned_proj_final_xrf, \
                    net_x_shifts_pcc_final, \
                    net_y_shifts_pcc_final, \
                    aligned_exp_proj_array, \
                    synth_proj_array, \
                    pcc_2d_array, \
                    recon_array, \
                    net_x_shifts_pcc_new, \
                    net_y_shifts_pcc_new, \
                    dx_array_new, \
                    dy_array_new = irprj(counts_xrt,
                                         opt_dens,
                                         counts_xrf,
                                         theta,
                                         I0_cts,
                                         n_iter_iter_reproj,
                                         return_aux_data = True)

                else:
                    aligned_proj_final_xrt, \
                    aligned_proj_final_opt_dens, \
                    aligned_proj_final_xrf, \
                    net_x_shifts_pcc_final, \
                    net_y_shifts_pcc_final = irprj(counts_xrt,
                                                   opt_dens,
                                                   counts_xrf,
                                                   theta,
                                                   I0_cts,
                                                   n_iter_iter_reproj)

        else:
            if norm_enabled:
                if norm_method == 'per_proj_mask':
                    if return_aux_data:
                        counts_xrt_norm, counts_xrf_norm, norm_array, I0_norm_cts, conv_mag_array = ppu.joint_fluct_norm(counts_xrt, counts_xrf, return_conv_mag_array = True)

                    else:
                        counts_xrt_norm, counts_xrf_norm, norm_array, I0_norm_cts = ppu.joint_fluct_norm(counts_xrt, counts_xrf)

                    opt_dens = -np.log10(counts_xrt_norm/I0_norm_cts)

            else:
                norm_array = np.ones(n_theta)
                
                if I0_cts_per_s is None or I0_cts_per_s < 0 or t_dwell_s is None or t_dwell_s < 0:
                    print('Error: Incident photon flux and dwell time must be positive values. Exiting program...')

                    sys.exit()
                
                opt_dens = -np.log10(counts_xrt*t_dwell_s/I0_cts_per_s)

        if mass_calibration_enabled:
            if synchrotron == 'aps':
                pass