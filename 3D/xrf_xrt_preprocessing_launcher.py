import numpy as np

from xrf_xrt_preprocessing_control_file import preprocess_xrf_xrt_data

preprocessing_inputs = {'synchrotron': 'nsls-ii',
                        'synchrotron_beamline': 'hxn',
                        'create_aggregate_xrf_xrt_files_enabled': False,
                        'aggregate_xrf_file_path': None,
                        'aggregate_xrt_file_path': None,
                        'pre_existing_align_norm_mass_calib_file_enabled': False,
                        'pre_existing_align_norm_mass_calib_file_path': None,
                        'norm_enabled': True,
                        'norm_method': 'incident_intensity_masking',
                        'I0_cts_per_s': None,
                        't_dwell_s': None,
                        'mass_calib_enabled': False,
                        'mass_calib_filepath': None,
                        'mass_calib_elements': np.array([['Ca', 'K'],
                                                         ['Fe', 'K'],
                                                         ['Cu', 'K']]),
                        'areal_mass_dens_mass_calib_elements_g_cm2': np.array([1.931, 0.504, 0.284])*1e-6,
                        'iter_reproj_enabled': True,
                        'n_iter_iter_reproj': 10,
                        'return_aux_data': True}

if __name__ == '__main__':
    preprocess_xrf_xrt_data(**preprocessing_inputs)