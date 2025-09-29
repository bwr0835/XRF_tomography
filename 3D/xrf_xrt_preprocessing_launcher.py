import numpy as np

from xrf_xrt_data_preprocessing_control_file import preprocess_xrf_xrt_data

preprocessing_inputs = {'synchrotron': 'aps',
                        'synchrotron_beamline': '2_ide',
                        'create_aggregate_xrf_xrt_files_enabled': True,
                        'pre_existing_alignment_norm_mass_calibration_file_enabled': False,
                        'pre_existing_alignment_norm_mass_calibration_file_path': None,
                        'norm_enabled': True,
                        'mass_calibration_enabled': False,
                        'mass_calibration_dir': None,
                        'mass_calibration_filepath': None,
                        'mass_calibration_elements': np.array([['Ca', 'K'],
                                                               ['Fe', 'K'],
                                                               ['Cu', 'K']]),
                        'areal_mass_density_mass_calibration_elements_g_cm2': np.array([1.931, 0.504, 0.284])*1e-6,
                        'iterative_reproj_enabled': True}

if __name__ == '__main__':
    preprocess_xrf_xrt_data(**preprocessing_inputs)