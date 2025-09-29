import numpy as np

from xrf_xrt_data_preprocessing_control_file import preprocess_xrf_xrt_data

preprocessing_inputs = {'synchrotron': 'aps',
                        'synchrotron_beamline': '2_ide',
                        'create_aggregate_xrf_xrt_files_enabled': True,
                        'xrf_aggregate_filepath': None,
                        'xrt_aggregate_filepath': None,
                        'mass_calibration_enabled': False,
                        'mass_calibration_dir': None,
                        'mass_calibration_filepath': None,
                        'mass_calibration_elements': np.array([['Ca', 'K'],
                                                               ['Fe', 'K'],
                                                               ['Cu', 'K']]),
                        'pre_existing_alignment_norm_mass_cal_file': False,
                        'iterative_reproj_enabled': True,
                        'norm_enabled': True}

if __name__ == '__main__':
    preprocess_xrf_xrt_data(**preprocessing_inputs)