import os, sys, file_util as futil

from xrf_xrt_preprocessing_control_file import preprocess_xrf_xrt_data

preprocessing_inputs = {'synchrotron': None,
                        'synchrotron_beamline': None,
                        'create_aggregate_xrf_xrt_files_enabled': None,
                        'aggregate_xrf_file_path': None,
                        'aggregate_xrt_file_path': None,
                        'pre_existing_align_norm_file_enabled': None,
                        'pre_existing_align_norm_file_path': None,
                        'norm_enabled': None,
                        'norm_method': None,
                        'I0_cts_per_s': None,
                        't_dwell_s': None,
                        'n_iter_iter_reproj': None,
                        'return_aux_data': None,
                        'output_dir_path': None}

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Error: Input file argument required. Exiting program...')

        sys.exit()
    
    if len(sys.argv) > 2:
        print('Error: More than one input argument detected. Exiting program...')

        sys.exit()
    
    input_param_file_path = sys.argv[1]

    if not os.path.isfile(input_param_file_path):
        print('Error: File does not exist. Exiting program...')

        sys.exit()

    preprocessing_inputs = futil.extract_csv_preprocessing_input_params(input_param_file_path)

    preprocess_xrf_xrt_data(**preprocessing_inputs)