import os, sys, xrf_xrt_preprocess_file_util as futil

from xrf_xrt_preprocessing_control_file import preprocess_xrf_xrt_data

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Error: Input CSV file argument required. Exiting program...')

        sys.exit()
    
    if len(sys.argv) > 2:
        print('Error: More than one input argument detected. Exiting program...')

        sys.exit()
    
    input_param_file_path = sys.argv[1]

    if not os.path.isfile(input_param_file_path):
        print('Error: File does not exist. Exiting program...')

        sys.exit()

    print('Extracting XRF/XRT pre-processing input parameters...')

    preprocessing_inputs = futil.extract_csv_preprocessing_input_params(input_param_file_path)
    
    # a = 0
    # if not a:
    #     sys.exit()

    _ = preprocess_xrf_xrt_data(**preprocessing_inputs)