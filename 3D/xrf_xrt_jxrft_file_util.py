import numpy as np, \
       pandas as pd, \
       xrf_xrt_input_param_names as ipn, \
       h5py, \
       ast, \
       os

from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

def extract_csv_input_jxrft_recon_params(file_path, fluor_lines, dev):
    if not os.path.isfile(file_path):
        print('Error: Cannot locate input file path. Exiting program...', flush = True)

        comm.abort(1)
    
    if not file_path.endswith('.csv'):
        print('Error: Reconstruction input parameter file must be CSV. Exiting program...', flush = True)

        comm.abort(1)
        
    try:
        input_params_csv = pd.read_csv(file_path,
                                       delimiter = '=', 
                                       header = None, 
                                       names = ['input_param', 'value'],
                                       dtype = str,
                                       keep_default_na = False)

        input_params = input_params_csv['input_param']
        values = input_params_csv['value'].str.strip().replace('', 'None') # Extract values while setting non-existent values to None

    except KeyboardInterrupt:
        print('\n\nKeyboardInterrupt occurred. Exiting program...', flush = True)
            
        comm.abort(1)

    except:
        print('Error: Unable to read in reconstruction input parameter CSV file. Exiting program...', flush = True)

        comm.abort(1)
    
    all_params_ordered = pd.Series(ipn.preprocessing_params_ordered)

    if not input_params.equals(all_params_ordered):
        print('Error: At least one parameter missing or at least one parameter too many.', flush = True)
        print('\nThe following input parameters are required:\n', flush = True)
        
        for param in all_params_ordered:
            print(rank, f"     -'{param}'", flush = True)
            
        print('\n\rEnding program...', flush = True)

    # available_synchrotrons = ipn.available_synchrotrons
    available_noise_models = ipn.available_noise_models
    numeric_scalar_params = ipn.recon_numeric_scalar_params
    numeric_array_params = ipn.recon_numeric_array_params
    dict_params = ipn.recon_dict_params
    bool_params = ipn.recon_bool_params
    
    for idx, val in enumerate(values):
        if val.lower() == 'none':
            values[idx] = None

        elif val.lower() == 'true' or val.lower() == 'false':
            values[idx] = val.lower() == 'true'
        
        elif input_params[idx] in numeric_array_params:
            try:
                values[idx] = np.array(ast.literal_eval(val))
            
            except:
                print('Error: At least one reconstruction input parameter value cannot be converted to a NumPy array. Exiting program...', flush = True)

                comm.abort(1)
        
        elif input_params[idx] in dict_params:
            try:
                values[idx] = ast.literal_eval(val)
            
            except:
                print('Error: Cannot convert value of at least one parameter to dictionary. Exiting program...', flush = True)

                comm.abort(1)
        
        else:
            try:
                values[idx] = int(val)
            
            except:
                try:
                    values[idx] = float(val)
                
                except:
                    continue
    
    input_param_dict = dict(zip(input_params, values))

    # if input_param_dict['synchrotron'] is None or input_param_dict['synchrotron_beamline'] is None:
        # print('Error: Synchrotron and/or synchrotron beamline fields empty. Exiting program...', flush = True)

        # comm.abort(1)
    
    # synchrotron = input_param_dict['synchrotron'].lower()
    noise_model = input_param_dict['noise_model'].lower()

    if noise_model not in available_noise_models:
        print('Error: Noise model unavailable. Exiting program...', flush = True)

        comm.abort(1)

    # if synchrotron not in available_synchrotrons:
        # print('Error: Synchrotron unavailable. Exiting program...', flush = True)

        # comm.abort(1)

    if not all(isinstance(input_param_dict[param], bool) for param in bool_params):
        print('Error: The following input parameters must all be set to True or False:', flush = True)
        
        for param in bool_params:
            print(f"     -'{param}'", flush = True)
            
        print('\n\rExiting program...', flush = True)

        comm.abort(1)

    for param in numeric_scalar_params:
        if isinstance(input_param_dict.get(param), str):
            print(f'Error: Expected a number for input parameter \'{param}\', but got a string. Exiting program...', flush = True)

            comm.abort(1)

    input_param_dict['probe_energy_kev'] = np.array([input_param_dict['probe_energy_kev']])
    
    input_param_dict['dev'] = dev # Device (GPU, CPU, etc.)
    input_param_dict['fl_K'] = fluor_lines['K']
    input_param_dict['fl_L'] = fluor_lines['L']
    input_param_dict['fl_M'] = fluor_lines['M']

    return input_param_dict

def extract_h5_aggregate_xrf_xrt_data(file_path, opt_dens_enabled, **kwargs):
    if not os.path.isfile(file_path):
        print('Error: Cannot locate aggregate XRF, XRT HDF5 file. Exiting program...', flush = True)

        comm.abort(1)
    
    if not file_path.endswith('.h5'):
        print(rank, 'Error: Aggregate XRF, XRT file extension must be \'.h5\'. Exiting program...', flush = True)

        comm.abort(1)
    
    try:
        with h5py.File(file_path, 'r') as h5:
            data = h5['exchange/data']
            elements = h5['exchange/elements']
            
            elements_xrf, elements_xrt = elements['xrf'][()], elements['xrt'][()]
            xrf_data, xrt_data = data['xrf'][()], data['xrt'][()]
            theta = h5['exchange/theta'][()]
    
    except KeyboardInterrupt:
        print(rank, 'Keyboard interrupt. Exiting program...', flush = True)

        comm.abort(1)
    
    except:
        print(rank, 'Error: Incorrect HDF5 file structure. Exiting program...', flush = True)

        comm.abort(1)
    
    elements_xrf_string = [element.decode() for element in elements_xrf]
    elements_xrt_string = [element.decode() for element in elements_xrt]

    element_lines_roi = kwargs.get('element_lines_roi')

    if element_lines_roi is not None:
        _element_lines_roi = np.array(element_lines_roi)

        element_lines_roi_idx = np.zeros(len(element_lines_roi), dtype = int)
        
        for idx, element_line in enumerate(_element_lines_roi):
            if element_line[1] == 'K' and '_K' not in elements_xrf_string[idx]:
                element_line_pair = element_line[0] + '_K'
            
            else:
                element_line_pair = element_line[0]
            
            element_line_idx = np.argwhere(_element_lines_roi == element_line_pair)
            
            element_lines_roi_idx[idx] = element_line_idx
        
        xrf_data_roi = xrf_data[element_lines_roi_idx]
    
    else:
        element_lines_roi_idx = np.arange(len(elements_xrf_string))
        xrf_data_roi = xrf_data

    if opt_dens_enabled:
        xrt_data_idx = elements_xrt_string.index('opt_dens')
        opt_dens = xrt_data[xrt_data_idx]

        return element_lines_roi_idx, xrf_data_roi, opt_dens, theta
    
    xrt_data_idx = elements_xrt_string.index('xrt_sig')
    xrt_sig = xrt_data[xrt_data_idx]

    return element_lines_roi_idx, xrf_data_roi, xrt_sig, theta