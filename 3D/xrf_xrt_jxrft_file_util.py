import numpy as np, \
       pandas as pd, \
       xrf_xrt_input_param_names as ipn, \
       csv, \
       h5py, \
       ast, \
       os

from mpi4py import MPI
from misc import print_flush_root

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

def extract_csv_input_jxrft_recon_params(file_path, fluor_lines, dev):
    if not os.path.isfile(file_path):
        print_flush_root(rank, 'Error: Cannot locate input file path. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()
    
    if not file_path.endswith('.csv'):
        print_flush_root(rank, 'Error: Reconstruction input parameter file must be CSV. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()
        
    try:
        input_params_csv = pd.read_csv(file_path,
                                       delimiter = ':', 
                                       header = None, 
                                       names = ['input_param', 'value'],
                                       dtype = str,
                                       keep_default_na = False)

        input_params = input_params_csv['input_param']
        values = input_params_csv['value'].str.strip().replace('', 'None') # Extract values while setting non-existent values to None

    except KeyboardInterrupt:
        print_flush_root(rank, '\n\nKeyboardInterrupt occurred. Exiting program...', save_stdout = False, print_terminal = True)
            
        comm.Abort()

    except:
        print_flush_root(rank, 'Error: Unable to read in reconstruction input parameter CSV file. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()
    
    all_params_ordered = pd.Series(ipn.preprocessing_params_ordered)

    if not input_params.equals(all_params_ordered):
        print_flush_root(rank, 'Error: At least one parameter missing or at least one parameter too many.', save_stdout = False, print_terminal = True)
        print_flush_root(rank, '\nThe following input parameters are required:\n', save_stdout = False, print_terminal = True)
        
        for param in all_params_ordered:
            print_flush_root(rank, f"     -'{param}'", save_stdout = False, print_terminal = True)
            
        print_flush_root(rank, '\n\rEnding program...', save_stdout = False, print_terminal = True)

    available_synchrotrons = ipn.available_synchrotrons
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
                print_flush_root(rank, 'Error: At least one reconstruction input parameter value cannot be converted to a NumPy array. Exiting program...', save_stdout = False, print_terminal = True)

                comm.Abort()
        
        elif input_params[idx] in dict_params:
            try:
                values[idx] = ast.literal_eval(val)
            
            except:
                print_flush_root(rank, 'Error: Cannot convert value of at least one parameter to dictionary. Exiting program...', save_stdout = False, print_terminal = True)

                comm.Abort()
        
        else:
            try:
                values[idx] = int(val)
            
            except:
                try:
                    values[idx] = float(val)
                
                except:
                    continue
    
    input_param_dict = dict(zip(input_params, values))

    if input_param_dict['synchrotron'] is None or input_param_dict['synchrotron_beamline'] is None:
        print_flush_root(rank, 'Error: Synchrotron and/or synchrotron beamline fields empty. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()
    
    synchrotron = input_param_dict['synchrotron'].lower()
    noise_model = input_param_dict['noise_model'].lower()

    if noise_model not in available_noise_models:
        print_flush_root(rank, 'Error: Noise model unavailable. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()

    if synchrotron not in available_synchrotrons:
        print_flush_root(rank, 'Error: Synchrotron unavailable. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()

    if not all(isinstance(input_param_dict[param], bool) for param in bool_params):
        print_flush_root(rank, 'Error: The following input parameters must all be set to True or False:', save_stdout = False, print_terminal = True)
        
        for param in bool_params:
            print_flush_root(rank, f"     -'{param}'", save_stdout = False, print_terminal = True)
            
        print_flush_root(rank, '\n\rExiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()

    for param in numeric_scalar_params:
        if isinstance(input_param_dict.get(param), str):
            print_flush_root(rank, f'Error: Expected a number for input parameter \'{param}\', but got a string. Exiting program...', save_stdout = False, print_terminal = True)

            comm.Abort()

    input_param_dict['probe_energy_kev'] = np.array([input_param_dict['probe_energy_kev']])
    
    if input_param_dict.get('downsample_factor') is None:
        input_param_dict['downsample_factor'] = 1
    
    if input_param_dict.get('upsample_factor') is None:
        input_param_dict['upsample_factor'] = 1

    input_param_dict['dev'] = dev # Device (GPU, CPU, etc.)
    input_param_dict['fl_K'] = fluor_lines['K']
    input_param_dict['fl_L'] = fluor_lines['L']
    input_param_dict['fl_M'] = fluor_lines['M']

    return input_param_dict

def extract_h5_aggregate_xrf_xrt_data(file_path, **kwargs):

    if not os.path.isfile(file_path):
        print_flush_root(rank, 'Error: Cannot locate aggregate XRF, XRT HDF5 file. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()
    
    if not file_path.endswith('.h5'):
        print_flush_root(rank, 'Error: Aggregate XRF, XRT file extension must be \'.h5\'. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()
    
    try:
        with h5py.File(file_path, 'r') as h5:
            elements_xrf = h5['exchange/elements_xrf'][()]
            elements_xrt = h5['exchange/elements_xrt'][()]
            xrf_data = h5['exchange/data_xrf'][()]
            xrt_data = h5['exchange/data_xrt'][()]
            theta = h5['exchange/theta'][()]
    
    except KeyboardInterrupt:
        print_flush_root(rank, 'Keyboard interrupt. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()
    
    except:
        print_flush_root(rank, 'Error: Incorrect HDF5 file structure. Exiting program...', save_stdout = False, print_terminal = True)

        comm.Abort()
    
    element_lines_roi = kwargs.get('element_lines_roi')

    elements_xrf_string = [element.decode() for element in elements_xrf]
    elements_xrt_string = [element.decode() for element in elements_xrt]

    if element_lines_roi is not None:
        try:
            element_lines_roi_idx = np.array([elements_xrf_string.index(element) for element in element_lines_roi[0]])
        
        except KeyboardInterrupt:
            print_flush_root(rank, 'Keyboard interrupt. Exiting program...', save_stdout = False, print_terminal = True)

            comm.Abort()
        
        except:
            print_flush_root(rank, 'Error: Unable to parse elements and/or line(s) of interest. Exiting program...', save_stdout = False, print_terminal = True)

            comm.Abort()

        xrf_data_roi = xrf_data[element_lines_roi_idx]
    
    else:
        element_lines_roi_idx = np.arange(len(elements_xrf_string))
        xrf_data_roi = xrf_data

    opt_dens_idx = elements_xrt_string.index('opt_dens')
    opt_dens = xrt_data[opt_dens_idx]

    return element_lines_roi_idx, xrf_data_roi, opt_dens, theta

