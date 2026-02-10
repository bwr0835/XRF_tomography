import numpy as np, \
       pandas as pd, \
       tkinter as tk, \
       xrf_xrt_input_param_names as ipn, \
       csv, \
       h5py, \
       ast, \
       os, \
       sys

from tkinter import filedialog as fd
from matplotlib import pyplot as plt
from imageio import v2 as iio2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

def extract_h5_xrf_xrt_data_file_lists_tk(synchrotron):
    root = tk.Tk()
    
    root.withdraw()
    
    if synchrotron == 'aps':
        xrf_file_array = list(fd.askopenfilenames(parent = root, title = "Choose XRF files to aggregate.", filetypes = [('HDF5 files', '*.h5')])) # List so .copy() can be used

        if xrf_file_array == '':
            print('Error: XRF/XRT filename array empty. Exiting program...')

            sys.exit()
        
        xrt_file_array = xrf_file_array.copy()

    else:
        xrf_file_array = fd.askopenfilenames(parent = root, title = "Choose XRF files to aggregate.", filetypes = [('HDF5 files', '*.h5')])
        xrt_file_array = fd.askopenfilenames(parent = root, title = "Choose XRT files to aggregate.", filetypes = [('HDF5 files', '*.h5')])

        if xrf_file_array == '' or xrt_file_array == '':
            print('Error: XRF and/or XRT filename array empty. Exiting program...')
            
        sys.exit()

    root.destroy()

    return xrf_file_array, xrt_file_array

def extract_h5_xrf_data(file_path, synchrotron, **kwargs):
    if not os.path.isfile(file_path):
        print('Error: HDF5 file path cannot be found. Exiting program...')

        sys.exit()

    if not file_path.endswith('.h5'):
        print('Error: File must be HDF5. Exiting program...')

        sys.exit()

    h5 = h5py.File(file_path, 'r')
    
    if synchrotron == 'aps':
        try:
            if "MAPS/XRF_Analyzed/NNLS" in h5.keys():
                counts_h5 = h5['MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec']
                elements_h5 = h5['MAPS/XRF_Analyzed/NNLS/Channel_Names']
            
            extra_pvs_h5 = h5['MAPS/Scan/Extra_PVs']
            
            nx_h5 = h5['MAPS/Scan/x_axis']
            ny_h5 = h5['MAPS/Scan/y_axis']
                
            counts = counts_h5[()]
            elements = elements_h5[()]
            extra_pvs_names = extra_pvs_h5['Names'][()]
            extra_pvs_values = extra_pvs_h5['Values'][()]

            nx_conv = ny_h5[()] # Width and height are reversed in the actual HDF5 data structure
            ny_conv = nx_h5[()] # Width and height are reversed in the actual HDF5 data structure

            h5.close()

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
            sys.exit()

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()    

        theta_idx = np.where(extra_pvs_names == b'2xfm:m58.VAL')[0]
        
        theta = float(extra_pvs_values[theta_idx][0].decode()) # Get the value of theta and decode it to a float (from a byte)

        nx = len(nx_conv)
        ny = len(ny_conv) - 2 # MAPS tacks on two extra values for whatever reason
                
        nx, ny = ny, nx
        
        elements_entries_to_ignore = [b'Ar_Ar',
                                      b'Si_Si',
                                      b'Fe_Fe', 
                                      b'COMPTON_AMPLITUDE', 
                                      b'COHERENT_SCT_AMPLITUDE', 
                                      b'Num_Iter', 
                                      b'Fit_Residual', 
                                      b'Total_Fluorescence_Yield',
                                      b'Sum_Elastic_Inelastic']
            
        counts_new = np.zeros((len(elements), ny, nx))
                
        idx_to_delete = []

        for element in elements:
                element_index = np.ndarray.item(np.where(elements == element)[0])
                    
                if element not in elements_entries_to_ignore:
                    counts_new[element_index] = counts[element_index, :, :-2] # MAPS tacks on two extra columns of zeroes post-scan for whatever reason
                    
                else:
                    idx_to_delete.append(element_index)

        counts_new = np.delete(counts_new, idx_to_delete, axis = 0) # Delete array elements corresponding to ignored element entries
        elements = np.delete(elements, idx_to_delete, axis = 0)

        # Get corresponding pixel spacings and convert from mm to cm
      
        dx_cm = 1e-1*np.abs([-1] - nx_conv[0])/(nx - 1)
        dy_cm = 1e-1*np.abs(ny_conv[-3] - ny_conv[0])/(ny - 1)

        elements_string = [element.decode() for element in elements] # Convert the elements array from a list of bytes to a list of strings

        return elements_string, counts_new, theta, nx, ny, dx_cm, dy_cm
        
    elif synchrotron == 'nsls-ii':
        try:
            counts_h5 = h5['xrfmap/detsum/xrf_fit']
            elements_h5 = h5['xrfmap/detsum/xrf_fit_name']
            axis_coords_h5 = h5['xrfmap/positions/pos']
            theta_h5 = h5['xrfmap/scan_metadata'].attrs['param_theta'] # Metadata stored as key-value pairs (attributes) (similar to a Python dictionary)

            if kwargs.get('us_ic_enabled') == True:
                scalers_names_h5 = h5['xrfmap/scalers/name']
                scalers_h5 = h5['xrfmap/scalers/val']

                scalers_names = scalers_names_h5[()]
                scalers = scalers_h5[()]

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()

        elements = elements_h5[()]
        counts = counts_h5[()]
        axis_coords = axis_coords_h5[()]
        theta = 1e-3*theta_h5[()] # Convert from mdeg to deg

        h5.close()

        x_um = axis_coords[0]
        y_um = axis_coords[1]

        nx = np.shape(x_um)[1]
        ny = np.shape(y_um)[0]

        elements_entries_to_ignore = [b'compton',
                                      b'elastic',
                                      b'snip_bkg',
                                      b'r_factor',
                                      b'sel_cnt',
                                      b'total_cnt']

        idx_to_delete = []

        for element in elements:
            element_index = np.ndarray.item(np.where(elements == element)[0])

            if element in elements_entries_to_ignore:
                idx_to_delete.append(element_index)
                
        counts = np.delete(counts, idx_to_delete, axis = 0)
        elements = np.delete(elements, idx_to_delete, axis = 0)
        
        dx_cm = 1e-4*(np.abs(x_um[0, -1] - x_um[0, 0])/(nx - 1)) # Convert from µm to cm
        dy_cm = 1e-4*(np.abs(y_um[-1, 0] - y_um[0, 0])/(ny - 1)) # Convert from µm to cm
            
        elements_string = [_str.split('_')[0] if '_K' in _str else _str for _str in (element.decode() for element in elements)] # Convert the elements array from a list of bytes to a list of strings

        if kwargs.get('us_ic_enabled') == True:
            us_ic_index = np.ndarray.item(np.where(scalers_names == b'sclr1_ch4')[0]) # Ion chamber upstream of zone plate, but downstream of X-ray beam slits
            
            us_ic = scalers[:, :, us_ic_index]

            return elements_string, counts, us_ic, theta, nx, ny, dx_cm, dy_cm

        else:
            return elements_string, counts, theta, nx, ny, dx_cm, dy_cm

def extract_h5_xrt_data(file_path, synchrotron, **kwargs):
    if not os.path.isfile(file_path):
        print('Error: HDF5 file path cannot be found. Exiting program...')

        sys.exit()

    if not file_path.endswith('.h5'):
        print('Error: File must be HDF5. Exiting program...')

        sys.exit()
    
    h5 = h5py.File(file_path, 'r')

    if synchrotron == 'aps':
        try:
            scalers_h5 = h5['MAPS/Scalers']
            extra_pvs_h5 = h5['MAPS/Scan/Extra_PVs']
            nx_h5 = h5['MAPS/Scan/x_axis']
            ny_h5 = h5['MAPS/Scan/y_axis']
        
            scaler_names = scalers_h5['Names'][()]
            scaler_values = scalers_h5['Values'][()]
            extra_pvs_names = extra_pvs_h5['Names'][()]
            extra_pvs_values = extra_pvs_h5['Values'][()]
        
        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
            sys.exit()

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()
        
        nx_conv = ny_h5[()] # Width and height are reversed in the actual HDF5 data structure
        ny_conv = nx_h5[()] # Width and height are reversed in the actual HDF5 data structure
        
        h5.close()

        # elements = ['empty', 'us_ic', 'ds_ic', 'abs_ic']
        elements = ['empty', 'us_ic', 'xrt_sig', 'empty']
        n_elements = len(elements)

        nx = len(nx_conv)
        ny = len(ny_conv) - 2 # MAPS tacks on two extra values for whatever reason
        
        try:
            us_ic_idx = np.where(scaler_names == b'US_IC')[0][0] # The second [0] converts the 1-element array into a scalar
            ds_ic_idx = np.where(scaler_names == b'DS_IC')[0][0]
            # abs_ic_idx = np.where(scaler_names == b'abs_ic')[0][0]
            theta_idx = np.where(extra_pvs_names == b'2xfm:m58.VAL')[0]
        
        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
            sys.exit()

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()

        nx, ny = ny, nx
        
        cts_combined = np.zeros((n_elements, ny, nx))
        
        cts_us_ic = scaler_values[us_ic_idx] 
        cts_ds_ic = scaler_values[ds_ic_idx]

        cts_combined[1] = cts_us_ic[:, :-2] # Remove last two columns since they are added after row is finished scanning
        cts_combined[2] = cts_ds_ic[:, :-2]

        dx_cm = 1e-1*np.abs([-1] - nx_conv[0])/(nx - 1)
        dy_cm = 1e-1*np.abs(ny_conv[-3] - ny_conv[0])/(ny - 1)

        theta = float(extra_pvs_values[theta_idx][0].decode())

        return elements, cts_combined, theta, nx, ny, dx_cm, dy_cm
    
    elif synchrotron == 'nsls-ii':
        # STXM calculations adapted from X. Huang, Brookhaven National Laboratory
        try:
            diffract_map_intensity = h5['diffamp'][()]
            theta = h5['angle'][()]

            dx_cm, dy_cm = h5['dr_x'][()], h5['dr_y'][()] # These are supposed to be different than for XRF due to ptychography requiring overlapping positions

            h5.close()

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
            sys.exit()

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()

        nx = kwargs.get('nx')
        ny = kwargs.get('ny')

        cts_stxm = diffract_map_intensity.sum(axis = (2, 1)) # Sum over axis = 2, then sum over axis = 1
        cts_stxm = cts_stxm.reshape((ny, nx))

        elements = ['empty', 'us_ic', 'xrt_sig', 'empty']
        n_elements = len(elements)
        
        cts_combined = np.zeros((n_elements, ny, nx))

        cts_combined[2] = cts_stxm

        return elements, cts_combined, theta, nx, ny, dx_cm, dy_cm

def create_aggregate_xrf_h5(file_path_array, 
                            output_h5_file, 
                            synchrotron,
                            sample_flipped_remounted_mid_experiment,
                            **kwargs):

    n_theta = len(file_path_array)

    theta_array = np.zeros(n_theta) 

    elements, counts, theta, nx, ny, _, _ = extract_h5_xrf_data(file_path_array[0], synchrotron) # Invoke the first time for getting the number of elements and the number of pixels
    
    n_elements = len(elements)
    
    counts_array = np.zeros((n_elements, n_theta, ny, nx))

    if synchrotron == 'nsls-ii' and kwargs.get('us_ic_enabled') == True:
        us_ic_array = np.zeros((n_theta, ny, nx))

    for theta_idx, file_path in enumerate(file_path_array):
        if theta_idx != len(file_path_array) - 1:
            print(f'\rHDF file {theta_idx + 1}/{len(file_path_array)} extracted', end = '', flush = True)
        
        else:
            print(f'\rHDF file {theta_idx + 1}/{len(file_path_array)} extracted', flush = True)
        
        if synchrotron != 'nsls-ii':
            elements_new, counts, theta, nx_new, ny_new, _, _ = extract_h5_xrf_data(file_path, synchrotron)
        
        else:
            if kwargs.get('us_ic_enabled') == True:
                elements_new, counts, us_ic, theta, nx_new, ny_new, _, _, = extract_h5_xrf_data(file_path, synchrotron, us_ic_enabled = True)
            
            else:
                elements_new, counts, theta, nx_new, ny_new, _, _, = extract_h5_xrf_data(file_path, synchrotron)
        
        assert nx == nx_new and ny == ny_new, f"Dimension mismatch in {file_path}." # Check that the dimensions of the new data match the dimensions of the first data set
        assert np.array_equal(elements, elements_new), f"Element mismatch in {file_path}." # Check that the elements are the same
        
        if synchrotron == 'nsls-ii' and kwargs.get('us_ic_enabled') == True:
            us_ic_array[theta_idx] = us_ic

        counts_array[:, theta_idx, :, :] = counts
        theta_array[theta_idx] = theta
        file_path_array[theta_idx] = os.path.basename(file_path)
    
    if sample_flipped_remounted_mid_experiment:
        neg_90_deg_idx = (np.where(theta_array == -90)[0])

        if len(neg_90_deg_idx) != 2:
            print('Error: Must have two -90° angles. Exiting program...')

            sys.exit()
        
        second_neg_90_deg_idx = neg_90_deg_idx[1]

        theta_array[:second_neg_90_deg_idx] -= 90 # Make all angles before flipping go from -180° to 0
        theta_array[second_neg_90_deg_idx:] += 90 # Make all angles after flipping go from 0 to 180°

        theta_array_sorted = theta_array
        counts_array_sorted = counts_array

        if synchrotron == 'nsls-ii' and kwargs.get('us_ic_enabled') == True:
            us_ic_array_sorted = us_ic_array

        counts_array_sorted[:, second_neg_90_deg_idx:] = np.flip(counts_array_sorted[:, second_neg_90_deg_idx:], axis = 2)

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in range(len(theta_array_sorted))]

    else:
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
        
        theta_array_sorted = theta_array[theta_idx_sorted]
        counts_array_sorted = counts_array[:, theta_idx_sorted]

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]

    with h5py.File(output_h5_file, 'w') as f:
        f.create_dataset('filenames', data = file_path_array_sorted)

        exchange = f.create_group('exchange')

        exchange.create_dataset('data', data = counts_array_sorted, compression = 'gzip', compression_opts = 6)
        exchange.create_dataset('elements', data = elements_new)
        exchange.create_dataset('theta', data = theta_array_sorted)
        
        exchange['data'].attrs['dataset_type'] = 'xrf'
        
        if synchrotron == 'aps':
            exchange.attrs['raw_spectrum_fitting_software'] = 'MAPS'
            exchange.attrs['raw_spectrum_fitting_method'] = 'NNLS'
        
        elif synchrotron == 'nsls-ii':
            exchange.attrs['raw_spectrum_fitting_software'] = 'PyMCA'
            exchange.attrs['raw_spectrum_fitting_method'] = 'NNLS'

    if synchrotron == 'nsls-ii' and kwargs.get('us_ic_enabled') == True:   
        return us_ic_array_sorted
    
    return

def create_aggregate_xrt_h5(file_path_array, 
                            output_h5_file, 
                            synchrotron, 
                            sample_flipped_remounted_mid_experiment, 
                            **kwargs):
    
    n_theta = len(file_path_array)

    theta_array = np.zeros(n_theta) 

    if synchrotron == 'aps':
        elements, counts, theta, nx, ny, _, _ = extract_h5_xrt_data(file_path_array[0], synchrotron) # Invoke the first time for getting the number of elements and the number of pixels
    
    elif synchrotron == 'nsls-ii':
        us_ic = kwargs.get('us_ic')

        if us_ic is None:
            print('Error: \'us_ic\' not provided. Exiting program...')

            sys.exit()
        
        print(us_ic.shape)

        kwargs['ny'], kwargs['nx'] = us_ic[0].shape

        elements, counts, theta, nx, ny, _, _ = extract_h5_xrt_data(file_path_array[0], synchrotron, **kwargs)

    n_elements = len(elements)
    
    counts_array = np.zeros((n_elements, n_theta, ny, nx))

    for theta_idx, file_path in enumerate(file_path_array):
        if theta_idx != len(file_path_array) - 1:
            print(f'\rHDF file {theta_idx + 1}/{len(file_path_array)} extracted', end = '', flush = True)
        
        else:
            print(f'\rHDF file {theta_idx + 1}/{len(file_path_array)} extracted', flush = True)
        
        if synchrotron == 'nsls-ii':
            elements_new, counts, theta, nx_new, ny_new, _, _ = extract_h5_xrt_data(file_path, synchrotron, **kwargs)
            
        else:
            elements_new, counts, theta, nx_new, ny_new, _, _ = extract_h5_xrt_data(file_path, synchrotron)
        
        assert nx == nx_new and ny == ny_new, f"Dimension mismatch in {file_path}." # Check that the dimensions of the new data match the dimensions of the first data set
        assert np.array_equal(elements, elements_new), f"Element mismatch in {file_path}." # Check that the elements are the same

        counts_array[:, theta_idx, :, :] = counts
        theta_array[theta_idx] = theta
        file_path_array[theta_idx] = os.path.basename(file_path)
    
    if sample_flipped_remounted_mid_experiment:
        neg_90_deg_idx = (np.where(theta_array == -90)[0])

        if len(neg_90_deg_idx) != 2:
            print('Error: Must have two -90° angles. Exiting program...')

            sys.exit()
        
        second_neg_90_deg_idx = neg_90_deg_idx[1]
        
        theta_array[:second_neg_90_deg_idx] -= 90 # Make all angles before flipping go from -180° to 0
        theta_array[second_neg_90_deg_idx:] += 90 # Make all angles after flipping go from 0 to 180°

        theta_array_sorted = theta_array
        counts_array_sorted = counts_array

        if synchrotron == 'nsls-ii':
            counts_array_sorted[1] = us_ic
        
        counts_array_sorted[:, second_neg_90_deg_idx:] = np.flip(counts_array_sorted[:, second_neg_90_deg_idx:], axis = 2)
        
        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in range(len(theta_array_sorted))]
    
    else:
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
    
        theta_array_sorted = theta_array[theta_idx_sorted]
        counts_array_sorted = counts_array[:, theta_idx_sorted]

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]
    
    with h5py.File(output_h5_file, 'w') as f:
        f.create_dataset('filenames', data = file_path_array_sorted)

        exchange = f.create_group('exchange')

        exchange.create_dataset('data', data = counts_array_sorted, compression = 'gzip', compression_opts = 6)
        exchange.create_dataset('elements', data = elements_new)
        exchange.create_dataset('theta', data = theta_array_sorted)

        exchange['data'].attrs['dataset_type'] = 'xrt'

        if synchrotron == 'aps':
            exchange['data'].attrs['us_ic_scaler_name'] = 'US_IC'
            exchange['data'].attrs['xrt_signal_name'] = 'DS_IC'
        
        elif synchrotron == 'nsls-ii':
            exchange['data'].attrs['us_ic_scaler_name'] = 'sclr1_ch4'
            exchange['data'].attrs['xrt_signal_name'] = 'stxm'

def extract_h5_aggregate_xrf_data(file_path, **kwargs):
    if not os.path.isfile(file_path):
        print('Error: HDF5 file path cannot be found. Exiting program...')

        sys.exit()

    if not file_path.endswith('.h5'):
        print('Error: File extension must be \'.h5\'. Exiting program...')

        sys.exit()
    
    h5 = h5py.File(file_path, 'r')
    
    try:
        counts_h5 = h5['exchange/data']
        theta_h5 = h5['exchange/theta']
        elements_h5 = h5['exchange/elements']

        if kwargs.get('filename_array') == True:
            filenames_h5 = h5['filenames']

            filenames = filenames_h5[()]

        dataset_type = counts_h5.attrs['dataset_type']
        raw_spectrum_fitting_method = h5['exchange'].attrs['raw_spectrum_fitting_method']
    
    except KeyboardInterrupt:
        print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
        sys.exit()

    except:
        print('Error: Incompatible HDF5 file structure. Exiting program...')

        sys.exit()

    counts = counts_h5[()]
    theta = theta_h5[()]
    elements = elements_h5[()]
        
    h5.close()

    elements_string = [element.decode() for element in elements]
    
    if kwargs.get('filename_array') == True:
        filename_array = [filename.decode() for filename in filenames]

        return elements_string, counts, theta, raw_spectrum_fitting_method, dataset_type, filename_array
    
    return elements_string, counts, theta, raw_spectrum_fitting_method, dataset_type

def extract_h5_aggregate_xrt_data(file_path, **kwargs):
    if not os.path.isfile(file_path):
        print('Error: HDF5 file path cannot be found. Exiting program...')

        sys.exit()

    if not file_path.endswith('.h5'):
        print('Error: File must be HDF5. Exiting program...')

        sys.exit()
    
    h5 = h5py.File(file_path, 'r')
    
    try:
        counts_h5 = h5['exchange/data']
        theta_h5 = h5['exchange/theta']
        elements_h5 = h5['exchange/elements']

        if kwargs.get('filename_array') == True:
            filenames_h5 = h5['filenames']

            filenames = filenames_h5[()]

        dataset_type = counts_h5.attrs['dataset_type']
        us_ic_scaler_name = counts_h5.attrs['us_ic_scaler_name']
    
    except KeyboardInterrupt:
        print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
        sys.exit()

    except:
        print('Error: Incompatible HDF file structure. Exiting program...')

        sys.exit()

    counts = counts_h5[()]
    theta = theta_h5[()]
    elements = elements_h5[()]
    
    h5.close()

    elements_string = [element.decode() for element in elements]

    if kwargs.get('filename_array') == True:
        filename_array = [filename.decode() for filename in filenames]

        return elements_string, counts, theta, us_ic_scaler_name, dataset_type, filename_array
    
    return elements_string, counts, theta, us_ic_scaler_name, dataset_type

def extract_csv_norm_net_shift_data(file_path, theta_array):
    if not os.path.isfile(file_path):
        print('Error: CSV file path cannot be found. Exiting program...')

        sys.exit()

    if not file_path.endswith('.csv'):
        print('Error: File must be CSV. Exiting program...')

        sys.exit()

    norm_mass_calibration_net_shift_data = pd.read_csv(file_path)
    
    try:
        thetas = norm_mass_calibration_net_shift_data['theta_deg'].to_numpy().astype(float)
    
    except KeyboardInterrupt:
        print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
        sys.exit()

    except:
        print('Error: Incorrect CSV file structure. Exiting program...')

        sys.exit()

    if not np.array_equal(thetas, theta_array):
        print('Error: Inconsistent projection angles and/or number of projection angles relative to input aggregate HDF files. Exiting program...')

        sys.exit()

    norm_array = norm_mass_calibration_net_shift_data['norm_factor'].to_numpy().astype(float)
    net_x_shifts = norm_mass_calibration_net_shift_data['net_x_pixel_shift'].to_numpy().astype(float)
    net_y_shifts = norm_mass_calibration_net_shift_data['net_y_pixel_shift'].to_numpy().astype(float)
    I0 = norm_mass_calibration_net_shift_data['I0_cts'].to_numpy()[0].astype(float)

    return norm_array, net_x_shifts, net_y_shifts, I0

def create_h5_aligned_aggregate_xrf_xrt(dir_path,
                                        elements_xrf,
                                        xrf_array, 
                                        xrt_array,
                                        opt_dens_array, 
                                        theta_array,
                                        # zero_idx_discarded,
                                        init_edge_info,
                                        final_edge_info):

    elements_xrt = ['xrt_sig', 'opt_dens']

    n_theta, n_slices, n_columns = xrt_array.shape

    xrt_array_new = np.zeros((len(elements_xrt), n_theta, n_slices, n_columns))

    xrt_array_new[0] = xrt_array
    xrt_array_new[1] = opt_dens_array

    output_subdir_name = 'aligned_data'
    
    os.makedirs(os.path.join(dir_path, output_subdir_name), exist_ok = True)
    
    output_file_path = os.path.join(dir_path, output_subdir_name, 'aligned_aggregate_xrf_xrt.h5')

    with h5py.File(output_file_path, 'w') as f:
        exchange = f.create_group('exchange')

        exchange.create_dataset('elements_xrf', data = elements_xrf)
        exchange.create_dataset('elements_xrt', data = elements_xrt)
        exchange.create_dataset('data_xrf', data = xrf_array)
        exchange.create_dataset('data_xrt', data = xrt_array_new)
        exchange.create_dataset('theta', data = theta_array)

        if init_edge_info is not None:
            exchange['data_xrt'].attrs['left_edge_cropped_init'] = init_edge_info['left']
            exchange['data_xrt'].attrs['right_edge_cropped_init'] = init_edge_info['right']
            exchange['data_xrt'].attrs['top_edge_cropped_init'] = init_edge_info['top']
            exchange['data_xrt'].attrs['bottom_edge_cropped_init'] = init_edge_info['bottom']
            
        if final_edge_info is not None:
            exchange['data_xrt'].attrs['left_edge_cropped_final'] = final_edge_info['left']
            exchange['data_xrt'].attrs['right_edge_cropped_final'] = final_edge_info['right']
            exchange['data_xrt'].attrs['top_edge_cropped_final'] = final_edge_info['top']
            exchange['data_xrt'].attrs['bottom_edge_cropped_final'] = final_edge_info['bottom']

        # if zero_idx_discarded is not None:
            # exchange['theta'].attrs['zero_deg_idx_discarded'] = zero_idx_discarded

    return

def extract_csv_preprocessing_input_params(file_path):
    if not os.path.isfile(file_path):
        print('Error: CSV file path cannot be found. Exiting program...')

        sys.exit()

    if not file_path.endswith('.csv'):
        print('Error: CSV file required for preprocessing input parameters. Exiting program...')

        sys.exit()

    try:
        input_params_csv = pd.read_csv(file_path,
                                       delimiter = '=', 
                                       header = None, 
                                       names = ['input_param', 'value'],
                                       dtype = str,
                                       keep_default_na = False)

        input_params = input_params_csv['input_param'].str.strip()
        values = input_params_csv['value'].str.strip().replace('', 'None') # Extract values while setting non-existent values to None
    
    except KeyboardInterrupt:
        print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
        sys.exit()

    except:
        print('Error: Unable to read in CSV file. Exiting program...')

        sys.exit()
    
    numeric_params = ipn.preprocessing_numeric_params
    bool_params = ipn.preprocessing_bool_params
    list_params = ipn.preprocessing_list_params
    dict_params = ipn.preprocessing_dict_params
    all_params_ordered = pd.Series(ipn.preprocessing_params_ordered)

    available_synchrotrons = ipn.available_synchrotrons
    edge_crop_dxns = ipn.edge_crop_dxns

    missing_data = set(all_params_ordered) - set(input_params)
    extra_data = set(input_params) - set(all_params_ordered)

    if bool(missing_data) or bool(extra_data):
        if bool(missing_data) and bool(extra_data):
            print('Error: The following input parameters are missing:\n')
            print(*(["'{}'".format(s) for s in missing_data]), sep = '\n')
            print('\nAdditionally, the following input parameters should be removed:\n')
            print(*(["'{}'".format(s) for s in extra_data]), sep = '\n')

        elif bool(missing_data):
            print('Error: The following input parameters are missing:\n')
            print(*(["'{}'".format(s) for s in missing_data]), sep = '\n')

        else:
            print('Error: The following input parameters should be removed:\n')
            print(*(["'{}'".format(s) for s in extra_data]), sep = '\n')

        sys.exit()

    for idx, val in enumerate(values): # Convert strings supposed to be numberic or Boolean to floats, ints, or bools
        if val.lower() == 'none':
            values[idx] = None
        
        elif input_params[idx] in bool_params and (val.lower() == 'true' or val.lower() == 'false'):
            values[idx] = (val.lower() == 'true')
        
        elif input_params[idx] in list_params:
            values[idx] = values[idx].split(',')

            for _idx, _str in enumerate(values[idx]):
                _str = _str.strip().lower()

                values[idx][_idx] = _str

        elif input_params[idx] in dict_params:
            try:
                values[idx] = ast.literal_eval(val)
            
            except KeyboardInterrupt:
                print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
                sys.exit()

            except:
                print('Error: Cannot convert value of at least one parameter to dictionary. Exiting program...', flush = True)

                sys.exit()
                
        else:
            try:
                values[idx] = int(val)
        
            except KeyboardInterrupt:
                print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
                sys.exit()

            except:
                try:
                    values[idx] = float(val)
            
                except KeyboardInterrupt:
                    print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
                    sys.exit()

                except:
                    continue

    input_param_dict = dict(zip(input_params, values)) # zip() creates tuples; dict() converts the tuples to a dictionary

    if not all(isinstance(input_param_dict[param], bool) for param in bool_params):
        print('Error: The following input parameters must all be set to True or False:\n', \
              '     \'create_aggregate_xrf_xrt_files_enabled\'\n', \
              '     \'pre_existing_aggregate_xrf_xrt_file_lists_enabled\'\n', \
              '     \'pre_existing_align_norm_file_enabled\'\n', \
              '     \'norm_enabled\'\n', \
              '     \'realignment_enabled\'\n\n', \
              '\rExiting program...')

        sys.exit()

    if input_param_dict['synchrotron'] is None or input_param_dict['synchrotron_beamline'] is None:
        print('Error: Synchrotron and/or synchrotron beamline fields empty. Exiting program...')

        sys.exit()

    synchrotron = input_param_dict['synchrotron'].lower()

    if synchrotron in available_synchrotrons:
        input_param_dict['synchrotron'] = synchrotron
    
    else:
        print('Error: Synchrotron unavailable. Exiting program...')

        sys.exit()
    
    for param in dict_params:
        if (param == 'init_edge_pixel_lengths_to_crop' or param == 'final_edge_pixel_lengths_to_crop') and input_param_dict[param] is not None:
            for key in input_param_dict[param]:
                if key not in edge_crop_dxns:
                    print(f"Error: Unable to identify at least one specified edge for '{param}'. Exiting program...")

                    sys.exit()
            
                if not isinstance(input_param_dict[param][key], int):
                    print(f"Error: All values in the input dictionary of '{param}' should be integers. Exiting program...")

                    sys.exit()
                
                if input_param_dict[param][key] <= 0:
                    print(f"Error: All values in the input dictionary of '{param}' should be positive integers. Exiting program...")

                    sys.exit()

            if len(param) != len(edge_crop_dxns):
                missing_edges = set(edge_crop_dxns) - set(input_param_dict[param].keys()) # Find all edges missing from edge_dict keys relative to edge_crop_dxns

                for edge in missing_edges:
                    input_param_dict[param][edge] = 0

    for param in numeric_params:
        if isinstance(input_param_dict[param], str):
            print(f'Error: Expected a number for input parameter \'{param}\'. Exiting program...')

            sys.exit()

    return input_param_dict

def create_csv_file_list(file_path_array,
                         dir_path, 
                         synchrotron, 
                         synchrotron_beamline, 
                         dataset_type = None):
    
    file_path_array_full = [[os.path.join(dir_path, file_path) for file_path in file_path_array]] # 2D list so .writerows() does not split each list element into separate characters
    
    if synchrotron == 'aps':
        with open(os.path.join(dir_path, f'{synchrotron_beamline}_xrf_xrt_file_list.csv'), 'w', newline = '') as f:
            writer = csv.writer(f)

            writer.writerows(file_path_array_full)
    
    else:
        if dataset_type is None:
            print('Error: Dataset type required. Exiting program...')

            sys.exit()
            
        with open(os.path.join(dir_path, f'{synchrotron_beamline}_{dataset_type}_file_list.csv'), 'w', newline = '') as f:
            writer = csv.writer(f)

            writer.writerows(file_path_array_full)
    
    return

def extract_csv_xrf_xrt_data_file_lists(file_path_1, file_path_2 = None, **kwargs):
    if not os.path.isfile(file_path_1):
        print('Error: CSV file path cannot be found. Exiting program...')

        sys.exit()
    
    with open(file_path_1, newline = '') as f:
        filename_array_1 = [fn for fn in (filename.strip() for filename in f) \
                            if (os.path.isfile(fn) and fn.endswith('.h5'))] # Nested for loop

        if len(filename_array_1) == 0:
            print('Error: No .h5 files in first file array. Exiting program...')

            sys.exit()

    if kwargs.get('synchrotron') is None:
        print('Error: Synchrotron keyword argument required. Exiting program...')

        sys.exit()

    if kwargs.get('synchrotron') == 'aps': 
        filename_array_2 = filename_array_1.copy()
    
    else:
        if file_path_2 is None:
            print('Error: Second CSV file path required. Exiting program...')

            sys.exit()
        
        if not os.path.isfile(file_path_2):
            print('Error: CSV file path cannot be found. Exiting program...')

            sys.exit()

        with open(file_path_2, newline = '') as f:
            filename_array_2 = [fn for fn in (filename.strip() for filename in f) \
                                if (os.path.isfile(fn) and fn.endswith('.h5'))]

            if len(filename_array_2) == 0:
                print('Error: No .h5 files in second file array. Exiting program...')

                sys.exit()
    
    return filename_array_1, filename_array_2

def create_aux_conv_mag_data_npy(dir_path, array):
    subdir_path = os.path.join(dir_path, 'aux_data')

    os.makedirs(subdir_path, exist_ok = True)

    np.save(os.path.join(subdir_path, 'conv_mag_array.npy'), array)

    return

def create_aux_opt_dens_data_npy(dir_path,
                                 aligned_exp_proj_array,
                                 recon_array,
                                 synth_proj_array,
                                 pcc_2d_array,
                                 dx_array,
                                 dy_array,
                                 net_x_shifts_pcc_array,
                                 net_y_shifts_pcc_array,
                                 **kwargs):
    
    subdir_path = os.path.join(dir_path, 'aux_data')

    os.makedirs(subdir_path, exist_ok = True)

    np.save(os.path.join(subdir_path, 'aligned_exp_proj_iter_array.npy'), aligned_exp_proj_array)
    np.save(os.path.join(subdir_path, 'recon_iter_array.npy'), recon_array)
    np.save(os.path.join(subdir_path, 'synth_proj_iter_array.npy'), synth_proj_array)
    np.save(os.path.join(subdir_path, 'pcc_2d_iter_array.npy'), pcc_2d_array)
    np.save(os.path.join(subdir_path, 'dx_iter_array.npy'), dx_array)
    np.save(os.path.join(subdir_path, 'dy_iter_array.npy'), dy_array)
    np.save(os.path.join(subdir_path, 'net_x_shifts_pcc_iter_array.npy'), net_x_shifts_pcc_array)
    np.save(os.path.join(subdir_path, 'net_y_shifts_pcc_iter_array.npy'), net_y_shifts_pcc_array)

    if kwargs.get('cropped_aligned_exp_proj_iter_array') is not None:
        np.save(os.path.join(subdir_path, 'cropped_aligned_exp_proj_iter_array.npy'), kwargs['cropped_aligned_exp_proj_iter_array'])

    return

def create_csv_norm_net_shift_data(dir_path,
                                   theta_array,
                                   norm_array,
                                   net_x_shifts,
                                   net_y_shifts,
                                   I0):

    file_path = os.path.join(dir_path, 'norm_net_shift_data.csv')

    df = pd.DataFrame({'theta_deg': theta_array,
                       'norm_factor': norm_array,
                       'net_x_pixel_shift': net_x_shifts,
                       'net_y_pixel_shift': net_y_shifts,
                       'I0_cts': I0})

    df.loc[1:, 'I0_cts'] = np.nan # To make sure no errors arise from not having unequal column lengths

    df.to_csv(file_path, index = False)

    return

def create_nonaligned_norm_non_cropped_proj_data_gif(dir_path,
                                                     xrf_element_array,
                                                     desired_xrf_element,
                                                     counts_xrf,
                                                     counts_xrf_norm = None,
                                                     counts_xrt = None,
                                                     counts_xrt_norm = None,
                                                     opt_dens = None,
                                                     convolution_mag_array = None,
                                                     norm_enabled = False,
                                                     data_percentile = None,
                                                     theta_array = None,
                                                     fps = None):

    n_theta, n_slices, n_columns = counts_xrf.shape[1:]

    if desired_xrf_element is None:
        print('Error: \'desired_xrf_element\' field empty. Exiting program...')

        sys.exit()
    
    if desired_xrf_element not in xrf_element_array:
        print(f'Error: Desired XRF element not an XRF element channel. Exiting program...')

        sys.exit()
    
    if fps is None or fps <= 0:
        print('Error: \'fps\' (frames per second) must be a positive number. Exiting program...')

        sys.exit()
    
    ref_element_idx_xrf = xrf_element_array.index(desired_xrf_element)

    counts_xrf_ref_element = counts_xrf[ref_element_idx_xrf]
    counts_xrf_ref_element_norm = counts_xrf_norm[ref_element_idx_xrf]
    
    vmin_xrf = counts_xrf_ref_element.min()
    vmax_xrf = counts_xrf_ref_element.max()

    vmin_xrt = counts_xrt.min()
    vmax_xrt = counts_xrt.max()

    vmin_opt_dens = opt_dens.min()
    vmax_opt_dens = opt_dens.max()

    theta_frames1 = []
    theta_frames3 = []

    if norm_enabled:
        print('Plotting non-aligned, non-cropped, normalized XRT, optical density, and convolution magnitude projection data...')
    
        vmin_xrt_norm = counts_xrt_norm.min()
        vmax_xrt_norm = counts_xrt_norm.max()

        vmin_xrf_norm = counts_xrf_ref_element_norm.min()
        vmax_xrf_norm = counts_xrf_ref_element_norm.max()

        if convolution_mag_array is not None:
            fig1, axs1 = plt.subplots(3, 2)

            vmin_conv = convolution_mag_array.min()
            vmax_conv = convolution_mag_array.max()

            threshold = np.percentile(convolution_mag_array[0], data_percentile)

            conv_mask = np.where(convolution_mag_array[0] >= threshold, convolution_mag_array[0], 0)

            im1_1 = axs1[0, 0].imshow(convolution_mag_array[0], vmin = vmin_conv, vmax = vmax_conv)
            im1_2 = axs1[0, 1].imshow(conv_mask, vmin = vmin_conv, vmax = vmax_conv)
            im1_3 = axs1[1, 0].imshow(counts_xrt[0], vmin = vmin_xrt, vmax = vmax_xrt)
            im1_4 = axs1[1, 1].imshow(counts_xrt_norm[0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm)
            im1_5 = axs1[2, 0].imshow(counts_xrf_ref_element[0], vmin = vmin_xrf, vmax = vmax_xrf)
            im1_6 = axs1[2, 1].imshow(counts_xrf_ref_element_norm[0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)

            axs1[0, 0].set_title(r'XRT conv.', fontsize = 14)
            axs1[0, 1].set_title(r'XRT conv. mask', fontsize = 14)
            axs1[1, 0].set_title(r'XRT data', fontsize = 14)
            axs1[1, 1].set_title(r'Norm. XRT data', fontsize = 14)
            axs1[2, 0].set_title(r'XRF data ({0})'.format(desired_xrf_element), fontsize = 14)
            axs1[2, 1].set_title(r'Norm. XRF data ({0})'.format(desired_xrf_element), fontsize = 14)

            text_1 = axs1[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[0, 0].transAxes, color = 'white')

            for axs in fig1.axes:
                axs.axis('off')
                axs.axvline(x = n_columns//2, color = 'red', linewidth = 2)

            for theta_idx in range(n_theta):
                threshold = np.percentile(convolution_mag_array[theta_idx], data_percentile)
            
                conv_mask = np.where(convolution_mag_array[theta_idx] >= threshold, convolution_mag_array[theta_idx], 0)

                im1_1.set_data(convolution_mag_array[theta_idx])
                im1_2.set_data(conv_mask)
                im1_3.set_data(counts_xrt[theta_idx])
                im1_4.set_data(counts_xrt_norm[theta_idx])
                im1_5.set_data(counts_xrf_ref_element[theta_idx])
                im1_6.set_data(counts_xrf_ref_element_norm[theta_idx])

                text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

                fig1.canvas.draw()

                frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]

                theta_frames1.append(frame1)
        
        else:
            fig1, axs1 = plt.subplots(2, 2)

            im1_1 = axs1[0, 0].imshow(counts_xrt[0], vmin = vmin_xrt, vmax = vmax_xrt)
            im1_2 = axs1[0, 1].imshow(counts_xrt_norm[0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm)
            im1_3 = axs1[1, 0].imshow(counts_xrf_ref_element[0], vmin = vmin_xrf, vmax = vmax_xrf)
            im1_4 = axs1[1, 1].imshow(counts_xrf_ref_element_norm[0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)

            axs1[0, 0].set_title(r'XRT data', fontsize = 14)
            axs1[0, 1].set_title(r'Norm. XRT data', fontsize = 14)
            axs1[1, 0].set_title(r'XRF data ({0})'.format(desired_xrf_element), fontsize = 14)
            axs1[1, 1].set_title(r'Norm. XRF data ({0})'.format(desired_xrf_element), fontsize = 14)

            text_1 = axs1[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[0, 0].transAxes, color = 'white')

            for axs in fig1.axes:
                axs.axis('off')
                axs.axvline(x = n_columns//2, color = 'red', linewidth = 2)

            for theta_idx in range(n_theta):
                im1_1.set_data(counts_xrt[theta_idx])
                im1_2.set_data(counts_xrt_norm[theta_idx])
                im1_3.set_data(counts_xrf_ref_element[theta_idx])
                im1_4.set_data(counts_xrf_ref_element_norm[theta_idx])

                text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

                fig1.canvas.draw()

                frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]

                theta_frames1.append(frame1)

        plt.close(fig1)

        gif_filename = os.path.join(dir_path, 'normalized_prealigned_conv_xrt_od_proj_data.gif')

        print('Saving data to GIF...')

        iio2.mimsave(gif_filename, theta_frames1, fps = fps)

        print('Plotting non-aligned, non-cropped, normalized XRT, optical density, XRF projection data...')

        fig2, axs2 = plt.subplots(1, 3)

        theta_frames2 = []

        im2_1 = axs2[0].imshow(counts_xrt_norm[0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm)
        im2_2 = axs2[1].imshow(opt_dens[0], vmin = vmin_opt_dens, vmax = vmax_opt_dens)
        im2_3 = axs2[2].imshow(counts_xrf_ref_element_norm[0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)
        
        text_2 = axs2[0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs2[0].transAxes, color = 'white')
        
        axs2[0].set_title(r'Norm. XRT', fontsize = 14)
        axs2[1].set_title(r'Norm. Opt. Dens.', fontsize = 14)
        axs2[2].set_title(r'Norm. XRF ({0})'.format(desired_xrf_element), fontsize = 14)

        for axs in fig2.axes:
            axs.axis('off')

        for theta_idx in range(n_theta):
            im2_1.set_data(counts_xrt_norm[theta_idx])
            im2_2.set_data(opt_dens[theta_idx])
            im2_3.set_data(counts_xrf_ref_element_norm[theta_idx])

            text_2.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

            fig2.canvas.draw()

            frame2 = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3]

            theta_frames2.append(frame2)
        
        plt.close(fig2)

        print('Saving data to GIF...')

        gif_filename = os.path.join(dir_path, 'normalized_prealigned_xrt_od_xrf_proj_data_comp.gif')

        iio2.mimsave(gif_filename, theta_frames2, fps = fps)

        print('Plotting non-aligned, non-cropped, normalized XRT, optical density, XRF sinograms...')

        fig3, axs3 = plt.subplots(3, 1)

        im3_1 = axs3[0].imshow(counts_xrt_norm[:, 0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm, aspect = 2)
        im3_2 = axs3[1].imshow(opt_dens[:, 0], vmin = vmin_opt_dens, vmax = vmax_opt_dens, aspect = 2)
        im3_3 = axs3[2].imshow(counts_xrf_ref_element_norm[:, 0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm, aspect = 2)

        text_3 = axs3[0].text(0.02, 0.02, r'Slice 0/{0}'.format(n_slices - 1), transform = axs3[0].transAxes, color = 'white')
        
        axs3[0].set_title(r'XRT', fontsize = 14)
        axs3[1].set_title(r'Opt. Dens.', fontsize = 14)
        axs3[2].set_title(r'XRF ({0})'.format(desired_xrf_element), fontsize = 14)

        for axs in fig3.axes:
            axs.axis('off')

        for slice_idx in range(n_slices):
            im3_1.set_data(counts_xrt_norm[:, slice_idx])
            im3_2.set_data(opt_dens[:, slice_idx])
            im3_3.set_data(counts_xrf_ref_element_norm[:, slice_idx])

            text_3.set_text(r'Slice {0}/{1}'.format(slice_idx, n_slices - 1))

            fig3.canvas.draw()

            frame3 = np.array(fig3.canvas.renderer.buffer_rgba())[:, :, :3]

            theta_frames3.append(frame3)
        
        plt.close(fig3)

        print('Saving data to GIF...')

        gif_filename = os.path.join(dir_path, 'normalized_prealigned_xrt_od_xrf_sinogram_data_comp.gif')
        
        iio2.mimsave(gif_filename, theta_frames3, fps = fps)

    else:
        print('Plotting non-aligned, non-cropped, non-normalized XRT, optical density, XRF projection data...')
        
        fig1, axs1 = plt.subplots(3, 1)

        im1_1 = axs1[0].imshow(counts_xrt[0], vmin = vmin_xrt, vmax = vmax_xrt)
        im1_2 = axs1[1].imshow(opt_dens[0], vmin = vmin_opt_dens, vmax = vmax_opt_dens)
        im1_3 = axs1[2].imshow(counts_xrf_ref_element[0], vmin = vmin_xrf, vmax = vmax_xrf)

        text_1 = axs1[0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[0].transAxes, color = 'white')

        axs1[0].set_title(r'XRT', fontsize = 14)
        axs1[1].set_title(r'Opt. Dens.', fontsize = 14)
        axs1[2].set_title(r'XRF', fontsize = 14)

        for axs in fig1.axes:
            axs.axis('off')

        for theta_idx in range(n_theta):
            im1_1.set_data(counts_xrt[theta_idx])
            im1_2.set_data(opt_dens[theta_idx])
            im1_3.set_data(counts_xrf_ref_element[theta_idx])

            text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

            fig1.canvas.draw()

            frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]

            theta_frames1.append(frame1)
        
        plt.close(fig1)

        print('Saving data to GIF...')

        gif_filename = os.path.join(dir_path, 'nonnormalized_prealigned_xrt_od_xrf_proj_data.gif')
        
        iio2.mimsave(gif_filename, theta_frames1, fps = fps)

        print('Plotting non-aligned, non-cropped, non-normalized XRT, optical density, XRF sinograms...')

        fig2, axs2 = plt.subplots(3, 1)

        im2_1 = axs2[0].imshow(counts_xrt[:, 0], vmin = vmin_xrt, vmax = vmax_xrt, aspect = 'auto')
        im2_2 = axs2[1].imshow(opt_dens[:, 0], vmin = vmin_opt_dens, vmax = vmax_opt_dens, aspect = 'auto')
        im2_3 = axs2[2].imshow(counts_xrf_ref_element[:, 0], vmin = vmin_xrf, vmax = vmax_xrf, aspect = 'auto')

        text_2 = axs2[0].text(0.02, 0.02, r'Slice 0/{0}'.format(n_slices - 1), transform = axs2[0].transAxes, color = 'white')
        
        axs2[0].set_title(r'XRT', fontsize = 14)
        axs2[1].set_title(r'Opt. Dens.', fontsize = 14)
        axs2[2].set_title(r'XRF', fontsize = 14)

        for axs in fig2.axes:
            axs.axis('off')

        for slice_idx in range(n_slices):
            im2_1.set_data(counts_xrt[:, slice_idx])
            im2_2.set_data(opt_dens[:, slice_idx])
            im2_3.set_data(counts_xrf_ref_element[:, slice_idx])

            text_2.set_text(r'Slice {0}/{1}'.format(slice_idx, n_slices - 1))

            fig2.canvas.draw()

            frame2 = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3]

            theta_frames2.append(frame2)
        
        plt.close(fig2)

        print('Saving data to GIF...')

        gif_filename = os.path.join(dir_path, 'nonnormalized_prealigned_xrt_od_xrf_sinogram_data.gif')
        
        iio2.mimsave(gif_filename, theta_frames2, fps = fps)
        
    return