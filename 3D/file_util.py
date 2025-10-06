import numpy as np, \
       pandas as pd, \
       h5py, \
       os, \
       sys

def extract_h5_xrf_data(file_path, synchrotron, **kwargs):
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
        
        dx_cm = 1e-4*(np.abs(x_um[0][-1] - x_um[0][0])/(nx - 1)) # Convert from µm to cm
        dy_cm = 1e-4*(np.abs(y_um[-1][0] - y_um[0][0])/(ny - 1)) # Convert from µm to cm
            
        elements_string = [element.decode() for element in elements] # Convert the elements array from a list of bytes to a list of strings

        # if kwargs.get('scan_coords') == True and kwargs.get('us_ic_enabled') == True:
        if kwargs.get('us_ic_enabled') == True:
            us_ic_index = np.ndarray.item(np.where(scalers_names == b'sclr1_ch4')[0]) # Ion chamber upstream of zone plate, but downstream of X-ray beam slits
            
            us_ic = scalers[:, :, us_ic_index]

            # print(us_ic)

            # return elements_string, counts, us_ic, theta, x_um, y_um, nx, ny, dx_cm, dy_cm
            return elements_string, counts, us_ic, theta, nx, ny, dx_cm, dy_cm

        else:
            return elements_string, counts, theta, nx, ny, dx_cm, dy_cm

def extract_h5_xrt_data(file_path, synchrotron, **kwargs):
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
        
        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()

        nx, ny = ny, nx
        
        cts_combined = np.zeros((n_elements, ny, nx))
        
        cts_us_ic = scaler_values[us_ic_idx] 
        cts_ds_ic = scaler_values[ds_ic_idx]
        # cts_abs_ic = scaler_values[abs_ic_idx]

        cts_combined[1] = cts_us_ic[:, :-2] # Remove last two columns since they are added after row is finished scanning
        cts_combined[2] = cts_ds_ic[:, :-2]
        # cts_combined[3] = cts_abs_ic[:, :-2]

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

def create_aggregate_xrf_h5(file_path_array, output_h5_file, synchrotron, **kwargs):
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
    
    if synchrotron == 'aps':
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
    
        theta_array_sorted = theta_array[theta_idx_sorted]
        counts_array_sorted = counts_array[:, theta_idx_sorted, :, :]

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]
    
    elif synchrotron == 'nsls-ii':
        # The following assumes all angles are in order over 360° (scan from -90° to 90°, flip sample, scan from -90° to 90°) AND -90° is included in BOTH sample orientations
        
        second_neg_90_deg_idx = (np.where(theta_array == -90)[0])[-1]

        theta_array[:second_neg_90_deg_idx] -= 90 # Make all angles before flipping go from -180° to 0
        theta_array[second_neg_90_deg_idx:] += 90 # Make all angles after flipping go from 0 to 180°
        
        theta_array_sorted = theta_array
        counts_array_sorted = counts_array

        if kwargs.get('us_ic_enabled') == True:
            us_ic_array_sorted = us_ic_array

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in range(n_theta)]
    
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

def create_aggregate_xrt_h5(file_path_array, output_h5_file, synchrotron, **kwargs):
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
    
    if synchrotron == 'aps':
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
    
        theta_array_sorted = theta_array[theta_idx_sorted]
        counts_array_sorted = counts_array[:, theta_idx_sorted, :, :]

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]
    
    elif synchrotron == 'nsls-ii':
        # This assumes all angles are in order over 360° (scan from -90° to 90°, flip sample, scan from -90° to 90°) AND -90° is included in BOTH sample orientations
        
        second_neg_90_deg_idx = (np.where(theta_array == -90)[0])[-1]

        theta_array[:second_neg_90_deg_idx] -= 90 # Make all angles before flipping go from -180° to 0
        theta_array[second_neg_90_deg_idx:] += 90 # Make all angles after flipping go from 0 to 180°

        theta_array_sorted = theta_array
        counts_array_sorted = counts_array

        counts_array_sorted[1] = us_ic

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in range(n_theta)]

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
    try:
        h5 = h5py.File(file_path, 'r')
    
    except:
        print('Error: File does not exist. Exiting program...')

        sys.exit()
    
    try:
        counts_h5 = h5['exchange/data']
        theta_h5 = h5['exchange/theta']
        elements_h5 = h5['exchange/elements']

        if kwargs.get('filename_array') == True:
            filenames_h5 = h5['filenames']

            filenames = filenames_h5[()]

        dataset_type = counts_h5.attrs['dataset_type']
        raw_spectrum_fitting_method = h5['exchange'].attrs['raw_spectrum_fitting_method']
    
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
    try:
        h5 = h5py.File(file_path, 'r')
    
    except:
        print('Error: File does not exist. Exiting program...')

        sys.exit()
    
    try:
        counts_h5 = h5['exchange/data']
        theta_h5 = h5['exchange/theta']
        elements_h5 = h5['exchange/elements']

        if kwargs.get('filename_array') == True:
            filenames_h5 = h5['filenames']

            filenames = filenames_h5[()]

        dataset_type = counts_h5.attrs['dataset_type']
        us_ic_scaler_name = counts_h5.attrs['us_ic_scaler_name']
    
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

def extract_norm_mass_calibration_net_shift_data(file_path, theta_array):
    try:
        norm_mass_calibration_net_shift_data = pd.read_csv(file_path)
    
    except:
        print('Error: File does not exist or incorrect file extension. Exiting program...')

        sys.exit()
    
    try:
        thetas = norm_mass_calibration_net_shift_data['theta'].to_numpy().astype(float)
    
    except:
        print('Error: Incorrect CSV file structure. Exiting program...')

        sys.exit()

    if not np.array_equal(thetas, theta_array):
        print('Error: Inconsistent projection angles and/or number of projection angles relative to input aggregate HDF files. Exiting program...')

        sys.exit()

    norm_array = norm_mass_calibration_net_shift_data['norm_factor'].to_numpy().astype(float)
    net_x_shifts = norm_mass_calibration_net_shift_data['net_x_shift'].to_numpy().astype(float)
    net_y_shifts = norm_mass_calibration_net_shift_data['net_y_shift'].to_numpy().astype(float)
    I0_norm = norm_mass_calibration_net_shift_data['I0_norm'].to_numpy()[0].astype(float)

    return norm_array, net_x_shifts, net_y_shifts, I0_norm

def create_h5_aligned_aggregate_xrf_xrt(elements_xrf, 
                                        elements_xrt,
                                        xrf_array, 
                                        xrt_array, 
                                        theta_array, 
                                        output_dir_path, 
                                        output_subdir_name):

    elements_xrt[3] = 'opt_dens'
    
    os.makedirs(os.path.join(output_dir_path, output_subdir_name), exist_ok = True)
    
    output_file_path = os.path.join(output_dir_path, output_subdir_name, 'aligned_aggregate_xrf_xrt.h5')

    with h5py.File(output_file_path, 'w') as f:
        exchange = f.create_group('exchange')

        exchange.create_dataset('element_xrf', data = elements_xrf)
        exchange.create_dataset('element_xrt', data = elements_xrt)
        exchange.create_dataset('data_xrf', data = xrf_array)
        exchange.create_dataset('data_xrt', data = xrt_array)
        exchange.create_dataset('theta', data = theta_array)
    
    return

def extract_csv_preprocessing_input_params(file_path):
    input_params_csv = pd.read_csv(file_path, delimiter = ':', header = None, names = ['input_param', 'value'])

    input_params = input_params_csv['input_param']
    values = input_params_csv['value'].str.strip().replace('', None) # Extract values while setting non-existent values to None
    
    for idx, val in enumerate(values): # Convert strings supposed to be numberic to floats or ints
        try:
            values[idx] = int(val)
        
        except:
            try:
                values[idx] = float(val)
            
            except:
                continue
    
    input_param_dict = dict(zip(input_params, values))

    for key, val in input_param_dict.items():
        print(f'{key}: {val} (dtype = {type(val).__name__})')

    sys.exit()
    # return input_param_dict

def create_aux_opt_dens_data_npy(dir_path,
                                 aligned_exp_proj_array,
                                 recon_array,
                                 synth_proj_array,
                                 pcc_2d_array,
                                 dx_array,
                                 dy_array,
                                 net_x_shifts_pcc_array,
                                 net_y_shifts_pcc_array):
    
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

    return

def create_intermed_aggregate_xrf_xrt_h5(file_path,
                                         f):
    pass
