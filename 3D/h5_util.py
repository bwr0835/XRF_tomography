import numpy as np, h5py, os, sys

def extract_h5_xrf_data(file_path, synchrotron, **kwargs):
    h5 = h5py.File(file_path, 'r')
    
    if synchrotron.lower() == 'aps':
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

            theta_idx = np.where(extra_pvs_names == b'2xfm:m58.VAL')[0]
            theta = float(extra_pvs_values[theta_idx][0].decode()) # Get the value of theta and decode it to a float (from a byte)

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()    

        nx_conv = ny_h5[()] # Width and height are reversed in the actual HDF5 data structure
        ny_conv = nx_h5[()] # Width and height are reversed in the actual HDF5 data structure

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
        
    elif synchrotron.lower() == 'nsls-ii':
        try:
            counts_h5 = h5['xrfmap/detsum/xrf_fit']
            elements_h5 = h5['xrfmap/detsum/xrf_fit_name']
            axis_coords_h5 = h5['xrfmap/positions/pos']
            theta_h5 = h5['xrfmap/scan_metadata'].attrs['param_theta'] # Metadata stored as key-value pairs (attributes) (similar to a Python dictionary)

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()

        elements = elements_h5[()]
        counts = counts_h5[()]
        axis_coords = axis_coords_h5[()]
        theta = 1e-3*theta_h5[()] # Convert from mdeg to deg

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

        # if kwargs.get('scan_coords') == True and kwargs.get('us_ic') == True:
        if kwargs.get('us_ic_enabled') == True:
            try:
                scalers_names_h5 = h5['xrfmap/scalers/name']
                scalers_h5 = h5['xrfmap/scalers/val']

                scalers_names = scalers_names_h5[()]
                scalers = scalers_h5[()]

                us_ic_index = np.ndarray.item(np.where(scalers_names == b'sclr1_ch4')[0]) # Ion chamber upstream of zone plate, but downstream of X-ray beam slits
            
            except:
                print('Error: Incompatible HDF5 file structure. Exiting program...')

                sys.exit()

            us_ic = scalers[us_ic_index]

            print(us_ic)

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
    
    elif synchrotron.lower() == 'nsls-ii':
        # STXM calculations adapted from X. Huang, Brookhaven National Laboratory
        try:
            diffract_map_intensity = h5['diffamp'][()]
            theta = h5['angle'][()]

            dx_cm, dy_cm = h5['dr_x'][()], h5['dr_y'][()] # These are supposed to be different than for XRF due to ptychography requiring overlapping positions

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()

        nx = kwargs.get('nx')
        ny = kwargs.get('ny')

        cts_stxm = diffract_map_intensity.sum(axis = (2, 1)) # Sum over axis = 2, then sum over axis = 1
        cts_stxm = cts_stxm.reshape((ny, nx))

        # cts_stxm /= np.max(cts_stxm) # TODO
# ----------------------------------------------------------------------------------------------
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

    if synchrotron.lower() == 'nsls-ii' and kwargs.get('us_ic_enabled') == True:
        us_ic_array = np.zeros((n_theta, ny, nx))

    for theta_idx, file_path in enumerate(file_path_array):
        if synchrotron.lower() != 'nsls-ii':
            elements_new, counts, theta, nx_new, ny_new, _, _ = extract_h5_xrf_data(file_path, synchrotron)
        
        else:
            if kwargs.get('us_ic_enabled') == True:
                elements_new, counts, us_ic, theta, nx_new, ny_new, _, _, = extract_h5_xrf_data(file_path, synchrotron, us_ic_enabled = kwargs.get('us_ic_enabled'))
            
            else:
                elements_new, counts, theta, nx_new, ny_new, _, _, = extract_h5_xrf_data(file_path, synchrotron)
        
        assert nx == nx_new and ny == ny_new, f"Dimension mismatch in {file_path}." # Check that the dimensions of the new data match the dimensions of the first data set
        assert np.array_equal(elements, elements_new), f"Element mismatch in {file_path}." # Check that the elements are the same
        
        if synchrotron.lower() == 'nsls-ii' and kwargs.get('us_ic') is not None:
            us_ic_array[theta_idx] = us_ic

            if us_ic_array[theta_idx].any():
                print(us_ic_array[theta_idx])

        counts_array[:, theta_idx, :, :] = counts
        theta_array[theta_idx] = theta
        file_path_array[theta_idx] = os.path.basename(file_path)
    
    if synchrotron.lower() != 'nsls-ii':
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
    
        theta_array_sorted = theta_array[theta_idx_sorted]
        counts_array_sorted = counts_array[:, theta_idx_sorted, :, :]

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]
    
    else:
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
        
        if synchrotron.lower() == 'aps':
            exchange.attrs['raw_spectrum_fitting_software'] = 'MAPS'
            exchange.attrs['raw_spectrum_fitting_method'] = 'NNLS'
        
        elif synchrotron.lower() == 'nsls-ii':
            exchange.attrs['raw_spectrum_fitting_software'] = 'PyMCA'
            exchange.attrs['raw_spectrum_fitting_method'] = 'NNLS'

    if synchrotron.lower() == 'nsls-ii' and kwargs.get('us_ic_enabled') == True:
        if np.any(us_ic_array):
            print(us_ic_array_sorted)

        sys.exit()
        
        return us_ic_array_sorted, nx, ny

def create_aggregate_xrt_h5(file_path_array, output_h5_file, synchrotron, **kwargs):
    n_theta = len(file_path_array)

    theta_array = np.zeros(n_theta) 

    if synchrotron.lower() == 'nsls-ii':
        us_ic = kwargs.get('us_ic')

        if us_ic is None:
            print('Error: \'us_ic\' not provided. Exiting program...')

            sys.exit()
        
        kwargs['ny'], kwargs['nx'] = us_ic[0].shape

        elements, counts, theta, nx, ny, _, _ = extract_h5_xrt_data(file_path_array[0], synchrotron, **kwargs)
    
    else:
        elements, counts, theta, nx, ny, _, _ = extract_h5_xrt_data(file_path_array[0], synchrotron) # Invoke the first time for getting the number of elements and the number of pixels
    
    n_elements = len(elements)
    
    counts_array = np.zeros((n_elements, n_theta, ny, nx))

    for theta_idx, file_path in enumerate(file_path_array):     
        if synchrotron.lower() == 'nsls-ii':
            print(f'HDF file {theta_idx + 1}/{len(file_path_array)} extracted')
            
            elements_new, counts, theta, nx_new, ny_new, _, _ = extract_h5_xrt_data(file_path, synchrotron, **kwargs)
            
        else:
            elements_new, counts, theta, nx_new, ny_new, _, _ = extract_h5_xrt_data(file_path, synchrotron)
        
        assert nx == nx_new and ny == ny_new, f"Dimension mismatch in {file_path}." # Check that the dimensions of the new data match the dimensions of the first data set
        assert np.array_equal(elements, elements_new), f"Element mismatch in {file_path}." # Check that the elements are the same

        counts_array[:, theta_idx, :, :] = counts
        theta_array[theta_idx] = theta
        file_path_array[theta_idx] = os.path.basename(file_path)
    
    if synchrotron.lower() != 'nsls-ii':
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
    
        theta_array_sorted = theta_array[theta_idx_sorted]
        counts_array_sorted = counts_array[:, theta_idx_sorted, :, :]

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]
    
    else:
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

        if synchrotron.lower() == 'aps':
            exchange['data'].attrs['us_ic_scaler_name'] = 'US_IC'
            exchange['data'].attrs['xrt_signal_name'] = 'DS_IC'
        
        elif synchrotron.lower() == 'nsls-ii':
            exchange['data'].attrs['us_ic_scaler_name'] = 'sclr1_ch4'
            exchange['data'].attrs['xrt_signal_name'] = 'stxm'

def extract_h5_aggregate_xrf_data(file_path, **kwargs):
    h5 = h5py.File(file_path, 'r')
    
    counts_h5 = h5['exchange/data']
    theta_h5 = h5['exchange/theta']
    elements_h5 = h5['exchange/elements']
    filenames_h5 = h5['filenames']
    
    dataset_type_h5 = counts_h5.attrs['dataset_type']
    raw_spectrum_fitting_method_h5 = counts_h5.attrs['raw_spectrum_fitting_method']

    counts = counts_h5[()]
    theta = theta_h5[()]
    elements = elements_h5[()]
    dataset_type = dataset_type_h5[()]
    raw_spectrum_fitting_method = raw_spectrum_fitting_method_h5[()]
    filenames = filenames_h5[()]

    elements_string = [element.decode() for element in elements]
    filename_array = [filename.decode() for filename in filenames]
    
    if kwargs.get('filename_array') == True:
        return elements_string, counts, theta, raw_spectrum_fitting_method.decode(), dataset_type.decode(), filename_array
    
    return elements_string, counts, theta, raw_spectrum_fitting_method.decode(), dataset_type.decode()

def extract_h5_aggregate_xrt_data(file_path, **kwargs):
    h5 = h5py.File(file_path, 'r')
    
    counts_h5 = h5['exchange/data']
    theta_h5 = h5['exchange/theta']
    elements_h5 = h5['exchange/elements']
    filenames_h5 = h5['filenames']

    dataset_type_h5 = counts_h5.attrs['dataset_type']
    us_ic_scaler_name_h5 = counts_h5.attrs['us_ic_scaler_name']

    counts = counts_h5[()]
    theta = theta_h5[()]
    elements = elements_h5[()]
    us_ic_scaler_name = us_ic_scaler_name_h5[()]
    dataset_type = dataset_type_h5[()]
    filenames = filenames_h5[()]

    elements_string = [element.decode() for element in elements]
    filename_array = [filename.decode() for filename in filenames]

    if kwargs.get('filename_array') == True:
        return elements_string, counts, theta, us_ic_scaler_name.decode(), dataset_type.decode(), filename_array
    
    return elements_string, counts, theta, us_ic_scaler_name.decode(), dataset_type.decode()