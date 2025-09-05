import numpy as np, h5py, os, sys

def extract_h5_xrf_data(file_path, synchrotron):
    h5 = h5py.File(file_path, 'r')
    
    if synchrotron == "Advanced Photon Source (APS)" or synchrotron == "APS" or synchrotron == "aps" or synchrotron == "Advanced Photon Source" or synchrotron == "advanced photon source" or synchrotron == "advanced photon source (aps)":
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
        
    elif synchrotron == "National Synchrotron Light Source II (NSLS-II)" or synchrotron == "National Synchrotron Light Source II" or synchrotron == "nsls-ii" or synchrotron == "NSLSII" or synchrotron == "nslsii":
        counts_h5 = h5['xrfmap/detsum/xrf_fit']
        elements_h5 = h5['xrfmap/detsum/xrf_fit_name']
        axis_coords_h5 = h5['xrfmap/positions/pos']
        theta_h5 = h5['xrfmap/scan_metadata'].attrs['param_theta'] # Metadata stored as key-value pairs (attributes) (similar to a Python dictionary)

        elements = elements_h5[()]
        counts = counts_h5[()]
        axis_coords = axis_coords_h5[()]
        theta = 1e-3*theta_h5[()] # Convert from mdeg to deg

        x = axis_coords[0]
        y = axis_coords[1]

        nx = np.shape(x)[1]
        ny = np.shape(y)[0]

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
        
        dx_cm = 1e-4*(np.abs(x[0][-1] - x[0][0])/(nx - 1)) # Convert from µm to cm
        dy_cm = 1e-4*(np.abs(y[-1][0] - y[0][0])/(ny - 1)) # Convert from µm to cm
            
        elements_string = [element.decode() for element in elements] # Convert the elements array from a list of bytes to a list of strings

        return elements_string, counts, theta, nx, ny, dx_cm, dy_cm

def extract_h5_xrt_data(file_path, synchrotron):
    h5 = h5py.File(file_path, 'r')

    if synchrotron == "Advanced Photon Source (APS)" or synchrotron == "APS" or synchrotron == "aps" or synchrotron == "Advanced Photon Source" or synchrotron == "advanced photon source" or synchrotron == "advanced photon source (aps)":
        scalers_h5 = h5['MAPS/Scalers']
        extra_pvs_h5 = h5['MAPS/Scan/Extra_PVs']
        nx_h5 = h5['MAPS/Scan/x_axis']
        ny_h5 = h5['MAPS/Scan/y_axis']
        
        scaler_names = scalers_h5['Names'][()]
        scaler_values = scalers_h5['Values'][()]
        extra_pvs_names = extra_pvs_h5['Names'][()]
        extra_pvs_values = extra_pvs_h5['Values'][()]
        nx_conv = ny_h5[()] # Width and height are reversed in the actual HDF5 data structure
        ny_conv = nx_h5[()] # Width and height are reversed in the actual HDF5 data structure
        
        elements = ['empty', 'us_ic', 'ds_ic', 'abs_ic']
        n_elements = len(elements)

        nx = len(nx_conv)
        ny = len(ny_conv) - 2 # MAPS tacks on two extra values for whatever reason
        
        us_ic_idx = np.where(scaler_names == b'US_IC')[0][0] # The second [0] converts the 1-element array into a scalar
        ds_ic_idx = np.where(scaler_names == b'DS_IC')[0][0]
        abs_ic_idx = np.where(scaler_names == b'abs_ic')[0][0]
        theta_idx = np.where(extra_pvs_names == b'2xfm:m58.VAL')[0]

        nx, ny = ny, nx
        
        cts_combined = np.zeros((n_elements, ny, nx))
        
        cts_us_ic = scaler_values[us_ic_idx] 
        cts_ds_ic = scaler_values[ds_ic_idx]
        cts_abs_ic = scaler_values[abs_ic_idx]

        cts_combined[1] = cts_us_ic[:, :-2] # Remove last two columns since they are added after row is finished scanning
        cts_combined[2] = cts_ds_ic[:, :-2]
        cts_combined[3] = cts_abs_ic[:, :-2]

        dx_cm = 1e-1*np.abs([-1] - nx_conv[0])/(nx - 1)
        dy_cm = 1e-1*np.abs(ny_conv[-3] - ny_conv[0])/(ny - 1)

        theta = float(extra_pvs_values[theta_idx][0].decode())

        return elements, cts_combined, theta, nx, ny, dx_cm, dy_cm

def create_aggregate_xrf_h5(file_path_array, output_h5_file, synchrotron):
    n_theta = len(file_path_array)

    theta_array = np.zeros(n_theta) 

    elements, counts, theta, nx, ny, _, _ = extract_h5_xrf_data(file_path_array[0], synchrotron) # Invoke the first time for getting the number of elements and the number of pixels
    
    n_elements = len(elements)
    
    counts_array = np.zeros((n_elements, n_theta, ny, nx))

    for theta_idx, file_path in enumerate(file_path_array):
        elements_new, counts, theta, nx_new, ny_new, _, _ = extract_h5_xrf_data(file_path, synchrotron)
        
        assert nx == nx_new and ny == ny_new, f"Dimension mismatch in {file_path}." # Check that the dimensions of the new data match the dimensions of the first data set
        assert np.array_equal(elements, elements_new), f"Element mismatch in {file_path}." # Check that the elements are the same
        
        counts_array[:, theta_idx, :, :] = counts
        theta_array[theta_idx] = theta
        file_path_array[theta_idx] = os.path.basename(file_path)
    
    if synchrotron != "National Synchrotron Light Source II (NSLS-II)" and synchrotron != "National Synchrotron Light Source II" and synchrotron != "nsls-ii" and synchrotron != "NSLSII" and synchrotron != "nslsii":
        print('Yes')

        sys.exit()
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
    
        theta_array_sorted = theta_array[theta_idx_sorted]
        counts_array_sorted = counts_array[:, theta_idx_sorted, :, :]
    
    else:
        # This assumes all angles are in order over 360° (scan from -90° to 90°, flip sample, scan from -90° to 90°) AND -90° is included in BOTH sample orientations
        
        second_neg_90_deg_idx = (np.where(theta_array == -90)[0])

        print(second_neg_90_deg_idx)

        sys.exit()

        theta_array[:second_neg_90_deg_idx] -= 90 # Make all angles before flipping go from -180° to 0
        theta_array[second_neg_90_deg_idx:] += 90 # Make all angles after flipping go from 0 to 180°

        theta_array_sorted = theta_array
        counts_array_sorted = counts_array
    
    file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]

    with h5py.File(output_h5_file, 'w') as f:
        exchange = f.create_group('exchange')
        file_info = f.create_group('corresponding_file_info')

        exchange.create_dataset('data', data = counts_array_sorted, compression = 'gzip', compression_opts = 6)
        exchange.create_dataset('elements', data = elements_new)
        exchange.create_dataset('theta', data = theta_array_sorted)
        
        file_info.create_dataset('filename', data = file_path_array_sorted)
        file_info.create_dataset('dataset_type', data = 'xrf')

def create_aggregate_xrt_h5(file_path_array, output_h5_file, synchrotron):
    n_theta = len(file_path_array)

    theta_array = np.zeros(n_theta) 

    elements, counts, theta, nx, ny, _, _ = extract_h5_xrt_data(file_path_array[0], synchrotron) # Invoke the first time for getting the number of elements and the number of pixels
    
    n_elements = len(elements)
    
    counts_array = np.zeros((n_elements, n_theta, ny, nx))

    for theta_idx, file_path in enumerate(file_path_array):
        elements_new, counts, theta, nx_new, ny_new, _, _ = extract_h5_xrt_data(file_path, synchrotron)
        
        assert nx == nx_new and ny == ny_new, f"Dimension mismatch in {file_path}." # Check that the dimensions of the new data match the dimensions of the first data set
        assert np.array_equal(elements, elements_new), f"Element mismatch in {file_path}." # Check that the elements are the same
        
        counts_array[:, theta_idx, :, :] = counts
        theta_array[theta_idx] = theta
        file_path_array[theta_idx] = os.path.basename(file_path)
    
    if synchrotron != "National Synchrotron Light Source II (NSLS-II)" or synchrotron != "National Synchrotron Light Source II" or synchrotron != "nsls-ii" or synchrotron != "NSLSII" or synchrotron != "nslsii":
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
    
        theta_array_sorted = theta_array[theta_idx_sorted]
        counts_array_sorted = counts_array[:, theta_idx_sorted, :, :]
    
    else:
        # This assumes all angles are in order over 360° (scan from -90° to 90°, flip sample, scan from -90° to 90°) AND -90° is included in BOTH sample orientations
        
        second_neg_90_deg_idx = (np.where(theta_array == -90)[0])[-1]

        theta_array[:second_neg_90_deg_idx] -= 90 # Make all angles before flipping go from -180° to 0
        theta_array[second_neg_90_deg_idx:] += 90 # Make all angles after flipping go from 0 to 180°

        theta_array_sorted = theta_array
        counts_array_sorted = counts_array
    
    file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]

    with h5py.File(output_h5_file, 'w') as f:
        exchange = f.create_group('exchange')
        file_info = f.create_group('corresponding_file_info')

        exchange.create_dataset('data', data = counts_array_sorted, compression = 'gzip', compression_opts = 6)
        exchange.create_dataset('elements', data = elements_new)
        exchange.create_dataset('theta', data = theta_array_sorted)
        
        file_info.create_dataset('filenames', data = file_path_array_sorted)
        file_info.create_dataset('dataset_type', data = 'xrt')

def extract_h5_aggregate_xrf_data(file_path):
    h5 = h5py.File(file_path, 'r')
    
    counts_h5 = h5['exchange/data']
    theta_h5 = h5['exchange/theta']
    elements_h5 = h5['exchange/elements']
    dataset_type_h5 = h5['corresponding_file_info/dataset_type']

    counts = counts_h5[()]
    theta = theta_h5[()]
    elements = elements_h5[()]
    dataset_type = dataset_type_h5[()]

    elements_string = [element.decode() for element in elements]

    return elements_string, counts, theta, dataset_type.decode()

def extract_h5_aggregate_xrt_data(file_path):
    h5 = h5py.File(file_path, 'r')
    
    counts_h5 = h5['exchange/data']
    theta_h5 = h5['exchange/theta']
    elements_h5 = h5['exchange/elements']
    dataset_type_h5 = h5['corresponding_file_info/dataset_type']
    filenames_h5 = h5['corresponding_file_info/filenames']

    counts = counts_h5[()]
    theta = theta_h5[()]
    elements = elements_h5[()]
    dataset_type = dataset_type_h5[()]
    filenames = filenames_h5[()]

    elements_string = [element.decode() for element in elements]
    filename_array = [filename.decode() for filename in filenames]

    return elements_string, counts, theta, dataset_type.decode(), filename_array