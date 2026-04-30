import numpy as np, \
       pandas as pd, \
       tkinter as tk, \
       xrf_xrt_input_param_names as ipn, \
       xrf_xrt_preprocess_utils as ppu, \
       csv, \
       h5py, \
       ast, \
       os, \
       sys

from tkinter import filedialog as fd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

    if synchrotron == 'aps':
        try:
            with h5py.File(file_path, 'r') as h5:
                if "MAPS/XRF_Analyzed/NNLS" in h5.keys():
                    intensity_h5 = h5['MAPS/XRF_Analyzed/NNLS/intensity_Per_Sec']
                    elements_h5 = h5['MAPS/XRF_Analyzed/NNLS/Channel_Names']
            
                extra_pvs_h5 = h5['MAPS/Scan/Extra_PVs']
            
                nx_h5 = h5['MAPS/Scan/x_axis']
                ny_h5 = h5['MAPS/Scan/y_axis']
                
                intensity = intensity_h5[()]
                elements = elements_h5[()]
                extra_pvs_names = extra_pvs_h5['Names'][()]
                extra_pvs_values = extra_pvs_h5['Values'][()]

                nx_conv = ny_h5[()] # Width and height are reversed in the actual HDF5 data structure
                ny_conv = nx_h5[()] # Width and height are reversed in the actual HDF5 data structure

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
            sys.exit()

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()    
        
        fitting_software = 'MAPS'
        
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

        intensity_new = np.zeros((len(elements), ny, nx))
                
        idx_to_delete = []

        for element in elements:
                element_index = np.ndarray.item(np.where(elements == element)[0])
                    
                if element not in elements_entries_to_ignore:
                    intensity_new[element_index] = intensity[element_index, :, :-2] # MAPS tacks on two extra columns of zeroes post-scan for whatever reason
                    
                else:
                    idx_to_delete.append(element_index)

        intensity_new = np.delete(intensity_new, idx_to_delete, axis = 0) # Delete array elements corresponding to ignored element entries
        elements = np.delete(elements, idx_to_delete, axis = 0)

        # Get corresponding pixel spacings and convert from mm to cm
      
        dx_cm = 1e-1*np.abs([-1] - nx_conv[0])/(nx - 1)
        dy_cm = 1e-1*np.abs(ny_conv[-3] - ny_conv[0])/(ny - 1)

        elements_string = [element.decode() for element in elements] # Convert the elements array from a list of bytes to a list of strings

        return elements_string, intensity_new, theta, nx, ny, dx_cm, dy_cm, fitting_software
        
    elif synchrotron == 'nsls-ii':
        try:
            with h5py.File(file_path, 'r') as h5:
                intensity_h5 = h5['xrfmap/detsum/xrf_fit']
                elements_h5 = h5['xrfmap/detsum/xrf_fit_name']
                axis_coords_h5 = h5['xrfmap/positions/pos']
                theta_h5 = h5['xrfmap/scan_metadata'].attrs['param_theta'] # Metadata stored as key-value pairs (attributes) (similar to a Python dictionary)
                incident_energy_keV = h5['xrfmap/scan_metadata'].attrs['instrument_mono_incident_energy']
                fitting_software = h5['xrfmap/scan_metadata'].attrs['file_software']

                if kwargs.get('us_ic_enabled') == True:
                    scalers_names_h5 = h5['xrfmap/scalers/name']
                    scalers_h5 = h5['xrfmap/scalers/val']

                    scalers_names = scalers_names_h5[()]
                    scalers = scalers_h5[()]

                elements = elements_h5[()]
                intensity = intensity_h5[()]
                axis_coords = axis_coords_h5[()]
                theta = 1e-3*theta_h5[()] # Convert from mdeg to deg

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()

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
                
        intensity = np.delete(intensity, idx_to_delete, axis = 0)
        elements = np.delete(elements, idx_to_delete, axis = 0)
        
        dx_cm = 1e-4*(np.abs(x_um[0, -1] - x_um[0, 0])/(nx - 1)) # Convert from µm to cm
        dy_cm = 1e-4*(np.abs(y_um[-1, 0] - y_um[0, 0])/(ny - 1)) # Convert from µm to cm
            
        elements_string = [_str.split('_')[0] if '_K' in _str else _str for _str in (element.decode() for element in elements)] # Convert the elements array from a list of bytes to a list of strings

        if kwargs.get('us_ic_enabled') == True:
            us_ic_index = np.ndarray.item(np.where(scalers_names == b'sclr1_ch4')[0]) # Ion chamber upstream of zone plate, but downstream of X-ray beam slits
            
            us_ic = scalers[:, :, us_ic_index]

            return elements_string, intensity, us_ic, theta, incident_energy_keV, nx, ny, dx_cm, dy_cm, fitting_software

        return elements_string, intensity, theta, incident_energy_keV, nx, ny, dx_cm, dy_cm, fitting_software

def extract_h5_xrt_data(file_path, synchrotron, **kwargs):
    if not os.path.isfile(file_path):
        print('Error: HDF5 file path cannot be found. Exiting program...')

        sys.exit()

    if not file_path.endswith('.h5'):
        print('Error: File must be HDF5. Exiting program...')

        sys.exit()
    
    if synchrotron == 'aps':
        try:
            with h5py.File(file_path, 'r') as h5:
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
        
        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
            sys.exit()

        except:
            print('Error: Incompatible XRF/XRT HDF5 file structure. Exiting program...')

            sys.exit()

        # elements = ['empty', 'us_ic', 'ds_ic', 'abs_ic']
        elements = ['empty', 'us_ic', 'xrt_sig', 'empty']
        n_elements = len(elements)

        nx = len(nx_conv)
        ny = len(ny_conv) - 2 # MAPS tacks on two extra values for whatever reason
        
        try:
            us_ic_idx = np.where(scaler_names == b'US_IC')[0][0] # The second [0] converts the 1-element array into a scalar
            ds_ic_idx = np.where(scaler_names == b'DS_IC')[0][0]
            theta_idx = np.where(extra_pvs_names == b'2xfm:m58.VAL')[0]
        
        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
            sys.exit()

        except:
            print('Error: Incompatible XRF/XRT HDF5 file structure. Exiting program...')

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
            with h5py.File(file_path, 'r') as h5:
                diffract_map = h5['diffamp'][()]
                theta = h5['angle'][()]

                dx_cm, dy_cm = h5['dr_x'][()], h5['dr_y'][()] # These are supposed to be different than for XRF due to ptychography requiring overlapping positions

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
            sys.exit()

        except:
            print('Error: Incompatible HDF5 file structure. Exiting program...')

            sys.exit()

        nx = kwargs.get('nx')
        ny = kwargs.get('ny')

        cts_stxm = diffract_map.sum(axis = (2, 1)) # Sum over axis = 2, then sum over axis = 1
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

    if synchrotron == 'nsls-ii':
        elements, intensity, theta, incident_energy_keV, nx, ny, _, _, fitting_software = extract_h5_xrf_data(file_path_array[0], synchrotron) # Invoke the first time for getting the number of elements and the number of pixels
    
    else:
        elements, intensity, theta, nx, ny, _, _, fitting_software = extract_h5_xrf_data(file_path_array[0], synchrotron) # Invoke the first time for getting the number of elements and the number of pixels
        
        if kwargs.get('incident_energy_keV') is None:
            print('Error: \'incident_energy_keV\' not provided. Exiting program...')

            sys.exit()
        
        incident_energy_keV = float(kwargs['incident_energy_keV'])
        
    n_elements = len(elements)
    
    intensity_array = np.zeros((n_elements, n_theta, ny, nx))

    if synchrotron == 'nsls-ii' and kwargs.get('us_ic_enabled') == True:
        us_ic_array = np.zeros((n_theta, ny, nx))

    for theta_idx, file_path in enumerate(file_path_array):
        if theta_idx != len(file_path_array) - 1:
            print(f'\rHDF5 file {theta_idx + 1}/{len(file_path_array)} extracted', end = '', flush = True)
        
        else:
            print(f'\rHDF5 file {theta_idx + 1}/{len(file_path_array)} extracted', flush = True)
        
        if synchrotron != 'nsls-ii':
            elements_new, intensity, theta, nx_new, ny_new, _, _, _ = extract_h5_xrf_data(file_path, synchrotron)
        
        else:
            if kwargs.get('us_ic_enabled') == True:
                elements_new, intensity, us_ic, theta, _, nx_new, ny_new, _, _, _ = extract_h5_xrf_data(file_path, synchrotron, us_ic_enabled = True)
            
            else:
                elements_new, intensity, theta, _, nx_new, ny_new, _, _, _ = extract_h5_xrf_data(file_path, synchrotron)
        
        assert nx == nx_new and ny == ny_new, f"Dimension mismatch in {file_path}." # Check that the dimensions of the new data match the dimensions of the first data set
        assert np.array_equal(elements, elements_new), f"Element mismatch in {file_path}." # Check that the elements are the same
        
        if synchrotron == 'nsls-ii' and kwargs.get('us_ic_enabled') == True:
            us_ic_array[theta_idx] = us_ic

        intensity_array[:, theta_idx, :, :] = intensity
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
        intensity_array_sorted = intensity_array
        intensity_array_sorted[second_neg_90_deg_idx:] = np.fliplr(intensity_array_sorted[second_neg_90_deg_idx:]) # Flip remounted sample data back to original orientation

        if synchrotron == 'nsls-ii' and kwargs.get('us_ic_enabled') == True:
            us_ic_array_sorted = us_ic_array
            us_ic_array_sorted[second_neg_90_deg_idx:] = np.fliplr(us_ic_array_sorted[second_neg_90_deg_idx:]) # Flip remounted sample data back to original orientation

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in range(len(theta_array_sorted))]

    else:
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
        
        theta_array_sorted = theta_array[theta_idx_sorted]
        intensity_array_sorted = intensity_array[:, theta_idx_sorted]

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]

    with h5py.File(output_h5_file, 'w') as f:
        f.create_dataset('filenames', data = file_path_array_sorted)

        exchange = f.create_group('exchange')

        exchange.create_dataset('data', data = intensity_array_sorted, compression = 'gzip', compression_opts = 6)
        exchange.create_dataset('elements', data = elements_new)
        exchange.create_dataset('theta', data = theta_array_sorted)
        
        exchange['data'].attrs['dataset_type'] = 'xrf'
        exchange['data'].attrs['raw_spectrum_fitting_software'] = fitting_software
        exchange['data'].attrs['raw_spectrum_fitting_method'] = 'NNLS'
        exchange['data'].attrs['incident_energy_keV'] = incident_energy_keV

    if synchrotron == 'nsls-ii':
        if kwargs.get('us_ic_enabled') == True:   
            return incident_energy_keV, us_ic_array_sorted
    
    return incident_energy_keV

def create_aggregate_xrt_h5(file_path_array, 
                            output_h5_file, 
                            synchrotron, 
                            sample_flipped_remounted_mid_experiment,
                            incident_energy_keV,
                            **kwargs):
    
    n_theta = len(file_path_array)

    theta_array = np.zeros(n_theta) 

    if synchrotron == 'aps':
        elements, intensity, theta, nx, ny, _, _ = extract_h5_xrt_data(file_path_array[0], synchrotron) # Invoke the first time for getting the number of elements and the number of pixels

        if incident_energy_keV is None:
            print('Error: \'incident_energy_keV\' not provided. Exiting program...')

            sys.exit()

    elif synchrotron == 'nsls-ii':
        us_ic = kwargs.get('us_ic')
        
        if us_ic is None:
            print('Error: \'us_ic\' not provided. Exiting program...')

            sys.exit()
        
        kwargs['ny'], kwargs['nx'] = us_ic[0].shape

        elements, intensity, theta, nx, ny, _, _ = extract_h5_xrt_data(file_path_array[0], synchrotron, **kwargs)

    n_elements = len(elements)
    
    intensity_array = np.zeros((n_elements, n_theta, ny, nx))

    for theta_idx, file_path in enumerate(file_path_array):
        if theta_idx != len(file_path_array) - 1:
            print(f'\rHDF file {theta_idx + 1}/{len(file_path_array)} extracted', end = '', flush = True)
        
        else:
            print(f'\rHDF file {theta_idx + 1}/{len(file_path_array)} extracted', flush = True)
        
        if synchrotron == 'nsls-ii':
            elements_new, intensity, theta, nx_new, ny_new, _, _ = extract_h5_xrt_data(file_path, synchrotron, **kwargs)
            
        else:
            elements_new, intensity, theta, nx_new, ny_new, _, _ = extract_h5_xrt_data(file_path, synchrotron)
        
        assert nx == nx_new and ny == ny_new, f"Dimension mismatch in {file_path}." # Check that the dimensions of the new data match the dimensions of the first data set
        assert np.array_equal(elements, elements_new), f"Element mismatch in {file_path}." # Check that the elements are the same

        intensity_array[:, theta_idx, :, :] = intensity
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
        intensity_array_sorted = intensity_array
        intensity_array_sorted[second_neg_90_deg_idx:] = np.fliplr(intensity_array_sorted[second_neg_90_deg_idx:]) # Flip remounted sample data back to original orientation

        if synchrotron == 'nsls-ii':
            intensity_array_sorted[1] = us_ic
        
        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in range(len(theta_array_sorted))]
    
    else:
        theta_idx_sorted = np.argsort(theta_array) # Get indices for angles for sorting them in ascending order
    
        theta_array_sorted = theta_array[theta_idx_sorted]
        intensity_array_sorted = intensity_array[:, theta_idx_sorted]

        file_path_array_sorted = [file_path_array[theta_idx] for theta_idx in theta_idx_sorted]
    
    with h5py.File(output_h5_file, 'w') as f:
        f.create_dataset('filenames', data = file_path_array_sorted)

        exchange = f.create_group('exchange')

        exchange.create_dataset('data', data = intensity_array_sorted, compression = 'gzip', compression_opts = 6)
        exchange.create_dataset('elements', data = elements_new)
        exchange.create_dataset('theta', data = theta_array_sorted)

        exchange['data'].attrs['dataset_type'] = 'xrt'
        exchange['data'].attrs['incident_energy_keV'] = incident_energy_keV

        if synchrotron == 'aps':
            exchange['data'].attrs['us_ic_scaler_name'] = 'US_IC'
            exchange['data'].attrs['xrt_signal_name'] = 'DS_IC'
            exchange['data'].attrs['xrt_photon_counting'] = False
            exchange['data'].attrs['xrt_instrument'] = 'ion_chamber'

        elif synchrotron == 'nsls-ii':
            exchange['data'].attrs['us_ic_scaler_name'] = 'sclr1_ch4'
            exchange['data'].attrs['xrt_signal_name'] = 'transmitted'
            exchange['data'].attrs['xrt_photon_counting'] = True
            exchange['data'].attrs['xrt_instrument'] = 'pixel_array_detector'

def extract_h5_aggregate_xrf_data(file_path, **kwargs):
    if not os.path.isfile(file_path):
        print('Error: HDF5 file path cannot be found. Exiting program...')

        sys.exit()

    if not file_path.endswith('.h5'):
        print('Error: File extension must be \'.h5\'. Exiting program...')

        sys.exit()
    
    try:
        with h5py.File(file_path, 'r') as h5:
            intensity_h5 = h5['exchange/data']
            theta_h5 = h5['exchange/theta']
            elements_h5 = h5['exchange/elements']

            intensity = intensity_h5[()]
            theta = theta_h5[()]
            elements = list(elements_h5.asstr()[:])

            if kwargs.get('filename_array') == True:
                filenames_h5 = h5['filenames']

                filenames = filenames_h5.asstr()[:]
            
            dataset_type = intensity_h5.attrs['dataset_type']
            raw_spectrum_fitting_method = intensity_h5.attrs['raw_spectrum_fitting_method']
            incident_energy_keV = intensity_h5.attrs['incident_energy_keV']
    
    except KeyboardInterrupt:
        print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
        sys.exit()

    except:
        print('Error: Incompatible XRF HDF5 file structure. Exiting program...')

        sys.exit()
    
    if kwargs.get('filename_array') == True:
        return elements, intensity, theta, raw_spectrum_fitting_method, dataset_type, filenames
    
    return elements, intensity, theta, incident_energy_keV, raw_spectrum_fitting_method, dataset_type

def extract_h5_aggregate_xrt_data(file_path, **kwargs):
    if not os.path.isfile(file_path):
        print('Error: HDF5 file path cannot be found. Exiting program...')

        sys.exit()

    if not file_path.endswith('.h5'):
        print('Error: File must be HDF5. Exiting program...')

        sys.exit()
    
    try:
        with h5py.File(file_path, 'r') as h5:
            intensity_h5 = h5['exchange/data']
            theta_h5 = h5['exchange/theta']
            elements_h5 = h5['exchange/elements']

            intensity = intensity_h5[()]
            theta = theta_h5[()]
            elements = list(elements_h5.asstr()[:])

            if kwargs.get('filename_array') == True:
                filenames_h5 = h5['filenames']

                filenames = filenames_h5.asstr()[:]

            dataset_type = intensity_h5.attrs['dataset_type']
            us_ic_scaler_name = intensity_h5.attrs['us_ic_scaler_name']
            xrt_photon_counting = intensity_h5.attrs['xrt_photon_counting']
            incident_energy_keV = intensity_h5.attrs['incident_energy_keV']
    
    except KeyboardInterrupt:
        print('\n\nKeyboardInterrupt occurred. Exiting program...')
            
        sys.exit()

    except:
        print('Error: Incompatible XRT HDF5 file structure. Exiting program...')

        sys.exit()

    if kwargs.get('filename_array') == True:
        return elements, intensity, theta, us_ic_scaler_name, dataset_type, filenames
    
    return elements, intensity, theta, incident_energy_keV, us_ic_scaler_name, dataset_type, xrt_photon_counting

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
        print('Error: Incorrect CSV file structure for normalization and net shift data. Exiting program...')

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
                                        edge_info,
                                        I0_photons,
                                        incident_energy_keV):

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
        data = exchange.create_group('data')
        elements = exchange.create_group('elements')
        
        elements.create_dataset('xrf', data = elements_xrf)
        elements.create_dataset('xrt', data = elements_xrt)
        data.create_dataset('xrf', data = xrf_array, compression = 'gzip', compression_opts = 6)
        data.create_dataset('xrt', data = xrt_array_new, compression = 'gzip', compression_opts = 6)
        exchange.create_dataset('theta', data = theta_array)

        data.attrs['incident_intensity_photons'] = I0_photons
        data.attrs['incident_energy_keV'] = incident_energy_keV

        if edge_info is not None:
            exchange['data'].attrs['left_edge_cropped'] = edge_info['left']
            exchange['data'].attrs['right_edge_cropped'] = edge_info['right']
            exchange['data'].attrs['top_edge_cropped'] = edge_info['top']
            exchange['data'].attrs['bottom_edge_cropped'] = edge_info['bottom']

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
    numeric_array_params = ipn.preprocessing_numeric_array_params
    bool_params = ipn.preprocessing_bool_params
    list_params = ipn.preprocessing_list_params
    dict_params = ipn.preprocessing_dict_params
    all_params_ordered = pd.Series(ipn.preprocessing_params_ordered)

    available_synchrotrons = ipn.available_synchrotrons
    available_cor_correction_algorithms = ipn.available_cor_correction_algorithms
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
        
        print('\n\rExiting program...')

        sys.exit()

    for idx, val in enumerate(values): # Convert strings supposed to be numberic or Boolean to floats, ints, or bools
        if val.lower() == 'none':
            values[idx] = None
        
        elif input_params[idx] in bool_params and (val.lower() == 'true' or val.lower() == 'false'):
            values[idx] = (val.lower() == 'true')
        
        elif input_params[idx] in numeric_array_params:
            try:
                values[idx] = np.array(ast.literal_eval(val))
            
            except:
                print('Error: At least one preprocessing input parameter value cannot be converted to a NumPy array. Exiting program...', flush = True)

                sys.exit()

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
        print('Error: The following input parameters must all be set to True or False:\n')
        print(*(["'{}'".format(s) for s in bool_params]), sep = '\n')
        print('\n\rExiting program...')

        sys.exit()

    if input_param_dict['synchrotron'] is None or input_param_dict['synchrotron_beamline'] is None:
        print('Error: Synchrotron and/or synchrotron beamline fields empty. Exiting program...')

        sys.exit()

    synchrotron = input_param_dict['synchrotron'].lower()
    cor_correction_alg = input_param_dict['cor_correction_alg'].lower()

    if synchrotron in available_synchrotrons:
        input_param_dict['synchrotron'] = synchrotron
    
    else:
        print('Error: Synchrotron unavailable. Exiting program...')

        sys.exit()

    if cor_correction_alg not in available_cor_correction_algorithms:
        print('Error: Correction algorithm unavailable. Exiting program...')

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

def create_post_adjacent_angle_jitter_correction_aux_data_npy(dir_path,
                                                              proj_img_array_element_to_align_with_orig,
                                                              adj_angle_jitter_corrected_proj_element_to_align_with,
                                                              phase_xcorr_2d_aggregate_aux,
                                                              phase_xcorr_2d_truncated_aggregate_aux,
                                                              net_x_shift_array,
                                                              net_y_shift_array):
    
    os.makedirs(dir_path, exist_ok = True)
    
    np.save(os.path.join(dir_path, 'proj_img_array_element_to_align_with_orig.npy'), proj_img_array_element_to_align_with_orig)
    np.save(os.path.join(dir_path, 'adj_angle_jitter_corrected_proj_element_to_align_with.npy'), adj_angle_jitter_corrected_proj_element_to_align_with)

    np.save(os.path.join(dir_path, 'phase_xcorr_2d_aggregate_aux.npy'), phase_xcorr_2d_aggregate_aux)
    
    if phase_xcorr_2d_truncated_aggregate_aux is not None:
        np.save(os.path.join(dir_path, 'phase_xcorr_2d_truncated_aggregate_aux.npy'), phase_xcorr_2d_truncated_aggregate_aux)
    
    np.save(os.path.join(dir_path, 'net_x_shift_array.npy'), net_x_shift_array)
    np.save(os.path.join(dir_path, 'net_y_shift_array.npy'), net_y_shift_array)

    return

def create_post_cor_correction_aux_data_npy(dir_path,
                                            shifted_proj_img_array_element_to_align_with,
                                            shifted_proj_img_array_element_to_align_with_aux,
                                            shifted_proj_img_array_element_to_align_with_orig):
    
    os.makedirs(dir_path, exist_ok = True)
    
    np.save(os.path.join(dir_path, 'shifted_proj_img_array_element_to_align_with_orig.npy'), shifted_proj_img_array_element_to_align_with_orig)
    np.save(os.path.join(dir_path, 'shifted_proj_img_array_element_to_align_with.npy'), shifted_proj_img_array_element_to_align_with)
    
    if shifted_proj_img_array_element_to_align_with_aux is not None:
        np.save(os.path.join(dir_path, 'shifted_proj_img_array_element_to_align_with_aux.npy'), shifted_proj_img_array_element_to_align_with_aux)

    return

def create_post_iter_reproj_aux_data_npy(dir_path,
                                         aligned_exp_proj_array,
                                         recon_array,
                                         synth_proj_array,
                                         pcc_2d_array,
                                         pcc_2d_truncated_array,
                                         dx_array,
                                         dy_array,
                                         net_x_shifts_pcc_array,
                                         net_y_shifts_pcc_array):

    os.makedirs(dir_path, exist_ok = True)

    np.save(os.path.join(dir_path, 'aligned_exp_proj_iter_array.npy'), aligned_exp_proj_array)
    np.save(os.path.join(dir_path, 'recon_iter_array.npy'), recon_array)
    np.save(os.path.join(dir_path, 'synth_proj_iter_array.npy'), synth_proj_array)
    np.save(os.path.join(dir_path, 'pcc_2d_iter_array.npy'), pcc_2d_array)
    np.save(os.path.join(dir_path, 'pcc_2d_truncated_iter_array.npy'), pcc_2d_truncated_array)
    np.save(os.path.join(dir_path, 'dx_iter_array.npy'), dx_array)
    np.save(os.path.join(dir_path, 'dy_iter_array.npy'), dy_array)
    np.save(os.path.join(dir_path, 'net_x_shifts_pcc_iter_array.npy'), net_x_shifts_pcc_array)
    np.save(os.path.join(dir_path, 'net_y_shifts_pcc_iter_array.npy'), net_y_shifts_pcc_array)

    return

def create_csv_raw_input_data(dir_path,
                              theta_array,
                              norm_factor_xrf,
                              norm_factor_xrt,
                              init_x_shifts,
                              init_y_shifts,
                              pixel_rad_pre_cor_jitter,
                              pixel_rad_cor,
                              pixel_rad_iter_reproj,
                              I0_photons,
                              data_percentile,
                              aligning_element):
    
    file_path = os.path.join(dir_path, 'raw_input_data.csv')

    pixel_rad_pre_cor_jitter_array = np.append(np.nan, pixel_rad_pre_cor_jitter)

    output_dict = {'theta_deg': theta_array,
                   'norm_factor_xrf': norm_factor_xrf,
                   'norm_factor_xrt': norm_factor_xrt,
                   'init_x_shifts': init_x_shifts,
                   'init_y_shifts': init_y_shifts,
                   'pixel_rad_pre_cor_jitter': pixel_rad_pre_cor_jitter_array,
                   'pixel_rad_cor': pixel_rad_cor,
                   'pixel_rad_iter_reproj': pixel_rad_iter_reproj,
                   'I0_photons': I0_photons,
                   'data_percentile': data_percentile,
                   'aligning_element': aligning_element}
    
    df = pd.DataFrame({key: pd.Series(value) for key, value in output_dict.items()})
    
    df.to_csv(file_path, index = False, na_rep = '')

    return

def extract_csv_raw_input_data(file_path):
    if not os.path.isfile(file_path):
        print('Error: Input data CSV file path cannot be found. Exiting program...')

        sys.exit()
    
    if not file_path.endswith('.csv'):
        print('Error: Input data file must be CSV. Exiting program...')

        sys.exit()
    
    try:
        df = pd.read_csv(file_path)
    
    except KeyboardInterrupt:
        print('\n\nKeyboardInterrupt occurred. Exiting program...')

        sys.exit()

    except:
        print('Error: Unable to read in input data CSV file. Exiting program...')

        sys.exit()

    norm_factor_xrt = df['norm_factor_xrt'].to_numpy().astype(float)
    norm_factor_xrf = df['norm_factor_xrf'].to_numpy().astype(float)
    init_x_shifts = df['init_x_shifts'].to_numpy().astype(float)
    init_y_shifts = df['init_y_shifts'].to_numpy().astype(float)
    pixel_rad_pre_cor_jitter = df['pixel_rad_pre_cor_jitter'][1:].to_numpy().astype(int)
    pixel_rad_cor = int(df['pixel_rad_cor'][0])
    pixel_rad_iter_reproj = df['pixel_rad_iter_reproj'].to_numpy().astype(int)
    I0_photons = df['I0_photons'][0]   
    aligning_element = df['aligning_element'][0]
    data_percentile = df['data_percentile'][0]

    print(pixel_rad_pre_cor_jitter.shape)

    if data_percentile == '':
        data_percentile = None
    
    print(data_percentile)

    return norm_factor_xrt, \
           norm_factor_xrf, \
           init_x_shifts, \
           init_y_shifts, \
           pixel_rad_pre_cor_jitter, \
           pixel_rad_cor, \
           pixel_rad_iter_reproj, \
           I0_photons, \
           data_percentile, \
           aligning_element

def create_csv_output_data(dir_path,
                           theta_array,
                           net_x_shifts,
                           net_y_shifts,
                           cor_correction_only = False):

    file_path = os.path.join(dir_path, 'output_net_shift_data.csv')

    print(net_x_shifts.shape)

    if cor_correction_only: # net_x_shifts.ndim = 2
        if net_x_shifts.ndim == 2:
            theta_array_new = np.repeat(theta_array, net_x_shifts.shape[1])
            net_x_shifts_new = net_x_shifts.ravel()
            net_y_shifts_new = np.repeat(net_y_shifts, net_x_shifts.shape[1])
        
        else:
            theta_array_new = theta_array
            net_x_shifts_new = net_x_shifts
            net_y_shifts_new = net_y_shifts

    elif net_x_shifts.ndim == 3:
        theta_array_new = np.repeat(theta_array, net_x_shifts.shape[2])
        
        net_x_shifts_new = (net_x_shifts[-1]).ravel()
        net_y_shifts_new = np.repeat(net_y_shifts[-1], net_x_shifts.shape[2])
    
    else:
        theta_array_new = theta_array
        net_x_shifts_new = net_x_shifts[-1]
        net_y_shifts_new = net_y_shifts[-1]

    output_dict = {'theta_deg': theta_array_new,
                   'net_x_pixel_shift': net_x_shifts_new,
                   'net_y_pixel_shift': net_y_shifts_new}
    
    df = pd.DataFrame({key: pd.Series(value) for key, value in output_dict.items()})

    df.to_csv(file_path, index = False, na_rep = '')

    return

def create_nonaligned_norm_non_cropped_proj_data_gif(dir_path,
                                                     xrf_element_array,
                                                     desired_xrf_element,
                                                     intensity_xrf,
                                                     intensity_xrf_norm = None,
                                                     intensity_xrt = None,
                                                     intensity_xrt_norm = None,
                                                     opt_dens = None,
                                                     convolution_mag_array = None,
                                                     norm_enabled = False,
                                                     data_percentile = None,
                                                     theta_array = None,
                                                     fps = None):

    n_theta, n_slices, n_columns = intensity_xrf.shape[1:]

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

    intensity_xrf_ref_element = intensity_xrf[ref_element_idx_xrf]
    intensity_xrf_ref_element_norm = intensity_xrf_norm[ref_element_idx_xrf]
    
    vmin_xrf = intensity_xrf_ref_element.min()
    vmax_xrf = intensity_xrf_ref_element.max()

    vmin_xrt = intensity_xrt.min()
    vmax_xrt = intensity_xrt.max()

    vmin_opt_dens = opt_dens.min()
    vmax_opt_dens = opt_dens.max()

    theta_frames1 = []
    theta_frames3 = []

    if norm_enabled:
        print('Plotting non-aligned, non-cropped, normalized XRT, optical density, and convolution magnitude projection data...')
    
        vmin_xrt_norm = intensity_xrt_norm.min()
        vmax_xrt_norm = intensity_xrt_norm.max()

        vmin_xrf_norm = intensity_xrf_ref_element_norm.min()
        vmax_xrf_norm = intensity_xrf_ref_element_norm.max()

        if convolution_mag_array is not None:
            fig1, axs1 = plt.subplots(3, 2)

            vmin_conv = convolution_mag_array.min()
            vmax_conv = convolution_mag_array.max()

            threshold = np.percentile(convolution_mag_array[0], data_percentile)

            conv_mask = np.where(convolution_mag_array[0] >= threshold, convolution_mag_array[0], 0)

            im1_1 = axs1[0, 0].imshow(convolution_mag_array[0], vmin = vmin_conv, vmax = vmax_conv)
            im1_2 = axs1[0, 1].imshow(conv_mask, vmin = vmin_conv, vmax = vmax_conv)
            im1_3 = axs1[1, 0].imshow(intensity_xrt[0], vmin = vmin_xrt, vmax = vmax_xrt)
            im1_4 = axs1[1, 1].imshow(intensity_xrt_norm[0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm)
            im1_5 = axs1[2, 0].imshow(intensity_xrf_ref_element[0], vmin = vmin_xrf, vmax = vmax_xrf)
            im1_6 = axs1[2, 1].imshow(intensity_xrf_ref_element_norm[0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)

            axs1[0, 0].set_title(r'XRT conv.', fontsize = 14)
            axs1[0, 1].set_title(r'XRT conv. mask', fontsize = 14)
            axs1[1, 0].set_title(r'XRT data', fontsize = 14)
            axs1[1, 1].set_title(r'Norm. XRT data', fontsize = 14)
            axs1[2, 0].set_title(r'XRF data ({0})'.format(desired_xrf_element), fontsize = 14)
            axs1[2, 1].set_title(r'Norm. XRF data ({0})'.format(desired_xrf_element), fontsize = 14)

            text_1 = axs1[2, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[2, 0].transAxes, color = 'white')

            for ax in fig1.axes:
                ax.axis('off')
                ax.axvline(x = n_columns//2, color = 'red', linewidth = 2)

            for theta_idx in range(n_theta):
                threshold = np.percentile(convolution_mag_array[theta_idx], data_percentile)
            
                conv_mask = np.where(convolution_mag_array[theta_idx] >= threshold, convolution_mag_array[theta_idx], 0)

                im1_1.set_data(convolution_mag_array[theta_idx])
                im1_2.set_data(conv_mask)
                im1_3.set_data(intensity_xrt[theta_idx])
                im1_4.set_data(intensity_xrt_norm[theta_idx])
                im1_5.set_data(intensity_xrf_ref_element[theta_idx])
                im1_6.set_data(intensity_xrf_ref_element_norm[theta_idx])

                text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

                fig1.canvas.draw()

                frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]

                theta_frames1.append(frame1)
        
        else:
            fig1, axs1 = plt.subplots(2, 2)

            im1_1 = axs1[0, 0].imshow(intensity_xrt[0], vmin = vmin_xrt, vmax = vmax_xrt)
            im1_2 = axs1[0, 1].imshow(intensity_xrt_norm[0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm)
            im1_3 = axs1[1, 0].imshow(intensity_xrf_ref_element[0], vmin = vmin_xrf, vmax = vmax_xrf)
            im1_4 = axs1[1, 1].imshow(intensity_xrf_ref_element_norm[0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)

            axs1[0, 0].set_title(r'XRT data', fontsize = 14)
            axs1[0, 1].set_title(r'Norm. XRT data', fontsize = 14)
            axs1[1, 0].set_title(r'XRF data ({0})'.format(desired_xrf_element), fontsize = 14)
            axs1[1, 1].set_title(r'Norm. XRF data ({0})'.format(desired_xrf_element), fontsize = 14)

            text_1 = axs1[1, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[1, 0].transAxes, color = 'white')

            for ax in fig1.axes:
                ax.axis('off')
                ax.axvline(x = n_columns//2, color = 'red', linewidth = 2)

            for theta_idx in range(n_theta):
                im1_1.set_data(intensity_xrt[theta_idx])
                im1_2.set_data(intensity_xrt_norm[theta_idx])
                im1_3.set_data(intensity_xrf_ref_element[theta_idx])
                im1_4.set_data(intensity_xrf_ref_element_norm[theta_idx])

                text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

                fig1.canvas.draw()

                frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]

                theta_frames1.append(frame1)

        plt.close(fig1)

        gif_filename = os.path.join(dir_path, 'normalized_prealigned_conv_xrt_od_proj_data.gif')

        print('Saving data to GIF...')

        iio2.mimsave(gif_filename, theta_frames1, fps = fps)

        print('Plotting non-aligned, non-cropped, normalized XRT, optical density, XRF projection data...')
        
        fig2, axs2 = plt.subplots(3, 1)

        theta_frames2 = []

        im2_1 = axs2[0].imshow(intensity_xrt_norm[0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm)
        im2_2 = axs2[1].imshow(opt_dens[0], vmin = vmin_opt_dens, vmax = vmax_opt_dens)
        im2_3 = axs2[2].imshow(intensity_xrf_ref_element_norm[0], vmin = vmin_xrf_norm, vmax = vmax_xrf_norm)
        
        text_2 = axs2[2].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs2[2].transAxes, color = 'white')
        
        axs2[0].set_title(r'Norm. XRT', fontsize = 14)
        axs2[1].set_title(r'Norm. Opt. Dens.', fontsize = 14)
        axs2[2].set_title(r'Norm. XRF ({0})'.format(desired_xrf_element), fontsize = 14)

        for ax in fig2.axes:
            ax.axis('off')

        for theta_idx in range(n_theta):
            im2_1.set_data(intensity_xrt_norm[theta_idx])
            im2_2.set_data(opt_dens[theta_idx])
            im2_3.set_data(intensity_xrf_ref_element_norm[theta_idx])

            text_2.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

            fig2.canvas.draw()

            frame2 = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3]

            theta_frames2.append(frame2)
        
        plt.close(fig2)

        print('Saving data to GIF...')

        gif_filename = os.path.join(dir_path, 'normalized_prealigned_xrt_od_xrf_proj_data_comp.gif')

        iio2.mimsave(gif_filename, theta_frames2, fps = fps)

        print('Plotting non-aligned, non-cropped, normalized XRT, optical density, XRF sinograms...')

        fig3, axs3 = plt.subplots(1, 3, figsize = (11, 6))

        im3_1 = axs3[0].imshow(intensity_xrt_norm[:, 0], vmin = vmin_xrt_norm, vmax = vmax_xrt_norm, origin = 'lower', extent = [0, n_columns - 1, -180, 180], aspect = 1.5)
        im3_2 = axs3[1].imshow(opt_dens[:, 0], vmin = vmin_opt_dens, vmax = vmax_opt_dens, origin = 'lower', extent = [0, n_columns - 1, -180, 180], aspect = 1.5)
        im3_3 = axs3[2].imshow(intensity_xrf_ref_element_norm[:, 0], vmin = vmin_xrf_norm, origin = 'lower', extent = [0, n_columns - 1, -180, 180], vmax = vmax_xrf_norm, aspect = 1.5)

        text_3 = axs3[2].text(0.02, 0.02, r'Slice index 0/{0}'.format(n_slices - 1), transform = axs3[2].transAxes, color = 'white')
        
        axs3[0].set_title(r'XRT', fontsize = 14)
        axs3[1].set_title(r'Opt. Dens.', fontsize = 14)
        axs3[2].set_title(r'XRF ({0})'.format(desired_xrf_element), fontsize = 14)

        for ax in fig3.axes:
            ax.set_xlabel(r'Pixel index', fontsize = 14)
            ax.set_ylabel(r'$\theta$ (\textdegree)', fontsize = 14)

        for slice_idx in range(n_slices):
            im3_1.set_data(intensity_xrt_norm[:, slice_idx])
            im3_2.set_data(opt_dens[:, slice_idx])
            im3_3.set_data(intensity_xrf_ref_element_norm[:, slice_idx])

            text_3.set_text(r'Slice index {0}/{1}'.format(slice_idx, n_slices - 1))

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

        im1_1 = axs1[0].imshow(intensity_xrt[0], vmin = vmin_xrt, vmax = vmax_xrt)
        im1_2 = axs1[1].imshow(opt_dens[0], vmin = vmin_opt_dens, vmax = vmax_opt_dens)
        im1_3 = axs1[2].imshow(intensity_xrf_ref_element[0], vmin = vmin_xrf, vmax = vmax_xrf)

        text_1 = axs1[0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[0].transAxes, color = 'white')

        axs1[0].set_title(r'XRT', fontsize = 14)
        axs1[1].set_title(r'Opt. Dens.', fontsize = 14)
        axs1[2].set_title(r'XRF', fontsize = 14)

        for ax in fig1.axes:
            ax.axis('off')

        for theta_idx in range(n_theta):
            im1_1.set_data(intensity_xrt[theta_idx])
            im1_2.set_data(opt_dens[theta_idx])
            im1_3.set_data(intensity_xrf_ref_element[theta_idx])

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

        im2_1 = axs2[0].imshow(intensity_xrt[:, 0], vmin = vmin_xrt, vmax = vmax_xrt, origin = 'lower', extent = [0, n_slices - 1, -180, 180], aspect = 20)
        im2_2 = axs2[1].imshow(opt_dens[:, 0], vmin = vmin_opt_dens, vmax = vmax_opt_dens, origin = 'lower', extent = [0, n_slices - 1, -180, 180], aspect = 20)
        im2_3 = axs2[2].imshow(intensity_xrf_ref_element[:, 0], vmin = vmin_xrf, vmax = vmax_xrf, origin = 'lower', extent = [0, n_slices - 1, -180, 180], aspect = 20)

        text_2 = axs2[0].text(0.02, 0.02, r'Slice index 0/{0}'.format(n_slices - 1), transform = axs2[0].transAxes, color = 'white')
        
        axs2[0].set_title(r'XRT', fontsize = 14)
        axs2[1].set_title(r'Opt. Dens.', fontsize = 14)
        axs2[2].set_title(r'XRF', fontsize = 14)

        for ax in fig2.axes:
            # ax.axis('off')
            ax.set_xlabel(r'Pixel index', fontsize = 14)
            ax.set_ylabel(r'$\theta$ (\textdegree)', fontsize = 14)

        for slice_idx in range(n_slices):
            im2_1.set_data(intensity_xrt[:, slice_idx])
            im2_2.set_data(opt_dens[:, slice_idx])
            im2_3.set_data(intensity_xrf_ref_element[:, slice_idx])

            text_2.set_text(r'Slice index {0}/{1}'.format(slice_idx, n_slices - 1))

            fig2.canvas.draw()

            frame2 = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3]

            theta_frames2.append(frame2)
        
        plt.close(fig2)

        print('Saving data to GIF...')

        gif_filename = os.path.join(dir_path, 'nonnormalized_prealigned_xrt_od_xrf_sinogram_data.gif')
        
        iio2.mimsave(gif_filename, theta_frames2, fps = fps)
        
    return

def create_adjacent_angle_jitter_corrected_norm_proj_data_npy(dir_path,
                                                              shifted_intensity_ref_element,
                                                              phase_xcorr_2d_adjacent_angle_jitter,
                                                              phase_xcorr_2d_adjacent_angle_jitter_truncated,
                                                              ref_element,
                                                              sigma,
                                                              alpha):
    
    subdir_path = os.path.join(dir_path, 'aux_data')

    os.makedirs(subdir_path, exist_ok = True)

    np.save(os.path.join(subdir_path, f'phase_xcorr_2d_adjacent_angle_jitter_{ref_element}_sigma_{sigma}_alpha_{alpha}.npy'), phase_xcorr_2d_adjacent_angle_jitter)
    np.save(os.path.join(subdir_path, f'phase_xcorr_2d_adjacent_angle_jitter_truncated_{ref_element}_sigma_{sigma}_alpha_{alpha}.npy'), phase_xcorr_2d_adjacent_angle_jitter_truncated)
    np.save(os.path.join(subdir_path, f'adj_angle_jitter_corrected_proj_element_to_align_with_{ref_element}_sigma_{sigma}_alpha_{alpha}.npy'), shifted_intensity_ref_element)

    return

def create_gridrec_density_maps_h5(dir_path,
                                   gridrec_density_maps,
                                   elements_xrf):
    
    file_name = os.path.join(dir_path, 'gridrec_density_maps.h5')

    elements_xrf_new = [element.split('_')[0] for element in elements_xrf]
    
    with h5py.File(file_name, 'w') as f:
        sample = f.create_group('sample')
        
        sample.create_dataset("densities", data = gridrec_density_maps.astype('f4'))
        sample.create_dataset("elements", data = np.array(elements_xrf_new).astype('S5'))

    return

def create_adjacent_angle_jitter_corrected_norm_proj_data_gif(dir_path,
                                                              ref_element,
                                                              intensity_ref_element,
                                                              shifted_intensity_ref_element,
                                                              sigma,
                                                              alpha,
                                                              theta_array,
                                                              fps):

    n_theta, n_slices, n_columns = intensity_ref_element.shape
    
    vmin = np.min([intensity_ref_element, shifted_intensity_ref_element])
    vmax = np.max([intensity_ref_element, shifted_intensity_ref_element])

    theta_frames1 = []
    # slice_frames = []

    fig1, axs1 = plt.subplots(2, 1)

    im1_1 = axs1[0].imshow(intensity_ref_element[0], vmin = vmin, vmax = vmax)
    im1_2 = axs1[1].imshow(shifted_intensity_ref_element[0], vmin = vmin, vmax = vmax)

    text_1 = axs1[0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[0].transAxes, color = 'white')

    axs1[0].set_title(r'{0}'.format(ref_element), fontsize = 14)
    axs1[1].set_title(r'{0} (adj. angle jitter-corrected, cropped)'.format(ref_element), fontsize = 14)

    for ax in fig1.axes:
        ax.axis('off')
        ax.axvline(x = n_columns//2, color = 'white', linestyle = '--', linewidth = 2)
        ax.axhline(y = n_slices//2, color = 'white', linestyle = '--', linewidth = 2)

    for theta_idx in range(n_theta):
        im1_1.set_data(intensity_ref_element[theta_idx])
        im1_2.set_data(shifted_intensity_ref_element[theta_idx])

        text_1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

        fig1.canvas.draw()

        frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]
        
        theta_frames1.append(frame1)

    plt.close(fig1)

    gif_filename = os.path.join(dir_path, f'adjacent_angle_jitter_corrected_proj_comp_sigma_{sigma}_alpha_{alpha}.gif')

    print('Saving projection data to GIF...')

    iio2.mimsave(gif_filename, theta_frames1, fps = fps)

    # print(f'Plotting common field of view original, adjacent angle jitter-corrected, cropped {ref_element} sinograms...')

    # fig2, axs2 = plt.subplots(1, 2, figsize = (11, 6))

    # im2_1 = axs2[0].imshow(intensity_ref_element[:, 0], vmin = vmin, vmax = vmax, origin = 'lower', extent = [0, n_slices - 1, -180, 180], aspect = 10)
    # im2_2 = axs2[1].imshow(shifted_intensity_ref_element[:, 0], vmin = vmin, vmax = vmax, origin = 'lower', extent = [0, n_slices - 1, -180, 180], aspect = 10)

    # text_2 = axs2[0].text(0.02, 0.02, r'Slice index 0/{0}'.format(n_slices - 1), transform = axs2[0].transAxes, color = 'white')
    
    # axs2[0].set_title(r'{0}'.format(ref_element), fontsize = 14)
    # axs2[1].set_title(r'{0} (vert. jitter-corrected, cropped)'.format(ref_element), fontsize = 14)

    # for ax in fig2.axes:
    #     ax.set_xlabel(r'Pixel index', fontsize = 14)
    #     ax.set_ylabel(r'$\theta$ (\textdegree)', fontsize = 14)

    # for slice_idx in range(n_slices):
    #     im2_1.set_data(intensity_ref_element[:, slice_idx])
    #     im2_2.set_data(shifted_intensity_ref_element[:, slice_idx])

    #     text_2.set_text(r'Slice index {0}/{1}'.format(slice_idx, n_slices - 1))

    #     fig2.canvas.draw()
        
    #     frame2 = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3]
        
    #     slice_frames.append(frame2)

    # plt.close(fig2)

    # gif_filename = os.path.join(dir_path, f'vert_jitter_corrected_sinogram_comp_sigma_{sigma}_alpha_{alpha}.gif')

    # print('Saving sinogram data to GIF...')

    # iio2.mimsave(gif_filename, slice_frames, fps = fps)
    
    return

def create_phase_xcorr_2d_gif(dir_path,
                              phase_xcorr_2d_orig,
                              phase_xcorr_2d_truncated_orig,
                              theta_array,
                              ref_element,
                              correction_type,
                              fps):

    theta_frames = []

    if correction_type == 'adjacent_angle_jitter':
        phase_xcorr_2d = phase_xcorr_2d_orig

        gif_filename = os.path.join(dir_path, f'phase_xcorr_2d_adjacent_angle_jitter.gif')
    
    elif correction_type == 'iter_reproj':
        phase_xcorr_2d = phase_xcorr_2d_orig[0]

        gif_filename = os.path.join(dir_path, f'phase_xcorr_2d_iter_reproj.gif')

    n_theta, n_slices, n_columns = phase_xcorr_2d.shape

    vmin = phase_xcorr_2d.min()
    vmax = phase_xcorr_2d.max()

    if phase_xcorr_2d_truncated_orig is not None:
        if correction_type == 'adjacent_angle_jitter':
            phase_xcorr_2d_truncated = phase_xcorr_2d_truncated_orig

        elif correction_type == 'iter_reproj':
            phase_xcorr_2d_truncated = phase_xcorr_2d_truncated_orig[0]

        fig, axs = plt.subplots(1, 2)
        
        im1_1 = axs[0].imshow(phase_xcorr_2d[0], vmin = vmin, vmax = vmax)
        im1_2 = axs[1].imshow(phase_xcorr_2d_truncated[0], vmin = vmin, vmax = vmax)
        
        if correction_type == 'adjacent_angle_jitter':
            text1 = axs[0].text(0.02, 0.02, r'$\left(\theta_{{1}} = {0}^{{\circ}}, \theta_{{2}} = {1}^{{\circ}}\right)$'.format(theta_array[0], theta_array[1]), transform = axs[0].transAxes, color = 'white')

        else:
            text1 = axs[0].text(0.02, 0.02, r'$\theta = {0}^{{\circ}}$'.format(theta_array[0]), transform = axs[0].transAxes, color = 'white')

        for ax in fig.axes:
            ax.axis('off')
            # ax.axvline(x = n_columns//2, color = 'white', linewidth = 2, linestyle = '--')
            # ax.axhline(y = n_slices//2, color = 'white', linewidth = 2, linestyle = '--')
        
        axs[0].set_title(r'{0} PCC'.format(ref_element), fontsize = 14)
        axs[1].set_title(r'{0} PCC (truncated)'.format(ref_element), fontsize = 14)
        
        for theta_idx in range(n_theta):
            im1_1.set_data(phase_xcorr_2d[theta_idx])
            im1_2.set_data(phase_xcorr_2d_truncated[theta_idx])
            
            if correction_type == 'adjacent_angle_jitter':
                text1.set_text(r'$\left(\theta_{{1}} = {0}^{{\circ}}, \theta_{{2}} = {1}^{{\circ}}\right)$'.format(theta_array[theta_idx], theta_array[theta_idx + 1]))
            
            else:
                text1.set_text(r'$\theta = {0}^{{\circ}}$'.format(theta_array[theta_idx]))

            fig.canvas.draw()

            frame1 = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

            theta_frames.append(frame1)
    
    else:
        fig, axs = plt.subplots()

        im1_1 = axs.imshow(phase_xcorr_2d[0], vmin = vmin, vmax = vmax)
    
        text1 = axs.text(0.02, 0.02, r'$\left(\theta_{{1}} = {0}^{{\circ}}, \theta_{{2}} = {1}^{{\circ}}\right)$'.format(theta_array[0], theta_array[1]), transform = axs.transAxes, color = 'white')
    
        axs.axis('off')
        # axs.axvline(x = n_columns//2, color = 'white', linewidth = 2, linestyle = '--')
        # axs.axhline(y = n_slices//2, color = 'white', linewidth = 2, linestyle = '--')
    
        for theta_idx in range(n_theta):
            im1_1.set_data(phase_xcorr_2d[theta_idx])
        
            text1.set_text(r'$\left(\theta_{{1}} = {0}^{{\circ}}, \theta_{{2}} = {1}^{{\circ}}\right)$'.format(theta_array[theta_idx], theta_array[theta_idx + 1]))
        
            fig.canvas.draw()
        
            frame1 = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
            theta_frames.append(frame1)
        
    plt.close(fig)
    
    iio2.mimsave(gif_filename, theta_frames, fps = fps)
    
    return

def create_center_of_rotation_figures(dir_path,
                                      shifted_proj_img_array_element_to_align_with,
                                      shifted_proj_img_array_element_to_align_with_aux,
                                      sample_flipped_remounted_mid_experiment,
                                      theta_array):
                                                
    if sample_flipped_remounted_mid_experiment:
        if np.count_nonzero(theta_array == 0) != 2:
            print('Error: Must have two 0° angles. Exiting program...')

            sys.exit()

        zero_deg_idx_array = np.where(theta_array == 0)[0]

        shifted_proj_img_array_element_to_align_with_theta_aux_0_0 = shifted_proj_img_array_element_to_align_with_aux[0]
        shifted_proj_img_array_element_to_align_with_theta_aux_0_1 = shifted_proj_img_array_element_to_align_with_aux[zero_deg_idx_array[0]]

        shifted_proj_img_array_element_to_align_with_theta_aux_1_0 = shifted_proj_img_array_element_to_align_with_aux[zero_deg_idx_array[1]]
        shifted_proj_img_array_element_to_align_with_theta_aux_1_1 = shifted_proj_img_array_element_to_align_with_aux[-1]
        
        shifted_proj_img_array_element_to_align_with_theta_0_0 = shifted_proj_img_array_element_to_align_with[0]
        shifted_proj_img_array_element_to_align_with_theta_0_1 = shifted_proj_img_array_element_to_align_with[zero_deg_idx_array[0]]
        shifted_proj_img_array_element_to_align_with_theta_1_0 = shifted_proj_img_array_element_to_align_with[zero_deg_idx_array[1]]
        shifted_proj_img_array_element_to_align_with_theta_1_1 = shifted_proj_img_array_element_to_align_with[-1]

        shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_aux_0_0)
        shifted_proj_img_array_element_to_align_with_theta_aux_0_1_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_aux_0_1)
        shifted_proj_img_array_element_to_align_with_theta_aux_1_0_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_aux_1_0)
        shifted_proj_img_array_element_to_align_with_theta_aux_1_1_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_aux_1_1)

        shifted_proj_img_array_element_to_align_with_theta_0_0_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_0_0)
        shifted_proj_img_array_element_to_align_with_theta_0_1_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_0_1)
        shifted_proj_img_array_element_to_align_with_theta_1_0_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_1_0)
        shifted_proj_img_array_element_to_align_with_theta_1_1_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_1_1)

        shifted_proj_img_array_element_to_align_with_theta_aux_0_0_rgb = np.dstack((shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm, np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm)))
        shifted_proj_img_array_element_to_align_with_theta_aux_0_1_rgb = np.dstack((np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_0_1_norm), np.fliplr(shifted_proj_img_array_element_to_align_with_theta_aux_0_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_0_1_norm)))
        shifted_proj_img_array_element_to_align_with_theta_aux_1_0_rgb = np.dstack((shifted_proj_img_array_element_to_align_with_theta_aux_1_0_norm, np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_1_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_1_0_norm)))
        shifted_proj_img_array_element_to_align_with_theta_aux_1_1_rgb = np.dstack((np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_1_1_norm), np.fliplr(shifted_proj_img_array_element_to_align_with_theta_aux_1_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_1_1_norm)))
    
        shifted_proj_img_array_element_to_align_with_theta_0_0_rgb = np.dstack((shifted_proj_img_array_element_to_align_with_theta_0_0_norm, np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm)))
        shifted_proj_img_array_element_to_align_with_theta_0_1_rgb = np.dstack((np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_1_norm), np.fliplr(shifted_proj_img_array_element_to_align_with_theta_0_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_1_norm)))
        shifted_proj_img_array_element_to_align_with_theta_1_0_rgb = np.dstack((shifted_proj_img_array_element_to_align_with_theta_1_0_norm, np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_1_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_1_0_norm)))
        shifted_proj_img_array_element_to_align_with_theta_1_1_rgb = np.dstack((np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_1_1_norm), np.fliplr(shifted_proj_img_array_element_to_align_with_theta_1_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_1_1_norm)))
        
        shifted_proj_img_array_element_to_align_with_sample_offset_1_aux_rgb = np.dstack((np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_1_0_norm), np.fliplr(shifted_proj_img_array_element_to_align_with_theta_aux_1_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm)))
        shifted_proj_img_array_element_to_align_with_sample_offset_2_aux_rgb = np.dstack((shifted_proj_img_array_element_to_align_with_theta_aux_0_1_norm, np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm)))
        
        shifted_proj_img_array_element_to_align_with_sample_offset_1_rgb = np.dstack((np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_1_0_norm), np.fliplr(shifted_proj_img_array_element_to_align_with_theta_1_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm)))
        shifted_proj_img_array_element_to_align_with_sample_offset_2_rgb = np.dstack((shifted_proj_img_array_element_to_align_with_theta_0_1_norm, np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm)))

        overlay_aux_shifted_0 = np.dstack((shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm, np.fliplr(shifted_proj_img_array_element_to_align_with_theta_aux_0_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm)))
        overlay_aux_shifted_1 = np.dstack((shifted_proj_img_array_element_to_align_with_theta_aux_1_0_norm, np.fliplr(shifted_proj_img_array_element_to_align_with_theta_aux_1_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_1_0_norm)))
        overlay_aux_shifted_2 = np.dstack((shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm, np.fliplr(shifted_proj_img_array_element_to_align_with_theta_aux_1_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_aux_0_0_norm)))
        overlay_aux_shifted_3 = np.dstack((shifted_proj_img_array_element_to_align_with_theta_aux_0_1_norm, np.fliplr(shifted_proj_img_array_element_to_align_with_theta_aux_1_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm)))

        overlay_shifted_0 = np.dstack((shifted_proj_img_array_element_to_align_with_theta_0_0_norm, np.fliplr(shifted_proj_img_array_element_to_align_with_theta_0_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm)))
        overlay_shifted_1 = np.dstack((shifted_proj_img_array_element_to_align_with_theta_1_0_norm, np.fliplr(shifted_proj_img_array_element_to_align_with_theta_1_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_1_0_norm)))
        overlay_shifted_2 = np.dstack((shifted_proj_img_array_element_to_align_with_theta_0_0_norm, np.fliplr(shifted_proj_img_array_element_to_align_with_theta_1_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm)))
        overlay_shifted_3 = np.dstack((shifted_proj_img_array_element_to_align_with_theta_0_1_norm, np.fliplr(shifted_proj_img_array_element_to_align_with_theta_1_1_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm)))

        fig1, axs1 = plt.subplots(4, 3)
        
        im1_1 = axs1[0, 0].imshow(shifted_proj_img_array_element_to_align_with_theta_aux_0_0_rgb)
        im1_2 = axs1[0, 1].imshow(shifted_proj_img_array_element_to_align_with_theta_aux_0_1_rgb)
        im1_3 = axs1[0, 2].imshow(overlay_aux_shifted_0)
        im1_4 = axs1[1, 0].imshow(shifted_proj_img_array_element_to_align_with_theta_aux_1_0_rgb)
        im1_5 = axs1[1, 1].imshow(shifted_proj_img_array_element_to_align_with_theta_aux_1_1_rgb)
        im1_6 = axs1[1, 2].imshow(overlay_aux_shifted_1)
        im1_7 = axs1[2, 0].imshow(shifted_proj_img_array_element_to_align_with_theta_aux_0_0_rgb)
        im1_8 = axs1[2, 1].imshow(shifted_proj_img_array_element_to_align_with_sample_offset_1_aux_rgb)
        im1_9 = axs1[2, 2].imshow(overlay_aux_shifted_2)
        im1_10 = axs1[3, 0].imshow(shifted_proj_img_array_element_to_align_with_sample_offset_2_aux_rgb)
        im1_11 = axs1[3, 1].imshow(shifted_proj_img_array_element_to_align_with_theta_aux_1_1_rgb)
        im1_12 = axs1[3, 2].imshow(overlay_aux_shifted_3)

        text1 = axs1[0, 0].text(0.02, 0.02, r'$\theta = -180$\textdegree', transform = axs1[0, 0].transAxes, color = 'white')
        text2 = axs1[0, 1].text(0.02, 0.02, r'$\theta = 0^{-}$', transform = axs1[0, 1].transAxes, color = 'white')
        text3 = axs1[1, 0].text(0.02, 0.02, r'$\theta = 0^{+}$', transform = axs1[1, 0].transAxes, color = 'white')
        text4 = axs1[1, 1].text(0.02, 0.02, r'$\theta = 180$\textdegree', transform = axs1[1, 1].transAxes, color = 'white')
        text5 = axs1[2, 0].text(0.02, 0.02, r'$\theta = -180$\textdegree', transform = axs1[2, 0].transAxes, color = 'white')
        text6 = axs1[2, 1].text(0.02, 0.02, r'$\theta = 0^{+}$', transform = axs1[2, 1].transAxes, color = 'white')
        text7 = axs1[3, 0].text(0.02, 0.02, r'$\theta = 0^{-}$', transform = axs1[3, 0].transAxes, color = 'white')
        text8 = axs1[3, 1].text(0.02, 0.02, r'$\theta = 180$\textdegree', transform = axs1[3, 1].transAxes, color = 'white')
       
        for ax in fig1.axes:
            ax.axis('off')
            ax.axvline(x = shifted_proj_img_array_element_to_align_with_theta_0_0.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
            ax.axhline(y = shifted_proj_img_array_element_to_align_with_theta_0_0.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')

        # fig1.tight_layout()

        # plt.show()
        plt.close(fig1)
        
        fig_filename = os.path.join(dir_path, f'aux_shifted_center_of_rotation_figure.svg')
        
        fig1.savefig(fig_filename)

        fig2, axs2 = plt.subplots(4, 3)
        
        im2_1 = axs2[0, 0].imshow(shifted_proj_img_array_element_to_align_with_theta_0_0_rgb)
        im2_2 = axs2[0, 1].imshow(shifted_proj_img_array_element_to_align_with_theta_0_1_rgb)
        im2_3 = axs2[0, 2].imshow(overlay_shifted_0)
        im2_4 = axs2[1, 0].imshow(shifted_proj_img_array_element_to_align_with_theta_1_0_rgb)
        im2_5 = axs2[1, 1].imshow(shifted_proj_img_array_element_to_align_with_theta_1_1_rgb)
        im2_6 = axs2[1, 2].imshow(overlay_shifted_1)
        im2_7 = axs2[2, 0].imshow(shifted_proj_img_array_element_to_align_with_theta_0_0_rgb)
        im2_8 = axs2[2, 1].imshow(shifted_proj_img_array_element_to_align_with_sample_offset_1_rgb)
        im2_9 = axs2[2, 2].imshow(overlay_shifted_2)
        im2_10 = axs2[3, 0].imshow(shifted_proj_img_array_element_to_align_with_sample_offset_2_rgb)
        im2_11 = axs2[3, 1].imshow(shifted_proj_img_array_element_to_align_with_theta_1_1_rgb)
        im2_12 = axs2[3, 2].imshow(overlay_shifted_3)

        text1 = axs2[0, 0].text(0.02, 0.02, r'$\theta = -180$\textdegree', transform = axs2[0, 0].transAxes, color = 'white')
        text2 = axs2[0, 1].text(0.02, 0.02, r'$\theta = 0^{-}$', transform = axs2[0, 1].transAxes, color = 'white')
        text3 = axs2[1, 0].text(0.02, 0.02, r'$\theta = 0^{+}$', transform = axs2[1, 0].transAxes, color = 'white')
        text4 = axs2[1, 1].text(0.02, 0.02, r'$\theta = 180$\textdegree', transform = axs2[1, 1].transAxes, color = 'white')
        text5 = axs2[2, 0].text(0.02, 0.02, r'$\theta = -180$\textdegree', transform = axs2[2, 0].transAxes, color = 'white')
        text6 = axs2[2, 1].text(0.02, 0.02, r'$\theta = 0^{+}$', transform = axs2[2, 1].transAxes, color = 'white')
        text7 = axs2[3, 0].text(0.02, 0.02, r'$\theta = 0^{-}$', transform = axs2[3, 0].transAxes, color = 'white')
        text8 = axs2[3, 1].text(0.02, 0.02, r'$\theta = 180$\textdegree', transform = axs2[3, 1].transAxes, color = 'white')

        for ax in fig2.axes:
            ax.axis('off')
            ax.axvline(x = shifted_proj_img_array_element_to_align_with_theta_0_0.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
            ax.axhline(y = shifted_proj_img_array_element_to_align_with_theta_0_0.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')

        # fig2.tight_layout()

        # plt.show()
        plt.close(fig2)

        fig_filename = os.path.join(dir_path, f'shifted_center_of_rotation_figure.svg')
        
        fig2.savefig(fig_filename)

    else:
        theta_pair_idx = ppu.find_theta_combos(theta_array)[0]

        shifted_proj_img_array_element_to_align_with_theta_0_0 = shifted_proj_img_array_element_to_align_with[theta_pair_idx[0]]
        shifted_proj_img_array_element_to_align_with_theta_0_1 = shifted_proj_img_array_element_to_align_with[theta_pair_idx[1]]

        shifted_proj_img_array_element_to_align_with_theta_0_0_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_0_0)
        shifted_proj_img_array_element_to_align_with_theta_0_1_norm = ppu.normalize_array_for_rgb(shifted_proj_img_array_element_to_align_with_theta_0_1)

        shifted_proj_img_array_element_to_align_with_theta_0_0_rgb = np.dstack((shifted_proj_img_array_element_to_align_with_theta_0_0_norm, np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm), np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm)))
        shifted_proj_img_array_element_to_align_with_theta_0_1_rgb = np.dstack((np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_1_norm), shifted_proj_img_array_element_to_align_with_theta_0_1_norm, np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_1_norm)))
        
        overlay_shifted_0 = np.dstack((shifted_proj_img_array_element_to_align_with_theta_0_0_norm, shifted_proj_img_array_element_to_align_with_theta_0_1_norm, np.zeros_like(shifted_proj_img_array_element_to_align_with_theta_0_0_norm)))

        fig1, axs1 = plt.subplots(1, 3)

        im1_1 = axs1[0].imshow(shifted_proj_img_array_element_to_align_with_theta_0_0_rgb)
        im1_2 = axs1[1].imshow(shifted_proj_img_array_element_to_align_with_theta_0_1_rgb)
        im1_3 = axs1[2].imshow(overlay_shifted_0)

        text1 = axs1[0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_pair_idx[0]]), transform = axs1[0].transAxes, color = 'white')
        text2 = axs1[1].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[theta_pair_idx[1]]), transform = axs1[1].transAxes, color = 'white')

        for ax in fig1.axes:
            ax.axis('off')
            ax.axvline(x = shifted_proj_img_array_element_to_align_with_theta_0_0.shape[1]//2, color = 'white', linewidth = 2, linestyle = '--')
            ax.axhline(y = shifted_proj_img_array_element_to_align_with_theta_0_0.shape[0]//2, color = 'white', linewidth = 2, linestyle = '--')
        
        fig1.tight_layout()

        plt.show()
        # plt.close(fig1)

        # fig_filename = os.path.join(dir_path, f'shifted_center_of_rotation_figure.svg')
        
        # fig1.savefig(fig_filename)

    return

def create_exp_synth_proj_data_gif(dir_path,
                                   exp_proj,
                                   synth_proj,
                                   theta_array,
                                   fps):
    
    n_theta, n_slices, n_columns = exp_proj[0].shape

    exp_proj_first_iter = exp_proj[0]
    synth_proj_first_iter = synth_proj[0]

    exp_proj_final_iter = exp_proj[-1]
    synth_proj_final_iter = synth_proj[-1]

    exp_proj_first_iter_norm = ppu.normalize_array_for_gif(exp_proj_first_iter[0])
    synth_proj_first_iter_norm = ppu.normalize_array_for_gif(synth_proj_first_iter[0])
    exp_proj_final_iter_norm = ppu.normalize_array_for_gif(exp_proj_final_iter[0])
    synth_proj_final_iter_norm = ppu.normalize_array_for_gif(synth_proj_final_iter[0])

    exp_proj_first_iter_rgb = np.dstack((exp_proj_first_iter_norm, np.zeros_like(exp_proj_first_iter_norm), np.zeros_like(exp_proj_first_iter_norm)))
    synth_proj_first_iter_rgb = np.dstack((np.zeros_like(synth_proj_first_iter_norm), synth_proj_first_iter_norm, np.zeros_like(synth_proj_first_iter_norm)))
    exp_proj_final_iter_rgb = np.dstack((exp_proj_final_iter_norm, np.zeros_like(exp_proj_final_iter_norm), np.zeros_like(exp_proj_final_iter_norm)))
    synth_proj_final_iter_rgb = np.dstack((np.zeros_like(synth_proj_final_iter_norm), synth_proj_final_iter_norm, np.zeros_like(synth_proj_final_iter_norm)))

    overlay_exp_synth_proj_first_iter = np.dstack((exp_proj_first_iter_norm, synth_proj_first_iter_norm, np.zeros_like(exp_proj_first_iter_norm)))
    overlay_exp_synth_proj_final_iter = np.dstack((exp_proj_final_iter_norm, synth_proj_final_iter_norm, np.zeros_like(exp_proj_final_iter_norm)))

    fig1, axs1 = plt.subplots(2, 3)

    img1_1 = axs1[0, 0].imshow(exp_proj_first_iter_rgb)
    img1_2 = axs1[0, 1].imshow(synth_proj_first_iter_rgb)
    img1_3 = axs1[0, 2].imshow(overlay_exp_synth_proj_first_iter)
    img1_4 = axs1[1, 0].imshow(exp_proj_final_iter_rgb)
    img1_5 = axs1[1, 1].imshow(synth_proj_final_iter_rgb)
    img1_6 = axs1[1, 2].imshow(overlay_exp_synth_proj_final_iter)

    text1 = axs1[0, 0].text(0.02, 0.02, r'$\theta = {0}$\textdegree'.format(theta_array[0]), transform = axs1[0, 0].transAxes, color = 'white')
    
    theta_frames = []
    
    for ax in fig1.axes:
        ax.axis('off')
        ax.axvline(x = n_columns//2, color = 'red', linewidth = 2)
        ax.axhline(y = n_slices//2, color = 'red', linewidth = 2)
    
    for theta_idx in range(n_theta):
        exp_proj_first_iter_norm = ppu.normalize_array_for_gif(exp_proj_first_iter[theta_idx])
        synth_proj_first_iter_norm = ppu.normalize_array_for_gif(synth_proj_first_iter[theta_idx])
        exp_proj_final_iter_norm = ppu.normalize_array_for_gif(exp_proj_final_iter[theta_idx])
        synth_proj_final_iter_norm = ppu.normalize_array_for_gif(synth_proj_final_iter[theta_idx])

        exp_proj_first_iter_rgb = np.dstack((exp_proj_first_iter_norm, np.zeros_like(exp_proj_first_iter_norm), np.zeros_like(exp_proj_first_iter_norm)))
        synth_proj_first_iter_rgb = np.dstack((np.zeros_like(synth_proj_first_iter_norm), synth_proj_first_iter_norm, np.zeros_like(synth_proj_first_iter_norm)))
        exp_proj_final_iter_rgb = np.dstack((exp_proj_final_iter_norm, np.zeros_like(exp_proj_final_iter_norm), np.zeros_like(exp_proj_final_iter_norm)))
        synth_proj_final_iter_rgb = np.dstack((np.zeros_like(synth_proj_final_iter_norm), synth_proj_final_iter_norm, np.zeros_like(synth_proj_final_iter_norm)))

        overlay_exp_synth_proj_first_iter = np.dstack((exp_proj_first_iter_norm, synth_proj_first_iter_norm, np.zeros_like(exp_proj_first_iter_norm)))
        overlay_exp_synth_proj_final_iter = np.dstack((exp_proj_final_iter_norm, synth_proj_final_iter_norm, np.zeros_like(exp_proj_final_iter_norm)))

        img1_1.set_data(exp_proj_first_iter_rgb)
        img1_2.set_data(synth_proj_first_iter_rgb)
        img1_3.set_data(overlay_exp_synth_proj_first_iter)
        img1_4.set_data(exp_proj_final_iter_rgb)
        img1_5.set_data(synth_proj_final_iter_rgb)
        img1_6.set_data(overlay_exp_synth_proj_final_iter)
        
        text1.set_text(r'$\theta = {0}$\textdegree'.format(theta_array[theta_idx]))

        fig1.canvas.draw()

        frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]

        theta_frames.append(frame1)
        
    plt.close(fig1)

    gif_filename = os.path.join(dir_path, f'exp_synth_proj_data.gif')

    iio2.mimsave(gif_filename, theta_frames, fps = fps)

    return

def create_incremental_shifts_vs_angle_plot(dir_path,
                                            net_x_shift_array,
                                            net_y_shift_array,
                                            correction_type,
                                            theta_array):
    
    fig1, axs1 = plt.subplots()

    vmin = np.min([net_x_shift_array, net_y_shift_array])
    vmax = np.max([net_x_shift_array, net_y_shift_array])
    
    if correction_type == 'iter_reproj':
        color_array = ['k', 'b', 'g', 'r', 'm']
    
        for iter_idx in range(net_x_shift_array.shape[0]):
            if iter_idx == 0:
                axs1.plot(theta_array, net_x_shift_array[iter_idx], marker = 'o', markersize = 5, linewidth = 2, color = color_array[iter_idx], label = r'$\delta x$')
                axs1.plot(theta_array, net_y_shift_array[iter_idx], linestyle = '--', marker = 'o', markersize = 3, linewidth = 2, color = color_array[iter_idx], label = r'$\delta y$')
            
            else:
                axs1.plot(theta_array, net_x_shift_array[iter_idx], marker = 'o', markersize = 5, linewidth = 2, color = color_array[iter_idx])
                axs1.plot(theta_array, net_y_shift_array[iter_idx], linestyle = '--', marker = 'o', markersize = 3, linewidth = 2, color = color_array[iter_idx])

        axs1.set_ylim(vmin, vmax)
    
    elif correction_type == 'adjacent_angle_jitter':
        axs1.scatter(theta_array, net_x_shift_array, s = 5, marker = 'o', linewidth = 2, color = 'k', label = r'$\delta x$')
        axs1.scatter(theta_array, net_y_shift_array, s = 5, marker = 'o', linewidth = 2, color = 'r', label = r'$\delta y$')

        axs1.set_ylim(vmin, vmax)
    
    axs1.set_xlim(-180, 180)
    axs1.tick_params(axis = 'both', which = 'major', labelsize = 14)
    axs1.tick_params(axis = 'both', which = 'minor', labelsize = 14)
    axs1.set_xlabel(r'$\theta$ (\textdegree{})', fontsize = 16)
    axs1.set_ylabel(r'Incremental shift', fontsize = 16)
    axs1.legend(frameon = False, fontsize = 14)

    fig1.tight_layout()
    
    # plt.show()
    plt.close(fig1)

    fig_filename = os.path.join(dir_path, f'incremental_shifts_vs_angle_plot.svg')
    
    fig1.savefig(fig_filename)

    return

def create_final_aligned_proj_data_gif(dir_path,
                                       aligning_element,
                                       raw_proj,
                                       aligned_proj,
                                       theta_array,
                                       fps):
    
    n_theta, n_slices, n_columns = aligned_proj.shape
    n_slices_raw, n_columns_raw = raw_proj.shape[0], raw_proj.shape[2]
    
    vmin = aligned_proj.min()
    vmax = aligned_proj.max()

    fig1, axs1 = plt.subplots(1, 2)

    img1_1 = axs1[0].imshow(raw_proj[0], vmin = vmin, vmax = vmax)
    img1_2 = axs1[1].imshow(aligned_proj[0], vmin = vmin, vmax = vmax)

    for ax in fig1.axes:
        ax.axis('off')
        ax.axvline(x = n_columns_raw//2, color = 'red', linewidth = 2)
        ax.axhline(y = n_slices_raw//2, color = 'red', linewidth = 2)
    
    axs1[0].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    axs1[1].set_title(r'{0} (aligned)'.format(aligning_element), fontsize = 14)
    
    text1 = axs1[0].text(0.02, 0.02, r'$\theta ={0}$\textdegree'.format(theta_array[0]), transform = axs1[0].transAxes, color = 'white')

    theta_frames = []

    for theta_idx in range(n_theta):
        img1_1.set_data(raw_proj[theta_idx])
        img1_2.set_data(aligned_proj[theta_idx])
        text1.set_text(r'$\theta ={0}$\textdegree'.format(theta_array[theta_idx]))

        fig1.canvas.draw()

        frame1 = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]

        theta_frames.append(frame1)

    plt.close(fig1)

    gif_filename = os.path.join(dir_path, f'final_aligned_proj_data_comp_aligning_element_{aligning_element}.gif')

    iio2.mimsave(gif_filename, theta_frames, fps = fps)

    # print('Creating final aligned projection sinogram GIF...')

    # fig2, axs2 = plt.subplots(1, 2)

    # im2_1 = axs2[0].imshow(raw_proj[:, 0], vmin = vmin, vmax = vmax, origin = 'lower', extent = [0, n_slices_raw - 1, -180, 180], aspect = 10)
    # im2_2 = axs2[1].imshow(aligned_proj[:, 0], vmin = vmin, vmax = vmax, origin = 'lower', extent = [0, n_slices_raw - 1, -180, 180], aspect = 10)

    # text2 = axs2[0].text(0.02, 0.02, r'Slice index 0/{0}'.format(n_slices_raw - 1), transform = axs2[0].transAxes, color = 'white')
    
    # axs2[0].set_title(r'{0}'.format(aligning_element), fontsize = 14)
    # axs2[1].set_title(r'{0} (aligned)'.format(aligning_element), fontsize = 14)

    # slice_frame_list = []

    # for ax in fig2.axes:
    #     ax.set_xlabel(r'Pixel index', fontsize = 14)
    #     ax.set_ylabel(r'$\theta$ (\textdegree)', fontsize = 14)

    # for slice_idx in range(n_slices_raw):
    #     im2_1.set_data(raw_proj[:, slice_idx])
    #     im2_2.set_data(aligned_proj[:, slice_idx])
    #     text2.set_text(r'Slice index {0}/{1}'.format(slice_idx, n_slices_raw - 1))

    #     fig2.canvas.draw()

    #     slice_frame = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3]

    #     slice_frame_list.append(slice_frame)

    # plt.close(fig2)

    # gif_filename = os.path.join(dir_path, f'final_aligned_sinogram_data_comp_aligning_element_{aligning_element}.gif')

    # iio2.mimsave(gif_filename, slice_frame_list, fps = fps)

    return

def create_gridrec_density_map_gif(dir_path,
                                   gridrec_density_maps, 
                                   desired_element, 
                                   elements_xrf, 
                                   fps):    

    desired_element_idx_1 = elements_xrf.index(desired_element[0])
    desired_element_idx_2 = elements_xrf.index(desired_element[1])
    desired_element_idx_3 = elements_xrf.index(desired_element[2])
    desired_element_idx_4 = elements_xrf.index(desired_element[3])
    
    density_map_1 = gridrec_density_maps[desired_element_idx_1]
    density_map_2 = gridrec_density_maps[desired_element_idx_2]
    density_map_3 = gridrec_density_maps[desired_element_idx_3]
    density_map_4 = gridrec_density_maps[desired_element_idx_4]

    n_slices = density_map_1.shape[0]

    vmin_1 = density_map_1.min()
    vmax_1 = density_map_1.max()
    vmin_2 = density_map_2.min()
    vmax_2 = density_map_2.max()
    vmin_3 = density_map_3.min()
    vmax_3 = density_map_3.max()
    vmin_4 = density_map_4.min()
    vmax_4 = density_map_4.max()

    fig, axs = plt.subplots(2, 2)

    img1_1 = axs[0, 0].imshow(density_map_1[0], vmin = vmin_1, vmax = vmax_1)
    img1_2 = axs[0, 1].imshow(density_map_2[0], vmin = vmin_2, vmax = vmax_2)
    img1_3 = axs[1, 0].imshow(density_map_3[0], vmin = vmin_3, vmax = vmax_3)
    img1_4 = axs[1, 1].imshow(density_map_4[0], vmin = vmin_4, vmax = vmax_4)
    
    text = axs.text(0.02, 0.02, r'Slice index 0/{0}'.format(n_slices - 1), transform = axs.transAxes, color = 'white')
    
    axs.axis('off')
    axs.set_title(r'{0}'.format(desired_element), fontsize = 14)

    divider1 = make_axes_locatable(axs[0, 0])
    divider2 = make_axes_locatable(axs[0, 1])
    divider3 = make_axes_locatable(axs[1, 0])
    divider4 = make_axes_locatable(axs[1, 1])

    cax1 = divider1.append_axes('right', size = '5%', pad = 0.05)
    cax2 = divider2.append_axes('right', size = '5%', pad = 0.05)
    cax3 = divider3.append_axes('right', size = '5%', pad = 0.05)
    cax4 = divider4.append_axes('right', size = '5%', pad = 0.05)

    cbar1 = fig.colorbar(img1_1, cax = cax1)
    cbar2 = fig.colorbar(img1_2, cax = cax2)
    cbar3 = fig.colorbar(img1_3, cax = cax3)
    cbar4 = fig.colorbar(img1_4, cax = cax4)

    cbar1.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)
    cbar2.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)
    cbar3.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)
    cbar4.ax.set_title(r'g/cm\textsuperscript{3}', fontsize = 16)

    for idx, ax in enumerate(axs):
        ax.set_title(r'{0}'.format(desired_element[idx]))
        ax.axis('off')

    slice_frames = []

    for slice_idx in range(n_slices):
        img1_1.set_data(density_map_1[slice_idx])
        img1_2.set_data(density_map_2[slice_idx])
        img1_3.set_data(density_map_3[slice_idx])
        img1_4.set_data(density_map_4[slice_idx])
        
        text.set_text(r'Slice index {0}/{1}'.format(slice_idx, n_slices - 1))

        fig.canvas.draw()
        
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        slice_frames.append(frame)

    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'gridrec_density_map_{desired_element}.gif')

    iio2.mimsave(gif_filename, slice_frames, fps = fps)

    return
    # plt.plot(theta, net_y_shift_array, 'ko', markersize = 3, linewidth = 2)
    # plt.xlim(-180, 180)
    # plt.ylim(-25, 25)
    # plt.xlabel(r'$\theta$ (\textdegree{})', fontsize = 16)
    # plt.ylabel(r'$\delta y$ (cumulative)', fontsize = 16)
    # plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
    # plt.tick_params(axis = 'both', which = 'minor', labelsize = 14)
    # plt.tight_layout()
    # plt.show()