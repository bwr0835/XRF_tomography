import numpy as np, \
       xrf_xrt_preprocess_file_util as futil, \
       h5py, \
       os, \
       sys


input_dir_path = '/raid/users/roter/Jacobsen/img.dat'

aggregate_xrf_h5_file_path = os.path.join(input_dir_path, '2_ide_aggregate_xrf.h5')
output_file_path = os.path.join(input_dir_path, '2_ide_aggregate_xrf_det_elements_0_1_sum.h5')

elements_xrf, intensity_xrf, theta, incident_energy_keV, _, dataset_type, filenames = futil.extract_h5_aggregate_xrf_data(aggregate_xrf_h5_file_path, filename_array = True)

n_det_elements = 2

n_elements, n_theta, n_slices, n_columns = intensity_xrf.shape

intensity_xrf_sum = np.zeros((n_elements, n_theta, n_slices, n_columns))

for theta_idx, filename in enumerate(filenames):
       print(f'Processing angle {theta_idx + 1}/{n_theta}', end = '\r', flush = True)

       for det in range(n_det_elements):
              with h5py.File(os.path.join(input_dir_path, f'{filename}{det}'), 'r') as f:
                     if theta_idx == 0:
                            elements_xrf_aux = list(f['MAPS/channel_names'].asstr()[:])

                            elements_of_interest_idx = [elements_xrf.index(element) for element in elements_xrf_aux if element in elements_xrf]
                     
                     intensity = f['MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec'][()][elements_of_interest_idx, :, :-2]
              
              intensity_xrf_sum[:, theta_idx] += intensity


with h5py.File(output_file_path, 'w') as f:
    f.create_dataset('filenames', data = filenames)

    exchange = f.create_group('exchange')

    exchange.create_dataset('data', data = intensity_xrf_sum, compression = 'gzip', compression_opts = 6)
    exchange.create_dataset('elements', data = elements_xrf)
    exchange.create_dataset('theta', data = theta)

    exchange['data'].attrs['dataset_type'] = 'xrf'
    exchange['data'].attrs['incident_energy_keV'] = incident_energy_keV

    exchange['data'].attrs['us_ic_scaler_name'] = 'US_IC'
    exchange['data'].attrs['xrt_signal_name'] = 'DS_IC'
    exchange['data'].attrs['xrt_photon_counting'] = False
    exchange['data'].attrs['xrt_instrument'] = 'ion_chamber'