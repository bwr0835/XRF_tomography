import numpy as np, xrf_xrt_preprocess_file_util as futil, h5py, os, sys

input_dir_path = '/home/bwr0835/2_ide_realigned_data_cor_only'
input_aux_and_output_dir_path = '/raid/users/roter/Jacobsen/img.dat'

n_det = 3

input_file_path_xrf_aux = os.path.join(input_aux_and_output_dir_path, '2_ide_aggregate_xrf.h5')
input_file_path_xrt_aux = os.path.join(input_aux_and_output_dir_path, '2_ide_aggregate_xrt.h5')

elements_xrf, intensity_xrf_aux, theta, incident_energy_keV, _, _, _ = futil.extract_h5_aggregate_xrf_data(input_file_path_xrf_aux)
elements_xrt, intensity_xrt, _, _, _, _, _ = futil.extract_h5_aggregate_xrt_data(input_file_path_xrt_aux)

n_elements, n_theta, n_slices, n_columns = intensity_xrf_aux.shape

n_slices -= 1 # Net change in slices due to padding and cropping
n_columns += 1 # For padding

with h5py.File(os.path.join(input_aux_and_output_dir_path, '2_ide_aggregate_xrf_xrt_aligned_two_det_elements.h5'), 'w') as f:
    exchange = f.create_group('exchange')

    data_h5 = exchange.create_group('data')
    elements_h5 = exchange.create_group('elements')
    theta_h5 = exchange.create_dataset('theta', data = theta)
    
    elements_xrf = elements_h5.create_dataset('xrf', data = elements_xrf)
    elements_xrt = elements_h5.create_dataset('xrt', data = elements_xrt)

    intensity_xrf_h5 = data_h5.create_dataset('xrf', shape = (n_det, n_elements, n_theta, n_slices, n_columns), dtype = 'float32')
    intensity_xrt_h5 = data_h5.create_dataset('xrt', data = intensity_xrt, dtype = 'float32')

    intensity_xrf_h5.attrs['det_elements_dim_0'] = ['0', '1', 'sum']

    data_h5.attrs['incident_energy_keV'] = incident_energy_keV

    for det in range(n_det):
        print(f'Processing det {det}...', end = '\r', flush = True)
        
        if det == 2:
            full_input_dir_path = f'{input_dir_path}_det_elements_0_1_sum_07_10_2026'

        else:
            full_input_dir_path = f'{input_dir_path}_det_element_{det}_07_10_2026'

        input_file_path = os.path.join(full_input_dir_path, 'aligned_data', 'aligned_aggregate_xrf_xrt.h5')

        with h5py.File(input_file_path, 'r') as ff:
            data = ff['exchange/data']
            
            intensity_xrf_h5[det] = data['xrf'][det]

            if det == 2:
                data_h5.attrs['bottom_edge_cropped_final'] = data['bottom_edge_cropped_final']
                data_h5.attrs['top_edge_cropped_final'] = data['top_edge_cropped_final']
                data_h5.attrs['left_edge_cropped_final'] = data['left_edge_cropped_final']
                data_h5.attrs['right_edge_cropped_final'] = data['right_edge_cropped_final']
                data_h5.attrs['incident_intensity_photons'] = data['incident_intensity_photons']
