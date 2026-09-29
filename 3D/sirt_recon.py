import numpy as np, tomopy as tomo, xraylib as xrl, h5py, os, sys

def extract_xrf_proj_data(dir_path):
    with h5py.File(os.path.join(dir_path, 'aligned_data', 'aligned_aggregate_xrf_xrt.h5'), "r") as f:
        exchange = f['exchange']

        elements = list(exchange['elements/xrf'].asstr()[:])
        

        data = exchange['data/xrf'][()]
        theta = exchange['theta'][()]

    return elements, data, theta

def extract_xrf_recon_data(dir_path):
    with h5py.File(os.path.join(dir_path, 'gridrec_density_maps.h5'), "r") as f:
        sample = f['sample']

        data = sample['densities'][()]
        elements = list(sample['elements'].asstr()[:])

    return elements, data

def downsample_data(data, row_start, row_stop, downsample_factor):
    new_row_stop = (row_stop//downsample_factor)*downsample_factor
    
    raw = data[:, :, row_start:new_row_stop]

    c, a, h, w = raw.shape

    h_new, w_new = h//downsample_factor, w//downsample_factor

    downsampled_data = (raw.reshape(c, a, h_new, downsample_factor, w_new, downsample_factor)).mean(axis = (3, 5))

    return downsampled_data

def create_density_map(recon_array, element):
    n_elements_xrf, _, n_slices, n_columns = recon_array.shape

    rho = np.zeros((n_elements_xrf, n_slices, n_columns, n_columns))
    
    if '_' in element:
        element = element.split('_')[0]

    vmin = recon_array.min()
    vmax = recon_array.max()
        
    rho = xrl.ElementDensity(xrl.SymbolToAtomicNumber(element))*(recon_array - vmin)/(vmax - vmin) # Initial guess of density based on 0-100% concentration of each element

    return rho

def export_density_maps(dir_path, density, elements):
    with h5py.File(os.path.join(dir_path, 'sirt_recon.h5'), "w") as f:
        sample = f.create_group('sample')

        sample.create_dataset('densities', data = density.astype('f4'))
        sample.create_dataset('elements', data = np.array(elements).astype('S5'))

downsample_factor = 4
row_start = 0
row_stop = 287

dir_path_det_element_0 = '/home/bwr0835/2_ide_realigned_data_cor_manual_09_03_2026_det_element_0_corrected_order_2'
dir_path_det_element_1 = '/home/bwr0835/2_ide_realigned_data_cor_manual_09_03_2026_det_element_1_corrected_order_2'
dir_path_det_elements_0_1_sum = '/home/bwr0835/2_ide_realigned_data_cor_manual_09_03_2026_det_elements_0_1_sum_corrected_order_2'

dir_path_list = [dir_path_det_element_0, dir_path_det_element_1, dir_path_det_elements_0_1_sum]

elements, proj_data_det_element_0, theta = extract_xrf_proj_data(dir_path_det_element_0)
_, proj_data_det_element_1, _ = extract_xrf_proj_data(dir_path_det_element_1)
_, proj_data_det_elements_0_1_sum, _ = extract_xrf_proj_data(dir_path_det_elements_0_1_sum)

# _, gridrec_recon_data_det_element_0 = extract_xrf_recon_data(dir_path_det_element_0)
# _, gridrec_recon_data_det_element_1 = extract_xrf_recon_data(dir_path_det_element_1)
# _, gridrec_recon_data_det_elements_0_1_sum = extract_xrf_recon_data(dir_path_det_elements_0_1_sum)

desired_elements = ['Si', 'Fe']

desired_elements_idx = [elements.index(element) for element in desired_elements]

proj_data_elements_of_interest_det_element_0 = proj_data_det_element_0[desired_elements_idx]
proj_data_elements_of_interest_det_element_1 = proj_data_det_element_1[desired_elements_idx]
proj_data_elements_of_interest_det_elements_0_1_sum = proj_data_det_elements_0_1_sum[desired_elements_idx]

# recon_data_elements_of_interest_det_element_0 = gridrec_recon_data_det_element_0[desired_elements_idx]
# recon_data_elements_of_interest_det_element_1 = gridrec_recon_data_det_element_1[desired_elements_idx]
# recon_data_elements_of_interest_det_elements_0_1_sum = gridrec_recon_data_det_elements_0_1_sum[desired_elements_idx]

proj_data_elements_of_interest_list = [proj_data_elements_of_interest_det_element_0, 
                                       proj_data_elements_of_interest_det_element_1, 
                                       proj_data_elements_of_interest_det_elements_0_1_sum]

# recon_data_elements_of_interest_list = [recon_data_elements_of_interest_det_element_0, 
#                                        recon_data_elements_of_interest_det_element_1, 
#                                        recon_data_elements_of_interest_det_elements_0_1_sum]

n_elements, n_theta, n_slices, n_columns = proj_data_elements_of_interest_det_element_0.shape

n_iterations = 100

for index, proj_dataset in enumerate(proj_data_elements_of_interest_list):
    print(f'Processing {dir_path_list[index]}...')
    
    downsampled_proj_dataset = downsample_data(proj_dataset, row_start, row_stop, downsample_factor)

    if index == 0:
        n_slices, n_columns = downsampled_proj_dataset.shape[2:]
        
        density = np.zeros((n_elements, n_slices, n_columns, n_columns))

    for idx, element in enumerate(desired_elements):
        sirt_recon = tomo.recon(downsampled_proj_dataset[idx], theta, algorithm = 'sirt', num_iter = n_iterations)

        density[idx] = create_density_map(sirt_recon, element)

        print(f'Processed {element}...')
    
    export_density_maps(dir_path_list[index], density, desired_elements)