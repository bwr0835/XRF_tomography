import numpy as np, tomopy as tomo, xraylib as xrl, h5py, os, sys

from skimage import measure as meas

def extract_proj_data(dir_path, xrt = False):
    with h5py.File(os.path.join(dir_path, 'aligned_data', 'aligned_aggregate_xrf_xrt.h5'), "r") as f:
        exchange = f['exchange']

        elements_xrf = list(exchange['elements/xrf'].asstr()[:])
        
        data_xrf = exchange['data/xrf'][()]
        
        if xrt:
            data_xrt = exchange['data/xrt'][()][1]
        
        else:
            data_xrt = None
        
        theta = exchange['theta'][()]

    return elements_xrf, data_xrf, data_xrt, theta

def extract_recon_data(dir_path):
    with h5py.File(os.path.join(dir_path, 'gridrec_density_maps.h5'), "r") as f:
        sample = f['sample']

        data = sample['densities'][()]
        elements = list(sample['elements'].asstr()[:])

    return elements, data

def downsample_data(data, row_start, row_stop, downsample_factor, xrt = False):
    if not xrt:
        raw = data[:, :, row_start:row_stop]

        downsampled_data = meas.block_reduce(raw, (1, 1, downsample_factor, downsample_factor), np.mean)
    
    else:
        raw = data[:, row_start:row_stop]

        downsampled_data = meas.block_reduce(raw, (1, downsample_factor, downsample_factor), np.mean)

    return downsampled_data

def create_density_map(recon_array, element):
    n_slices, n_columns = recon_array.shape[1:]

    rho = np.zeros((n_slices, n_columns, n_columns))
    
    if '_' in element:
        element = element.split('_')[0]

    vmin = recon_array.min()
    vmax = recon_array.max()
    
    if element == 'Si':
        rho_g_cm3 = 2.26 # Ideal amorphous Si (97% of crystalline Si density)

    else:
        rho_g_cm3 = xrl.ElementDensity(xrl.SymbolToAtomicNumber(element))

    density_map_g_cm3 = rho_g_cm3*(recon_array - vmin)/(vmax - vmin) # Initial guess of density based on 0-100% concentration of each element

    return density_map_g_cm3

def export_recon(dir_path, xrf_density, opt_dens, elements_xrf):
    with h5py.File(os.path.join(dir_path, 'mlem_recon_downsampled.h5'), "w") as f:
        sample = f.create_group('sample')

        xrf = sample.create_group('xrf')
        xrt = sample.create_group('xrt')

        xrf.create_dataset('densities_ug_cm3', data = xrf_density.astype('f4'))
        xrf.create_dataset('elements', data = np.array(elements_xrf).astype('S5'))
        xrt.create_dataset('opt_dens', data = opt_dens.astype('f4'))

def export_recon_append(dir_path, xrf_density, xrt_recon, elements_xrf):
    with h5py.File(os.path.join(dir_path, 'mlem_recon_downsampled.h5'), "r+") as f:
        xrf = f['sample/xrf']
        xrt = f['sample/xrt']
        
        elements = list(xrf['elements'].asstr()[:]) + list(elements_xrf)
        
        density = np.concatenate((xrf['densities_ug_cm3'][()], xrf_density), axis = 0)
        
        order = np.argsort([xrl.SymbolToAtomicNumber(element.split('_')[0]) for element in elements])
        
        elements = [elements[i] for i in order]
        density = density[order]
        
        del xrf['elements']
        del xrf['densities_ug_cm3']
        
        xrf.create_dataset('densities_ug_cm3', data = density.astype('f4'))
        xrf.create_dataset('elements', data = np.array(elements).astype('S5'))
        
        if 'opt_dens' in xrt:
            del xrt['opt_dens']
        
        xrt.create_dataset('opt_dens', data = xrt_recon.astype('f4'))

def overwrite_opt_dens_recon(dir_path, xrt_recon):
    with h5py.File(os.path.join(dir_path, 'mlem_recon_downsampled.h5'), "r+") as f:
        xrt = f['sample/xrt']

        data = np.asarray(xrt_recon, dtype = 'f4')

        if 'opt_dens' in xrt and xrt['opt_dens'].shape == data.shape:
            xrt['opt_dens'][...] = data

        else:
            if 'opt_dens' in xrt:
                del xrt['opt_dens']

            xrt.create_dataset('opt_dens', data = data)

downsample_factor = 4
row_start = 0
row_stop = 287

I0 = 8.6776e6

dir_path_det_element_0 = '/home/bwr0835/2_ide_realigned_data_cor_manual_09_03_2026_det_element_0_corrected_order_2'
dir_path_det_element_1 = '/home/bwr0835/2_ide_realigned_data_cor_manual_09_03_2026_det_element_1_corrected_order_2'
dir_path_det_elements_0_1_sum = '/home/bwr0835/2_ide_realigned_data_cor_manual_09_03_2026_det_elements_0_1_sum_corrected_order_2'

dir_path_list = [dir_path_det_element_0, dir_path_det_element_1, dir_path_det_elements_0_1_sum]



elements, xrf_proj_data_det_element_0, xrt_proj_data, theta = extract_proj_data(dir_path_det_element_0, xrt = True)
_, xrf_proj_data_det_element_1, _, _ = extract_proj_data(dir_path_det_element_1)
_, xrf_proj_data_det_elements_0_1_sum, _, _ = extract_proj_data(dir_path_det_elements_0_1_sum)

# _, gridrec_recon_data_det_element_0 = extract_recon_data(dir_path_det_element_0)
# _, gridrec_recon_data_det_element_1 = extract_recon_data(dir_path_det_element_1)
# _, gridrec_recon_data_det_elements_0_1_sum = extract_recon_data(dir_path_det_elements_0_1_sum)

desired_elements_xrf = ['Si', 'Ti', 'Fe', 'Ba_L']

desired_elements_idx_xrf = [elements.index(element) for element in desired_elements_xrf]

xrf_proj_data_elements_of_interest_det_element_0 = xrf_proj_data_det_element_0[desired_elements_idx_xrf]
xrf_proj_data_elements_of_interest_det_element_1 = xrf_proj_data_det_element_1[desired_elements_idx_xrf]
xrf_proj_data_elements_of_interest_det_elements_0_1_sum = xrf_proj_data_det_elements_0_1_sum[desired_elements_idx_xrf]

# recon_data_elements_of_interest_det_element_0 = gridrec_recon_data_det_element_0[desired_elements_idx]
# recon_data_elements_of_interest_det_element_1 = gridrec_recon_data_det_element_1[desired_elements_idx]
# recon_data_elements_of_interest_det_elements_0_1_sum = gridrec_recon_data_det_elements_0_1_sum[desired_elements_idx]

xrf_proj_data_elements_of_interest_list = [xrf_proj_data_elements_of_interest_det_element_0, 
                                           xrf_proj_data_elements_of_interest_det_element_1, 
                                           xrf_proj_data_elements_of_interest_det_elements_0_1_sum]

# recon_data_elements_of_interest_list = [recon_data_elements_of_interest_det_element_0, 
#                                         recon_data_elements_of_interest_det_element_1, 
#                                         recon_data_elements_of_interest_det_elements_0_1_sum]

n_elements_xrf, n_theta, n_slices, n_columns = xrf_proj_data_elements_of_interest_det_element_0.shape

# data/xrt[1] is already -log(I/I0). Do not apply that conversion again.
opt_dens = np.array(xrt_proj_data, dtype = np.float32, copy = True)

n_neg = int(np.count_nonzero(opt_dens < 0))

opt_dens[~np.isfinite(opt_dens)] = 0
opt_dens[opt_dens < 0] = 0

print(f'Clipped {n_neg} negative optical-density pixels ({100*n_neg/opt_dens.size:.2f}%) to 0')

n_iterations = 100

for index, proj_dataset in enumerate(xrf_proj_data_elements_of_interest_list):
    print(f'Processing {dir_path_list[index]}...')
    
    downsampled_proj_dataset = downsample_data(proj_dataset, row_start, row_stop, downsample_factor)
    
    # downsampled_proj_dataset = proj_dataset
    # downsampled_xrt_proj_dataset = opt_dens
    if index == 0:
        downsampled_xrt_proj_dataset = downsample_data(opt_dens, row_start, row_stop, downsample_factor)

        mlem_recon_xrt = tomo.recon(downsampled_xrt_proj_dataset, theta*np.pi/180, algorithm = 'mlem', num_iter = n_iterations)

    if index == 0:
        n_slices, n_columns = downsampled_proj_dataset.shape[2:]
        
        density_xrf = np.zeros((n_elements_xrf, n_slices, n_columns, n_columns))

    for idx, element in enumerate(desired_elements_xrf):
        mlem_recon_xrf = tomo.recon(downsampled_proj_dataset[idx], theta*np.pi/180, algorithm = 'mlem', num_iter = n_iterations)

        density_xrf[idx] = create_density_map(mlem_recon_xrf, element)

        print(f'Processed {element}...')
    
    export_recon(dir_path_list[index], density_xrf, mlem_recon_xrt, desired_elements_xrf)
    # overwrite_opt_dens_recon(dir_path_list[index], mlem_recon_xrt)