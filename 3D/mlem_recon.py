import numpy as np, tomopy as tomo, xraylib as xrl, h5py, os, sys

from skimage import measure as meas

def extract_proj_data(dir_path, xrt = False):
    with h5py.File(os.path.join(dir_path, 'aligned_data', 'aligned_aggregate_xrf_xrt.h5'), "r") as f:
        exchange = f['exchange']

        elements_xrf = list(exchange['elements/xrf'].asstr()[:])
        
        data_xrf = exchange['data/xrf'][()]
        
        if xrt:
            data_xrt = exchange['data/xrt'][()]

            # elements/xrt is ['xrt_sig', 'opt_dens']
            xrt_sig = data_xrt[0]
            opt_dens = data_xrt[1]
        
        else:
            xrt_sig = None
            opt_dens = None
        
        theta = exchange['theta'][()]

    return elements_xrf, data_xrf, xrt_sig, opt_dens, theta

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

def create_density_map(recon_array, element, rho_g_cm3):
    n_slices, n_columns = recon_array.shape[1:]

    rho = np.zeros((n_slices, n_columns, n_columns))
    
    if '_' in element:
        element = element.split('_')[0]

    vmin = recon_array.min()
    vmax = recon_array.max()
    
    # if element == 'Si':
    #     rho_g_cm3 = 2.26 # Ideal amorphous Si (97% of crystalline Si density)

    # else:
    #     rho_g_cm3 = xrl.ElementDensity(xrl.SymbolToAtomicNumber(element))

    density_map_g_cm3 = rho_g_cm3*(recon_array - vmin)/(vmax - vmin) # Initial guess of density based on 0-100% concentration of each element

    return density_map_g_cm3

def write_xrt_dataset(xrt, name, recon):
    data = np.asarray(recon, dtype = 'f4')

    if name in xrt and xrt[name].shape == data.shape:
        xrt[name][...] = data
    else:
        if name in xrt:
            del xrt[name]

        xrt.create_dataset(name, data = data)

def export_recon(dir_path, xrf_density, xrt_data, opt_dens, elements_xrf):
    with h5py.File(os.path.join(dir_path, 'mlem_recon.h5'), "w") as f:
        sample = f.create_group('sample')

        xrf = sample.create_group('xrf')
        xrt = sample.create_group('xrt')

        xrf.create_dataset('densities_ug_cm3', data = xrf_density.astype('f4'))
        xrf.create_dataset('elements', data = np.array(elements_xrf).astype('S5'))
        xrt.create_dataset('opt_dens', data = opt_dens.astype('f4'))
        xrt.create_dataset('xrt_sig_photons', data = xrt_data.astype('f4'))

def export_recon_append(dir_path, xrf_density = None, elements_xrf = None, opt_dens = None, xrt_sig = None):
    if (xrf_density is None) != (elements_xrf is None):
        raise ValueError('xrf_density and elements_xrf must be passed together')

    if xrf_density is None and opt_dens is None and xrt_sig is None:
        return

    with h5py.File(os.path.join(dir_path, 'mlem_recon_downsampled.h5'), "r+") as f:
        sample = f.require_group('sample')

        if xrf_density is not None:
            xrf = sample.require_group('xrf')
            elements_new = list(elements_xrf)
            density_new = np.asarray(xrf_density)

            if 'elements' in xrf and 'densities_ug_cm3' in xrf:
                elements = list(xrf['elements'].asstr()[:])
                keep = [i for i, element in enumerate(elements_new) if element not in elements]

                if keep:
                    elements = elements + [elements_new[i] for i in keep]
                    density = np.concatenate((xrf['densities_ug_cm3'][()], density_new[keep]), axis = 0)

                    del xrf['elements']
                    del xrf['densities_ug_cm3']
                else:
                    elements = None
            else:
                elements = elements_new
                density = density_new

            if elements is not None:
                order = np.argsort([xrl.SymbolToAtomicNumber(element.split('_')[0]) for element in elements])

                elements = [elements[i] for i in order]
                density = density[order]

                xrf.create_dataset('densities_ug_cm3', data = density.astype('f4'))
                xrf.create_dataset('elements', data = np.array(elements).astype('S5'))

        if opt_dens is not None or xrt_sig is not None:
            xrt = sample.require_group('xrt')

            if opt_dens is not None and 'opt_dens' not in xrt:
                write_xrt_dataset(xrt, 'opt_dens', opt_dens)

            if xrt_sig is not None and 'xrt_sig' not in xrt:
                write_xrt_dataset(xrt, 'xrt_sig', xrt_sig)

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
row_stop = 288

I0 = 8.6776e6

dir_path_det_element_0 = '/home/bwr0835/2_ide_realigned_data_cor_manual_09_03_2026_det_element_0_corrected_order_2'
dir_path_det_element_1 = '/home/bwr0835/2_ide_realigned_data_cor_manual_09_03_2026_det_element_1_corrected_order_2'
dir_path_det_elements_0_1_sum = '/home/bwr0835/2_ide_realigned_data_cor_manual_09_03_2026_det_elements_0_1_sum_corrected_order_2'

dir_path_list = [dir_path_det_element_0, dir_path_det_element_1, dir_path_det_elements_0_1_sum]
# _, _, xrt_sig_proj, _, theta = extract_proj_data(dir_path_det_element_0, xrt = True)

# data/xrt[0] is transmission. Existing recon files already contain XRF and opt_dens.
# xrt_sig = np.array(xrt_sig_proj, dtype = np.float32, copy = True)

n_iterations = 100

# downsampled_xrt_sig = downsample_data(xrt_sig, row_start, row_stop, downsample_factor, xrt = True)
# downsampled_xrt_sig = xrt_sig
# mlem_recon_xrt_sig = tomo.recon(downsampled_xrt_sig, theta*np.pi/180, algorithm = 'mlem', num_iter = n_iterations)

# for dir_path in dir_path_list:
#     print(f'Appending transmission reconstruction to {dir_path}...')

#     export_recon_append(dir_path, xrt_sig = mlem_recon_xrt_sig)

elements, xrf_proj_data_det_element_0, xrt_proj, opt_dens_proj, theta = extract_proj_data(dir_path_det_element_0, xrt = True)
_, xrf_proj_data_det_element_1, _, _, _ = extract_proj_data(dir_path_det_element_1)
_, xrf_proj_data_det_elements_0_1_sum, _, _, _ = extract_proj_data(dir_path_det_elements_0_1_sum)

desired_elements_xrf = ['Si', 'Ti', 'Cr', 'Fe', 'Ni', 'Ba_L']
densities_xrf = np.array([0.813, 0.867, 1.404, 5.538, 0.624, 2.745])
#
desired_elements_idx_xrf = [elements.index(element) for element in desired_elements_xrf]
#
xrf_proj_data_elements_of_interest_det_element_0 = xrf_proj_data_det_element_0[desired_elements_idx_xrf]
xrf_proj_data_elements_of_interest_det_element_1 = xrf_proj_data_det_element_1[desired_elements_idx_xrf]
xrf_proj_data_elements_of_interest_det_elements_0_1_sum = xrf_proj_data_det_elements_0_1_sum[desired_elements_idx_xrf]
#
xrf_proj_data_elements_of_interest_list = [xrf_proj_data_elements_of_interest_det_element_0,
                                           xrf_proj_data_elements_of_interest_det_element_1,
                                           xrf_proj_data_elements_of_interest_det_elements_0_1_sum]
#
n_elements_xrf, n_theta, n_slices, n_columns = xrf_proj_data_elements_of_interest_det_element_0.shape
#
opt_dens = np.array(opt_dens_proj, dtype = np.float32, copy = True)
#
n_neg = int(np.count_nonzero(opt_dens < 0))
#
opt_dens[~np.isfinite(opt_dens)] = 0
opt_dens[opt_dens < 0] = 0
#
print(f'Clipped {n_neg} negative optical-density pixels ({100*n_neg/opt_dens.size:.2f}%) to 0')
#
for index, proj_dataset in enumerate(xrf_proj_data_elements_of_interest_list):
    print(f'Processing {dir_path_list[index]}...')

    # downsampled_proj_dataset = downsample_data(proj_dataset, row_start, row_stop, downsample_factor)
    downsampled_proj_dataset = proj_dataset

    if index == 0:
        # downsampled_xrt = downsample_data(xrt_proj, row_start, row_stop, downsample_factor, xrt = True)
        # downsampled_opt_dens = downsample_data(opt_dens, row_start, row_stop, downsample_factor, xrt = True)
        
        downsampled_xrt = xrt_proj
        downsampled_opt_dens = opt_dens

        mlem_recon_xrt = tomo.recon(downsampled_xrt, theta*np.pi/180, algorithm = 'mlem', num_iter = n_iterations)
        mlem_recon_opt_dens = tomo.recon(downsampled_opt_dens, theta*np.pi/180, algorithm = 'mlem', num_iter = n_iterations)
        
        n_slices, n_columns = downsampled_proj_dataset.shape[2:]

        density_xrf = np.zeros((n_elements_xrf, n_slices, n_columns, n_columns))

    for idx, element in enumerate(desired_elements_xrf):
        mlem_recon_xrf = tomo.recon(downsampled_proj_dataset[idx], theta*np.pi/180, algorithm = 'mlem', num_iter = n_iterations)

        density_xrf[idx] = create_density_map(mlem_recon_xrf, element, densities_xrf[idx])

        print(f'Processed {element}...')

    export_recon(dir_path_list[index], density_xrf, mlem_recon_opt_dens, desired_elements_xrf)