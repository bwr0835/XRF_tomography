import numpy as np, \
       xrf_xrt_jxrft_file_util as recon_futil, \
       xrf_xrt_preprocess_file_util as pp_futil, \
       xrf_xrt_preprocess_utils as ppu, \
       tomopy as tomo, \
       h5py, \
       os, \
       sys

from skimage import measure as meas
from matplotlib import pyplot as plt
from imageio import v2 as iio2

def extract_h5_aggregate_xrf_xrt_data_for_recon(file_path):
    if not os.path.isfile(file_path):
        print('Error: Cannot locate aggregate XRF, XRT HDF5 file. Exiting program...', flush = True)

        sys.exit()
    
    if not file_path.endswith('.h5'):
        print('Error: Aggregate XRF, XRT file extension must be \'.h5\'. Exiting program...', flush = True)

        sys.exit()
    
    # try:
    with h5py.File(file_path, 'r') as h5:
        data = h5['exchange/data']
        elements = h5['exchange/elements']
            
        elements_xrf, elements_xrt = list(elements['xrf'].asstr()[:]), list(elements['xrt'].asstr()[:])
        xrf_data, xrt_data = data['xrf'][()], data['xrt'][()]
        theta = h5['exchange/theta'][()]

        num_slices_cropped_top = h5['exchange/data'].attrs['top_edge_cropped_final']
        num_slices_cropped_bottom = h5['exchange/data'].attrs['bottom_edge_cropped_final']
    
    # except KeyboardInterrupt:
    #     print('Keyboard interrupt. Exiting program...', flush = True)

    #     sys.exit()
    
    # except:
    #     print('Error: Incorrect HDF5 file structure. Exiting program...', flush = True)

    #     sys.exit()
    
    xrt_sig_data = xrt_data[elements_xrt.index('xrt_sig')]

    return elements_xrf, xrf_data, xrt_sig_data, theta, num_slices_cropped_top, num_slices_cropped_bottom

def extract_h5_scan_coords(file_path, synchrotron): 
    if not os.path.isfile(file_path):
        print('Error: Cannot locate scan data HDF5 file. Exiting program...')

        sys.exit()
    
    if not file_path.endswith('.h5'):
        print('Error: Scan data file extension must be \'.h5\'. Exiting program...')

        sys.exit()
    
    if synchrotron == 'aps':
        with h5py.File(file_path, 'r') as h5:
            x = h5['MAPS/Scan/x_axis'][()]

            x = x[:-2]
    
    elif synchrotron == 'nsls-ii':
        with h5py.File(file_path, 'r') as h5:
            x = h5['xrfmap/positions/pos'][()][0]

    return x

def create_xrf_proj_movie(dir_path, xrf_data, elements_of_interest, theta, fps):
    _, n_theta, n_slices, n_columns = xrf_data.shape

    n_elements = len(elements_of_interest)

    if n_elements != 4:
        print('Error: Number of elements of interest must be 4. Exiting program...')

        sys.exit()

    element_idx = [elements_of_interest.index(element) for element in elements_of_interest]

    el_1 = xrf_data[element_idx[0]]
    el_2 = xrf_data[element_idx[1]]
    el_3 = xrf_data[element_idx[2]]
    el_4 = xrf_data[element_idx[3]]

    fig, axs = plt.subplots(2, 2)

    img1_1 = axs[0, 0].imshow(el_1[0], vmin = el_1.min(), vmax = el_1.max())
    img1_2 = axs[0, 1].imshow(el_2[0], vmin = el_2.min(), vmax = el_2.max())
    img1_3 = axs[1, 0].imshow(el_3[0], vmin = el_3.min(), vmax = el_3.max())
    img1_4 = axs[1, 1].imshow(el_4[0], vmin = el_4.min(), vmax = el_4.max())

    text = axs[0, 0].text(0.02, 0.02, r'$\theta$ = {0}$\textdegree'.format(theta[0]), transform = axs[0, 0].transAxes, color = 'white', fontsize = 14)
    
    for ax in fig.axes:
        ax.axis('off')
        # ax.axvline(x = n_columns//2, color = 'red', linewidth = 2)
        # ax.axhline(y = n_slices//2, color = 'red', linewidth = 2)
    
    axs[0, 0].set_title(r'{0}'.format(elements_of_interest[element_idx[0]]), fontsize = 14)
    axs[0, 1].set_title(r'{0}'.format(elements_of_interest[element_idx[1]]), fontsize = 14)
    axs[1, 0].set_title(r'{0}'.format(elements_of_interest[element_idx[2]]), fontsize = 14)
    axs[1, 1].set_title(r'{0}'.format(elements_of_interest[element_idx[3]]), fontsize = 14)

    theta_frames = []

    for theta_idx in range(n_theta):
        img1_1.set_data(el_1[theta_idx])
        img1_2.set_data(el_2[theta_idx])
        img1_3.set_data(el_3[theta_idx])
        img1_4.set_data(el_4[theta_idx])
        
        text.set_text(r'$\theta$ = {0}$\textdegree'.format(theta[theta_idx]))

        fig.canvas.draw()

        frame1 = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        theta_frames.append(frame1)

    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'xrf_proj_movie.gif')

    iio2.mimsave(gif_filename, theta_frames, fps = fps)

    return

def pad_col(array):
    n_slices, _ = array.shape

    final_column = array[:, -1].reshape(-1, 1) + np.ones((n_slices, 1))
                
    array = np.hstack((array, final_column))

    return array

def downsample(array, downsample_factor_1, downsample_factor_2 = None, data_type = None, func = np.mean):
    if downsample_factor_1 <= 0:
        print('Error: Downsampling factor must be positive. Exiting program...')

        sys.exit()

    if downsample_factor_2 is None:
        downsample_factor_2 = downsample_factor_1
    
    elif downsample_factor_2 <= 0:
        print('Error: Downsampling factor must be positive. Exiting program...')

        sys.exit()

    if data_type == 'xrt' or data_type == 'xrf':
        if array.ndim != 3:
            print('Error: Input XRF/XRT projection data must be exactly 3D. Exiting program...')

            sys.exit()
        
        _, n_slices, n_columns = array.shape
    
    elif data_type == 'scan_coords':
        if array.ndim not in (1, 2):
            print('Error: Input scan coordinates must be 1D or 2D. Exiting program...')

            sys.exit()
        
        if array.ndim == 1:
            array_length = len(array)

            if array_length % downsample_factor_1 != 0:
                print('Error: Downsampling factor results in non-integer number of scan positions. Exiting program...')

                sys.exit()

            if (array_length//downsample_factor_1) % 2:
                print('Warning: Odd number of scan positions resulting from downsampling. Consider switching to even number of scan positions being output.')
        
            return meas.block_reduce(array, block_size = downsample_factor_1, func = func)
        
        else:
            n_rows, n_columns = array.shape

            if n_rows % downsample_factor_1 != 0 or n_columns % downsample_factor_2 != 0:
                print('Error: Downsampling factor results in non-integer number of rows and/or columns. Exiting program...')

                sys.exit()

            if (n_rows//downsample_factor_1) % 2 or (n_columns//downsample_factor_2) % 2:
                print('Warning: Odd number of rows and/or columns resulting from downsampling. Consider switching to even number of rows and/or columns being output.')
        
            return meas.block_reduce(array, block_size = (downsample_factor_1, downsample_factor_2), func = func)
        
    else:
        print('Error: Invalid data type. Exiting program...')
        
        sys.exit()
    
    if downsample_factor_1 == 1 and downsample_factor_2 == 1:
        return array

    if n_slices % downsample_factor_1 != 0 or n_columns % downsample_factor_2 != 0:
        print('Error: Downsampling factor results in non-integer number of rows/slices and/or columns/scan positions. Exiting program...')

        sys.exit()

    if (n_slices//downsample_factor_1) % 2 or (n_columns//downsample_factor_2) % 2:
        print('Warning: Odd number of rows/slices and/or columns/scan positions resulting from downsampling. Consider switching to even number of slices and/or scan positions being output.')

    return meas.block_reduce(array, block_size = (1, downsample_factor_1, downsample_factor_2), func = func)

def create_h5_recon(dir_path, element, xrf_data, x, y, downsample_factor, algorithm, synchrotron):
    if not os.path.isdir(dir_path):
        print('Error: Cannot locate directory. Exiting program...')

        sys.exit()
    
    if synchrotron == 'aps':
        with h5py.File(os.path.join(dir_path, f'recon_downsample_{downsample_factor}_{element}_{algorithm}.h5'), 'w') as h5:
            maps = h5.create_group('MAPS')
        
            nnls = maps.create_group('XRF_Analyzed/NNLS')
            nnls.create_dataset('Counts_Per_Sec', data = xrf_data[None])
            nnls.create_dataset('Channel_Names', data = [element])

            scan = maps.create_group('Scan')
            scan.create_dataset('x_axis', data = x)
            scan.create_dataset('y_axis', data = y)
    
    elif synchrotron == 'nsls-ii':
        scan_coords = np.stack((x, y), axis = 0)

        with h5py.File(os.path.join(dir_path, f'recon_downsample_{downsample_factor}_{element}_{algorithm}.h5'), 'w') as h5:
            xrfmap = h5.create_group('xrfmap')

            detsum = xrfmap.create_group('detsum')
            
            detsum.create_dataset('xrf_fit', data = xrf_data[None])
            detsum.create_dataset('xrf_fit_name', data = [element])

            positions = xrfmap.create_group('positions')

            positions.create_dataset('name', data = ['x', 'y'])
            positions.create_dataset('pos', data = scan_coords)

def create_middle_slice_recon_figure(recon, downsample_factors, slice_idx):
    if len(downsample_factors) != 4:
        print('Error: Number of downsample factors must be 4. Exiting program...')

        sys.exit()
    
    fig, axs = plt.subplots(2, 2)

    vmin = recon.min()
    vmax = recon.max()
    n_columns = recon.shape[-1]

    for idx, ax in enumerate(fig.axes):
        n_columns_downsampled = n_columns//downsample_factors[idx]
        
        print(n_columns_downsampled)
       
        ax.imshow(recon[idx, :n_columns_downsampled, :n_columns_downsampled], vmin = vmin, vmax = vmax)
        
        ax.axis('off')
        ax.set_title(r'DSF = {0}'.format(downsample_factors[idx]), fontsize = 14)

    fig.tight_layout()
    fig.suptitle(r'Slice {0}'.format(slice_idx), fontsize = 16)

    plt.show()

    return

input_proj_dir_path = '/home/bwr0835/2_ide_realigned_data_03_27_2026_iter_reproj_cor_correction_only_final/xrt_od_xrf_realignment'
input_proj_scan_data_file_path = '/raid/users/roter/Jacobsen/img.dat/2xfm_0096.mda.h5' # There is no 0° projection, so use closest to zero (-5°)

# input_proj_dir_path = '/home/bwr0835/3_idrealigned_data_04_19_2026_diff_cor_correction'
# input_proj_scan_data_file_path = '/Users/bwr0835/Documents/2_ide_realigned_data_03_27_2026_iter_reproj_cor_correction_only_final/2xfm_0097.mda.h5'

proj_data_h5_path = os.path.join(input_proj_dir_path, 'aligned_data', 'aligned_aggregate_xrf_xrt.h5')

synchrotron = 'aps'
element_of_interest = 'Fe'
algorithm = 'gridrec'

save_recon = True
save_proj = False

downsample_factors_1 = np.array([1, 2, 5, 10])
downsample_factors_2 = np.array([1, 2, 5, 10])

print('Extracting projection data...')

elements_xrf, xrf_data, xrt_sig_data, theta, num_slices_cropped_top, num_slices_cropped_bottom = extract_h5_aggregate_xrf_xrt_data_for_recon(proj_data_h5_path)

print('Extracting scan data...')

x = extract_h5_scan_coords(input_proj_scan_data_file_path, synchrotron)

if save_proj:
    elements_of_interest_hxn = ['Ni', 'Cu', 'Zn', 'Ce_L']

    print(f'Saving projection data...')

    create_xrf_proj_movie(input_proj_dir_path, xrf_data, elements_of_interest_hxn, theta, fps = 10)

xrf_data_element_of_interest = xrf_data[elements_xrf.index(element_of_interest)]

n_theta, n_slices, n_columns = xrf_data_element_of_interest.shape

if synchrotron == 'aps': # Append additional scan position to ensure matching number of scan positions between coordinate arrays, aligned projections
    x = np.append(x, x[-1] + (x[-1] - x[0])/(len(x) - 1))

start_slice = num_slices_cropped_top
end_slice = n_slices - num_slices_cropped_bottom

recon = np.zeros((len(downsample_factors_1), n_slices, n_columns, n_columns))
middle_slice_recons = np.zeros((len(downsample_factors_1), n_columns, n_columns))

if x.ndim == 1:
    # x_cropped_downsampled_array = np.zeros((len(downsample_factors), n_columns))

    x_cropped = x

else:
    x_cropped = x[start_slice:end_slice] # Since reconstructed object slices are square and are related to scan positions, only need to worry about per-pixel scan distance in x

    # x_cropped_downsampled_array = np.zeros((len(downsample_factors), n_columns, n_columns))
    # y_cropped_downsampled_array = np.zeros((len(downsample_factors), n_columns, n_columns))

for idx, downsample_factor_1 in enumerate(downsample_factors_1):
    print(f'Downsampling projection data by factor of {downsample_factor_1}...')

    xrf_data_element_of_interest_downsampled = downsample(xrf_data_element_of_interest, downsample_factor_1, data_type = 'xrf')
    x_cropped_downsampled = downsample(x_cropped, downsample_factor_1, data_type = 'scan_coords')

    print('Creating downsampled x and y scan data arrays that mimick scanning the middle reconstructed slice...')

    n_slices = xrf_data_element_of_interest_downsampled.shape[1]
    # middle_slice = n_slices//2
    middle_slice = 90//downsample_factor_1

    if x_cropped_downsampled.ndim == 1:
        n_columns = len(x_cropped_downsampled)

        dx = (x_cropped_downsampled[-1] - x_cropped_downsampled[0])/(n_columns - 1)

        # x_cropped_downsampled_array[idx, :n_columns] = dx*np.arange(n_columns)
        x_cropped_downsampled_array = dx*np.arange(n_columns)
        y_cropped_downsampled_array = x_cropped_downsampled_array

    else:
        n_slices, n_columns = x_cropped_downsampled.shape

        dx = (x_cropped_downsampled[middle_slice, -1] - x_cropped_downsampled[middle_slice, 0])/(n_columns - 1)

        y_cropped_downsampled_array, x_cropped_downsampled_array = dx*np.mgrid[0:n_columns, 0:n_columns] # dy = dx since reconstructed object slices are square
        
        # x_cropped_downsampled_array[idx, :n_columns, :n_columns] = dx*x_grid
        # y_cropped_downsampled_array[idx, :n_columns, :n_columns] = dx*y_grid # dy = dx since reconstructed object slices are square
        
    print('Reconstructing downsampled XRF projection data...')

    if algorithm == 'gridrec':
        recon[idx, :n_slices, :n_columns, :n_columns] = tomo.recon(xrf_data_element_of_interest_downsampled, theta*np.pi/180, algorithm = algorithm, filter_name = 'ramlak')

    elif algorithm == 'mlem':
        recon[idx, :n_slices, :n_columns, :n_columns] = tomo.recon(xrf_data_element_of_interest_downsampled, theta*np.pi/180, algorithm = algorithm, num_iter = 70)

    else:
        print('Error: Algorithm not available. Exiting program...')

        sys.exit()

    middle_slice_recons[idx] = recon[idx, middle_slice]/downsample_factors_1[idx]
    # middle_slice_recons[idx] = recon[idx, middle_slice]

    if save_recon:
        print('Saving reconstruction and downsampled scan data to HDF5 file for middle slice...')

        create_h5_recon(input_proj_dir_path, element_of_interest, middle_slice_recons[idx], x_cropped_downsampled_array, y_cropped_downsampled_array, downsample_factor_1, algorithm, synchrotron)

print('Creating figure comparing middle slices of downsampled reconstructions...')

create_middle_slice_recon_figure(middle_slice_recons, downsample_factors_1, middle_slice)