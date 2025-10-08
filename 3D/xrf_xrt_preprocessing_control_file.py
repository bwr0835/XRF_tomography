import numpy as np, \
       tkinter as tk, \
       file_util as futil, \
       xrf_xrt_preprocess_utils as ppu, \
       sys, \
       os

from tkinter import filedialog as fd
from realignment_final import iter_reproj as irprj

def preprocess_xrf_xrt_data(synchrotron,
                            synchrotron_beamline,
                            create_aggregate_xrf_xrt_files_enabled,
                            pre_existing_aggregate_xrf_xrt_file_lists_enabled,
                            aggregate_xrf_csv_file_path,
                            aggregate_xrt_csv_file_path,
                            aggregate_xrf_h5_file_path,
                            aggregate_xrt_h5_file_path,
                            pre_existing_align_norm_file_enabled,
                            pre_existing_align_norm_file_path,
                            norm_enabled,
                            data_percentile,
                            return_aux_data,
                            I0_cts_per_s,
                            t_dwell_s,
                            realignment_enabled,
                            n_iter_iter_reproj,
                            aligned_data_output_dir_path):

    available_synchrotrons = ['aps', 'nsls-ii']
    
    if synchrotron is None or synchrotron_beamline is None:
        print('Error: Synchrotron and/or synchrotron beamline fields empty. Exiting program...')

        sys.exit()
    
    synchrotron = synchrotron.lower()

    if synchrotron not in available_synchrotrons:
        print('Error: Synchrotron unavailable. Exiting program...')

        sys.exit()
    
    bool_params = [create_aggregate_xrf_xrt_files_enabled,
                   pre_existing_aggregate_xrf_xrt_file_lists_enabled, 
                   pre_existing_align_norm_file_enabled,
                   realignment_enabled]

    if not any(isinstance(val, bool) for val in bool_params):
        print('Error: \'create_aggregate_xrf_xrt_files_enabled\', \
                      \'pre_existing_aggregate_xrf_xrt_file_lists_enabled\', \
                      \'pre_existing_align_norm_file_enabled\', \
                      and \'realignment_enabled\' must be set to True or False. Exiting program...')

        sys.exit()
    
    if not create_aggregate_xrf_xrt_files_enabled and not isinstance(norm_enabled, bool):
        print('Error: \'norm_enabled\' must be set to True or False. Exiting program...')

        sys.exit()

    if create_aggregate_xrf_xrt_files_enabled:
        if pre_existing_aggregate_xrf_xrt_file_lists_enabled:
            if synchrotron == 'aps':
                xrf_file_array = futil.extract_csv_file_list(aggregate_xrf_csv_file_path)
                xrt_file_array = xrf_file_array.copy()
            
            else:
                xrf_file_array = futil.extract_csv_file_list(aggregate_xrf_csv_file_path)
                xrt_file_array = futil.extract_csv_file_list(aggregate_xrt_csv_file_path)
        
        else:   
            root = tk.Tk()

            xrf_file_array = fd.askopenfilenames(parent = root, title = "Choose XRF files to aggregate.", filetypes = [('HDF5 files', '*.h5')])
            xrt_file_array = fd.askopenfilenames(parent = root, title = "Choose XRT files to aggregate.", filetypes = [('HDF5 files', '*.h5')])

        if xrf_file_array == '' or xrt_file_array == '':
            print('Error: XRF and/or XRT filename array empty. Exiting program...')
            
            sys.exit()

        xrf_array_dir = os.path.dirname(xrf_file_array[0])
        xrt_array_dir = os.path.dirname(xrt_file_array[0])

        output_xrf_filepath = os.path.join(xrf_array_dir, f'{synchrotron_beamline}_aggregate_xrf.h5')
        output_xrt_filepath = os.path.join(xrt_array_dir, f'{synchrotron_beamline}_aggregate_xrt.h5')

        print('Creating aggregate XRF data file...')

        if synchrotron == 'aps':
            futil.create_aggregate_xrf_h5(xrf_file_array, 
                                        output_xrf_filepath, 
                                        synchrotron)

            print('Creating aggregate XRT data file...')

            futil.create_aggregate_xrt_h5(xrt_file_array, 
                                        output_xrt_filepath, 
                                        synchrotron)
        
        elif synchrotron == 'nsls-ii':
            us_ic = futil.create_aggregate_xrf_h5(xrf_file_array, 
                                                  output_xrf_filepath, 
                                                  synchrotron, 
                                                  us_ic_enabled = True) # us_ic_array only returned since that data is not present in NSLS-II ptychography files

            print('Creating aggregate XRT data file...')

            futil.create_aggregate_xrt_h5(xrt_file_array, 
                                          output_xrt_filepath, 
                                          synchrotron,
                                          us_ic = us_ic)

        print('Creating XRF file list CSV file...')

        sys.exit()

    else:
        if not os.path.dirname(aligned_data_output_dir_path):
            print('Error: Output directory does not exist. Exiting program...')

            sys.exit()

        elements_xrf, counts_xrf, theta, _, dataset_type = futil.extract_h5_aggregate_xrf_data(aggregate_xrf_h5_file_path)
        elements_xrt, counts_xrt, _, _, dataset_type = futil.extract_h5_aggregate_xrt_data(aggregate_xrt_h5_file_path)

        _, n_theta, n_slices, n_columns = counts_xrf.shape

        if (n_slices % 2) or (n_columns % 2):
            if (n_slices % 2) and (n_columns % 2):
                print('Odd number of slices (rows) and scan positions (columns) detected. Padding one additional slice and scan position column to XRF and XRT data...')

                counts_xrt = ppu.pad_col_row(counts_xrt, dataset_type)
                counts_xrf = ppu.pad_col_row(counts_xrf, dataset_type)
            
                n_slices += 1
                n_columns += 1
        
            elif n_slices % 2:
                print('Odd number of slices (rows) detected. Padding one additional slice to XRF and XRT data...')
                
                counts_xrt = ppu.pad_row(counts_xrt, dataset_type)
                counts_xrf = ppu.pad_row(counts_xrf, dataset_type)

                n_slices += 1

            else:
                print('Odd number of scan positions (columns) detected. Padding one additional scan position column to XRF and XRT data...')
                
                counts_xrt = ppu.pad_col(counts_xrt, dataset_type)
                counts_xrf = ppu.pad_col(counts_xrf, dataset_type)

                n_columns += 1

        counts_xrt_sig_idx = elements_xrt.index('xrt_sig')
        counts_xrt_sig = counts_xrt[counts_xrt_sig_idx]

        if pre_existing_align_norm_file_enabled:
            norm_array, \
            net_x_shift_array, \
            net_y_shift_array, \
            I0_cts = futil.extract_csv_norm_net_shift_data(pre_existing_align_norm_file_path, theta)

            print('Applying pre-existing per-projection normalizations to XRF, XRT arrays...')

            counts_xrf_norm = counts_xrf*norm_array
            counts_xrt_norm = counts_xrt_sig*norm_array
                
        else:
            net_x_shift_array = np.zeros(n_theta)
            net_y_shift_array = np.zeros(n_theta)
            
            if norm_enabled:
                print('Normalizing XRF, XRT data via per-projection XRT masks...')
                
                if data_percentile is not None or data_percentile < 0 or data_percentile > 100:
                    print('Error: \'data_percentile\' must be between 0 and 100. Exiting program...')

                    sys.exit()

                if return_aux_data:
                    counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts, conv_mag_array = ppu.joint_fluct_norm(counts_xrt,
                                                                                                                counts_xrf, 
                                                                                                                data_percentile, 
                                                                                                                return_conv_mag_array = True)

                else:
                    counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts = ppu.joint_fluct_norm(counts_xrt, 
                                                                                                counts_xrf,
                                                                                                data_percentile)

            else:
                if I0_cts_per_s is None or I0_cts_per_s < 0 or t_dwell_s is None or t_dwell_s < 0:
                    print('Error: Incident photon flux and dwell time must be positive values. Exiting program...')

                    sys.exit()
                
                norm_array = np.ones(n_theta)

                counts_xrf_norm = counts_xrf
                counts_xrt_norm = counts_xrt_sig
                
                I0_cts = I0_cts_per_s*t_dwell_s

        print('Calculating optical densities...')
        
        opt_dens = -np.log(counts_xrt_norm/I0_cts)

    if realignment_enabled:
        if return_aux_data:
            aligned_proj_final_xrt_sig, \
            aligned_proj_final_opt_dens, \
            aligned_proj_final_xrf, \
            net_x_shifts_pcc_final, \
            net_y_shifts_pcc_final, \
            aligned_exp_proj_array, \
            synth_proj_array, \
            pcc_2d_array, \
            recon_array, \
            net_x_shifts_pcc_array, \
            net_y_shifts_pcc_array, \
            dx_pcc_array, \
            dy_pcc_array = irprj(counts_xrt_norm,
                                 opt_dens,
                                 counts_xrf_norm,
                                 theta,
                                 I0_cts,
                                 n_iter_iter_reproj,
                                 net_x_shift_array,
                                 net_y_shift_array,
                                 return_aux_data = True)

            print('Writing convolution magnitude array to NumPy (.npy) file')
            
            futil.create_aux_conv_mag_data_npy(aligned_data_output_dir_path, conv_mag_array)

            print('Writing the following auxiliary, per-iteration data to NumPy (.npy) files (NOTE: Python is needed to view these!):')
            print('     -Experimental optical density projection data')
            print('     -Reconstructed optical density data')
            print('     -Reprojected optical density data')
            print('     -2D phase cross-correlation data')
            print('     -Incremental x shifts')
            print('     -Incremental y shifts')
            print('     -Net x shifts')
            print('     -Net y shifts')

            futil.create_aux_opt_dens_data_npy(aligned_data_output_dir_path,
                                               aligned_exp_proj_array,
                                               recon_array,
                                               synth_proj_array,
                                               pcc_2d_array,
                                               dx_pcc_array,
                                               dy_pcc_array,
                                               net_x_shifts_pcc_array,
                                               net_y_shifts_pcc_array)                           
                    
        else:
            aligned_proj_final_xrt_sig, \
            aligned_proj_final_opt_dens, \
            aligned_proj_final_xrf, \
            net_x_shifts_pcc_final, \
            net_y_shifts_pcc_final = irprj(counts_xrt_norm,
                                           opt_dens,
                                           counts_xrf_norm,
                                           theta,
                                           I0_cts,
                                           n_iter_iter_reproj)
            
            print('Writing final aligned XRF, XRT, and optical density projection data to HDF5 file...')
            
            futil.create_h5_aligned_aggregate_xrf_xrt(aligned_data_output_dir_path,
                                                      elements_xrf,
                                                      aligned_proj_final_xrf, 
                                                      aligned_proj_final_xrt_sig,
                                                      aligned_proj_final_opt_dens, 
                                                      theta)
            
            print('Writing per-projection normalization, final net x and y shifts, and incident intensity to CSV file...')

            futil.create_csv_norm_net_shift_data(aligned_data_output_dir_path,
                                                 theta,
                                                 norm_array,
                                                 net_x_shifts_pcc_final,
                                                 net_y_shifts_pcc_final,
                                                 I0_cts)
            
            print('Done')

    else:
        if return_aux_data:
            print('Writing per-projection convolution magnitudes to NumPy (.npy) file...')

            futil.create_aux_conv_mag_data_npy(aligned_data_output_dir_path, conv_mag_array)
        
        print('Writing normalized XRF, XRT, and optical density projection data to HDF5 file...')

        futil.create_h5_aligned_aggregate_xrf_xrt(aligned_data_output_dir_path,
                                                  elements_xrf,
                                                  counts_xrf_norm, 
                                                  counts_xrt_norm,
                                                  opt_dens, 
                                                  theta)

        print('Writing per-projection normalization, final net x and y shifts, and incident intensity to CSV file...')

        futil.create_csv_norm_net_shift_data(aligned_data_output_dir_path,
                                             theta,
                                             norm_array,
                                             net_x_shift_array,
                                             net_y_shift_array,
                                             I0_cts)
            
        print('Done')