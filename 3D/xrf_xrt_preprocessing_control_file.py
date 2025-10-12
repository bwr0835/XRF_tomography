import numpy as np, \
       file_util as futil, \
       xrf_xrt_preprocess_utils as ppu, \
       sys, \
       os

from realignment_final import realign_proj as rap

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
                            xrt_data_percentile,
                            return_aux_data,
                            I0_cts_per_s,
                            t_dwell_s,
                            realignment_enabled,
                            n_iter_iter_reproj,
                            sigma,
                            alpha,
                            upsample_factor,
                            eps,
                            aligned_data_output_dir_path):

    print('OK')

    if create_aggregate_xrf_xrt_files_enabled:
        if pre_existing_aggregate_xrf_xrt_file_lists_enabled:
            print('Extracting pre-existing XRF, XRT HDF5 file lists...')
            
            if synchrotron == 'aps':
                xrf_file_array, xrt_file_array = futil.extract_csv_xrf_xrt_data_file_lists(aggregate_xrf_csv_file_path, synchrotron = synchrotron)
            
            else:
                xrf_file_array, xrt_file_array = futil.extract_csv_xrf_xrt_data_file_lists(aggregate_xrf_csv_file_path, aggregate_xrt_csv_file_path, synchrotron = synchrotron)
               
        else:
            print('Opening file dialog window for opening XRF, XRT HDF5 file extraction...')

            xrf_file_array, xrt_file_array = futil.extract_h5_xrf_xrt_data_file_lists_tk(synchrotron)

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

        print('Creating aggregate XRF, XRT file list CSV files...')

        if synchrotron == 'aps':
            futil.create_csv_file_list(xrf_file_array,
                                       xrf_array_dir, 
                                       synchrotron, 
                                       synchrotron_beamline)

        else:
            futil.create_csv_file_list(xrf_file_array,
                                       xrf_array_dir, 
                                       synchrotron, 
                                       synchrotron_beamline,
                                       'xrf')

            futil.create_csv_file_list(xrf_file_array,
                                       xrf_array_dir, 
                                       synchrotron, 
                                       synchrotron_beamline,
                                       'xrt')

        sys.exit()

    else:
        if not os.path.dirname(aligned_data_output_dir_path):
            print('Error: Unable to locate output file directory. Exiting program...')

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
            print('Extracting pre-existing normalizations, net x pixel shifts, net y pixel shifts, and incident intensity...')
            
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

                if return_aux_data:
                    counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts, conv_mag_array = ppu.joint_fluct_norm(counts_xrt,
                                                                                                                counts_xrf, 
                                                                                                                xrt_data_percentile, 
                                                                                                                return_conv_mag_array = True)

                else:
                    counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts = ppu.joint_fluct_norm(counts_xrt,
                                                                                                counts_xrf,
                                                                                                xrt_data_percentile)

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

    xrt_od_xrf_realignment_subdir_path = os.path.join(aligned_data_output_dir_path, 'xrt_od_xrf_realignment')

    os.makedirs(xrt_od_xrf_realignment_subdir_path, exist_ok = True)

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
            dy_pcc_array = rap(synchrotron,
                               counts_xrt_norm,
                               opt_dens,
                               counts_xrf_norm,
                               theta,
                               I0_cts,
                               n_iter_iter_reproj,
                               net_x_shift_array,
                               net_y_shift_array,
                               sigma,
                               alpha,
                               upsample_factor,
                               eps,
                               return_aux_data = True)

            print('Writing convolution magnitude array to NumPy (.npy) file')
            
            futil.create_aux_conv_mag_data_npy(xrt_od_xrf_realignment_subdir_path, conv_mag_array)

            print('Writing the following auxiliary, per-iteration data to NumPy (.npy) files (NOTE: Python is needed to view these!):')
            print('     -Experimental optical density projection data')
            print('     -Reconstructed optical density data')
            print('     -Reprojected optical density data')
            print('     -2D phase cross-correlation data')
            print('     -Incremental x shifts')
            print('     -Incremental y shifts')
            print('     -Net x shifts')
            print('     -Net y shifts')

            futil.create_aux_opt_dens_data_npy(xrt_od_xrf_realignment_subdir_path,
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
            net_y_shifts_pcc_final = rap(synchrotron,
                                         counts_xrt_norm,
                                         opt_dens,
                                         counts_xrf_norm,
                                         theta,
                                         I0_cts,
                                         n_iter_iter_reproj,
                                         net_x_shift_array,
                                         net_y_shift_array,
                                         sigma,
                                         alpha,
                                         upsample_factor,
                                         eps)
            
            print('Writing final aligned XRF, XRT, and optical density projection data to HDF5 file...')
            
            futil.create_h5_aligned_aggregate_xrf_xrt(xrt_od_xrf_realignment_subdir_path,
                                                      elements_xrf,
                                                      aligned_proj_final_xrf, 
                                                      aligned_proj_final_xrt_sig,
                                                      aligned_proj_final_opt_dens, 
                                                      theta)
            
            print('Writing per-projection normalization, final net x and y shifts, and incident intensity to CSV file...')

            futil.create_csv_norm_net_shift_data(xrt_od_xrf_realignment_subdir_path,
                                                 theta,
                                                 norm_array,
                                                 net_x_shifts_pcc_final,
                                                 net_y_shifts_pcc_final,
                                                 I0_cts)
            
            print('Done')

    else:
        if return_aux_data:
            print('Writing per-projection convolution magnitudes to NumPy (.npy) file...')

            futil.create_aux_conv_mag_data_npy(xrt_od_xrf_realignment_subdir_path, conv_mag_array)
        
        print('Writing normalized XRF, XRT, and optical density projection data to HDF5 file...')

        futil.create_h5_aligned_aggregate_xrf_xrt(xrt_od_xrf_realignment_subdir_path,
                                                  elements_xrf,
                                                  counts_xrf_norm, 
                                                  counts_xrt_norm,
                                                  opt_dens, 
                                                  theta)

        print('Writing per-projection normalization, final net x and y shifts, and incident intensity to CSV file...')

        futil.create_csv_norm_net_shift_data(xrt_od_xrf_realignment_subdir_path,
                                             theta,
                                             norm_array,
                                             net_x_shift_array,
                                             net_y_shift_array,
                                             I0_cts)
            
        print('Done')