import numpy as np, \
       xrf_xrt_preprocess_file_util as futil, \
       xrf_xrt_preprocess_utils as ppu, \
       sys, \
       os

from realignment_final import realign_proj as rap
from matplotlib import pyplot as plt
from imageio import v2 as iio_v2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

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
                            desired_xrf_element,
                            xrt_data_percentile,
                            return_aux_data,
                            I0_cts_per_s,
                            t_dwell_s,
                            init_edge_crop_enabled,
                            init_edge_pixel_lengths_to_crop,
                            realignment_enabled,
                            n_iter_iter_reproj,
                            # zero_idx_to_discard,
                            sample_flipped_remounted_mid_experiment,
                            n_iterations_cor_correction,
                            eps_cor_correction,
                            sigma,
                            alpha,
                            upsample_factor,
                            eps_iter_reproj,
                            final_edge_crop_enabled,
                            final_edge_pixel_lengths_to_crop,
                            aligned_data_output_dir_path,
                            fps):

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
                                          synchrotron,
                                          sample_flipped_remounted_mid_experiment)

            print('Creating aggregate XRT data file...')

            futil.create_aggregate_xrt_h5(xrt_file_array,
                                          output_xrt_filepath,
                                          synchrotron,
                                          sample_flipped_remounted_mid_experiment)
        
        elif synchrotron == 'nsls-ii':
            us_ic = futil.create_aggregate_xrf_h5(xrf_file_array,
                                                  output_xrf_filepath, 
                                                  synchrotron,
                                                  sample_flipped_remounted_mid_experiment,
                                                  us_ic_enabled = True) # us_ic_array only returned since that data is not present in NSLS-II ptychography files

            print('Creating aggregate XRT data file...')

            futil.create_aggregate_xrt_h5(xrt_file_array,
                                          output_xrt_filepath, 
                                          synchrotron,
                                          sample_flipped_remounted_mid_experiment,
                                          us_ic = us_ic)

        if not pre_existing_aggregate_xrf_xrt_file_lists_enabled:
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

        print('Done. Exiting program...')

        sys.exit()

    else:
        if not os.path.dirname(aligned_data_output_dir_path):
            print('Error: Unable to locate output file directory. Exiting program...')

            sys.exit()

        elements_xrf, counts_xrf, theta, _, dataset_type = futil.extract_h5_aggregate_xrf_data(aggregate_xrf_h5_file_path)
        elements_xrt, counts_xrt, theta_xrt, _, dataset_type = futil.extract_h5_aggregate_xrt_data(aggregate_xrt_h5_file_path)

        if not np.array_equal(theta, theta_xrt):
            print('Error: Inconsistent XRF, XRT projection angles. Exiting program...')

            sys.exit()
        
        if counts_xrf.shape[2:] != counts_xrt.shape[2:]:
            print('Error: Inconsistent number of XRF, XRT slices (rows) and/or scan positions (columns). Exiting program...')

            sys.exit()

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
        counts_xrt_sig_backup = counts_xrt_sig.copy()

        if pre_existing_align_norm_file_enabled:
            print('Extracting pre-existing normalizations, net x pixel shifts, net y pixel shifts, and incident intensity...')

            norm_array, \
            net_x_shift_array, \
            net_y_shift_array, \
            I0_cts = futil.extract_csv_norm_net_shift_data(pre_existing_align_norm_file_path, theta)

            print('Applying pre-existing per-projection normalizations to XRF, XRT arrays...')

            counts_xrf_norm = counts_xrf*norm_array[None, :, None, None]
            counts_xrt_norm = counts_xrt_sig*norm_array[:, None, None]
                
        else:
            net_x_shift_array = np.zeros(n_theta)
            net_y_shift_array = np.zeros(n_theta)
            
            if norm_enabled:
                print('Normalizing XRF, XRT data via per-projection XRT masks...')

                if return_aux_data:
                    counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts, conv_mag_array = ppu.joint_fluct_norm(counts_xrt_sig,
                                                                                                                counts_xrf, 
                                                                                                                xrt_data_percentile, 
                                                                                                                return_conv_mag_array = True)
                    opt_dens = -np.log(counts_xrt_sig_backup/I0_cts)
                
                else:
                    counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts = ppu.joint_fluct_norm(counts_xrt_sig,
                                                                                                counts_xrf,
                                                                                                xrt_data_percentile)

            else:
                if I0_cts_per_s is None or t_dwell_s is None:
                    print('Warning: Incident photon flux and/or dwell time not provided. Optical density will be calculated using the mean incident intensity of empty space around sample.')
                   
                    _, _, _, I0_cts, _ = ppu.joint_fluct_norm(counts_xrt_sig, counts_xrf, xrt_data_percentile)
                
                else:
                    if I0_cts_per_s < 0 or t_dwell_s < 0:
                        print('Error: Incident photon flux and dwell time must be positive values. Exiting program...')

                        sys.exit()
            
                    I0_cts = I0_cts_per_s*t_dwell_s
                
                norm_array = np.ones(n_theta)

                counts_xrf_norm = counts_xrf
                counts_xrt_norm = counts_xrt_sig
                
        print('Calculating optical densities...')
        
        opt_dens_norm = -np.log(counts_xrt_norm/I0_cts)
    
    xrt_od_xrf_realignment_subdir_path = os.path.join(aligned_data_output_dir_path, 'xrt_od_xrf_realignment')

    os.makedirs(xrt_od_xrf_realignment_subdir_path, exist_ok = True)

    if realignment_enabled:
        if init_edge_crop_enabled:
            print('Creating auxilliary cropped XRF, optical density projection images...')

            if init_edge_pixel_lengths_to_crop is None:
                print("Error: Empty field for 'init_edge_pixel_lengths_to_crop'. Exiting program...")

                sys.exit()
            
            init_cropped_xrf_array, init_cropped_xrt_array, init_cropped_opt_dens_array = ppu.crop_array(counts_xrf_norm, counts_xrt_norm, opt_dens_norm, init_edge_pixel_lengths_to_crop)
        
        else:
            init_cropped_xrf_array = None
            init_cropped_xrt_array = None
            init_cropped_opt_dens_array = None
            
            if init_edge_pixel_lengths_to_crop is not None:
                print("Warning: Non-empty 'init_edge_pixel_lengths_to_crop' dictionary detected. Removing items from 'init_edge_pixel_lengths_to_crop'...")
                
                init_edge_pixel_lengths_to_crop = None
        
        if return_aux_data:
            if init_edge_crop_enabled:
                aligned_proj_final_xrt_sig, \
                aligned_proj_final_opt_dens, \
                aligned_proj_final_xrf, \
                net_x_shifts_pcc_final, \
                net_y_shifts_pcc_final, \
                aligned_exp_proj_array, \
                cropped_aligned_exp_proj_array, \
                synth_proj_array, \
                pcc_2d_array, \
                recon_array, \
                theta_final, \
                net_x_shifts_pcc_array, \
                net_y_shifts_pcc_array, \
                dx_pcc_array, \
                dy_pcc_array = rap(counts_xrt_norm,
                                   init_cropped_xrt_array,
                                   opt_dens_norm,
                                   init_cropped_opt_dens_array,
                                   counts_xrf_norm,
                                   init_cropped_xrf_array,
                                   theta,
                                #    zero_idx_to_discard,
                                   sample_flipped_remounted_mid_experiment,
                                   n_iterations_cor_correction,
                                   eps_cor_correction,
                                   I0_cts,
                                   n_iter_iter_reproj,
                                   net_x_shift_array,
                                   net_y_shift_array,
                                   sigma,
                                   alpha,
                                   upsample_factor,
                                   eps_iter_reproj,
                                   init_edge_pixel_lengths_to_crop,
                                   return_aux_data = True)
            
            else:
                aligned_proj_final_xrt_sig, \
                aligned_proj_final_opt_dens, \
                aligned_proj_final_xrf, \
                net_x_shifts_pcc_final, \
                net_y_shifts_pcc_final, \
                aligned_exp_proj_array, \
                synth_proj_array, \
                pcc_2d_array, \
                recon_array, \
                theta_final, \
                net_x_shifts_pcc_array, \
                net_y_shifts_pcc_array, \
                dx_pcc_array, \
                dy_pcc_array = rap(counts_xrt_norm,
                                   init_cropped_xrt_array,
                                   opt_dens_norm,
                                   init_cropped_opt_dens_array,
                                   counts_xrf_norm,
                                   init_cropped_xrf_array,
                                   theta,
                                #    zero_idx_to_discard,
                                   sample_flipped_remounted_mid_experiment,
                                   n_iterations_cor_correction,
                                   eps_cor_correction,
                                   I0_cts,
                                   n_iter_iter_reproj,
                                   net_x_shift_array,
                                   net_y_shift_array,
                                   sigma,
                                   alpha,
                                   upsample_factor,
                                   eps_iter_reproj,
                                   init_edge_pixel_lengths_to_crop,
                                   return_aux_data = True)

            print('Writing the following auxiliary, per-iteration data to NumPy (.npy) files (NOTE: Python is needed to view these!):')
            
            if init_edge_crop_enabled:
                print('     -Remapped experimental optical density projection data')
                print('     -Cropped experimental optical density projection data')
            
            else:
                print('     -Experimental optical density projection data')

            print('     -Reconstructed optical density data')
            print('     -Reprojected optical density data')
            print('     -2D phase cross-correlation data')
            print('     -Incremental x shifts')
            print('     -Incremental y shifts')
            print('     -Net x shifts')
            print('     -Net y shifts')

            if init_edge_crop_enabled:
                futil.create_aux_opt_dens_data_npy(xrt_od_xrf_realignment_subdir_path,
                                                   aligned_exp_proj_array,
                                                   recon_array,
                                                   synth_proj_array,
                                                   pcc_2d_array,
                                                   dx_pcc_array,
                                                   dy_pcc_array,
                                                   net_x_shifts_pcc_array,
                                                   net_y_shifts_pcc_array,
                                                   cropped_aligned_exp_proj_iter_array = cropped_aligned_exp_proj_array)
            
            else:
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
            theta_final, \
            net_x_shifts_pcc_final, \
            net_y_shifts_pcc_final = rap(counts_xrt_norm,
                                         init_cropped_xrt_array,
                                         opt_dens_norm,
                                         init_cropped_opt_dens_array,
                                         counts_xrf_norm,
                                         init_cropped_xrf_array,
                                         theta,
                                        #  zero_idx_to_discard,
                                         sample_flipped_remounted_mid_experiment,
                                         n_iterations_cor_correction,
                                         eps_cor_correction,
                                         I0_cts,
                                         n_iter_iter_reproj,
                                         net_x_shift_array,
                                         net_y_shift_array,
                                         sigma,
                                         alpha,
                                         upsample_factor,
                                         eps_iter_reproj)

        if final_edge_crop_enabled:
            if final_edge_pixel_lengths_to_crop is None:
                print("Error: Empty field for 'final_edge_pixel_lengths_to_crop'. Exiting program...")

                sys.exit()

            print('Cropping aligned XRF, XRT, and optical density projection images...')

            aligned_proj_final_xrf_cropped, \
            aligned_proj_final_xrt_sig_cropped, \
            aligned_proj_final_opt_dens_cropped = ppu.crop_array(aligned_proj_final_xrf,
                                                                 aligned_proj_final_xrt_sig,
                                                                 aligned_proj_final_opt_dens,
                                                                 final_edge_pixel_lengths_to_crop)

        else:
            aligned_proj_final_xrf_cropped = aligned_proj_final_xrf
            aligned_proj_final_xrt_sig_cropped = aligned_proj_final_xrt_sig
            aligned_proj_final_opt_dens_cropped = aligned_proj_final_opt_dens
            
            if final_edge_pixel_lengths_to_crop is not None:
                print("Warning: Non-empty 'init_edge_pixel_lengths_to_crop' dictionary detected. Removing items from 'final_edge_pixel_lengths_to_crop'...")

            final_edge_pixel_lengths_to_crop = None

        print('Writing final aligned XRF, XRT, and optical density projection data to HDF5 file...')
            
        futil.create_h5_aligned_aggregate_xrf_xrt(xrt_od_xrf_realignment_subdir_path,
                                                  elements_xrf,
                                                  aligned_proj_final_xrf_cropped, 
                                                  aligned_proj_final_xrt_sig_cropped,
                                                  aligned_proj_final_opt_dens_cropped,
                                                  theta_final,
                                                #   zero_idx_to_discard,
                                                  init_edge_pixel_lengths_to_crop,
                                                  final_edge_pixel_lengths_to_crop)
            
        print('Writing per-projection normalization, final net x and y shifts, and incident intensity to CSV file...')

        futil.create_csv_norm_net_shift_data(xrt_od_xrf_realignment_subdir_path,
                                             theta_final,
                                             norm_array,
                                             net_x_shifts_pcc_final,
                                             net_y_shifts_pcc_final,
                                             I0_cts)
            
        print('Done')

    else:
        if return_aux_data:
            if norm_enabled:
                print('Writing per-projection convolution magnitudes to NumPy (.npy) file...')

                futil.create_aux_conv_mag_data_npy(xrt_od_xrf_realignment_subdir_path, conv_mag_array)
            
                print('Preparing pre-aligned, non-cropped, normalized XRF, XRT, and optical density projection data for GIF creation...')

                futil.create_nonaligned_norm_non_cropped_proj_data_gif(dir_path = xrt_od_xrf_realignment_subdir_path,
                                                                       xrf_element_array = elements_xrf,
                                                                       desired_xrf_element = desired_xrf_element,
                                                                       counts_xrf = counts_xrf_norm,
                                                                       counts_xrf_norm = counts_xrf_norm,
                                                                       counts_xrt = counts_xrt_sig_backup,
                                                                       counts_xrt_norm = counts_xrt_norm,
                                                                       opt_dens = opt_dens,
                                                                       opt_dens_norm = opt_dens_norm,
                                                                       convolution_mag_array = conv_mag_array,
                                                                       norm_enabled = norm_enabled,
                                                                       data_percentile = xrt_data_percentile, #
                                                                       theta_array = theta,
                                                                       fps = fps)

            else:
                print('Preparing pre-aligned, non-cropped, non-normalized XRF, XRT, and optical density projection data for GIF creation...')
            
                futil.create_nonaligned_norm_non_cropped_proj_data_gif(dir_path = xrt_od_xrf_realignment_subdir_path,
                                                                       xrf_element_array = elements_xrf,
                                                                       desired_xrf_element = desired_xrf_element,
                                                                       counts_xrf = counts_xrf,
                                                                       counts_xrt = counts_xrt_sig,
                                                                       opt_dens = opt_dens_norm, # NOTE: opt_dens is the same as opt_dens_norm since norm_enabled is False
                                                                       norm_enabled = norm_enabled,
                                                                       theta_array = theta,
                                                                       fps = fps)
        
        # if zero_idx_to_discard is not None:
        #     final_xrf, \
        #     final_xrt, \
        #     final_opt_dens, \
        #     theta_final = ppu.remove_zero_deg_proj_no_realignment(counts_xrf_norm,
        #                                                           counts_xrt_norm,
        #                                                           opt_dens,
        #                                                           zero_idx_to_discard,
        #                                                           theta)

        # else:
            final_xrf = counts_xrf_norm
            final_xrt = counts_xrt_norm
            final_opt_dens = opt_dens_norm
            theta_final = theta

        if final_edge_crop_enabled:
            if final_edge_pixel_lengths_to_crop is None:
                print("Error: Empty field for 'final_edge_pixel_lengths_to_crop'. Exiting program...")

                sys.exit()
            
            print('Cropping XRF, XRT, and optical density projection images...')

            final_xrf_cropped, \
            final_xrt_sig_cropped, \
            final_opt_dens_cropped = ppu.crop_array(final_xrf, final_xrt, final_opt_dens)

        else:
            final_xrf_cropped = counts_xrf_norm
            final_xrt_sig_cropped = counts_xrt_norm
            final_opt_dens_cropped = opt_dens_norm

            if final_edge_pixel_lengths_to_crop is not None:
                print("Warning: Non-empty 'final_edge_pixel_lengths_to_crop' dictionary detected. Removing items from 'final_edge_pixel_lengths_to_crop'...")

                final_edge_pixel_lengths_to_crop = None

        print('Writing normalized XRF, XRT, and optical density projection data to HDF5 file (NOTE: These contain edge crop information as well)...')

        futil.create_h5_aligned_aggregate_xrf_xrt(xrt_od_xrf_realignment_subdir_path,
                                                  elements_xrf,
                                                  final_xrf_cropped, 
                                                  final_xrt_sig_cropped,
                                                  final_opt_dens_cropped, 
                                                  theta_final,
                                                #   zero_idx_to_discard,
                                                  init_edge_pixel_lengths_to_crop,
                                                  final_edge_pixel_lengths_to_crop)

        print('Writing per-projection normalization, final net x and y shifts, and incident intensity to CSV file...')

        futil.create_csv_norm_net_shift_data(xrt_od_xrf_realignment_subdir_path,
                                             theta_final,
                                             norm_array,
                                             net_x_shift_array,
                                             net_y_shift_array,
                                             I0_cts)
            
        print('Done')
    
    return