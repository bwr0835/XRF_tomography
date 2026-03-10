import numpy as np, \
       xrf_xrt_preprocess_file_util as futil, \
       xrf_xrt_preprocess_utils as ppu, \
       realignment_final as realign, \
       sys, \
       os

from matplotlib import pyplot as plt

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
                            aligning_element,
                            pre_cor_correction_adjacent_angle_jitter_correction_enabled,
                            init_edge_crop_enabled,
                            init_edge_pixel_lengths_to_crop,
                            realignment_enabled,
                            cor_correction_only,
                            n_iter_iter_reproj,
                            sample_flipped_remounted_mid_experiment,
                            n_iterations_cor_correction,
                            eps_cor_correction,
                            sigma,
                            alpha,
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
        print(elements_xrt)
        counts_xrt_sig_idx = elements_xrt.index('xrt_sig')
        counts_xrt_sig = counts_xrt[counts_xrt_sig_idx]

        if pre_existing_align_norm_file_enabled:
            print('Extracting pre-existing normalizations, net x pixel shifts, net y pixel shifts, pixel radii for adjacent angle jitter correction and iterative reprojection, and incident intensity from CSV file...')

            norm_array, \
            init_x_shift_array, \
            init_y_shift_array, \
            pixel_rad_adjacent_angle_jitter, \
            pixel_rad_cor_correction, \
            pixel_rad_iter_reproj, \
            I0_cts = futil.extract_csv_raw_input_data(pre_existing_align_norm_file_path, theta)

            print('Applying pre-existing per-projection normalizations to XRF, XRT arrays...')

            counts_xrf_norm = counts_xrf*norm_array[None, :, None, None]
            counts_xrt_norm = counts_xrt_sig*norm_array[:, None, None]

        else:
            init_x_shift_array = np.zeros(n_theta)
            init_y_shift_array = np.zeros(n_theta)
            
            pixel_rad_adjacent_angle_jitter = None
            pixel_rad_cor_correction = None
            pixel_rad_iter_reproj = None
            
            if norm_enabled:
                print('Normalizing XRF, XRT data via per-projection XRT masks...')

                if return_aux_data:
                    counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts, conv_mag_array = ppu.joint_fluct_norm(counts_xrt_sig,
                                                                                                                counts_xrf, 
                                                                                                                xrt_data_percentile, 
                                                                                                                return_conv_mag_array = True)
                
                else:
                    counts_xrt_norm, counts_xrf_norm, norm_array, I0_cts = ppu.joint_fluct_norm(counts_xrt_sig,
                                                                                                counts_xrf,
                                                                                                xrt_data_percentile)
                
                # for theta_idx in range(n_theta):
                #     plt.imshow(counts_xrt_norm[theta_idx], vmin = counts_xrt_norm.min(), vmax = counts_xrt_norm.max())
                #     plt.show()

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
        if pre_cor_correction_adjacent_angle_jitter_correction_enabled:
            print('Calculating required shifts for vertical jitter correction pre-center of rotation error correction...')
            
            if aligning_element == 'opt_dens':
                proj_img_array_element_to_align_with = opt_dens_norm
            
            elif aligning_element in elements_xrf:
                proj_img_array_element_to_align_with = counts_xrf_norm[elements_xrf.index(aligning_element)]
            
            elif aligning_element == 'xrt':
                proj_img_array_element_to_align_with = counts_xrt_norm
            
            else:
                print('Error: \'aligning_element\' must be in \'elements_xrf\' or \'opt_dens\'. Exiting program...')

                sys.exit()

            init_y_shift_array, \
            start_slice_aux, \
            end_slice_aux, \
            phase_xcorr_2d_aggregate_aux, \
            phase_xcorr_2d_truncated_aggregate_aux, \
            adj_angle_jitter_corrected_proj_element_to_align_with_aux = realign.correct_adjacent_angle_jitter_pre_cor_correction(proj_img_array_element_to_align_with, 
                                                                                                                                 init_y_shift_array,
                                                                                                                                 sigma,
                                                                                                                                 alpha,
                                                                                                                                 pixel_rad_adjacent_angle_jitter,
                                                                                                                                 theta,
                                                                                                                                 common_field_of_view_axes = 'y',
                                                                                                                                 return_aux_data = return_aux_data)
            
            if not init_edge_crop_enabled:
                start_slice = start_slice_aux
                end_slice = end_slice_aux
                # start_column = start_column_aux
                # end_column = end_column_aux
            
            else:
                if init_edge_pixel_lengths_to_crop is not None:
                    start_slice = init_edge_pixel_lengths_to_crop['top'] + start_slice_aux
                    end_slice = end_slice_aux - init_edge_pixel_lengths_to_crop['bottom']
                    
                    # start_column = init_edge_pixel_lengths_to_crop['left'] + start_column_aux
                    # end_column = end_column_aux - init_edge_pixel_lengths_to_crop['right']
                
                else:
                    print("Error: Empty field for 'init_edge_pixel_lengths_to_crop'. Exiting program...")

                    sys.exit()
            
            # edge_info = {'top': start_slice, 'bottom': end_slice, 'left': start_column, 'right': end_column}
            edge_info = {'top': start_slice, 'bottom': end_slice}

            if pre_existing_align_norm_file_enabled:
                print('Updating input normalization, net y shift, and incident intensity CSV file...')

                futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                                theta,
                                                norm_array,
                                                init_x_shift_array,
                                                init_y_shift_array,
                                                pixel_rad_adjacent_angle_jitter,
                                                pixel_rad_cor_correction,
                                                pixel_rad_iter_reproj,
                                                I0_cts)

            else:
                print('Creating input normalization, net y shift, and incident intensity CSV file...')

                futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                                theta,
                                                norm_array,
                                                init_x_shift_array,
                                                init_y_shift_array,
                                                pixel_rad_pre_cor_jitter = np.zeros(proj_img_array_element_to_align_with.shape[0] - 1),
                                                pixel_rad_cor = 0,
                                                pixel_rad_iter_reproj = np.zeros(proj_img_array_element_to_align_with.shape[0] - 1),
                                                I0_cts = I0_cts)

            if return_aux_data:
                print('Writing the following auxiliary, pre-COR-corrected, adjacent angle jitter-corrected, cropped per-projection data to NumPy (.npy) files (NOTE: Python is needed to view these!) and .gif files:')
                print('     -Phase cross-correlation data')
                print('     -Phase cross-correlation data (truncated)')
                print('     -Shifted, pre-COR-corrected, adjacent angle, vertical jitter-corrected, cropped projection data')
                
                # if common_field_of_view_axes == 'both':
                #     print('     -Shifted, pre-COR-corrected, adjacent angle jitter-corrected, cropped projection data')
                
                # elif common_field_of_view_axes == 'x':
                #     print('     -Shifted, pre-COR-corrected, adjacent angle jitter-corrected (horizontal), cropped projection data')
                
                # elif common_field_of_view_axes == 'y':
                #     print('     -Shifted, pre-COR-corrected, adjacent angle jitter-corrected (vertical), cropped projection data')
            
                # futil.create_adjacent_angle_jitter_corrected_norm_proj_data_npy(xrt_od_xrf_realignment_subdir_path,
                #                                                                 adj_angle_jitter_corrected_proj_element_to_align_with_aux,
                #                                                                 phase_xcorr_2d_aggregate_aux,
                #                                                                 phase_xcorr_2d_truncated_aggregate_aux,
                #                                                                 aligning_element,
                #                                                                 sigma,
                #                                                                 alpha)
                
                # proj_cropped_to_common_fov = proj_img_array_element_to_align_with[:, start_slice_aux:end_slice_aux, :]
                # futil.create_adjacent_angle_jitter_corrected_norm_proj_data_gif(xrt_od_xrf_realignment_subdir_path,
                #                                                                 aligning_element,
                #                                                                 proj_cropped_to_common_fov,
                #                                                                 adj_angle_jitter_corrected_proj_element_to_align_with_aux,
                #                                                                 sigma,
                #                                                                 alpha,
                #                                                                 theta,
                #                                                                 fps)

        counts_xrf_norm[:, np.where(theta == 0)[1]:] = np.flip(counts_xrf_norm[:, np.where(theta == 0)[1]:], axis = 0)

        aligned_proj_final_xrt_sig, \
        aligned_proj_final_opt_dens, \
        aligned_proj_final_xrf, \
        net_x_shifts_pcc_final, \
        net_y_shifts_pcc_final, \
        recon_array, \
        aligned_exp_proj_array, \
        synth_proj_array, \
        pcc_2d_array_iter_reproj, \
        pcc_2d_array_iter_reproj_truncated, \
        dx_array_pcc, \
        dy_array_pcc = realign.realign_proj(cor_correction_only,
                                            aligning_element,
                                            elements_xrf,
                                            counts_xrt_norm,
                                            opt_dens_norm,
                                            counts_xrf_norm,
                                            theta,
                                            sample_flipped_remounted_mid_experiment,
                                            n_iterations_cor_correction,
                                            pixel_rad_cor_correction,
                                            eps_cor_correction,
                                            I0_cts,
                                            n_iter_iter_reproj,
                                            init_x_shift_array,
                                            init_y_shift_array,
                                            sigma,
                                            alpha,
                                            pixel_rad_iter_reproj,
                                            eps_iter_reproj,
                                            edge_info,
                                            return_aux_data)
        
        print('Creating gridrec-based density maps from aligned XRF data...')
        
        if return_aux_data:
            print('Writing per-iteration auxiliary data to NumPy (.npy) files (NOTE: Python is needed to view these!)...')

            futil.create_post_iter_reproj_aux_data_npy(xrt_od_xrf_realignment_subdir_path,
                                                       cor_correction_only,
                                                       aligned_exp_proj_array,
                                                       recon_array,
                                                       synth_proj_array,
                                                       pcc_2d_array_iter_reproj,
                                                       pcc_2d_array_iter_reproj_truncated,
                                                       dx_array_pcc,
                                                       dy_array_pcc,
                                                       net_x_shifts_pcc_final,
                                                       net_y_shifts_pcc_final)

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
        
        print('Creating gridrec-based density maps from aligned and cropped XRF data...')
        
        gridrec_density_maps = ppu.create_gridrec_density_maps(aligned_proj_final_xrf_cropped,
                                                               elements_xrf,
                                                               theta)
        
        print('Writing gridrec-based density map data to HDF5 file...')
        
        futil.create_gridrec_density_maps_h5(xrt_od_xrf_realignment_subdir_path,
                                             gridrec_density_maps,
                                             elements_xrf)
        
        print('Writing final aligned XRF, XRT, and optical density projection data to HDF5 file...')
            
        futil.create_h5_aligned_aggregate_xrf_xrt(xrt_od_xrf_realignment_subdir_path,
                                                  elements_xrf,
                                                  aligned_proj_final_xrf_cropped, 
                                                  aligned_proj_final_xrt_sig_cropped,
                                                  aligned_proj_final_opt_dens_cropped,
                                                  theta,
                                                  init_edge_pixel_lengths_to_crop,
                                                  final_edge_pixel_lengths_to_crop)
            
        print('Writing per-projection normalization, final net x and y shifts, and incident intensity to CSV file...')

        futil.create_csv_output_data(xrt_od_xrf_realignment_subdir_path,
                                     theta,
                                     net_x_shifts_pcc_final,
                                     net_y_shifts_pcc_final,
                                     I0_cts)
            
        print('Done')

    else:
        if return_aux_data:
            if norm_enabled:
                if not pre_existing_align_norm_file_enabled:
                    print('Writing per-projection convolution magnitudes to NumPy (.npy) file...')

                    futil.create_aux_conv_mag_data_npy(xrt_od_xrf_realignment_subdir_path, conv_mag_array)
            
                    print('Preparing pre-aligned, non-cropped, normalized XRF, XRT, and optical density projection data for GIF creation...')

                    futil.create_nonaligned_norm_non_cropped_proj_data_gif(dir_path = xrt_od_xrf_realignment_subdir_path,
                                                                           xrf_element_array = elements_xrf,
                                                                           desired_xrf_element = desired_xrf_element,
                                                                           counts_xrf = counts_xrf_norm,
                                                                           counts_xrf_norm = counts_xrf_norm,
                                                                           counts_xrt = counts_xrt_sig,
                                                                           counts_xrt_norm = counts_xrt_norm,
                                                                           opt_dens = opt_dens_norm,
                                                                           convolution_mag_array = conv_mag_array,
                                                                           norm_enabled = norm_enabled,
                                                                           data_percentile = xrt_data_percentile, #
                                                                           theta_array = theta,
                                                                           fps = fps)
                else:
                    print('Preparing pre-aligned, non-cropped, normalized XRF, XRT, and optical density projection data for GIF creation...')
                    print('(NOTE: Using pre-existing normalization file; thus, no updated convolution array will be output.)')

                    futil.create_nonaligned_norm_non_cropped_proj_data_gif(dir_path = xrt_od_xrf_realignment_subdir_path,
                                                                           xrf_element_array = elements_xrf,
                                                                           desired_xrf_element = desired_xrf_element,
                                                                           counts_xrf = counts_xrf_norm,
                                                                           counts_xrf_norm = counts_xrf_norm,
                                                                           counts_xrt = counts_xrt_sig,
                                                                           counts_xrt_norm = counts_xrt_norm,
                                                                           opt_dens = opt_dens_norm,
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
                                                                       opt_dens_norm = opt_dens_norm, # NOTE: opt_dens is the same as opt_dens_norm since norm_enabled is False
                                                                       norm_enabled = norm_enabled,
                                                                       theta_array = theta,
                                                                       fps = fps)
                                                                       
            final_xrf = counts_xrf_norm
            final_xrt = counts_xrt_norm
            final_opt_dens = opt_dens_norm

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
                                                  theta,
                                                  init_edge_pixel_lengths_to_crop,
                                                  final_edge_pixel_lengths_to_crop)

        print('Writing per-projection normalization, final net x and y shifts, and incident intensity to CSV file...')

        futil.create_csv_output_data(xrt_od_xrf_realignment_subdir_path,
                                     theta,
                                     net_x_shift_array = np.zeros(final_xrt_sig_cropped.shape[0] - 1),
                                     net_y_shift_array = np.zeros(final_xrt_sig_cropped.shape[0] - 1),
                                     I0_cts = I0_cts)
            
        print('Done')
    
    return