import numpy as np, \
       xrf_xrt_preprocess_file_util as futil, \
       xrf_xrt_preprocess_utils as ppu, \
       realignment_final_ultimate as realign, \
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
                            incident_energy_keV,
                            pre_existing_aggregate_xrf_xrt_file_lists_enabled,
                            aggregate_xrf_csv_file_path,
                            aggregate_xrt_csv_file_path,
                            aggregate_xrf_h5_file_path,
                            aggregate_xrt_h5_file_path,
                            pre_existing_align_norm_file_enabled,
                            pre_existing_align_norm_dir_path,
                            norm_enabled,
                            desired_xrf_element,
                            data_percentile,
                            return_aux_data,
                            incident_flux_photons_per_s,
                            t_dwell_s,
                            aligning_element,
                            pre_cor_correction_adjacent_angle_jitter_correction_enabled,
                            realignment_enabled,
                            cor_correction_enabled,
                            cor_correction_alg,
                            iter_reproj_enabled,
                            n_iter_iter_reproj,
                            sample_flipped_remounted_mid_experiment,
                            sample_flipped_remounted_correction_type,
                            sigma,
                            alpha,
                            eps_iter_reproj,
                            create_final_aligned_proj_enabled,
                            edge_crop_enabled,
                            edge_pixel_lengths_to_crop,
                            desired_xrf_element_list,
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
                                          sample_flipped_remounted_mid_experiment,
                                          incident_energy_keV = incident_energy_keV)

            print('Creating aggregate XRT data file...')

            futil.create_aggregate_xrt_h5(xrt_file_array,
                                          output_xrt_filepath,
                                          synchrotron,
                                          sample_flipped_remounted_mid_experiment,
                                          incident_energy_keV)
        
        elif synchrotron == 'nsls-ii':
            incident_energy_keV, us_ic = futil.create_aggregate_xrf_h5(xrf_file_array,
                                                                       output_xrf_filepath, 
                                                                       synchrotron,
                                                                       sample_flipped_remounted_mid_experiment,
                                                                       us_ic_enabled = True) # us_ic_array only returned since that data is not present in NSLS-II ptychography files

            print('Creating aggregate XRT data file...')

            futil.create_aggregate_xrt_h5(xrt_file_array,
                                          output_xrt_filepath, 
                                          synchrotron,
                                          sample_flipped_remounted_mid_experiment,
                                          incident_energy_keV,
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

    
    if not os.path.dirname(aligned_data_output_dir_path):
        print('Error: Unable to locate output file directory. Exiting program...')

        sys.exit()

    elements_xrf, intensity_xrf, theta, incident_energy_keV, _, dataset_type = futil.extract_h5_aggregate_xrf_data(aggregate_xrf_h5_file_path)
    elements_xrt, intensity_xrt, theta_xrt, _, _, dataset_type, xrt_photon_counting = futil.extract_h5_aggregate_xrt_data(aggregate_xrt_h5_file_path)
        
    if not np.array_equal(theta, theta_xrt):
        print('Error: Inconsistent XRF, XRT projection angles. Exiting program...')

        sys.exit()
        
    if intensity_xrf.shape[2:] != intensity_xrt.shape[2:]:
        print('Error: Inconsistent number of XRF, XRT slices (rows) and/or scan positions (columns). Exiting program...')

        sys.exit()

    _, n_theta, n_slices, n_columns = intensity_xrf.shape

    if (n_slices % 2) or (n_columns % 2):
        if (n_slices % 2) and (n_columns % 2):
            print('Odd number of slices (rows) and scan positions (columns) detected. Padding one additional slice and scan position column to XRF and XRT data...')

            intensity_xrt = ppu.pad_col_row(intensity_xrt, dataset_type)
            intensity_xrf = ppu.pad_col_row(intensity_xrf, dataset_type)
            
            n_slices += 1
            n_columns += 1
        
        elif n_slices % 2:
            print('Odd number of slices (rows) detected. Padding one additional slice to XRF and XRT data...')
                
            intensity_xrt = ppu.pad_row(intensity_xrt, dataset_type)
            intensity_xrf = ppu.pad_row(intensity_xrf, dataset_type)

            n_slices += 1

        else:
            print('Odd number of scan positions (columns) detected. Padding one additional scan position column to XRF and XRT data...')
                
            intensity_xrt = ppu.pad_col(intensity_xrt, dataset_type)
            intensity_xrf = ppu.pad_col(intensity_xrf, dataset_type)

            n_columns += 1
        
    intensity_xrt_sig_idx = elements_xrt.index('xrt_sig')
    intensity_xrt_sig = intensity_xrt[intensity_xrt_sig_idx]

    if pre_existing_align_norm_file_enabled:
        print('Extracting pre-existing normalizations, net x pixel shifts, net y pixel shifts, pixel radii for adjacent angle jitter correction and iterative reprojection, and incident intensity from CSV file...')
        
        pre_existing_align_norm_file_path = os.path.join(pre_existing_align_norm_dir_path, 'raw_input_data.csv')
        print(pre_existing_align_norm_file_path)
        norm_array_xrt, \
        norm_array_xrf, \
        init_x_shift_array, \
        init_y_shift_array, \
        pixel_rad_adjacent_angle_jitter, \
        pixel_rad_cor_correction, \
        pixel_rad_iter_reproj, \
        I0_photons, \
        data_percentile_aux, \
        aligning_element_aux = futil.extract_csv_raw_input_data(pre_existing_align_norm_file_path)

        file_number = int(pre_existing_align_norm_file_path.split('/')[-2].split('_')[-1]) + 1 # Extract file number from pre-existing alignment normalization file path and increment by 1
        file_number = f'{file_number:03d}'

        xrt_od_xrf_realignment_subdir_path = os.path.join(aligned_data_output_dir_path, f'xrt_od_xrf_realignment_{file_number}')

        os.makedirs(xrt_od_xrf_realignment_subdir_path, exist_ok = True)                

        if (data_percentile_aux is None)^(data_percentile is None): # Exclusive OR to see if one is None and the other is not
            print('Error: Inconsistent data percentile between input CSV files. Exiting program...')

            sys.exit()
            
        if data_percentile_aux != data_percentile:
            print('Error: Inconsistent data percentile between input CSV files. Exiting program...')

            sys.exit()

        if aligning_element_aux != aligning_element:
            print('Error: Inconsistent aligning element between input CSV files. Exiting program...')

            sys.exit()

        print('Applying pre-existing per-projection normalizations to XRF, XRT arrays...')

        intensity_xrf_norm = intensity_xrf*norm_array_xrf[None, :, None, None]
        intensity_xrt_norm = intensity_xrt_sig*norm_array_xrt[:, None, None]

        conv_mag_array = None

    else:
        init_x_shift_array = np.zeros(n_theta)
        init_y_shift_array = np.zeros(n_theta)
            
        pixel_rad_adjacent_angle_jitter = None
        pixel_rad_cor_correction = None
        pixel_rad_iter_reproj = None

        file_number = '001'
            
        xrt_od_xrf_realignment_subdir_path = os.path.join(aligned_data_output_dir_path, f'xrt_od_xrf_realignment_{file_number}')

        os.makedirs(xrt_od_xrf_realignment_subdir_path, exist_ok = True)
            
        if norm_enabled:
            print('Normalizing XRF, XRT data via per-projection XRT masks...')

            intensity_xrt_norm, intensity_xrf_norm, norm_array_xrt, norm_array_xrf, I0_photons, conv_mag_array = ppu.joint_fluct_norm(intensity_xrt_sig,
                                                                                                                                      intensity_xrf,
                                                                                                                                      data_percentile,
                                                                                                                                      xrt_photon_counting,
                                                                                                                                      incident_flux_photons_per_s,
                                                                                                                                      t_dwell_s,
                                                                                                                                      return_aux_data)

        else:
            if xrt_photon_counting:
                print('Calculating incident intensity from XRT data...')
                   
                I0_photons = ppu.calculate_abs_incident_intensity_photons(intensity_xrt_sig, data_percentile)
                
            else:
                if incident_flux_photons_per_s is None or t_dwell_s is None:
                    print('Error: Incident photon flux and dwell time must be provided. Exiting program...')

                    sys.exit()
                    
                if incident_flux_photons_per_s < 0 or t_dwell_s < 0:
                    print('Error: Incident photon flux and dwell time must be positive values. Exiting program...')

                    sys.exit()
            
                I0_photons = incident_flux_photons_per_s*t_dwell_s
                
            norm_array_xrt = np.ones(n_theta)
            norm_array_xrf = np.ones(n_theta)

            intensity_xrf_norm = intensity_xrf
            intensity_xrt_norm = intensity_xrt_sig
                
    print('Calculating optical densities...')
        
    opt_dens_norm = -np.log(intensity_xrt_norm/I0_photons)
    
    if realignment_enabled:
        if aligning_element == 'opt_dens':
            proj_img_array_element_to_align_with = opt_dens_norm

        elif aligning_element in elements_xrf:
            proj_img_array_element_to_align_with = intensity_xrf_norm[elements_xrf.index(aligning_element)]
            
        elif aligning_element == 'xrt':
            proj_img_array_element_to_align_with = intensity_xrt_norm
            
        else:
            print('Error: \'aligning_element\' must be in \'elements_xrf\' or \'opt_dens\'. Exiting program...')

            sys.exit()
        
        print(f'Vignetting \'{aligning_element}\' projection images...')
        
        vignetted_proj_array_element_to_align_with = np.zeros_like(proj_img_array_element_to_align_with)
        cval_array = np.zeros(n_theta)

        for theta_idx in range(n_theta):
            vignetted_proj_array_element_to_align_with[theta_idx], cval_array[theta_idx] = ppu.edge_gauss_filter(proj_img_array_element_to_align_with[theta_idx], sigma, alpha, nx = n_columns, ny = n_slices)
            
        if pre_cor_correction_adjacent_angle_jitter_correction_enabled:
            print('Calculating shifts for adjacent angle jitter correction pre-center of rotation error correction...')
            
            net_x_shift_array, \
            net_y_shift_array, \
            phase_xcorr_2d_aggregate_aux, \
            phase_xcorr_2d_truncated_aggregate_aux, \
            adj_angle_jitter_corrected_proj_element_to_align_with, \
            proj_img_array_element_to_align_with_orig = realign.correct_adjacent_angle_jitter_pre_cor_correction(proj_img_array_element_to_align_with, 
                                                                                                                 aligning_element,
                                                                                                                 init_x_shift_array,
                                                                                                                 init_y_shift_array,
                                                                                                                 sigma,
                                                                                                                 alpha,
                                                                                                                 pixel_rad_adjacent_angle_jitter,
                                                                                                                 theta,
                                                                                                                 cval_array,
                                                                                                                 return_aux_data = return_aux_data)
            if return_aux_data:
                print('Writing the following auxiliary data to NumPy (.npy) files (NOTE: Python is needed to view these!) files:')
                print('     -Original (vignetted) projection data')
                print('     -Adjacent angle jitter-corrected (vignetted) projection data')
                print('     -Phase cross-correlation data')
                
                if not np.any(pixel_rad_adjacent_angle_jitter == 0):
                    print('     -Phase cross-correlation data (truncated)')
                
                print('     -Net x, y shifts')
                
                futil.create_post_adjacent_angle_jitter_correction_aux_data_npy(xrt_od_xrf_realignment_subdir_path,
                                                                                proj_img_array_element_to_align_with_orig,
                                                                                adj_angle_jitter_corrected_proj_element_to_align_with,
                                                                                phase_xcorr_2d_aggregate_aux,
                                                                                phase_xcorr_2d_truncated_aggregate_aux,
                                                                                net_x_shift_array, 
                                                                                net_y_shift_array)

                print(f'Creating \'{aligning_element}\' adjacent angle jitter-corrected, per-projection data GIF...')

                futil.create_adjacent_angle_jitter_corrected_norm_proj_data_gif(xrt_od_xrf_realignment_subdir_path,
                                                                                aligning_element,
                                                                                proj_img_array_element_to_align_with_orig,
                                                                                adj_angle_jitter_corrected_proj_element_to_align_with,
                                                                                sigma,
                                                                                alpha,
                                                                                theta,
                                                                                fps)
                
                print(f'Creating \'{aligning_element}\' aggregate phase cross-correlation data GIF...')

                futil.create_phase_xcorr_2d_gif(xrt_od_xrf_realignment_subdir_path,
                                                phase_xcorr_2d_aggregate_aux,
                                                phase_xcorr_2d_truncated_aggregate_aux,
                                                theta,
                                                aligning_element,
                                                'adjacent_angle_jitter',
                                                fps)
                
                print('Creating plot of incremental shifts vs. angle...')

                futil.create_incremental_shifts_vs_angle_plot(xrt_od_xrf_realignment_subdir_path,
                                                              net_x_shift_array,
                                                              net_y_shift_array,
                                                              'adjacent_angle_jitter',
                                                              theta)

            print(f'Creating new raw input data CSV file with file number {file_number}...')
    
            futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                            theta,
                                            norm_array_xrf,
                                            norm_array_xrt,
                                            net_x_shift_array,
                                            net_y_shift_array,
                                            pixel_rad_adjacent_angle_jitter,
                                            pixel_rad_cor_correction,
                                            pixel_rad_iter_reproj,
                                            I0_photons,
                                            data_percentile,
                                            aligning_element)
            
            return

        elif cor_correction_enabled:
            print('Calculating shifts needed for center of rotation correction...')

            net_x_shift_array, \
            net_y_shift_array, \
            shifted_proj_img_array_element_to_align_with, \
            shifted_proj_img_array_element_to_align_with_aux, \
            shifted_proj_img_array_element_to_align_with_orig = realign.correct_center_of_rotation(proj_img_array_element_to_align_with,
                                                                                                   net_x_shift_array,
                                                                                                   net_y_shift_array,
                                                                                                   theta,
                                                                                                   cor_correction_alg,
                                                                                                   aligning_element,
                                                                                                   cval_array,
                                                                                                   sigma,
                                                                                                   alpha,
                                                                                                   pixel_rad_cor_correction,
                                                                                                   sample_flipped_remounted_mid_experiment,
                                                                                                   sample_flipped_remounted_correction_type,
                                                                                                   return_aux_data)

            if return_aux_data:
                print(f'Writing the following auxiliary {aligning_element} data to NumPy (.npy) files (NOTE: Python is needed to view these!) files:')
                print('     -Original (vignetted) projection data')

                if sample_flipped_remounted_mid_experiment:
                    print('     -Shifted, pre-COR-corrected, pre- and post-sample remount offset-corrected (vignetted) projection data')
                
                else:    
                    print('     -Shifted, pre-COR-corrected (vignetted) projection data')

                
                futil.create_post_cor_correction_aux_data_npy(xrt_od_xrf_realignment_subdir_path,
                                                              shifted_proj_img_array_element_to_align_with,
                                                              shifted_proj_img_array_element_to_align_with_aux,
                                                              shifted_proj_img_array_element_to_align_with_orig)
                
                print('Creating figure comparing original, new centers of rotation:')
            
                futil.create_center_of_rotation_figures(xrt_od_xrf_realignment_subdir_path,
                                                        shifted_proj_img_array_element_to_align_with,
                                                        shifted_proj_img_array_element_to_align_with_aux,
                                                        sample_flipped_remounted_mid_experiment,
                                                        theta)
            
            print(f'Creating new raw input data CSV file with file number {file_number}...')

            futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                            theta,
                                            norm_array_xrf,
                                            norm_array_xrt,
                                            net_x_shift_array,
                                            net_y_shift_array,
                                            pixel_rad_cor_correction,
                                            pixel_rad_iter_reproj,
                                            I0_photons,
                                            data_percentile,
                                            aligning_element)

        elif iter_reproj_enabled:
            print('Calculating shifts needed for jitter correction via iterative reprojection...')
            
            net_x_shifts, \
            net_y_shifts, \
            aligned_proj_desired_element, \
            aligned_proj_orig, \
            recon_array, \
            aligned_exp_proj_array, \
            aligned_synth_proj_array, \
            pcc_2d_array, \
            pcc_2d_truncated_array, \
            dx_array_pcc, \
            dy_array_pcc = realign.iter_reproj(proj_img_array_element_to_align_with,
                                               theta,
                                               aligning_element,
                                               n_iter_iter_reproj,
                                               init_x_shift_array,
                                               init_y_shift_array,
                                               cval_array,
                                               pixel_rad_iter_reproj,
                                               eps_iter_reproj,
                                               return_aux_data)
            
            if return_aux_data:
                print('Writing the following per-iteration auxiliary data to NumPy (.npy) files (NOTE: Python is needed to view these!)...')
                print('     -Aligned (vignetted) experimental projection data')
                print('     -Reconstruction slices from vignetted projections')
                print('     -Aligned (vignetted) synthetic projection data')
                print('     -Phase cross-correlation data')
                print('     -Phase cross-correlation data (truncated)')
                print('     -Net x shifts')
                print('     -Net y shifts')
                
                futil.create_post_iter_reproj_aux_data_npy(xrt_od_xrf_realignment_subdir_path,
                                                           aligned_exp_proj_array,
                                                           recon_array,
                                                           aligned_synth_proj_array,
                                                           pcc_2d_array,
                                                           pcc_2d_truncated_array,
                                                           dx_array_pcc,
                                                           dy_array_pcc,
                                                           net_x_shifts,
                                                           net_y_shifts)
                
                print('Creating 2D phase cross-correlation data GIF for FIRST iteration...')
                
                futil.create_phase_xcorr_2d_gif(xrt_od_xrf_realignment_subdir_path,
                                                pcc_2d_array,
                                                pcc_2d_truncated_array,
                                                theta,
                                                aligning_element,
                                                'iter_reproj',
                                                fps)
                
                print('Creating experimental, synthetic projection data GIF for FIRST, FINAL iterations...')
                
                futil.create_exp_synth_proj_data_gif(xrt_od_xrf_realignment_subdir_path,
                                                     aligned_exp_proj_array,
                                                     aligned_synth_proj_array,
                                                     theta,
                                                     fps)
                
                if n_iter_iter_reproj <= 5:
                    print(f'Creating plot of incremental shifts vs. angle for {net_x_shifts.shape[0]} iterations...')

                    futil.create_incremental_shifts_vs_angle_plot(xrt_od_xrf_realignment_subdir_path,
                                                                  net_x_shifts,
                                                                  net_y_shifts,
                                                                  'iter_reproj',
                                                                  theta)
                
                else:
                    print('Warning: Number of iterations is greater than 5. Plot of incremental shifts vs. angle will not be created...')

                print(f'Creating new raw input data CSV file with file number {file_number}...')

            futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                            theta,
                                            norm_array_xrf,
                                            norm_array_xrt,
                                            net_x_shifts,
                                            net_y_shifts,
                                            pixel_rad_cor_correction,
                                            pixel_rad_iter_reproj,
                                            I0_photons,
                                            data_percentile,
                                            aligning_element)

        elif create_final_aligned_proj_enabled:
            print('Shifting all XRF, XRT, optical density projection images by final net shifts...')

            shifted_xrf_proj_img_array, \
            shifted_xrt_proj_img_array, \
            shifted_opt_dens_proj_img_array = realign.realign_proj_final(intensity_xrf_norm, intensity_xrt_norm, opt_dens_norm, theta, net_x_shifts, net_y_shifts)
            
            if np.any(init_y_shift_array != 0):
                print('Cropping final aligned XRF, XRT, optical density projection images to vertical common field of view...')

                if edge_crop_enabled:
                    if edge_pixel_lengths_to_crop is None:
                        print("Error: Empty field for 'edge_pixel_lengths_to_crop'. Exiting program...")

                        sys.exit()

                    print('...and cropping additional rows and/or columns...')
            
            elif edge_crop_enabled:
                if edge_pixel_lengths_to_crop is None:
                    print("Error: Empty field for 'edge_pixel_lengths_to_crop'. Exiting program...")

                    sys.exit()

                print('Cropping final aligned XRF, XRT, optical density projection based on \'edge_pixel_lengths_to_crop\'...')

            cropped_xrf_proj_img_array, \
            cropped_xrt_proj_img_array, \
            cropped_opt_dens_proj_img_array, \
            start_slice, \
            end_slice, \
            start_column, \
            end_column = ppu.joint_array_crop(shifted_xrf_proj_img_array, 
                                              shifted_xrt_proj_img_array, 
                                              shifted_opt_dens_proj_img_array, 
                                              net_y_shift_array, 
                                              edge_pixel_lengths_to_crop)

            if return_aux_data:
                print(f'Creating final aligned projection data GIF for {aligning_element} in common field of view...')

                if aligning_element == 'opt_dens':
                    cropped_intensity_ref_element = cropped_opt_dens_proj_img_array
                
                elif aligning_element in elements_xrf:
                    cropped_intensity_ref_element = cropped_xrf_proj_img_array[elements_xrf.index(aligning_element)]
                
                elif aligning_element == 'xrt':
                    cropped_intensity_ref_element = cropped_xrt_proj_img_array

                else:
                    print('Error: \'aligning_element\' must be in \'elements_xrf\' or \'opt_dens\'. Exiting program...')

                    sys.exit()
                
                cropped_proj_ref_element, \
                _, \
                _, \
                _, \
                _ = ppu.crop_array(proj_img_array_element_to_align_with_orig, 
                                   net_y_shift_array, 
                                   edge_pixel_lengths_to_crop)
                
                futil.create_final_aligned_proj_data_gif(xrt_od_xrf_realignment_subdir_path,
                                                         aligning_element,
                                                         cropped_proj_ref_element,
                                                         cropped_intensity_ref_element,
                                                         theta,
                                                         fps)
            
            print('Writing final aligned XRF, XRT, and optical density projection data to HDF5 file...')

            if (start_slice, start_column) != (0, 0) or (end_slice, end_column) != (n_slices, n_columns):
                edge_info = {'left': start_column, 
                             'right': n_columns - end_column, 
                             'top': start_slice,
                             'bottom': n_slices - end_slice}

            else:
                edge_info = None

            futil.create_h5_aligned_aggregate_xrf_xrt(aligned_data_output_dir_path,
                                                      elements_xrf,
                                                      cropped_xrf_proj_img_array,
                                                      cropped_xrt_proj_img_array,
                                                      cropped_opt_dens_proj_img_array,
                                                      theta,
                                                      edge_info,
                                                      I0_photons,
                                                      incident_energy_keV)
            
            print('Creating gridrec-based density maps from aligned XRF data...')

            gridrec_density_maps = ppu.create_gridrec_density_maps(cropped_xrf_proj_img_array,
                                                                   elements_xrf,
                                                                   theta)
            
            print('Writing gridrec-based density map data to HDF5 file...')
            
            futil.create_gridrec_density_maps_h5(aligned_data_output_dir_path,
                                                 gridrec_density_maps,
                                                 elements_xrf)
            
            if return_aux_data:
                print('Creating gridrec-based density map GIF...')

                if len(desired_xrf_element_list) != 4:
                    print('Error: \'desired_xrf_element_list\' must contain 4 elements. Exiting program...')

                    sys.exit()

                for desired_xrf_element in desired_xrf_element_list:
                    if desired_xrf_element not in elements_xrf:
                        print(f'Error: \'desired_xrf_element\' {desired_xrf_element} not in \'elements_xrf\'. Exiting program...')

                        sys.exit()

                futil.create_gridrec_density_map_gif(aligned_data_output_dir_path,
                                                     gridrec_density_maps,
                                                     desired_xrf_element_list,
                                                     elements_xrf,
                                                     fps)

            print('Done')

            return
        
        else:
            print('Error: Alignment stage not selected. Exiting program...')
    
            sys.exit()
    
    else:
        print('Skipping alignment...')
        print('Preparing pre-aligned, non-cropped, normalized XRF, XRT, and optical density projection data for GIF creation...')

        if conv_mag_array is None:
            print('(NOTE: Using pre-existing normalization file or normalization disabled; thus, no updated convolution array will be output.)')

        futil.create_nonaligned_norm_non_cropped_proj_data_gif(dir_path = xrt_od_xrf_realignment_subdir_path,
                                                               xrf_element_array = elements_xrf,
                                                               desired_xrf_element = desired_xrf_element,
                                                               intensity_xrf = intensity_xrf_norm,
                                                               intensity_xrf_norm = intensity_xrf_norm,
                                                               intensity_xrt = intensity_xrt_sig,
                                                               intensity_xrt_norm = intensity_xrt_norm,
                                                               opt_dens = opt_dens_norm,
                                                               convolution_mag_array = conv_mag_array,
                                                               norm_enabled = norm_enabled,
                                                               data_percentile = data_percentile, #
                                                               theta_array = theta,
                                                               fps = fps)
        
        print('Creating new raw input data CSV file...')

        futil.create_csv_raw_input_data(dir_path = xrt_od_xrf_realignment_subdir_path,
                                        theta_array = theta,
                                        norm_factor_xrf = norm_array_xrf,
                                        norm_factor_xrt = norm_array_xrt,
                                        init_x_shifts = np.zeros(n_theta),
                                        init_y_shifts = np.zeros(n_theta),
                                        pixel_rad_pre_cor_jitter = np.zeros(max(n_theta - 1, 0)),
                                        pixel_rad_cor = 0,
                                        pixel_rad_iter_reproj = np.zeros(n_theta),
                                        I0_photons = I0_photons,
                                        data_percentile = data_percentile,
                                        aligning_element = aligning_element)
        
        print('Done')
    
        return