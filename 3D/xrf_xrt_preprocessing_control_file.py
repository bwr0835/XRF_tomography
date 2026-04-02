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
                            incident_energy_keV,
                            pre_existing_aggregate_xrf_xrt_file_lists_enabled,
                            aggregate_xrf_csv_file_path,
                            aggregate_xrt_csv_file_path,
                            aggregate_xrf_h5_file_path,
                            aggregate_xrt_h5_file_path,
                            pre_existing_align_norm_file_enabled,
                            pre_existing_align_norm_file_path,
                            norm_enabled,
                            desired_xrf_element,
                            data_percentile,
                            return_aux_data,
                            incident_flux_photons_per_s,
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

    else:
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
        intensity_xrt_sig_orig = intensity_xrt_sig.copy()

        intensity_xrf_orig = intensity_xrf.copy()

        if pre_existing_align_norm_file_enabled:
            print('Extracting pre-existing normalizations, net x pixel shifts, net y pixel shifts, pixel radii for adjacent angle jitter correction and iterative reprojection, and incident intensity from CSV file...')

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

        else:
            init_x_shift_array = np.zeros(n_theta)
            init_y_shift_array = np.zeros(n_theta)
            
            pixel_rad_adjacent_angle_jitter = None
            pixel_rad_cor_correction = None
            pixel_rad_iter_reproj = None
            
            if norm_enabled:
                print('Normalizing XRF, XRT data via per-projection XRT masks...')

                if return_aux_data:
                    intensity_xrt_norm, intensity_xrf_norm, norm_array_xrt, norm_array_xrf, I0_photons, conv_mag_array = ppu.joint_fluct_norm(intensity_xrt_sig,
                                                                                                                                              intensity_xrf, 
                                                                                                                                              data_percentile,
                                                                                                                                              xrt_photon_counting, 
                                                                                                                                              incident_flux_photons_per_s,
                                                                                                                                              t_dwell_s,
                                                                                                                                              return_conv_mag_array = True)
                
                else:
                    intensity_xrt_norm, intensity_xrf_norm, norm_array_xrt, norm_array_xrf, I0_photons = ppu.joint_fluct_norm(intensity_xrt_sig,
                                                                                                                              intensity_xrf,
                                                                                                                              data_percentile,
                                                                                                                              xrt_photon_counting,
                                                                                                                              incident_flux_photons_per_s,
                                                                                                                              t_dwell_s)

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
    
    xrt_od_xrf_realignment_subdir_path = os.path.join(aligned_data_output_dir_path, 'xrt_od_xrf_realignment')

    os.makedirs(xrt_od_xrf_realignment_subdir_path, exist_ok = True)

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
            
        proj_img_array_element_to_align_with_orig = proj_img_array_element_to_align_with.copy()
        
        if pre_cor_correction_adjacent_angle_jitter_correction_enabled:
            print('Calculating required shifts for vertical jitter correction pre-center of rotation error correction...')
        
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
                                                                                                                                 return_aux_data = return_aux_data)
            
            if not init_edge_crop_enabled:
                start_slice = start_slice_aux
                end_slice = end_slice_aux

                
            
            else:
                if init_edge_pixel_lengths_to_crop is not None:
                    start_slice = init_edge_pixel_lengths_to_crop['top'] + start_slice_aux
                    end_slice = end_slice_aux - init_edge_pixel_lengths_to_crop['bottom']

                    adj_angle_jitter_corrected_proj_element_to_align_with_aux = adj_angle_jitter_corrected_proj_element_to_align_with_aux[:, init_edge_pixel_lengths_to_crop['top']:(end_slice_aux - init_edge_pixel_lengths_to_crop['bottom'])]
                    
                else:
                    print("Error: Empty field for 'init_edge_pixel_lengths_to_crop'. Exiting program...")

                    sys.exit()
            
            edge_info = {'top': start_slice, 'bottom': end_slice}

            for key in edge_info.keys():
                if key == 'top':
                    init_edge_pixel_lengths_to_crop[key] = edge_info[key]
                
                elif key == 'bottom':
                    init_edge_pixel_lengths_to_crop[key] = n_slices - edge_info[key]

            if pre_existing_align_norm_file_enabled:
                print('Updating input normalization, net y shift CSV file...')
        
                futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                                theta,
                                                norm_array_xrf,
                                                norm_array_xrt,
                                                init_x_shift_array,
                                                init_y_shift_array,
                                                pixel_rad_adjacent_angle_jitter,
                                                pixel_rad_cor_correction,
                                                pixel_rad_iter_reproj,
                                                I0_photons,
                                                data_percentile,
                                                aligning_element)

            else:
                print('Creating input normalization, net y shift, and incident intensity CSV file...')

                futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                                theta,
                                                norm_array_xrf,
                                                norm_array_xrt,
                                                init_x_shift_array,
                                                init_y_shift_array,
                                                pixel_rad_pre_cor_jitter = np.zeros(proj_img_array_element_to_align_with.shape[0] - 1),
                                                pixel_rad_cor = 0,
                                                pixel_rad_iter_reproj = np.zeros(proj_img_array_element_to_align_with.shape[0]),
                                                I0_photons = I0_photons,
                                                data_percentile = data_percentile,
                                                aligning_element = aligning_element)

            if return_aux_data:
                print('Writing the following auxiliary, pre-COR-corrected, adjacent angle jitter-corrected, cropped per-projection data to NumPy (.npy) files (NOTE: Python is needed to view these!) and .gif files:')
                print('     -Phase cross-correlation data')
                print('     -Phase cross-correlation data (truncated)')
                print('     -Shifted, pre-COR-corrected, adjacent angle, vertical jitter-corrected, cropped projection data')
            
                futil.create_adjacent_angle_jitter_corrected_norm_proj_data_npy(xrt_od_xrf_realignment_subdir_path,
                                                                                adj_angle_jitter_corrected_proj_element_to_align_with_aux,
                                                                                phase_xcorr_2d_aggregate_aux,
                                                                                phase_xcorr_2d_truncated_aggregate_aux,
                                                                                aligning_element,
                                                                                sigma,
                                                                                alpha)
                
                print('Creating vertical jitter-corrected, cropped per-projection data GIF...')
                
                print(proj_img_array_element_to_align_with_orig.shape)
                print(adj_angle_jitter_corrected_proj_element_to_align_with_aux.shape)

                futil.create_adjacent_angle_jitter_corrected_norm_proj_data_gif(dir_path = xrt_od_xrf_realignment_subdir_path,
                                                                                ref_element = aligning_element,
                                                                                intensity_ref_element = proj_img_array_element_to_align_with_orig[:, start_slice:end_slice],
                                                                                shifted_intensity_ref_element = adj_angle_jitter_corrected_proj_element_to_align_with_aux,
                                                                                sigma = sigma,
                                                                                alpha = alpha,
                                                                                theta_array = theta,
                                                                                fps = fps)

        else:
            if pre_existing_align_norm_file_enabled:
                print('Updating input data CSV file...')

                futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                                theta,
                                                norm_array_xrf,
                                                norm_array_xrt,
                                                init_x_shift_array,
                                                init_y_shift_array,
                                                pixel_rad_adjacent_angle_jitter,
                                                pixel_rad_cor_correction,
                                                pixel_rad_iter_reproj,
                                                I0_photons,
                                                data_percentile,
                                                aligning_element)

            else:
                print('Creating input data CSV file...')

                futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                                theta,
                                                norm_array_xrf,
                                                norm_array_xrt,
                                                init_x_shift_array,
                                                init_y_shift_array,
                                                pixel_rad_pre_cor_jitter = np.zeros(proj_img_array_element_to_align_with.shape[0] - 1),
                                                pixel_rad_cor = 0,
                                                pixel_rad_iter_reproj = np.zeros(proj_img_array_element_to_align_with.shape[0]),
                                                I0_photons = I0_photons,
                                                data_percentile = data_percentile,
                                                aligning_element = aligning_element)

            if not init_edge_crop_enabled:
                if np.any(init_y_shift_array != 0):
                    edge_info = {'top': int(np.clip(np.ceil(np.max(init_y_shift_array)), 0, n_slices)), 
                                 'bottom': int(np.clip(n_slices + np.floor(np.min(init_y_shift_array)), 0, n_slices))}
                    
                    init_edge_pixel_lengths_to_crop = {'top': 0, 'bottom': 0}
                    
                    for key in edge_info.keys():
                        if key == 'top':
                            init_edge_pixel_lengths_to_crop[key] = edge_info[key]
                        
                        elif key == 'bottom':
                            init_edge_pixel_lengths_to_crop[key] = n_slices - edge_info[key]
                
                else:
                    edge_info = None
            
            elif init_edge_pixel_lengths_to_crop is not None and np.any(init_y_shift_array != 0):
                edge_info = {'top': int(np.clip(np.ceil(np.max(init_y_shift_array)), 0, n_slices) + init_edge_pixel_lengths_to_crop['top']), 
                             'bottom': int(np.clip(n_slices + np.floor(np.min(init_y_shift_array)), 0, n_slices) - init_edge_pixel_lengths_to_crop['bottom'])}
                
                for key in edge_info.keys():
                    if key == 'top':
                        init_edge_pixel_lengths_to_crop[key] = edge_info[key]
                    
                    elif key == 'bottom':
                        init_edge_pixel_lengths_to_crop[key] = n_slices - edge_info[key]
                    
                    
                
                
            else:
                print("Error: Empty field for 'init_edge_pixel_lengths_to_crop'. Exiting program...")

                sys.exit()
        
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
                                            intensity_xrt_norm,
                                            opt_dens_norm,
                                            intensity_xrf_norm,
                                            theta,
                                            sample_flipped_remounted_mid_experiment,
                                            n_iterations_cor_correction,
                                            pixel_rad_cor_correction,
                                            eps_cor_correction,
                                            I0_photons,
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
                print("Warning: Non-empty 'final_edge_pixel_lengths_to_crop' dictionary detected. Removing items from 'final_edge_pixel_lengths_to_crop'...")

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
                                                  final_edge_pixel_lengths_to_crop,
                                                  I0_photons,
                                                  incident_energy_keV)
            
        print('Writing per-projection normalization, final net x and y shifts, and incident intensity to CSV file...')

        futil.create_csv_output_data(xrt_od_xrf_realignment_subdir_path,
                                     theta,
                                     net_x_shifts_pcc_final,
                                     net_y_shifts_pcc_final,
                                     cor_correction_only = cor_correction_only)
            
        print('Done')

    else:
        if pre_existing_align_norm_file_enabled:
            print('Updating input data CSV file...')

            futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                            theta,
                                            norm_array_xrf,
                                            norm_array_xrt,
                                            init_x_shift_array,
                                            init_y_shift_array,
                                            pixel_rad_adjacent_angle_jitter,
                                            pixel_rad_cor_correction,
                                            pixel_rad_iter_reproj,
                                            I0_photons,
                                            data_percentile,
                                            aligning_element)

        else:
            print('Creating input data CSV file...')

            futil.create_csv_raw_input_data(xrt_od_xrf_realignment_subdir_path,
                                            theta,
                                            norm_array_xrf,
                                            norm_array_xrt,
                                            init_x_shift_array,
                                            init_y_shift_array,
                                            pixel_rad_pre_cor_jitter = np.zeros(opt_dens_norm.shape[0] - 1),
                                            pixel_rad_cor = 0,
                                            pixel_rad_iter_reproj = np.zeros(opt_dens_norm.shape[0]),
                                            I0_photons = I0_photons,
                                            data_percentile = data_percentile,
                                            aligning_element = aligning_element)

        if return_aux_data:
            if norm_enabled:
                if not pre_existing_align_norm_file_enabled:
                    print('Writing per-projection convolution magnitudes to NumPy (.npy) file...')

                    futil.create_aux_conv_mag_data_npy(xrt_od_xrf_realignment_subdir_path, conv_mag_array)
            
                    print('Preparing pre-aligned, non-cropped, normalized XRF, XRT, and optical density projection data for GIF creation...')

                    futil.create_nonaligned_norm_non_cropped_proj_data_gif(dir_path = xrt_od_xrf_realignment_subdir_path,
                                                                           xrf_element_array = elements_xrf,
                                                                           desired_xrf_element = desired_xrf_element,
                                                                           intensity_xrf = intensity_xrf_orig,
                                                                           intensity_xrf_norm = intensity_xrf_norm,
                                                                           intensity_xrt = intensity_xrt_sig_orig,
                                                                           intensity_xrt_norm = intensity_xrt_norm,
                                                                           opt_dens = opt_dens_norm,
                                                                           convolution_mag_array = conv_mag_array,
                                                                           norm_enabled = norm_enabled,
                                                                           data_percentile = data_percentile, #
                                                                           theta_array = theta,
                                                                           fps = fps)
                else:
                    print('Preparing pre-aligned, non-cropped, normalized XRF, XRT, and optical density projection data for GIF creation...')
                    print('(NOTE: Using pre-existing normalization file; thus, no updated convolution array will be output.)')

                    futil.create_nonaligned_norm_non_cropped_proj_data_gif(dir_path = xrt_od_xrf_realignment_subdir_path,
                                                                           xrf_element_array = elements_xrf,
                                                                           desired_xrf_element = desired_xrf_element,
                                                                           intensity_xrf = intensity_xrf_norm,
                                                                           intensity_xrf_norm = intensity_xrf_norm,
                                                                           intensity_xrt = intensity_xrt_sig,
                                                                           intensity_xrt_norm = intensity_xrt_norm,
                                                                           opt_dens = opt_dens_norm,
                                                                           norm_enabled = norm_enabled,
                                                                           data_percentile = data_percentile, #
                                                                           theta_array = theta,
                                                                           fps = fps)

            else:
                print('Preparing pre-aligned, non-cropped, non-normalized XRF, XRT, and optical density projection data for GIF creation...')
            
                futil.create_nonaligned_norm_non_cropped_proj_data_gif(dir_path = xrt_od_xrf_realignment_subdir_path,
                                                                       xrf_element_array = elements_xrf,
                                                                       desired_xrf_element = desired_xrf_element,
                                                                       intensity_xrf = intensity_xrf,
                                                                       intensity_xrt = intensity_xrt_sig,
                                                                       opt_dens_norm = opt_dens_norm, # NOTE: opt_dens is the same as opt_dens_norm since norm_enabled is False
                                                                       norm_enabled = norm_enabled,
                                                                       theta_array = theta,
                                                                       fps = fps)
                                                                       
            final_xrf = intensity_xrf_norm
            final_xrt = intensity_xrt_norm
            final_opt_dens = opt_dens_norm

        if final_edge_crop_enabled:
            if final_edge_pixel_lengths_to_crop is None:
                print("Error: Empty field for 'final_edge_pixel_lengths_to_crop'. Exiting program...")

                sys.exit()
            
            print('Cropping XRF, XRT, and optical density projection images...')

            final_xrf_cropped, \
            final_xrt_sig_cropped, \
            final_opt_dens_cropped = ppu.crop_array(final_xrf, 
                                                    final_xrt, 
                                                    final_opt_dens,
                                                    final_edge_pixel_lengths_to_crop)

        else:
            final_xrf_cropped = intensity_xrf_norm
            final_xrt_sig_cropped = intensity_xrt_norm
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
                                                  final_edge_pixel_lengths_to_crop,
                                                  I0_photons,
                                                  incident_energy_keV)

        print('Final net x and y shifts to CSV file...')

        futil.create_csv_output_data(xrt_od_xrf_realignment_subdir_path,
                                     theta,
                                     net_x_shifts = np.zeros(final_xrt_sig_cropped.shape[0]),
                                     net_y_shifts = np.zeros(final_xrt_sig_cropped.shape[0]))

        print('Creating gridrec-based density maps from aligned and cropped XRF data...')
        
        gridrec_density_maps = ppu.create_gridrec_density_maps(final_xrf_cropped,
                                                               elements_xrf,
                                                               theta)
        
        print('Writing gridrec-based density map data to HDF5 file...')
        
        futil.create_gridrec_density_maps_h5(xrt_od_xrf_realignment_subdir_path,
                                             gridrec_density_maps,
                                             elements_xrf)
            
        print('Done')
    
    return