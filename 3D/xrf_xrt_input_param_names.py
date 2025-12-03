available_synchrotrons = ['aps',
                          'nsls-ii']

available_noise_models = ['gaussian',
                          'poisson']

preprocessing_params_ordered = ['synchrotron',
                                'synchrotron_beamline',
                                'create_aggregate_xrf_xrt_files_enabled',
                                'pre_existing_aggregate_xrf_xrt_file_lists_enabled',
                                'aggregate_xrf_csv_file_path',
                                'aggregate_xrt_csv_file_path',
                                'aggregate_xrf_h5_file_path',
                                'aggregate_xrt_h5_file_path',
                                'pre_existing_align_norm_file_enabled',
                                'pre_existing_align_norm_file_path',
                                'norm_enabled',
                                'xrt_data_percentile',
                                'return_aux_data',
                                'I0_cts_per_s',
                                't_dwell_s',
                                'realignment_enabled',
                                'n_iter_iter_reproj',
                                'sigma',
                                'alpha',
                                'upsample_factor',
                                'eps',
                                'aligned_data_output_dir_path']

preprocessing_numeric_params = ['xrt_data_percentile',
                                'I0_cts_per_s',
                                't_dwell_s',
                                'sigma',
                                'alpha',
                                'eps']

preprocessing_bool_params = ['create_aggregate_xrf_xrt_files_enabled',
                             'pre_existing_aggregate_xrf_xrt_file_lists_enabled', 
                             'pre_existing_align_norm_file_enabled',
                             'norm_enabled',
                             'realignment_enabled']

recon_params_ordered = ['synchrotron',
                        'synchrotron_beamline',
                        'f_recon_parameters',
                        'probe_intensity',
                        'selfAb',
                        'cont_from_check_point',
                        'use_saved_initial_guess',
                        'downsample_factor',
                        'upsample_factor',
                        'ini_kind',
                        'init_const',
                        'ini_rand_amp',
                        'recon_path',
                        'f_initial_guess',
                        'f_recon_grid',
                        'data_path'
                        'f_XRF_XRT_data'
                        'this_aN_dic',
                        'element_lines_roi',
                        'n_line_group_each_element',
                        'sample_size_n',
                        'sample_height_n',
                        'sample_size_cm',
                        'probe_energy_keV',
                        'n_epochs',
                        'save_every_n_epochs',
                        'minibatch_size',
                        'b1',
                        'b2',
                        'manual_det_coord',
                        'set_det_coord_cm',
                        'det_on_which_side',
                        'det_from_sample_cm',
                        'det_ds_spacing_cm',
                        'manual_det_area',
                        'det_area_eff_cm2',
                        'det_dia_cm',
                        'P_folder',
                        'f_P']

recon_numeric_scalar_params = ['probe_intensity',
                               'downsample_factor',
                               'upsample_factor',
                               'init_const',
                               'ini_rand_amp'
                               'sample_size_n',
                               'sample_height_n',
                               'sample_size_cm',
                               'n_epochs',
                               'save_every_n_epochs',
                               'minibatch_size',
                               'b1',
                               'b2',
                               'lr',
                               'det_from_sample_cm',
                               'det_ds_spacing_cm',
                               'det_area_eff_cm2',
                               'det_dia_cm']

recon_numeric_array_params = ['element_lines_roi',
                              'n_line_group_each_element'
                              'probe_energy_keV',
                              'set_det_coord_cm']

recon_dict_params = ['this_aN_dict']

recon_bool_params = ['cont_from_check_point',
                     'selfAb',
                     'use_saved_initial_guess',
                     'manual_det_coord',
                     'manual_det_area']