import tkinter as tk, sys, os, h5_util as h5u

from tkinter import filedialog as fd

def preprocess_xrf_xrt_data(synchrotron,
                            synchrotron_beamline,
                            create_aggregate_xrf_xrt_files_enabled,
                            pre_existing_alignment_norm_mass_calibration_file_enabled,
                            pre_existing_alignment_norm_mass_calibration_file_path,
                            norm_enabled,
                            mass_calibration_enabled,
                            mass_calibration_dir,
                            mass_calibration_filepath,
                            mass_calibration_elements,
                            areal_mass_density_mass_calibration_elements_g_cm2,
                            iterative_reproj_enabled):
        
    if create_aggregate_xrf_xrt_files_enabled:
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

        if synchrotron.lower() == 'nsls-ii':
            us_ic, nx, ny = h5u.create_aggregate_xrf_h5(xrf_file_array, 
                                                              output_xrf_filepath, 
                                                              synchrotron, 
                                                              us_ic = True) # us_ic_array only returned since that data is not present in NSLS-II ptychography files

            print('Creating aggregate XRT data file...')

            h5u.create_aggregate_xrt_h5(xrt_file_array, 
                                        output_xrt_filepath, 
                                        synchrotron, 
                                        nx = nx, 
                                        ny = ny, 
                                        us_ic = us_ic)
        
        else:
            h5u.create_aggregate_xrf_h5(xrf_file_array, 
                                        output_xrf_filepath, 
                                        synchrotron)

            print('Creating aggregate XRT data file...')

            h5u.create_aggregate_xrt_h5(xrt_file_array, 
                                        output_xrt_filepath, 
                                        synchrotron)
        
        # sys.exit()

    # if mass_calibration_enabled:
    #     print('Mass calibrating incident beam intensity...')

    #     if synchrotron == 'aps':
            