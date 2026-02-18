#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:58:57 2020

@author: panpanhuang
"""
import numpy as np, \
       torch as tc, \
       xraylib as xlib, \
       xraylib_np as xlib_np, \
       xrf_xrt_jxrft_file_util as futil, \
       xrl_fluorline_macros, \
       Atomic_number as atom_num, \
       os, \
       shutil, \
       datetime, \
       time, \
       matplotlib, \
       h5py, \
       dxchange, \
       warnings

from util import downsample_proj_data, \
                 upsample_recon_data, \
                 rotate, \
                 MakeFLlinesDictionary_manual, \
                 intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual, \
                 find_lines_roi_idx_from_dataset, \
                 upsample_recon_data
from mpi4py import MPI
from torch import nn
from tqdm import tqdm
from standard_calibration import calibrate_incident_probe_intensity
from array_ops import initialize_guess_3d
from forward_model import PPM
from misc import print_flush_root, print_flush_all
from matplotlib import pyplot as plt, ticker as mtick, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

tc.set_default_tensor_type(tc.FloatTensor)

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)

warnings.filterwarnings("ignore")

# fl = {"K": np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
#                      xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE]),
#       "L": np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
#                      xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
#                      xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE]),              
#       "M": np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])               
#      }

fl = xrl_fluorline_macros.fl

# def reconstruct_jXRFT_tomography(
#         synchrotron,
#         # ______________________________________
#         # |Raw data and experimental parameters|________________________________
#         sample_size_n, # Set sample_size_n (sample_height_n) to the number of pixels along the direction
#                        # perpendicular (parallel) to # the rotational axis of the sample;
#         sample_height_n, 
#         sample_size_cm, # Size of the sample size (in unit cm) along the direction 
#                         # perpendicular to the rotational axis of the sample
#         probe_energy_keV = None,
#         probe_intensity = None, # Set the value of incident probe intensity (in photons or photons/s) 
#                                 # For simulations, set the value to some estimated probe intensity
#         probe_att = True,
#         manual_det_coord = True, # True when using a exp. data; 
#                                  # False when using a simulation data; 
#                                  # For simulation data, auto-distribute detecting points on a circular sensing area
#                                  # given det_dia_cm and det_ds_spacing_cm;
#         set_det_coord_cm = None, # Set to None for simulation data;
#                                  # For exp. data, a np array with dimension (# of detecing points, 3)
#                                  # 3 for (z, x, y) coordinates, probe propagate along +y axis; sample rotates about +z axis;
#                                  # The sign of x determines which side the detector locates relative to the sample;
#         det_on_which_side = 'negative', # Choose from 'positive' or 'negative' depending on the side that the detector locates 
#                                   # relative the sample; TODO
#         manual_det_area = True, # True when using a exp. data;
#                                 # False when using a simulation data;
#                                 # If set to True, the program caculated the signal collecting solid angle of XRF using 
#                                 # det_area_eff_cm2;
#                                 # If set to False, the program uses provided det_dia_cm and det_from_sample_cm;
#                                 #### Note ####
#                                 # For exp. data, the factor of signal collecting solid angle is included in probe_intensity
#                                 # which is calculated from the calibration data;
#         det_area_eff_cm2 = None, # For exp. data only. Set the value of the total sensing area;
#                                  # For simulation data, set to None (program calculates sensing area with given det_dia_cm)
#         det_dia_cm = None,  # For simulation data only. Diameter of the sensor assuming a circular sensing area.
#                             # Only used when manual_det_area is False. 
#                             # Need to use the same diameter setting, as it's used to generate simulation data  
#         det_ds_spacing_cm = None, # For simulation data only. used to distribute detecting points on the XRF detecting plane.
#         det_from_sample_cm = None, # For simulation data only.
#                                    # For exp. data, the distance between the detector and the sample is given in set_det_coord_cm
#         # |Probe Intensity calibration data|____________________________________    
#         use_std_calibation = False, # Set use_std_calibration to True if the calibration measurement exist otherwise set to False.
#         std_path = None, # Density_std_elements unit in g/cm^2
#         f_std = None, 
#         std_element_lines_roi = None, 
#         density_std_elements = None, 
#         fitting_method = None, # Set fitting method to  'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus'
#         # |Reconstruction parameters|___________________________________________
#         n_epochs = 50, 
#         save_every_n_epochs = 10, 
#         minibatch_size = None,
#         f_recon_parameters = "recon_parameters.txt", 
#         dev = None,
#         selfAb = False, 
#         cont_from_check_point = False, 
#         use_saved_initial_guess = False, 
#         ini_kind = 'const',  # Set ini_kind to 'const', 'rand' or 'randn'
#         init_const = 0.5, 
#         ini_rand_amp = 0.1,
#         recon_path='./', 
#         f_initial_guess = None, 
#         f_recon_grid = None,  # Name of the file that saves the most recent reconstructed result 
#         # data_path = None, 
#         # f_XRF_data = None, 
#         f_XRF_XRT_data = None,
#         # f_XRT_data = None,
#         scaler_counts_us_ic_dataset_idx = None,
#         scaler_counts_ds_ic_dataset_idx = None,  # the index of us_ic in the dataset MAPS/scaler_names
#         XRT_ratio_dataset_idx = None, # the index of ds_ic in the dataset MAPS/scaler_names
#         theta_ls_dataset = 'exchange/theta', 
#         channel_names = 'exchange/elements',  # the index of abs_ic in the dataset MAPS/scaler_names
#         this_aN_dic = None, 
#         element_lines_roi = None, 
#         n_line_group_each_element = None,
#         b1 = None, 
#         b2 = None, 
#         lr = None,
#         P_folder = None, 
#         f_P = None, 
#         fl_K = fl["K"], 
#         fl_L = fl["L"], 
#         fl_M = fl["M"],
#         **kwargs):
    
def reconstruct_jXRFT_tomography(sample_size_n,
                                 sample_height_n, 
                                 sample_size_cm,
                                 probe_energy_keV = None,
                                 probe_intensity = None,
                                 probe_att = True,
                                 manual_det_coord = True,
                                 set_det_coord_cm = None,
                                 det_on_which_side = 'negative',
                                 manual_det_area = True,
                                 det_area_eff_cm2 = None, 
                                 det_dia_cm = None,
                                 det_ds_spacing_cm = None,
                                 det_from_sample_cm = None,
                                 det_window_element = None,
                                 det_window_thickness_um = None,
                                 n_epochs = 50, 
                                 save_every_n_epochs = 10, 
                                 minibatch_size = None,
                                 f_recon_parameters = "recon_parameters.txt", 
                                 dev = None,
                                 selfAb = True,
                                 noise_model = None,
                                 cont_from_check_point = False, 
                                 use_saved_initial_guess = False, 
                                 ini_kind = 'const',
                                 init_const = 0.5, 
                                 ini_rand_amp = 0.1,
                                 recon_path = None, 
                                 f_initial_guess = None, 
                                 f_recon_grid = None,
                                 f_XRF_XRT_data = None,
                                 opt_dens_enabled = True,
                                 downsample_factor = 1,
                                 upsample_factor = 1,
                                 this_aN_dic = None, 
                                 element_lines_roi = None, 
                                 n_line_group_each_element = None,
                                 b1 = None, 
                                 b2 = None, 
                                 lr = None,
                                 P_folder = None, 
                                 f_P = None, 
                                 fl_K = fl["K"], 
                                 fl_L = fl["L"], 
                                 fl_M = fl["M"],
                                 **kwargs):

    '''
    Perform joint iterative X-ray fluorescence (XRF) and X-ray transmission (XRT) reconstruction via automatic differentiation 
    while correcting for incident probe/beam attenuation and XRF self-absorption [1].

    Parameters
    ----------
    synchrotron : str
        Name of synchrotron light source
    synchrotron_beamline : str 
        Name of synchrotron light source beamline
    sample_size_n : int
        Number of pixels along width of projection images when rotation axis is up-down (e.g. # of columns/scan positions)
    sample_height_n: int 
        Number of pixels along height of projection images when rotation axis is up-down (e.g. # of rows/slices)
    sample_size_cm : float
        Size of sample_n (in cm) along direction perpendicular to sample axis of rotation
    probe_energy_keV : ndarray
        Energy of incident probe/beam (in keV)
    probe_intensity : float
        Incident probe/beam intensity (in photons/s or photons)
    probe_att : bool 
        Flag for including incident probe/beam attenuation
    manual_det_coord : bool 
        Flag for setting detector coordinates manually (True for experimental data, False for simulated data)
    set_det_coord_cm : ndarray or None 
        Detector coordinates (cm)
        Coordinate array structure: [[0, z_0, x_0, y_0], [1, z_1, x_1, y_1], ..., [N_det - 1, z_(N_det - 1), x_(N_det - 1), y_(N_det - 1)]]
            z: Axis of rotation
            x: Detection axis (sign of x dictates which side detector located on relative to sample)
            y: Incident beam axis
            N_det: Number of detection points
        Set to None for simulated data (program calculates coordinates using det_dia_cm and det_from_sample_cm)
    det_on_which_side : str
        Which side of a sample the detector is on
        Set to 'negative' or 'positive' (TODO!!!!!)
    det_area_eff_cm2: float 
        Effective detector area (cm^2)
    det_dia_cm : float 
        Detector diameter (cm) assuming a circular detection area
            For simulated data only (or if manual_det_area = False)!
            For simulated data, value must be the same as when generating such data
    n_epochs : int
        Number of epochs to execute function for
    save_every_n_epochs : int
        Save data for every n epochs only
    minibatch_size : int
        Size of randomly-selected subset of XRF, XRT data (TODO!!!!!)
    f_recon_parameters : str
        File name for writing out reconstruction parameters
    dev : str or None
        Device (for parallelization nature of the reconstruction algorithm)
        If None, then only CPU involved in function execution
    selfAb : bool
        Flag for enabling self-absorption correction
    noise_model : str
        Noise model to use with cost function to be minimized
        Set to 'gaussian' or 'poisson'
    cont_from_check_point : bool
        Flag for continuing reconstruction algorithm from (????) (TODO!!!!!)
    use_saved_initial_guess : bool
        Flag for using saved initial XRF, XRT guesses of object (unsure about this) (TODO!!!!!)
    ini_kind : str
        TODO!!!!!
        Set to 'const' (default), 'rand', or 'randn'
    init_const : float
        TODO!!!!!
    ini_rand_amp : float
        TODO!!!!!
    recon_path : str
        Directory path for reconstructed data
    f_initial_guess : str
        File name for initial guess of reconstructed data
    f_recon_grid : str
        File name for most recent reconstructed result
    f_XRF_XRT_data : str
        File name for joint XRF, XRT data (currently supporting .h5 format only)
    this_aN_dic : dict
        Elements and atomic numbers Z in reconstructed object
        Dictionary structure: {Element: Z, ...}
        Example: {'Ca': 20, 'Fe', 26, 'Ba', 56}
    element_lines_roi : ndarray
        Elements of interest and shells of interest
        Array structure: np.array([[Element, subshell], ...])
        Example: np.array([['Si', 'K], ['Ca', 'K'], ['Ca', 'L'], ['Fe', 'K'], ['Cu', 'K'], ['Cu', 'L'], ['Ba', 'L']])
    n_line_group_each_element : ndarray
        Number of shells of interest for each element of interest (array-like; dtype: str)
        Example using above element_lines_roi example: np.array([1, 2, 1, 2, 1])
    b1 : float
        Regularization prefactor of XRT cost term
    b2 : float
        Second prefactor inside XRT cost term [2]
    lr : float
        Learning rate
    P_folder : str, optional (????) (TODO!!!!!)
        Directory path for info on XRF-detectorlet intersecting lengths (TODO!!!!!)
    f_P : str
        File name for XRF-detectorlet intersecting lengths (TODO!!!!!)
    fl_K : dict
        K fluorescence lines
        See XRF_tomography.py for example/default dictionary
        See xraylib documentation for more information on XRF line macros [3]
    fl_L : dict
        L fluorescence lines
        See XRF_tomography.py for default dictionary
        See xraylib documentation for more information on XRF line macros [3]
    fl_M : dict
        M fluorescence lines
        See XRF_tomography.py for default dictionary
        See xraylib documentation for more information on XRF line macros [3]


    References
    ----------
    [1] P. Huang, “Toward Large-scale X-ray Microscopy for Ptychography and Fluorescence Tomography”, Ph.D. Thesis (Northwestern University, May 2022).

    [2] Z. W. Di, S. Chen, Y. P. Hong, C. Jacobsen, S. Leyffer, and S. M. Wild, Opt. Express 25, 13107 (2017).
    
    [3] T. Schoonjans, A. Brunetti, B. Golosio, M. S. D. Rio, V. A. Solé, C. Ferrero, and L. Vincze, Spectrochim. Acta, Part B 66, 776 (2011).
    '''

    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    
    if noise_model == 'gaussian':
        loss_fn = nn.MSELoss()
    
    elif noise_model == 'poisson':
        loss_fn = nn.PoissonNLLLoss()
    
    elif rank == 0:
        msg = 'Error: Unable to extract noise model option. Exiting program...'

        print_flush_root(rank, msg, save_stdout = False, print_terminal = True)

        comm.abort(1)
    
    dia_len_n = int(1.2*(sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5) # dev
    n_voxel_minibatch = minibatch_size*sample_size_n # dev
    n_voxel = sample_height_n*sample_size_n**2 # dev
    
    #### create the file handle for experimental data; y1: channel data, y2: scalers data ####
    
    # y1_true_handle, y2_true_handle = 

    # y1_true_handle = h5py.File(os.path.join(data_path, f_XRF_data), 'r') # XRF data
    # y2_true_handle = h5py.File(os.path.join(data_path, f_XRT_data), 'r') # XRT data
    ####----------------------------------------------------------------------------------####
    
    # TODO Check parallel computing aspect of this function
    # TODO Check with Chris about this_aN_dic and element_line_roi
    

    print_flush_root(rank, "Extracting XRF, XRT data from aggregate file...", save_stdout = False, print_terminal = True)
    
    elements_xrf, \
    xrf_data, \
    xrt_data, \
    theta_tomo = futil.extract_h5_aggregate_xrf_xrt_data(f_XRF_XRT_data, opt_dens_enabled = opt_dens_enabled, element_lines_roi = element_lines_roi)
    
    # TODO Check parallel computing aspect of downsampling and upsampling blocks

    if rank == 0:
        msg = f'Downsampling XRF, optical density projection images by factor of {downsample_factor}'
        
        print_flush_root(rank, msg, save_stdout = False, print_terminal = True)

        xrf_data_roi = downsample_proj_data(xrf_data, downsample_factor)
        xrt_data_new = downsample_proj_data(xrt_data, downsample_factor)

        sample_height_n /= downsample_factor
        sample_size_n /= downsample_factor  

    else:
        xrf_data_roi = None
        xrt_data_new = None
    
    comm.bcast(xrf_data_roi)
    comm.bcast(xrt_data_new)

    dia_len_n = int(1.2*(sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5) # dev
    n_voxel_minibatch = minibatch_size*sample_size_n # dev
    n_voxel = sample_height_n*sample_size_n**2 # dev

    #### Calculate the number of elements in the reconstructed object, list the atomic numbers ####
    n_element = len(elements_xrf)

    aN_ls = np.zeros(n_element, dtype = int)

    for element, idx in enumerate(elements_xrf):
        if '_' in element:
            aN_ls[idx] = xlib.SymbolToAtomicNumber(element.split('_')[0])
            
        else:
            aN_ls[idx] = xlib.SymbolToAtomicNumber(element)

    # aN_ls = np.array(list(this_aN_dic.values()))
    ####--------------------------------------------------------------####
    
    #### Make the lookup table of the fluorescence lines of interests ####

    print_flush_root(rank, "Making lookup table of fluorescence lines of interests...", save_stdout = False, print_terminal = True)

    fl_all_lines_dic = MakeFLlinesDictionary_manual(element_lines_roi,                           
                                                    n_line_group_each_element, 
                                                    probe_energy_keV, 
                                                    sample_size_n, 
                                                    sample_size_cm) #cpu
    
    stdout_options = {'root': 0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
    
    # FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, fl_all_lines_dic["fl_energy"])).float().to(dev) #dev
    FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total_Kissel(aN_ls, fl_all_lines_dic["fl_energy"])).float().to(dev) #dev
    
    detected_fl_unit_concentration = tc.as_tensor(fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(dev)
    n_line_group_each_element = tc.IntTensor(fl_all_lines_dic["n_line_group_each_element"]).to(dev)
    n_lines = fl_all_lines_dic["n_lines"] #scalar
    ####--------------------------------------------------------------####
    
    if det_window_element is not None:
        Z_window = xlib.SymbolToAtomicNumber(det_window_element)
    
        FL_line_det_attCS_ls = tc.as_tensor(xlib_np.CS_Total_Kissel(Z_window, fl_all_lines_dic["fl_energy"])).float().to(dev)

    else:
        FL_line_det_attCS_ls = tc.zeros(n_lines).float().to(dev)

    #### Calculate the MAC of probe ####
    # probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, probe_energy_keV).flatten()).to(dev)
    probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total_Kissel(aN_ls, probe_energy_keV).flatten()).to(dev)
    ####----------------------------####
    
    #### Load all object angles ####
    # theta_ls = tc.from_numpy(y1_true_handle[theta_ls_dataset][...]*np.pi/180).float()  #unit: rad #cpu
    theta_ls = tc.from_numpy(theta_tomo*np.pi/180).float()  #unit: rad #cpu
    n_theta = len(theta_ls)  
    ####------------------------####
   
    # element_lines_roi_idx = find_lines_roi_idx_from_dataset(data_path, f_XRF_data, element_lines_roi, std_sample = False)
    
    #### pick only the element lines of interests from the channel data. flatten the data to strips
    #### original dim = (n_lines_roi, n_theta, sample_height_n, sample_size_n)
    
    # y1_true = tc.from_numpy(y1_true_handle['exchange/data'][element_lines_roi_idx]).view(len(element_lines_roi_idx), n_theta, sample_height_n*sample_size_n).to(dev)
    y1_true = tc.from_numpy(xrf_data_roi).view(n_element, n_theta, sample_height_n*sample_size_n).to(dev)
#     #### pick the probe photon counts after the ion chamber from the scalers data as the transmission data
    # y2_true = tc.from_numpy(y2_true_handle['exchange/data'][scaler_counts_ds_ic_dataset_idx]).view(n_theta, sample_height_n * sample_size_n).to(dev)
    
    ## Use this y2_true if using the attenuating expoenent in the XRT loss calculation
    # y2_true = tc.from_numpy(y2_true_handle['exchange/data'][XRT_ratio_dataset_idx]).view(n_theta, sample_height_n * sample_size_n).to(dev)

    # y2_true = -tc.log(y2_true)

    y2_true = tc.from_numpy(xrt_data_new).view(n_theta, sample_height_n*sample_size_n).to(dev)
    
    #### pick the probe photon counts calibrated for all optics and detectors
    # if use_std_calibation:
    #     probe_cts = calibrate_incident_probe_intensity(std_path, f_std, fitting_method, std_element_lines_roi, density_std_elements, probe_energy_keV)
    #     # TODO Remove since we can extract incident flux from normalization of incident flux fluctuations or photodiode or XRT measurements
    # else:
    #     probe_cts = probe_intensity

    probe_cts = probe_intensity

    minibatch_ls_0 = tc.arange(n_ranks).to(dev) #dev
    n_batch = (sample_height_n*sample_size_n)//(n_ranks*minibatch_size) #scalar
    
    if manual_det_area == True:
        det_solid_angle_ratio = det_area_eff_cm2/(4*np.pi*det_from_sample_cm**2)
        signal_attenuation_factor = 1.0
    
    else:
        #### det_solid_angle_ratio is used only for simulated dataset (use_std_calibation: False, manual_det_area: False, manual_det_coord: False)
        #### in which the incident probe intensity is not calibrated with the axo_std file.
        #### The simulated collected XRF photon number is estimated by multiplying the generated
        #### fluorescence photon number by "det_solid_angle_ratio" to account for the limited solid angle and the detecting efficiency of the detector
        
#         #### Calculate the detecting solid angle covered by the area of the spherical cap covered by the detector #### 
#         #### OPTION A: estimate the solid angle by the curved surface
#         # The distance from the sample to the boundary of the detector
#         r = (det_from_sample_cm**2 + (det_dia_cm/2)**2)**0.5   
#         # The height of the cap
#         h =  r - det_from_sample_cm
#         # The area of the cap area
#         fl_sig_collecting_cap_area = np.pi*((det_dia_cm/2)**2 + h**2)
#         # The ratio of the detecting solid angle / full soilid angle
#         det_solid_angle_ratio = fl_sig_collecting_cap_area / (4*np.pi*r**2)

        #### OPTION B: estimate the solid angle by the flat surface
        det_solid_angle_ratio = (np.pi*(det_dia_cm/2)**2)/(4*np.pi*det_from_sample_cm**2)
        
        #### signal_attenuation_factor is used to account for other factors that cause the attenuation of the XRF
        #### except the limited solid angle and self-absorption
        signal_attenuation_factor = 1.0
   
    checkpoint_path = os.path.join(recon_path, "checkpoint")
   
    if rank == 0: 
        if not os.path.exists(recon_path):
            os.makedirs(recon_path)  
        
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)  
    
    P_save_path = os.path.join(P_folder, f_P)
    
    # Check if the P array exists; if it doesn't exist, call the function to calculate the P array and store it as a .h5 file.
    if not os.path.isfile(P_save_path + ".h5"):
        print_flush_root(rank, "Calculating P array...", save_stdout = False, print_terminal = True)
        
        intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual(n_ranks, minibatch_size, rank,
                                                                    manual_det_coord, set_det_coord_cm, det_on_which_side,
                                                                    manual_det_area, det_dia_cm, det_from_sample_cm, det_ds_spacing_cm,
                                                                    sample_size_n, sample_size_cm,
                                                                    sample_height_n, P_folder, f_P) #cpu
    
    comm.Barrier()
    
    P_handle = h5py.File(P_save_path + ".h5", 'r')

    # TODO Check parallel computing aspect with upsampling data

    if cont_from_check_point == False: 
        # load the saved_initial_guess to rank0 cpu
        if use_saved_initial_guess:
            if rank == 0:
                with h5py.File(os.path.join(recon_path, f_initial_guess + '.h5'), "r") as s:
                    X_init = s["sample/densities"][...].astype(np.float32)                    

                if upsample_factor > 1:
                    msg = f'Upsampling initial guess of reconstruction slices by factor of {upsample_factor}...'
        
                    print_flush_root(rank, msg, save_stdout = False, print_terminal = True)
                   
                    X = upsample_recon_data(X_init, upsample_factor)
                
                elif upsample_factor == 1:
                    X = X_init

                X = tc.from_numpy(X)
                
                shutil.copy(os.path.join(recon_path, f_initial_guess + '.h5'), os.path.join(recon_path, f_recon_grid + '.h5'))
                
            else:
                X = None
                
        # create the initial_guess in rank0 cpu
        else:
            if rank == 0:
                X = initialize_guess_3d("cpu", ini_kind, n_element, sample_size_n, sample_height_n, recon_path, f_recon_grid, f_initial_guess, init_const, ini_rand_amp) #cpu         
                ## Save the initial guess for future reference
                with h5py.File(os.path.join(recon_path, f_initial_guess +'.h5'), 'w') as s:
                    sample = s.create_group("sample")
                
                    sample_v = sample.create_dataset("densities", shape = (n_element, sample_height_n, sample_size_n, sample_size_n), dtype = "f4")
                    sample_e = sample.create_dataset("elements", shape = (n_element,), dtype = 'S5')

                    sample_v[...] = X
                    sample_e[...] = np.array(elements_xrf).astype('S5')
 
                ## Save the initial guess which will be used in reconstruction and will be updated to the current reconstructing result 
                shutil.copy(os.path.join(recon_path, f_initial_guess + '.h5'), os.path.join(recon_path, f_recon_grid + '.h5'))

            else:
                X = None

        comm.Barrier()
            
        if rank == 0:
            XRF_loss_whole_obj = tc.zeros(n_epochs*n_theta)
            XRT_loss_whole_obj = tc.zeros(n_epochs*n_theta)
            loss_whole_obj = tc.zeros(n_epochs*n_theta)
            
            with open(os.path.join(recon_path, f_recon_parameters), "w") as recon_params:
                recon_params.write("starting_epoch = 0\n")
                recon_params.write("n_epochs = %d\n" %n_epochs)
                recon_params.write("n_ranks = %d\n" %n_ranks)
                recon_params.write("element_line:\n" + str(element_lines_roi)+"\n") # element_lines_roi is effectively a 2D array => each array row gets its own line
                recon_params.write("proj_downsample_factor = %d\n" %downsample_factor)
                recon_params.write("recon_upsample_factor = %d\n" %upsample_factor)
                recon_params.write("noise_model = %s\n" %noise_model)
                recon_params.write("b1 = %.9f\n" %b1)
                recon_params.write("b2 = %.9f\n" %b2)
                recon_params.write("learning rate = %f\n" %lr)
                recon_params.write("theta_st = %.2f\n" %theta_ls[0])
                recon_params.write("theta_end = %.2f\n" %theta_ls[-1])
                recon_params.write("n_theta = %d\n" %n_theta)
                recon_params.write("sample_size_n = %d\n" %sample_size_n)
                recon_params.write("sample_height_n = %d\n" %sample_height_n)
                recon_params.write("sample_size_cm = %.2f\n" %sample_size_cm)

                if det_window_element is not None:
                    recon_params.write("detector_window_element = %s\n" %det_window_element)
                    recon_params.write("detector_window_thickness_um = %.2f\n" %det_window_thickness_um)
                
                else:
                    recon_params.write("detector_window_element = windowless\n")
                    recon_params.write("detector_window_thickness_um = n/a\n")

                recon_params.write("probe_energy_keV = %.2f\n" %probe_energy_keV[0])
                recon_params.write("incident_probe_cts = %.2e\n" %probe_cts)             
                
                if not manual_det_area:
                    recon_params.write("det_dia_cm = %.2f\n" %det_dia_cm)
                
                else:
                    recon_params.write("det_dia_cm = n/a\n")
                
                if not manual_det_coord:
                    recon_params.write("det_from_sample_cm = %.2f\n" %det_from_sample_cm)
                    recon_params.write("det_ds_spacing_cm = %.2f\n" %det_ds_spacing_cm)
                
                else:
                    recon_params.write("det_from_sample_cm = n/a\n")
                    recon_params.write("det_ds_spacing_cm = n/a\n")
        
        comm.Barrier()          
        
        for epoch in range(n_epochs):
            t0_epoch = time.perf_counter()
            
            if rank == 0:
                rand_idx = tc.randperm(n_theta)
                theta_ls_rand = theta_ls[rand_idx]  
            
            else:
                rand_idx = tc.ones(n_theta)
                theta_ls_rand = tc.ones(n_theta)

            comm.Barrier() 
            rand_idx = comm.bcast(rand_idx, root = 0).to(dev) 
            theta_ls_rand = comm.bcast(theta_ls_rand, root = 0).to(dev)         
            comm.Barrier() 
            
            stdout_options = {'root': 0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': True}
            timestr = str(datetime.datetime.today())     
            print_flush_root(rank, val = f"epoch: {epoch}, time: {timestr}", output_file = '', **stdout_options)
 
            for idx, theta in enumerate(theta_ls_rand):
                this_theta_idx = rand_idx[idx] 
                                          
                # The updated X read by all ranks only at each new obj. angle
                # Because updating the remaining slices in the current obj. angle doesn't require the info of the previous updated slices.   
                ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
                with h5py.File(os.path.join(recon_path, f_recon_grid +'.h5'), "r") as s:
                    X = s["sample/densities"][...].astype(np.float32)
                    X = tc.from_numpy(X).to(dev) #dev
                    
                if selfAb == True:               
                    X_ap_rot = rotate(X, theta, dev) #dev
                    lac = X_ap_rot.view(n_element, 1, 1, n_voxel)*FL_line_attCS_ls.view(n_element, n_lines, 1, 1) #dev
                    lac = lac.expand(-1, -1, n_voxel_minibatch, -1).float() #dev
                
                else:
                    lac = 0.
                
                if rank == 0:
                    XRF_loss_n_batch = tc.zeros(n_batch)
                    XRT_loss_n_batch = tc.zeros(n_batch)
                    total_loss_n_batch = tc.zeros(n_batch)
                    
                for m in range(n_batch):                    
                    minibatch_ls = n_ranks*m + minibatch_ls_0  #dev, e.g. [5,6,7,8]
                    p = minibatch_ls[rank]
                    
                    if selfAb == True:
                        P_minibatch = tc.from_numpy(P_handle['P_array'][:, :, p*dia_len_n*minibatch_size*sample_size_n: \
                                                                        (p + 1)*dia_len_n*minibatch_size*sample_size_n]).to(dev)
                        n_det = P_minibatch.shape[0] 
                    
                    else:
                        P_minibatch = 0
                        n_det = 0
                    
#                     stdout_options = {'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
#                     print_flush_all(rank, val=p * dia_len_n * minibatch_size * sample_size_n, output_file=f'P_start_idx_{rank}.csv', **stdout_options)
#                     print_flush_all(rank, val=(p+1) * dia_len_n * minibatch_size * sample_size_n, output_file=f'P_end_idx_{rank}.csv', **stdout_options)
                  
#                     stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
#                     print_flush_root(rank, val=minibatch_ls, output_file='minibatch_ls.csv', **stdout_options)
                    
                    ## Load us_ic as the incoming probe count in this minibatch
                    model = PPM(dev, selfAb, lac, X, p, n_element, n_lines, FL_line_attCS_ls, FL_line_det_attCS_ls,
                                detected_fl_unit_concentration, n_line_group_each_element,
                                sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                                probe_energy_keV, probe_cts, probe_att, probe_attCS_ls,
                                theta, signal_attenuation_factor,
                                n_det, P_minibatch, det_dia_cm, det_from_sample_cm, det_solid_angle_ratio)
                    
                    optimizer = tc.optim.Adam(model.parameters(), lr = lr)              
                                    
                    ## load true data, y1: XRF_data, y2: XRT data
                    #dev #Take all lines_roi, this_theta_idx, and strips in this minibatch
                    y1_hat, y2_hat = model()
                    
                    XRF_loss = loss_fn(y1_hat, y1_true[:, this_theta_idx, minibatch_size*p:minibatch_size*(p + 1)])
                    XRT_loss = loss_fn(y2_hat, b2*y2_true[this_theta_idx, minibatch_size*p:minibatch_size*(p + 1)])
                    loss = XRF_loss + b1*XRT_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                         
                    updated_minibatch = model.xp.detach().cpu()
                    updated_minibatch = tc.clamp(updated_minibatch, 0, float('inf'))
                    
                    comm.Barrier()
                       
                    XRF_loss = XRF_loss.detach().item()
                    XRF_loss_sum = comm.reduce(XRF_loss, op = MPI.SUM, root = 0)
                    
                    XRT_loss = XRT_loss.detach().item() 
                    XRT_loss_sum = comm.reduce(XRT_loss, op = MPI.SUM, root = 0)                    
                                       
                    loss = loss.detach().item()           
                    loss_sum = comm.reduce(loss, op = MPI.SUM, root = 0)
                    comm.Barrier()                    
                    
                    with h5py.File(os.path.join(recon_path, f_recon_grid +'.h5'), 'r+', driver = 'mpio', comm = comm) as s:
                        s["sample/densities"][:, minibatch_size*p//sample_size_n:minibatch_size*(p + 1)//sample_size_n, :, :] = updated_minibatch.numpy()
                    
                    comm.Barrier()
                    
                    if rank == 0: 
                        XRF_loss_n_batch[m] = XRF_loss_sum/n_ranks
                        XRT_loss_n_batch[m] = XRT_loss_sum/n_ranks
                        total_loss_n_batch[m] = loss_sum/n_ranks
               
                    del model 
                    
                    tc.cuda.empty_cache()

                if rank == 0:
                    loss_whole_obj[n_theta*epoch + idx] = tc.mean(total_loss_n_batch)
                    XRF_loss_whole_obj[n_theta*epoch + idx] = tc.mean(XRF_loss_n_batch)
                    XRT_loss_whole_obj[n_theta*epoch + idx] = tc.mean(XRT_loss_n_batch)
                    
                comm.Barrier()                    
                
                del lac

            stdout_options = {'root': 0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
            per_epoch_time = time.perf_counter() - t0_epoch
            
            print_flush_root(rank, val = per_epoch_time, output_file = f'per_epoch_time_mb_size_{minibatch_size}.csv', **stdout_options)
            
            comm.Barrier()
 
            if rank == 0:
                with h5py.File(os.path.join(recon_path, f_recon_grid + '.h5'), "r") as s:
                    X_cpu = s["sample/densities"][...].astype(np.float32)
                
            if rank == 0 and epoch != 0:
                epsilon = np.mean((X_cpu - X_previous)**2)
                
                print_flush_root(rank, val = epsilon, output_file = f'model_change_mse_epoch.csv', **stdout_options)
                
                if epsilon < 1e-12:             
                    if rank == 0:
                        with h5py.File(os.path.join(recon_path, f_recon_grid + "_" + str(epoch)+"_ending_condition" +'.h5'), "w") as s:
                            sample = s.create_group("sample")

                            sample_v = sample.create_dataset("densities", shape = (n_element, sample_height_n, sample_size_n, sample_size_n), dtype="f4")
                            sample_e = sample.create_dataset("elements", shape = (n_element,), dtype = 'S5')
                            
                            s["sample/densities"][...] = X_cpu
                            # s["sample/elements"][...] = np.array(list(this_aN_dic.keys())).astype('S5')
                            s["sample/elements"][...] = np.array(elements_xrf).astype('S5')
                        
                        dxchange.write_tiff(X_cpu, os.path.join(recon_path, f_recon_grid) + "_" + str(epoch) + "_ending_condition", dtype = 'float32', overwrite = True)                     
                    
                    break
                
                else:
                    pass
            
            else:
                pass
            
            comm.Barrier()
            
            if rank == 0:
                X_previous = X_cpu
            
            comm.Barrier()
            
            if rank == 0 and ((epoch + 1) % save_every_n_epochs == 0 and (epoch + 1)//save_every_n_epochs != 0 or epoch + 1 == n_epochs):
                with h5py.File(os.path.join(checkpoint_path, f_recon_grid + "_" + str(epoch) + '.h5'), "w") as s:
                    sample = s.create_group("sample")
                                        
                    sample_v = sample.create_dataset("densities", shape = (n_element, sample_height_n, sample_size_n, sample_size_n), dtype = "f4")
                    sample_e = sample.create_dataset("elements", shape = (n_element,), dtype = 'S5')
                    
                    s["sample/densities"][...] = X_cpu
                    s["sample/elements"][...] = np.array(list(this_aN_dic.keys())).astype('S5')
#                 dxchange.write_tiff(X_cpu, os.path.join(recon_path, f_recon_grid)+"_"+str(epoch), dtype='float32', overwrite=True)  
                
        ## It's important to close the hdf5 file handle in the end of the reconstruction.
        P_handle.close()
        # y1_true_handle.close()
        # y2_true_handle.close()
        
        comm.Barrier()
        
        if rank == 0:            
            fig6 = plt.figure(figsize=(10,15))
            gs6 = gridspec.GridSpec(nrows=3, ncols=1, width_ratios=[1])

            fig6_ax1 = fig6.add_subplot(gs6[0,0])
            fig6_ax1.plot(loss_whole_obj.numpy())
            fig6_ax1.set_xlabel('theta_iteration')
            fig6_ax1.set_ylabel('loss')
            fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

            fig6_ax2 = fig6.add_subplot(gs6[1,0])
            fig6_ax2.plot(XRF_loss_whole_obj.numpy())
            fig6_ax2.set_xlabel('theta_iteration')
            fig6_ax2.set_ylabel('XRF loss')
            fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

            fig6_ax3 = fig6.add_subplot(gs6[2,0])
            fig6_ax3.plot(XRT_loss_whole_obj.numpy())
            fig6_ax3.set_xlabel('theta_iteration')
            fig6_ax3.set_ylabel('XRT loss')
            fig6_ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            
            plt.savefig(os.path.join(recon_path, 'loss_signal.pdf'))
            
            np.save(os.path.join(recon_path, 'XRF_loss_signal.npy'), XRF_loss_whole_obj.numpy())
            np.save(os.path.join(recon_path, 'XRT_loss_signal.npy'), XRT_loss_whole_obj.numpy())
            np.save(os.path.join(recon_path, 'loss_signal.npy'), loss_whole_obj.numpy())
            
        comm.Barrier()
        
    if cont_from_check_point == True:
        if rank == 0:           
            with h5py.File(os.path.join(recon_path, f_recon_grid + ".h5"), "r") as s:
                X_init = s["sample/densities"][...].astype(np.float32)

                if upsample_factor > 1:
                    msg = f'Upsampling initial guess of reconstruction slices by factor of {upsample_factor}...'
        
                    print_flush_root(rank, msg, save_stdout = False, print_terminal = True)
                   
                    X = upsample_recon_data(X_init, upsample_factor)
                
                elif upsample_factor == 1:
                    X = X_init
                
                else:
                    msg = f"Error: 'upsample_factor' must be a positive integer. Exiting program..."

                    print_flush_root(rank, msg, save_stdout = False, print_terminal = True)

                    comm.abort(1)

                if (X.shape[0], X.shape[1]) != (y1_true.shape[2], y1_true.shape[3]): # If the number of slices and scan positions are not identical after
                                                                                         # downsampling projection data and upsampling initial reconstruction data, throw error and terminate program
                        
                    msg = 'Error: Number of slices and scan positions do not match between upsampled initial reconstruction guess and downsampled projection data. Exiting program...'
                        
                    print_flush_root(rank, msg, save_stdout = False, print_terminal = True)

                    comm.abort(1)

                X = tc.from_numpy(X)
            
        else:
            X = None
                   
        if rank == 0:
            XRF_loss_whole_obj = tc.from_numpy(np.load(os.path.join(recon_path, 'XRF_loss_signal.npy')).astype(np.float32))
            XRT_loss_whole_obj = tc.from_numpy(np.load(os.path.join(recon_path, 'XRT_loss_signal.npy')).astype(np.float32))
            loss_whole_obj = tc.from_numpy(np.load(os.path.join(recon_path, 'loss_signal.npy')).astype(np.float32))
            
            XRF_loss_whole_obj_cont = tc.zeros(n_epochs*n_theta)
            XRT_loss_whole_obj_cont = tc.zeros(n_epochs*n_theta)
            loss_whole_obj_cont = tc.zeros(n_epochs*n_theta)
            
            with open(os.path.join(recon_path, f_recon_parameters), "r") as recon_params:
                params_list = []
                
                for line in recon_params.readlines():
                    params_list.append(line.rstrip("\n"))
                
                n_ending = len(params_list)

            with open(os.path.join(recon_path, f_recon_parameters), "a") as recon_params:
                n_start_last = n_ending - 23 - len(element_lines_roi)

                previous_starting_epoch = int(params_list[n_start_last][(params_list[n_start_last].find("=") + 1):])
                previous_n_epoch = int(params_list[n_start_last + 1][(params_list[n_start_last + 1].find("=") + 1):])
                starting_epoch = previous_starting_epoch + previous_n_epoch
                recon_params.write("\n")
                recon_params.write("###########################################\n")
                recon_params.write("starting_epoch = %d\n" %starting_epoch)
                recon_params.write("n_epochs = %d\n" %n_epochs)
                recon_params.write("n_ranks = %d\n" %n_ranks)
                recon_params.write("element_line:\n" + str(element_lines_roi)+"\n") # element_lines_roi is effectively a 2D array => each array row gets its own line
                recon_params.write("proj_downsample_factor = %d\n" %downsample_factor)
                recon_params.write("recon_upsample_factor = %d\n" %upsample_factor)
                recon_params.write("noise_model = %s\n" %noise_model)
                recon_params.write("b1 = %.9f\n" %b1)
                recon_params.write("b2 = %.9f\n" %b2)
                recon_params.write("learning rate = %f\n" %lr)
                recon_params.write("theta_st = %.2f\n" %theta_ls[0])
                recon_params.write("theta_end = %.2f\n" %theta_ls[-1])
                recon_params.write("n_theta = %d\n" %n_theta)
                recon_params.write("sample_size_n = %d\n" %sample_size_n)
                recon_params.write("sample_height_n = %d\n" %sample_height_n)
                recon_params.write("sample_size_cm = %.2f\n" %sample_size_cm)

                if det_window_element is not None:
                    recon_params.write("detector_window_element = %s\n" %det_window_element)
                    recon_params.write("detector_window_thickness_um = %.2f\n" %det_window_thickness_um)
                
                else:
                    recon_params.write("detector_window_element = windowless\n")
                    recon_params.write("detector_window_thickness_um = 0\n")

                recon_params.write("probe_energy_keV = %.2f\n" %probe_energy_keV[0])
                recon_params.write("incident_probe_cts = %.2e\n" %probe_cts)             
                
                if not manual_det_area:
                    recon_params.write("det_dia_cm = %.2f\n" %det_dia_cm)
                
                else:
                    recon_params.write("det_dia_cm = n/a\n")
                
                if not manual_det_coord:
                    recon_params.write("det_from_sample_cm = %.2f\n" %det_from_sample_cm)
                    recon_params.write("det_ds_spacing_cm = %.2f\n" %det_ds_spacing_cm)
                
                else:
                    recon_params.write("det_from_sample_cm = n/a\n")
                    recon_params.write("det_ds_spacing_cm = n/a\n")
        
        comm.Barrier()  
       
        for epoch in range(n_epochs):
            t0_epoch = time.perf_counter()
            
            if rank == 0:
                rand_idx = tc.randperm(n_theta)
                theta_ls_rand = theta_ls[rand_idx]  
            
            else:
                rand_idx = tc.ones(n_theta)
                theta_ls_rand = tc.ones(n_theta)

            comm.Barrier() 
            
            rand_idx = comm.bcast(rand_idx, root = 0).to(dev) 
            theta_ls_rand = comm.bcast(theta_ls_rand, root = 0).to(dev)         
            
            comm.Barrier()     
             
            stdout_options = {'root': 0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': True}
            timestr = str(datetime.datetime.today())
            
            print_flush_root(rank, f"epoch: {epoch}, time: {timestr}", output_file = '', **stdout_options)
            
            for idx, theta in enumerate(theta_ls_rand):
                this_theta_idx = rand_idx[idx]

                # The updated X read by all ranks only at each new obj. angle
                # Because updating the remaining slices in the current obj. angle doesn't require the info of the previous updated slices.   
                ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
                with h5py.File(os.path.join(recon_path, f_recon_grid + '.h5'), "r") as s:
                    X = s["sample/densities"][...].astype(np.float32)
                    X = tc.from_numpy(X).to(dev) #dev
                    
                ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
                if selfAb == True:
                    X_ap_rot = rotate(X, theta, dev) #dev
                    lac = X_ap_rot.view(n_element, 1, 1, n_voxel)*FL_line_attCS_ls.view(n_element, n_lines, 1, 1) #dev
                    lac = lac.expand(-1, -1, n_voxel_minibatch, -1).float() #dev
                
                else:
                    lac = 0.
                
                if rank == 0:
                    XRF_loss_n_batch = tc.zeros(n_batch)
                    XRT_loss_n_batch = tc.zeros(n_batch)
                    total_loss_n_batch = tc.zeros(n_batch)
                
                for m in range(n_batch):
                    minibatch_ls = n_ranks*m + minibatch_ls_0  #dev
                    p = minibatch_ls[rank]

                    if selfAb == True:
                        P_minibatch = tc.from_numpy(P_handle['P_array'][:, :, p*dia_len_n*minibatch_size*sample_size_n: \
                                                                        (p + 1)*dia_len_n*minibatch_size*sample_size_n]).to(dev)
                        n_det = P_minibatch.shape[0] 
                    
                    else:
                        P_minibatch = 0
                        n_det = 0                    
                                       
                    model = PPM(dev, selfAb, lac, X, p, n_element, n_lines, FL_line_attCS_ls, FL_line_det_attCS_ls,
                                 detected_fl_unit_concentration, n_line_group_each_element,
                                 sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                                 probe_energy_keV, probe_cts, probe_att, probe_attCS_ls,
                                 theta, signal_attenuation_factor,
                                 n_det, P_minibatch, det_dia_cm, det_from_sample_cm, det_solid_angle_ratio)

                    optimizer = tc.optim.Adam(model.parameters(), lr=lr)   
            
                    ## load true data, y1: XRF_data, y2: XRT data
                    #dev #Take all lines_roi, this_theta_idx, and strips in this minibatch                    
                    y1_hat, y2_hat = model()
                    XRF_loss = loss_fn(y1_hat, y1_true[:, this_theta_idx, minibatch_size*p:minibatch_size*(p + 1)])
                    XRT_loss = loss_fn(y2_hat, b2*y2_true[this_theta_idx, minibatch_size*p:minibatch_size*(p + 1)])
                    loss = XRF_loss + b1 * XRT_loss
            
                    optimizer.zero_grad()
                    loss.backward()             
                    optimizer.step()
                
                    updated_minibatch = model.xp.detach().cpu()
                    updated_minibatch = tc.clamp(updated_minibatch, 0, float('inf'))
                    
                    comm.Barrier()
                    
                    XRF_loss = XRF_loss.detach().item() 
                    XRF_loss_sum = comm.reduce(XRF_loss, op = MPI.SUM, root = 0)
                    
                    XRT_loss = XRT_loss.detach().item() 
                    XRT_loss_sum = comm.reduce(XRT_loss, op = MPI.SUM, root = 0)                    
                                       
                    loss = loss.detach().item()           
                    loss_sum = comm.reduce(loss, op = MPI.SUM, root = 0)
                    
                    comm.Barrier()                  
                
                    with h5py.File(os.path.join(recon_path, f_recon_grid +'.h5'), 'r+', driver = 'mpio', comm = comm) as s:
                        s["sample/densities"][:, minibatch_size*p//sample_size_n:minibatch_size*(p + 1)//sample_size_n, :, :] = updated_minibatch.numpy()
                        
                    if rank == 0:
                        XRF_loss_n_batch[m] = XRF_loss_sum/n_ranks
                        XRT_loss_n_batch[m] = XRT_loss_sum/n_ranks
                        total_loss_n_batch[m] = loss_sum/n_ranks                        
                        
                        # Note that we need to detach the voxels in the updated_batch of the current iteration.
                        # Otherwise Pytorch will keep calculating the gradient of the updated_batch of the current iteration in the NEXT iteration

                    del model 
             
                if rank == 0:  
                    loss_whole_obj_cont[n_theta*epoch + idx] = tc.mean(total_loss_n_batch)
                    XRF_loss_whole_obj_cont[n_theta*epoch + idx] = tc.mean(XRF_loss_n_batch)
                    XRT_loss_whole_obj_cont[n_theta*epoch + idx] = tc.mean(XRT_loss_n_batch)

                comm.Barrier()                    
                
                del lac
#                 tc.cuda.empty_cache()
                         
            stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
            per_epoch_time = time.perf_counter() - t0_epoch
            
            print_flush_root(rank, val = per_epoch_time, output_file = f'per_epoch_time_mb_size_{minibatch_size}.csv', **stdout_options)
            
            comm.Barrier()

            if rank == 0:
                with h5py.File(os.path.join(recon_path, f_recon_grid + '.h5'), "r") as s:
                    X_cpu = s["sample/densities"][...].astype(np.float32)
                    
            if rank == 0 and epoch != 0:
                epsilon = np.mean((X_cpu - X_previous)**2)
                
                print_flush_root(rank, val = epsilon, output_file = f'model_change_mse_epoch.csv', **stdout_options)
                
                if epsilon < 1e-12:
                    if rank == 0:
                        with h5py.File(os.path.join(recon_path, f_recon_grid + "_" + str(epoch) + "_ending_condition" + '.h5'), "w") as s:
                            sample = s.create_group("sample")
                            
                            sample_v = sample.create_dataset("densities", shape = (n_element, sample_height_n, sample_size_n, sample_size_n), dtype = "f4")
                            sample_e = sample.create_dataset("elements", shape = (n_element,), dtype = 'S5')
                            
                            s["sample/densities"][...] = X_cpu
                            s["sample/elements"][...] = np.array(list(this_aN_dic.keys())).astype('S5')
                        
                        dxchange.write_tiff(X_cpu, os.path.join(recon_path, f_recon_grid) + "_" + str(epoch) + "_ending_condition", dtype = 'float32', overwrite = True)                         
                    
                    break
                
                else:
                    pass
            
            else:
                pass
            
            comm.Barrier()
            
            if rank == 0:
                X_previous = X_cpu            
            
            comm.Barrier()            
            
            checkpoint_path = os.path.join(recon_path, "checkpoint")
            
            if rank == 0 and ((epoch + 1) % save_every_n_epochs == 0 and (epoch + 1)//save_every_n_epochs != 0 or epoch + 1 == n_epochs):
                with h5py.File(os.path.join(checkpoint_path, f_recon_grid + "_" + str(starting_epoch + epoch) + '.h5'), "w") as s:
                    sample = s.create_group("sample")

                    sample_v = sample.create_dataset("densities", shape = (n_element, sample_height_n, sample_size_n, sample_size_n), dtype = "f4")
                    sample_e = sample.create_dataset("elements", shape = (n_element,), dtype = 'S5')
                    
                    s["sample/densities"][...] = X_cpu
                    s["sample/elements"][...] = np.array(list(this_aN_dic.keys())).astype('S5')
#                 dxchange.write_tiff(X_cpu, os.path.join(recon_path, f_recon_grid)+"_"+str(epoch), dtype='float32', overwrite=True)     
            
        ## It's important to close the hdf5 file handle in the end of the reconstruction.
        P_handle.close()  
        # y1_true_handle.close()
        # y2_true_handle.close()
        
        comm.Barrier()
        
        if rank == 0:                           
            loss_whole_obj = tc.cat((loss_whole_obj, loss_whole_obj_cont))
            XRF_loss_whole_obj = tc.cat((XRF_loss_whole_obj, XRF_loss_whole_obj_cont))
            XRT_loss_whole_obj = tc.cat((XRT_loss_whole_obj, XRT_loss_whole_obj_cont))
 
            fig6 = plt.figure(figsize = (10, 15))
            gs6 = gridspec.GridSpec(nrows = 3, ncols = 1, width_ratios = [1])

            fig6_ax1 = fig6.add_subplot(gs6[0, 0])
            fig6_ax1.plot(loss_whole_obj.numpy())
            fig6_ax1.set_xlabel('theta_iteration')
            fig6_ax1.set_ylabel('loss')
            fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

            fig6_ax2 = fig6.add_subplot(gs6[1, 0])
            fig6_ax2.plot(XRF_loss_whole_obj.numpy())
            fig6_ax2.set_xlabel('theta_iteration')
            fig6_ax2.set_ylabel('XRF loss')
            fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

            fig6_ax3 = fig6.add_subplot(gs6[2, 0])
            fig6_ax3.plot(XRT_loss_whole_obj.numpy())
            fig6_ax3.set_xlabel('theta_iteration')
            fig6_ax3.set_ylabel('XRT loss')
            fig6_ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            
            plt.savefig(os.path.join(recon_path, 'loss_signal.pdf'))
            
            np.save(os.path.join(recon_path, 'XRF_loss_signal.npy'), XRF_loss_whole_obj.numpy())
            np.save(os.path.join(recon_path, 'XRT_loss_signal.npy'), XRT_loss_whole_obj.numpy())
            np.save(os.path.join(recon_path, 'loss_signal.npy'), loss_whole_obj.numpy())
             
