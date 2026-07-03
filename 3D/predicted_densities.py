import numpy as np, \
       predicted_density_line_info as pdli, \
       h5py, \
       json

from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

def extract_h5_data(dir_path, mda, synchrotron, n_det, aux_mda = None, param_names = None):
    if synchrotron == 'aps':
        if n_det > 1 and type(n_det) == int:
            raw_cts = np.zeros((n_det, n_energy_bins, n_slices, n_columns))
            energy_bins_kev = np.zeros((n_det, n_energy_bins))
            fwhm_offset = np.zeros(n_det)
            fwhm_fanoprime = np.zeros(n_det)

            for det in range(n_det):
                filename = f'{dir_path}/2xfm_{mda}.mda.h5{det}'

                with h5py.File(filename, 'r') as f:
                    if det == 0:
                        xrf_analyzed = f['MAPS/XRF_Analyzed']
                        fit_param_override = f['MAPS/Fit_Parameters_Override']
                        
                        elements = list(xrf_analyzed['Channel_Names'].asstr()[:])
                        fit_param_names = list(fit_param_override['Names'].asstr()[:])

                    raw_cts[det] = xrf_analyzed['mca_arr'][()][:, :, :, :n_columns]
                    energy_bins_kev[det] = xrf_analyzed['Energy'][()]

                    fit_param_values = fit_param_override['Values'][()]

                    fwhm_offset[det] = fit_param_values[fit_param_names.index('FWHM_OFFSET')]
                    fwhm_fanoprime[det] = fit_param_values[fit_param_names.index('FWHM_FANOPRIME')]
    
    elif synchrotron == 'nsls-ii':
        if n_det == 1:
            filename = f'{dir_path}/xrf/scan2D_{mda}.h5'

            with h5py.File(filename, 'r') as f:
                detsum = f['xrfmap/detsum']

                elements = list(detsum['xrf_fit_name'].asstr()[:])
                raw_cts = np.transpose(detsum['counts'][()], (1, 2, 0))
            
            aux_filename = f'{dir_path}/xrf/scan2D_{aux_mda}_sum_out.txt'
            _dict = {}

            i = 0

            param_names_init = ['e_linear', 'e_offset', 'e_quadratic', 'fwhm_fanoprime', 'fwhm_offset']
            
            with open(aux_filename, 'r') as f:
                param_names = param_names_init.copy()

                for line in f:
                    for param_name in param_names:
                        if param_name in line:
                            _dict[param_name] = float(line.split(':')[1].strip().split('==')[0])
                    
                        param_names.remove(param_name)
                    
                    if not param_names:
                        break
            
            e_linear = _dict['e_linear']
            e_offset = _dict['e_offset']
            e_quadratic = _dict['e_quadratic']
            fwhm_fanoprime = _dict['fwhm_fanoprime']
            fwhm_offset = _dict['fwhm_offset']

            energy_channel_idx_float = np.arange(4096, dtype = float)

            energy_bins_kev = e_offset + e_linear*energy_channel_idx_float + e_quadratic*energy_channel_idx_float**2

    return elements, raw_cts, energy_bins_kev, np.array([fwhm_offset]), np.array([fwhm_fanoprime])

synchrotron = 'aps'
dir_path = '/raid/users/roter/Jacobsen/img.dat'
mda = '0097'
aux_mda = None
n_det = 3
param_names = None

n_columns = 599
n_slices = 301
n_energy_bins = 2048

max_photon_count = 100
inc_flux_norm_factor = 0.8195217622400474

E_eh_pair_keV = 0.00365

elements_of_interest = ['Si', 'Ti', 'Mn', 'Cr', 'Fe', 'Ba_L']

raw_cts = np.zeros((n_det, n_energy_bins, n_slices, n_columns))
energy_bins_kev = np.zeros((n_det, n_energy_bins))
fwhm_offset = np.zeros(n_det)
fwhm_fanoprime = np.zeros(n_det)

total_photons = np.zeros((n_det, n_slices*n_columns))
total_photons_aggregate_list = []

elements, raw_cts, energy_bins_kev, fwhm_offset, fwhm_fanoprime = extract_h5_data(dir_path, mda, synchrotron, n_det, aux_mda, param_names)

sig_to_fwhm_factor = 2*np.sqrt(2*np.log(2))

for idx, element in enumerate(elements_of_interest):
    energy_centroids_element_kev = pdli.centroids[elements_of_interest[idx]]*1e-3
    
    element_idx = pdli.elements_of_interest.index(element)

    # Ignore overlapping peaks for now

    for det in range(n_det):
        raw_cts_element = raw_cts[det, element_idx].ravel()
        
        sigma = np.sqrt((fwhm_offset[det]/sig_to_fwhm_factor)**2 + E_eh_pair_keV*fwhm_fanoprime[det]*energy_centroids_element_kev)
        
        dE_fwhm = sigma*sig_to_fwhm_factor
        
        energy_bins_kev_det = energy_bins_kev[det]

        E_window = np.vstack(energy_centroids_element_kev - dE_fwhm/2, energy_centroids_element_kev + dE_fwhm/2)

        energy_window_mask = energy_bins_kev_det >= E_window[0, 0] and energy_bins_kev_det <= E_window[1, 0]

        for i in range(1, len(energy_centroids_element_kev)):
            energy_window_mask = energy_window_mask | ((energy_bins_kev_det >= E_window[0, i]) & (energy_bins_kev_det <= E_window[1, i]))

        energy_idx = np.argwhere(energy_window_mask).squeeze()

        selected_energies = raw_cts_element[energy_idx]

        total_photons[det] = selected_energies.sum(axis = 0)
    
    total_photons_aggregate = inc_flux_norm_factor*total_photons.sum(axis = 0)

    total_photons_aggregate = total_photons_aggregate[total_photons_aggregate <= max_photon_count]
    
    total_photons_aggregate_list.append(total_photons_aggregate.copy())

total_photons_aggregate_array = np.array(total_photons_aggregate_list, dtype = object)

# np.save('total_photons_aggregate_array.npy', total_photons_aggregate_array)

fig, axs = plt.subplots(2, 3)

photon_max = 0

for idx, ax in enumerate(fig.axes):
    bins = np.arange(total_photons_aggregate_array[idx].min() - 0.5, total_photons_aggregate_array[idx].max() + 0.5)

    _hist, _, _ = ax.hist(total_photons_aggregate_array[idx], bins = bins//100, edgecolor = 'black', facecolor = 'blue')

    if total_photons_aggregate_array[idx].max() > photon_max:
        photon_max = total_photons_aggregate_array[idx].max()
    
    ax.set_title(r'{0}'.format(elements_of_interest[idx]), fontsize = 16)

    ax.minorticks_on()
    ax.tick_params(axis = 'y', which = 'both', right = True, labelright = False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 14)

    ax.set_xlabel(r'$N_{\mathrm{fluor}}$ (photons)', fontsize = 14)
    ax.set_ylabel(r'Counts', fontsize = 14)

fig.suptitle(r'Predicted densities', fontsize = 16)
fig.tight_layout()

plt.show()