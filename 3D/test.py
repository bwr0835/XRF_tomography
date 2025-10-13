import numpy as np, h5py, file_util, xraylib_np as xrl

from matplotlib import pyplot as plt

# mda_array = ['0116', '0117', '0118', '0119', '0120', '0121', '0122', '0123', '0124', '0125', '0126']
# mda_array = ['0124', '0125', '0126']

# for mda, MDA in enumerate(mda_array):
#     try:
#         h5 = h5py.File(f'/raid/users/roter/Jacobsen/img.dat/2xfm_{MDA}.mda.h5')

#         extra_pv_names = h5['MAPS/Scan/Extra_PVs/Names'][()]
#         extra_pv_values = h5['MAPS/Scan/Extra_PVs/Values'][()]
        
#         scalers_names = h5['MAPS/Scalers/Names'][()]
#         scalers_values = h5['MAPS/Scalers/Values'][()]

#         z_idx = np.ndarray.item(np.where(extra_pv_names == b'2xfm:m26.VAL')[0])
#         us_ic_idx = np.ndarray.item(np.where(scalers_names == b'US_IC')[0])

#         z = extra_pv_values[z_idx]
#         us_ic = scalers_values[us_ic_idx].astype(float)

#         print(f'MDA {MDA}: z = {z}; US_IC Total = {us_ic.sum()}')

#     except:
#         print('None')

# sid = 235324
# sid_end = 235675

# filename_array = []

# # Commented out SIDs in sid_array are for if they don't have Si maps (no idea how they didn't get recorded)

# sid_array = [235324,
#              235327,
#              235330,
#              235333,
#              235336,
#              235339,
#              235342,
#              235345,
#              235348,
#              235351,
#              235354,
#              235358,
#              235364,
#              235374,
#              235377,
#              235386,
#              235389,
#              235392,
#              235395,
#              235398,
#              235401,
#              235404,
#              235407,
#              235410,
#              235413,
#              235416,
#              235419,
#              235422,
#              235425,
#              235428,
#              235431,
#              235434,
#              235437,
#              235440,
#              235443,
#              235446,
#              235449,
#              235452,
#              235455,
#              235458,
#              235461,
#              235464,
#              235467,
#              235470,
#              235473,
#              235476,
#              235479,
#              235482,
#              235485,
#              235488,
#              235491,
#              235494,
#              235497,
#              235500,
#              235503,
#              235506,
#              235509,
#              235512,
#              235515,
#              235518,
#              235546,
#              235549,
#              235552,
#              235555,
#              235558,
#              235561,
#              235564,
#              235567,
#              235570,
#              235573,
#              235576,
#              235579,
#              235582,
#              235585,
#              235588,
#              235592,
#             #  235595,
#              235598,
#              235601,
#              235604,
#              235607,
#              235610,
#              235613,
#              235616,
#              235619,
#              235622,
#              235625,
#              235628,
#              235631,
#             #  235634,
#              235637,
#              235640,
#              235643,
#              235646,
#              235649,
#              235652,
#              235655,
#              235658,
#             #  235661,
#              235664,
#              235667,
#              235670,
#              235673]

# while sid < sid_end:
#     filename = f'/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_{sid}.h5'

#     try:
#         elements_string, _, theta, nx, ny, _, _ = h5_util.extract_h5_xrf_data(filename, synchrotron = 'nsls-ii')

#         if 'Si_K' not in elements_string:
#             print(f'SID: {sid}; n_elements = {len(elements_string)}; {nx} x {ny} (theta = {theta}); No Si')
        
#         else:
#             print(f'SID: {sid}; n_elements = {len(elements_string)}; {nx} x {ny} (theta = {theta})')

#     except:
#         pass

#     sid += 1

# for sid in sid_array:
#     filename = f'/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_{sid}.h5'

#     filename_array.append(filename)

# output_filename = '/home/bwr0835/hxn_aggregate_xrf.h5'

# h5_util.create_aggregate_xrf_h5(filename_array, output_filename, synchrotron = "nsls-ii")
