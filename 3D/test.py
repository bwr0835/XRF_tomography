import numpy as np, h5py

from matplotlib import pyplot as plt

mda_array = ['0117', '0118', '0119', '0120', '0121', '0122', '0123', '0124', '0125', '0126']

for mda, MDA in enumerate(mda_array):
    try:
        h5 = h5py.File(f'/raid/users/roter/Jacobsen/img.dat/2xfm_{MDA}.mda.h5')

        extra_pv_names = h5['MAPS/Scan/Extra_PVs/Names'][()]
        extra_pv_values = h5['MAPS/Scan/Extra_PVs/Values'][()]

        z_idx = np.ndarray.item(np.where(extra_pv_names == b'2xfm:m26.VAL')[0])

        z = extra_pv_values[z_idx]

        print(f'MDA {MDA}: z = {z}')

    except:
        print('None')

 



