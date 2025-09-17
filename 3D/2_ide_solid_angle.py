import numpy as np, sys

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cf
from lmfit import Model

import h5_util

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def i_obj(x, A, B):
    return A/(x + B)**2

def round_correct(num, ndec): # CORRECTLY round a number (num) to chosen number of decimal places (ndec)
    if ndec == 0:
        return int(num + 0.5)
    
    else:
        digit_value = 10**ndec
        
        if num > 0:
            return int(num*digit_value + 0.5)/digit_value
        
        else:
            return int(num*digit_value - 0.5)/digit_value

# Solid angle info for 2-ID-E data
# XRF detector z position indicator in HDF5 file: 2xfm:m26.VAL (no idea why MAPS has this as an x-coord., but y- and z-coords. were constant with MDA # - including for final tomography scan)

file_name_0124 = "/raid/users/roter/Jacobsen/img.dat/2xfm_0124.mda.h5"
file_name_0125 = "/raid/users/roter/Jacobsen/img.dat/2xfm_0125.mda.h5"
file_name_0126 = "/raid/users/roter/Jacobsen/img.dat/2xfm_0126.mda.h5"

elements_xrf, cts_xrf_0124, theta, _, _, _, _ = h5_util.extract_h5_xrf_data(file_name_0124, synchrotron = 'aps')
_, cts_xrf_0125, _, _, _, _, _ = h5_util.extract_h5_xrf_data(file_name_0125, synchrotron = 'aps')
_, cts_xrf_0126, _, _, _, _, _ = h5_util.extract_h5_xrf_data(file_name_0126, synchrotron = 'aps')

elements_xrt, cts_xrt_0124, _, _, _, _, _ = h5_util.extract_h5_xrt_data(file_name_0124, synchrotron = 'aps')
_, cts_xrt_0125, _, _, _, _, _ = h5_util.extract_h5_xrt_data(file_name_0125, synchrotron = 'aps')
_, cts_xrt_0126, _, _, _, _, _ = h5_util.extract_h5_xrt_data(file_name_0126, synchrotron = 'aps')

print(f'Theta = {theta} degrees')

element_index_xrf = elements_xrf.index('Fe')
element_index_xrt = elements_xrt.index('us_ic')

fe_0124 = cts_xrf_0124[element_index_xrf] # z = 12 mm (dz = 0)
fe_0125 = cts_xrf_0125[element_index_xrf] # z = 17 mm (dz = 5 mm)
fe_0126 = cts_xrf_0126[element_index_xrf] # z = 20 mm (dz = 8 mm)

i_tot_xrf_0124 = fe_0124.sum()
i_tot_xrf_0125 = fe_0125.sum()
i_tot_xrf_0126 = fe_0126.sum()

us_ic_0124 = cts_xrt_0124[element_index_xrt]
us_ic_0125 = cts_xrt_0125[element_index_xrt]
us_ic_0126 = cts_xrt_0126[element_index_xrt]

i_tot_us_ic_0124 = us_ic_0124.sum()
i_tot_us_ic_0125 = us_ic_0125.sum()
i_tot_us_ic_0126 = us_ic_0126.sum()

dz = [0, 5, 8]

i_tot_xrf = [i_tot_xrf_0124, i_tot_xrf_0125, i_tot_xrf_0126]
i_tot_us_ic = [i_tot_us_ic_0124, i_tot_us_ic_0125, i_tot_us_ic_0126]

print(i_tot_us_ic)
print(i_tot_xrf)

sys.exit()

# x_kb, y_kb = dz_kb, i_tot_kb
# x_cap, y_cap = dz_cap, i_tot_cap

# fit_kb = cf(i_obj_kb, dz_kb, i_tot_kb)
# mod = Model(i_obj_kb)

mod = Model(i_obj)

params = mod.make_params(A = 1, B = 1)
params['A'].min = 0
params['B'].min = 0

# result = mod.fit(i_tot_kb, params, x=dz_kb)

# a_kb = result.params['A'].value
# a_kb = f"{a_kb:.2e}"
# a_kb_coeff, a_kb_exponent = a_kb.split("e+0")

# z_det_nom_kb = result.params['B'].value
# print(z_det_nom_kb)

result = mod.fit(i_tot, params, x = dz)

a = result.params['A'].value
a = f"{a:.2e}"
a_coeff, a_cap_exponent = a.split("e+0")

z_det_nom = result.params['B'].value

# print(z_det_nom_cap)

# x1 = np.linspace(np.min(dz_cap), np.max(dz_cap), 1000)
# yfit = result.eval(x = x1)

x1 = np.linspace(np.min(dz), np.max(dz), 1000)
yfit = result.eval(x = x1)

plt.scatter(dz, i_tot, color = 'black')
# plt.scatter(dz_kb, i_tot_kb, color = 'black')
# plt.plot(x1, yfit, 'r--', label = r'$I = {0}/\left({1} + \Delta d\right)^{{2}}$'.format(rc(a_kb, ndec = 1), rc(z_det_nom_kb, ndec = 1)))
plt.plot(x1, yfit, 'r--', label = r'$I = {0}\times 10^{1}/\left({2} + \Delta d\right)^{{2}}$'.format(a_coeff, a_cap_exponent, round_correct(z_det_nom, ndec = 1)))
# plt.plot(x1, yfit, 'r--', label = r'$I = {0}\times 10^{1}/\left({2} + \Delta d\right)^{{2}}$'.format(a_kb_coeff, a_kb_exponent, rc(z_det_nom_kb, ndec = 1)))
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tick_params('both', length = 9)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel(r'$\Delta d$ (mm)', fontsize = 16)
plt.ylabel(r'$I$ (a.u.)', fontsize = 16)
plt.gca().yaxis.get_offset_text().set_fontsize(14)  # Adjust as needed
plt.legend(frameon = False, fontsize = 14)
plt.tight_layout()
# plt.savefig("/Users/bwr0835/Documents/GitHub/gradresearch/8_bm_res_exp/writeup/solid_angle_kb.svg")
plt.show()


# def i_obj_cap(dz_cap, a_cap, z_det_nom_cap):
    # return a_cap/(z_det_nom_cap + dz_cap)**2

# x_kb, y_kb = dz_kb, i_tot_kb
# # x_cap, y_cap = dz_cap, i_tot_cap

# fit_kb = cf(i_obj_kb, dz_kb, i_tot_kb)
# # fit_cap = cf(i_obj_cap, dz_cap, i_tot_cap)

# a_kb, z_det_nom_kb = fit_kb[0]
# # a_cap, z_det_nom_cap = fit_cap[0]
# print(a_kb)
# print(z_det_nom_kb)

# x = np.linspace(np.min(dz_kb), np.max(dz_kb), 1000)
# y = i_obj_kb(x, a_kb, z_det_nom_kb)

# # print(a_cap)
# # print(z_det_nom_cap)

# # print(np.shape(fe_0160))
# # print(np.shape(fe_0161))

# # x = np.linspace(np.min(dz_cap), np.max(dz_cap), 1000)
# # y = i_obj_cap(x, a_cap, z_det_nom_cap)

# # x = np.linspace(np.min(dz_cap), np.max(dz_cap), 1000)
# # y = i_obj_cap(x, a_cap, z_det_nom_cap)

# # plt.imshow(fe_0161 - fe_0160)

# plt.plot(x, y, 'r--', label = r'$\tilde{{I}} = {0}/\left({1} + \Delta d\right)^{{2}}$'.format(rc(a_kb, ndec = 1), rc(z_det_nom_kb, ndec = 1)))
# plt.scatter(dz_kb, i_tot_kb, color = 'black')
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.tick_params('both', length = 9)
# plt.xlabel(r'$\Delta d$ (mm)', fontsize = 16)
# plt.ylabel(r'$\tilde{I}$ (a.u.)', fontsize = 16)
# plt.legend(frameon = False, fontsize = 14)
# plt.tight_layout()
# # plt.savefig("/Users/bwr0835/Documents/GitHub/gradresearch/8_bm_res_exp/writeup/solid_angle_kb.svg")
# plt.show()