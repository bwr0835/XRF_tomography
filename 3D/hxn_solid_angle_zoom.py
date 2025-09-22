import numpy as np, h5py, h5_util as util, sys

from matplotlib import pyplot as plt, patches as pat

plt.rcParams["figure.autolayout"] = True

coarse_scan_filename = '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235675.h5'
fine_scan_filename = '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235676.h5'

elements, counts_coarse, _,  _, x_um_coarse, y_um_coarse, nx_coarse, ny_coarse, dx_cm_coarse, dy_cm_coarse = util.extract_h5_xrf_data(coarse_scan_filename, synchrotron = 'nsls-ii', scan_coords = True, US_IC = True)
_, counts_fine, us_ic, _, x_um_fine, y_um_fine, nx_fine, ny_fine, dx_cm_fine, dy_cm_fine = util.extract_h5_xrf_data(fine_scan_filename, synchrotron = 'nsls-ii', scan_coords = True, US_IC = True)

desired_element = 'Fe_K'
desired_element_index = elements.index(desired_element)

fe_coarse = counts_coarse[desired_element_index]
fe_fine = counts_fine[desired_element_index]

x0_fine, y0_fine = x_um_fine[0, 0], y_um_fine[0, 0] # Top left corner
x1_fine, y1_fine = x_um_fine[0, -1], y_um_fine[0, -1] # Top right corner
x2_fine, y2_fine = x_um_fine[-1, 0], y_um_fine[-1, 0] # Bottom left corner
x3_fine, y3_fine = x_um_fine[-1, -1], y_um_fine[-1, -1] # Bottom right corner

x0_coarse, y0_coarse = x_um_coarse[0, 0], y_um_coarse[0, 0] # Top left corner
x1_coarse, y1_coarse = x_um_coarse[0, -1], y_um_coarse[0, -1] # Top right corner
x2_coarse, y2_coarse = x_um_coarse[-1, 0], y_um_coarse[-1, 0] # Bottom left corner
x3_coarse, y3_coarse = x_um_coarse[-1, -1], y_um_coarse[-1, -1] # Bottom right corner

print(f'{x0_fine}, {y0_fine}')
print(f'{x1_fine}, {y1_fine}')
print(f'{x2_fine}, {y2_fine}')
print(f'{x3_fine}, {y3_fine}')
print('---------------------')
# 
# x0_coarse_idx, y0_coarse_idx = np.unravel_index(np.argmin(np.abs(x_um_coarse - x0_fine)), x_um_coarse.shape), np.unravel_index(np.argmin(np.abs(y_um_coarse - y0_fine)), y_um_coarse.shape)
# x1_coarse_idx, y1_coarse_idx = np.unravel_index(np.argmin(np.abs(x_um_coarse - x1_fine)), x_um_coarse.shape), np.unravel_index(np.argmin(np.abs(y_um_coarse - y1_fine)), y_um_coarse.shape)
# x2_coarse_idx, y2_coarse_idx = np.unravel_index(np.argmin(np.abs(x_um_coarse - x2_fine)), x_um_coarse.shape), np.unravel_index(np.argmin(np.abs(y_um_coarse - y2_fine)), y_um_coarse.shape)
# x3_coarse_idx, y3_coarse_idx = np.unravel_index(np.argmin(np.abs(x_um_coarse - x3_fine)), x_um_coarse.shape), np.unravel_index(np.argmin(np.abs(y_um_coarse - y3_fine)), y_um_coarse.shape)

y0_coarse_idx, x0_coarse_idx = np.unravel_index(np.argmin(np.sqrt((x_um_coarse - x0_fine)**2 + (y_um_coarse - y0_fine)**2)), x_um_coarse.shape)
y1_coarse_idx, x1_coarse_idx = np.unravel_index(np.argmin(np.sqrt((x_um_coarse - x1_fine)**2 + (y_um_coarse - y1_fine)**2)), x_um_coarse.shape)
y2_coarse_idx, x2_coarse_idx = np.unravel_index(np.argmin(np.sqrt((x_um_coarse - x2_fine)**2 + (y_um_coarse - y2_fine)**2)), x_um_coarse.shape)
y3_coarse_idx, x3_coarse_idx = np.unravel_index(np.argmin(np.sqrt((x_um_coarse - x3_fine)**2 + (y_um_coarse - y3_fine)**2)), x_um_coarse.shape)

print(f'{x0_coarse_idx}, {y0_coarse_idx}')
print(f'{x1_coarse_idx}, {y1_coarse_idx}')
print(f'{x2_coarse_idx}, {y2_coarse_idx}')
print(f'{x3_coarse_idx}, {y3_coarse_idx}')

# sys.exit()

fig, axs = plt.subplots(1, 2)

pixel_width = x1_coarse_idx - x0_coarse_idx
pixel_height = y3_coarse_idx - y1_coarse_idx

# pixel_width = x1_fine - x0_fine
# pixel_height = y1_fine - y0_fine



axs[0].imshow(fe_coarse)
axs[1].imshow(fe_fine/us_ic)

rect = pat.Rectangle((x0_coarse_idx - 0.5, y1_coarse_idx - 0.5), pixel_width, pixel_height, edgecolor = 'white', facecolor = 'none')

axs[0].add_patch(rect)

for axes in fig.axes:
    axes.axis('off')

fig.tight_layout()

plt.show()