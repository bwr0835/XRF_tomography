import numpy as np, h5py, h5_util as util, sys

from matplotlib import pyplot as plt, patches as pat

coarse_scan_filename = '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235675.h5'
fine_scan_filename = '/raid/users/roter/Jacobsen-nslsii/data/xrf/scan2D_235676.h5'

elements, counts_coarse, _, x_um_coarse, y_um_coarse, nx_coarse, ny_coarse, dx_cm_coarse, dy_cm_coarse = util.extract_h5_xrf_data(coarse_scan_filename, synchrotron = 'nsls-ii', scan_coords = True)
_, counts_fine, _, x_um_fine, y_um_fine, nx_fine, ny_fine, dx_cm_fine, dy_cm_fine = util.extract_h5_xrf_data(coarse_scan_filename, synchrotron = 'nsls-ii', scan_coords = True)

desired_element = 'Fe_K'
desired_element_index = elements.index(desired_element)

fe_coarse = counts_coarse[desired_element_index]
fe_fine = counts_fine[desired_element_index]

x0_fine, y0_fine = x_um_fine[0, 0], y_um_fine[0, 0] # Top left corner
x1_fine, y1_fine = x_um_fine[0, -1], y_um_fine[0, -1] # Top right corner
x2_fine, y2_fine = x_um_fine[-1, 0], y_um_fine[-1, 0] # Bottom left corner
x3_fine, y3_fine = x_um_fine[-1, -1], y_um_fine[-1, -1] # Bottom right corner

x0_coarse_idx, y0_coarse_idx = np.unravel(np.argmin(np.abs(x_um_coarse - x0_fine)), x_um_coarse.shape), np.argmin(np.abs(y_um_coarse - y0_fine))
x1_coarse_idx, y1_coarse_idx = np.argmin(np.abs(x_um_coarse - x1_fine)), np.argmin(np.abs(y_um_coarse - y1_fine))
x2_coarse_idx, y2_coarse_idx = np.argmin(np.abs(x_um_coarse - x2_fine)), np.argmin(np.abs(y_um_coarse - y2_fine))
x3_coarse_idx, y3_coarse_idx = np.argmin(np.abs(x_um_coarse - x3_fine)), np.argmin(np.abs(y_um_coarse - y3_fine))

print(x0_coarse_idx)

sys.exit()

fig, axs = plt.subplots(1, 2)

pixel_width = x1_coarse_idx[1] - x0_coarse_idx[1]
pixel_height = y1_coarse_idx[0] - y0_coarse_idx[0]

rect = pat.Rectangle(x0_coarse_idx[1] - 0.5, y0_coarse_idx[0] - 0.5, pixel_width, pixel_height, edge_color = 'white', facecolor = 'none')

axs[0].imshow(fe_coarse)
axs[1].imshow(fe_fine)

axs[0].add_patch(rect)

for axes in fig.axes:
    axes.axis('off')

fig.tight_layout()

plt.show()