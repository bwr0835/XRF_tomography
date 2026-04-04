import numpy as np, imageio as iio, os, matplotlib as mpl

from matplotlib import pyplot as plt, colors

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

def create_gif(tiff_filename_array, output_filepath, fps):
    writer = iio.get_writer(output_filepath, mode = 'I', duration = 1/fps)
    
    for filename in tiff_filename_array:
        img = iio.imread(filename)

        writer.append_data(img)
    
    writer.close()

    for filename in tiff_filename_array:
        os.remove(filename)

input_dir_path = '/Users/bwr0835/Documents/2_ide_realigned_data_04__2026_cor_correction_iter_reproj/xrt_od_xrf_realignment'
output_dir_path = '/home/bwr0835/iter_reproj/xrt_gridrec_alignment_comp_no_shift_find_cor_pc_shift_no_log_july_14_2025'