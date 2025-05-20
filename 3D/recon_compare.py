import numpy as np, imageio as iio, os

from matplotlib import pyplot as plt

def create_gif(tiff_filename_array, output_filepath, fps):
    writer = iio.get_writer(output_filepath, mode = 'I', duration = 1/fps)
    
    for filename in tiff_filename_array:
        img = iio.imread(filename)

        writer.append_data(img)
    
    writer.close()

    for filename in tiff_filename_array:
        os.remove(filename)

output_dir_path = '/home/bwr0835/iter_reproj/gridrec_mlem_comp_no_shift_shift_20_may_20_2025'

file_1 = '/home/bwr0835/iter_reproj/mlem_1_iter_manual_shift_20_tomopy_default_cor_w_padding_05_06_2025/recon_array_iter_Fe.npy'
file_2 = '/home/bwr0835/iter_reproj/gridrec_1_iter_manual_shift_20_tomopy_default_cor_w_padding_05_06_2025/recon_array_iter_Fe.npy'
file_3 = '/home/bwr0835/iter_reproj/gridrec_1_iter_no_cor_shift_tomopy_default_cor_w_padding_05_06_2025/recon_array_iter_Fe.npy'
file_4 = '/home/bwr0835/iter_reproj/mlem_1_iter_no_cor_shift_tomopy_default_cor_w_padding_05_06_2025/recon_array_iter_Fe.npy'

recon_gridrec_no_shift = np.load(file_3)[0]
recon_gridrec_shift_20 = np.load(file_2)[0]
recon_mlem_no_shift = np.load(file_1)[0]
recon_mlem_shift_20 = np.load(file_4)[0]

recon_array = [recon_gridrec_no_shift, recon_gridrec_shift_20, recon_mlem_no_shift, recon_mlem_shift_20]

n_slices = recon_gridrec_no_shift.shape[0]

tiff_array = []

fig1, axs1 = plt.subplots(4, 4)

im1_1 = axs1[0, 0].imshow(recon_gridrec_no_shift[0])
im2_1 = axs1[0, 1].imshow(recon_gridrec_shift_20[0])
im3_1 = axs1[1, 0].imshow(recon_mlem_no_shift[0])
im4_1 = axs1[1, 1].imshow(recon_mlem_shift_20[0])

text_1 = axs1[0, 0].text(0.02, 0.02, r'Slice 0', transform = axs1[0, 0].transAxes, color = 'white')

for slice_idx in range(n_slices):
    print(f'Creating frame {slice_idx}...')

    n = 0

    for axes in fig1.axes:
        axes.set_data(recon_array[n][slice_idx])
    
    text_1.set_text(r'Slice index {0}'.format(slice_idx))

    filename_1 = os.path.join(output_dir_path, f'recon_compare_slice_{slice_idx:03d}.tiff')

    fig1.tight_layout()
    fig1.savefig(filename_1, dpi = 400)

    tiff_array.append()

print('Creating reconstruction comparison GIF...')

create_gif(tiff_array, os.path.join(output_dir_path, 'recon_compare.gif'), fps = 25)
