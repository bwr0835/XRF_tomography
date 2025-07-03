import numpy as np, imageio as iio, os

from matplotlib import pyplot as plt

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

output_dir_path = '/home/bwr0835/iter_reproj/xrt_gridrec_mlem_comp_no_shift_shift_20_post_recon_log_july_03_2025'

os.makedirs(output_dir_path, exist_ok = True)

file_1 = '/home/bwr0835/iter_reproj/xrt_mlem_1_iter_manual_shift_20_no_log_tomopy_default_cor_w_padding_07_03_2025/recon_array_iter_ds_ic.npy'
file_2 = '/home/bwr0835/iter_reproj/xrt_gridrec_1_iter_manual_shift_20_no_log_tomopy_default_cor_w_padding_07_03_2025/recon_array_iter_ds_ic.npy'
file_3 = '/home/bwr0835/iter_reproj/xrt_gridrec_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025/recon_array_iter_ds_ic.npy'
file_4 = '/home/bwr0835/iter_reproj/xrt_mlem_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025/recon_array_iter_ds_ic.npy'

file_5 = '/home/bwr0835/iter_reproj/xrt_gridrec_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025/orig_exp_proj_ds_ic.npy'
file_6 = '/home/bwr0835/iter_reproj/xrt_gridrec_1_iter_manual_shift_20_no_log_tomopy_default_cor_w_padding_07_03_2025/aligned_proj_array_iter_ds_ic.npy'

file_7 = '/home/bwr0835/iter_reproj/xrt_gridrec_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025/synth_proj_array_iter_ds_ic.npy'
file_8 = '/home/bwr0835/iter_reproj/xrt_gridrec_1_iter_manual_shift_20_no_log_tomopy_default_cor_w_padding_07_03_2025/synth_proj_array_iter_ds_ic.npy'
file_9 = '/home/bwr0835/iter_reproj/xrt_mlem_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025/synth_proj_array_iter_ds_ic.npy'
file_10 = '/home/bwr0835/iter_reproj/xrt_mlem_1_iter_manual_shift_20_no_log_tomopy_default_cor_w_padding_07_03_2025/synth_proj_array_iter_ds_ic.npy'

file_11 = '/home/bwr0835/iter_reproj/xrt_mlem_1_iter_no_shift_no_log_tomopy_default_cor_w_padding_07_03_2025/theta_array.npy'

recon_gridrec_no_shift = np.load(file_3)[0]
recon_gridrec_shift_20 = np.load(file_2)[0]
recon_mlem_no_shift = np.load(file_4)[0]
recon_mlem_shift_20 = np.load(file_1)[0]

orig_proj = np.load(file_5)
aligned_proj_array_shift_20 = np.load(file_6)[0]

synth_proj_array_gridrec_no_shift = np.load(file_7)[0]
synth_proj_array_gridrec_shift_20 = np.load(file_8)[0]
synth_proj_array_mlem_no_shift = np.load(file_9)[0]
synth_proj_array_mlem_shift_20 = np.load(file_10)[0]

recon_gridrec_no_shift[recon_gridrec_no_shift > 0] = np.log(recon_gridrec_no_shift[recon_gridrec_no_shift > 0])
recon_gridrec_shift_20[recon_gridrec_shift_20 > 0] = np.log(recon_gridrec_shift_20[recon_gridrec_shift_20 > 0])
recon_mlem_no_shift[recon_mlem_no_shift > 0] = np.log(recon_mlem_no_shift[recon_mlem_no_shift > 0])
recon_mlem_shift_20[recon_mlem_shift_20 > 0] = np.log(recon_mlem_shift_20[recon_mlem_shift_20 > 0])

orig_proj[orig_proj > 0] = np.log(orig_proj[orig_proj > 0])
aligned_proj_array_shift_20[aligned_proj_array_shift_20 > 0] = np.log(aligned_proj_array_shift_20[aligned_proj_array_shift_20 > 0])

synth_proj_array_gridrec_no_shift[synth_proj_array_gridrec_no_shift > 0] = np.log(synth_proj_array_gridrec_no_shift[synth_proj_array_gridrec_no_shift > 0])
synth_proj_array_gridrec_shift_20[synth_proj_array_gridrec_shift_20 > 0] = np.log(synth_proj_array_gridrec_shift_20[synth_proj_array_gridrec_shift_20 > 0])
synth_proj_array_mlem_no_shift[synth_proj_array_mlem_no_shift > 0] = np.log(synth_proj_array_mlem_no_shift[synth_proj_array_mlem_no_shift > 0])
synth_proj_array_mlem_shift_20[synth_proj_array_mlem_shift_20 > 0] = np.log(synth_proj_array_mlem_shift_20[synth_proj_array_mlem_shift_20 > 0])

theta_array = np.load(file_11)

recon_array = [recon_gridrec_no_shift, recon_gridrec_shift_20, recon_mlem_no_shift, recon_mlem_shift_20]

n_slices = recon_gridrec_no_shift.shape[0]
n_theta = len(theta_array)

tiff_array_1 = []
tiff_array_2 = []

fig1, axs1 = plt.subplots(2, 2)
fig2, axs2 = plt.subplots(2, 3)

im1_1 = axs1[0, 0].imshow(recon_gridrec_no_shift[0])
im1_2 = axs1[0, 1].imshow(recon_gridrec_shift_20[0])
im1_3 = axs1[1, 0].imshow(recon_mlem_no_shift[0])
im1_4 = axs1[1, 1].imshow(recon_mlem_shift_20[0])

im2_1 = axs2[0, 0].imshow(orig_proj[0])
im2_2 = axs2[0, 1].imshow(synth_proj_array_gridrec_no_shift[0])
im2_3 = axs2[0, 2].imshow(synth_proj_array_mlem_no_shift[0])
im2_4 = axs2[1, 0].imshow(aligned_proj_array_shift_20[0])
im2_5 = axs2[1, 1].imshow(synth_proj_array_gridrec_shift_20[0])
im2_6 = axs2[1, 2].imshow(synth_proj_array_mlem_shift_20[0])

axs1[0, 0].set_title(r'No COR shift, GR')
axs1[0, 1].set_title(r'+20 shift, GR')
axs1[1, 0].set_title(r'No COR shift, MLEM')
axs1[1, 1].set_title(r'+20 shift, MLEM')

axs2[0, 0].set_title(r'Exp., Orig $\rightarrow$')
axs2[0, 1].set_title(r'Synth., GR')
axs2[0, 2].set_title(r'Synth., MLEM')
axs2[1, 0].set_title(r'Exp., +20 shift $\rightarrow$')
axs2[1, 1].set_title(r'Synth., GR')
axs2[1, 2].set_title(r'Synth., MLEM')

text_1 = axs1[0, 0].text(0.02, 0.02, r'Slice 0', transform = axs1[0, 0].transAxes, color = 'white')
text_2 = axs2[0, 0].text(0.02, 0.02, r'$\theta = {0}$'.format(theta_array[0]), transform = axs2[0, 0].transAxes, color = 'white')

for slice_idx in range(n_slices):
    print(f'Creating frame for slice {slice_idx}...')

    im1_1.set_data(recon_gridrec_no_shift[slice_idx])
    im1_2.set_data(recon_gridrec_shift_20[slice_idx])
    im1_3.set_data(recon_mlem_no_shift[slice_idx])
    im1_4.set_data(recon_mlem_shift_20[slice_idx])
    
    text_1.set_text(r'Slice index {0}'.format(slice_idx))

    filename_1 = os.path.join(output_dir_path, f'recon_compare_slice_{slice_idx:03d}.tiff')

    fig1.tight_layout()
    fig1.savefig(filename_1, dpi = 400)

    tiff_array_1.append(filename_1)

plt.close(fig1)

for theta_idx in range(n_theta):
    print(f'Creating frame for theta = {theta_array[theta_idx]} degrees...')

    im2_1.set_data(orig_proj[theta_idx])
    im2_2.set_data(synth_proj_array_gridrec_no_shift[theta_idx])
    im2_3.set_data(synth_proj_array_mlem_no_shift[theta_idx])
    im2_4.set_data(aligned_proj_array_shift_20[theta_idx])
    im2_5.set_data(synth_proj_array_gridrec_shift_20[theta_idx])
    im2_6.set_data(synth_proj_array_mlem_shift_20[theta_idx])

    text_2.set_text(r'$\theta = {0}$'.format(theta_array[theta_idx]))

    filename_2 = os.path.join(output_dir_path, f'proj_compare_theta_idx_{theta_idx:03d}.tiff')

    fig2.tight_layout()
    fig2.savefig(filename_2, dpi = 400)

    tiff_array_2.append(filename_2)

plt.close(fig2)

# print('Creating reconstruction comparison GIF...')

create_gif(tiff_array_1, os.path.join(output_dir_path, 'recon_compare.gif'), fps = 25)

print('Creating projection comparison GIF...')

create_gif(tiff_array_2, os.path.join(output_dir_path, 'proj_compare.gif'), fps = 25)
