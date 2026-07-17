import numpy as np, \
       h5py, \
       os, \
       sys

from imageio import v2 as iio2
from matplotlib import pyplot as plt
from numpy.ma import true_divide

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.minor.size'] = 4.5

dir_path = '/home/bwr0835'

output_path_xrf = os.path.join(dir_path, 'simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64.h5')
output_path_xrt = os.path.join(dir_path, 'simulated_proj_data_xrt_64_64_64.h5')

proj_data_xrf = np.zeros((4, 200, 64, 64))
proj_data_xrt = np.zeros((1, 200, 64, 64))

proj_data_xrt[0] = np.load(os.path.join(dir_path, 'simulated_proj_data_xrt_64_64_64.npy')).reshape(200, 64, 64)

theta = np.linspace(-180, 180, 201)[:-1]
dtheta = theta[1] - theta[0]

elements_xrf = ['Ca', 'Ca_L', 'Sc', 'Sc_L']
elements_xrt = ['xrt_sig']

xrt_proj_img_enabled = False
xrt_sino_enabled = False

xrf_proj_img_enabled = True
xrf_sino_enabled = False

remove_files_enabled = False

for theta_idx in range(200):
    file_path = f'{dir_path}/simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64_{theta_idx}.npy'

    proj_data_xrf[:, theta_idx] = np.load(file_path).reshape(4, 64, 64)

with h5py.File(output_path_xrt, 'w') as f:
    exchange = f.create_group('exchange')

    exchange.create_dataset('data', data = proj_data_xrt)
    exchange.create_dataset('elements', data = elements_xrt)
    exchange.create_dataset('theta', data = theta)

if remove_files_enabled:
    for theta_idx in range(200):
        file_path = f'{dir_path}/simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64_{theta_idx}.npy'

        os.remove(file_path)

with h5py.File(output_path_xrf, 'w') as f:
    exchange = f.create_group('exchange')

    exchange.create_dataset('data', data = proj_data_xrf)
    exchange.create_dataset('elements', data = elements_xrf)
    exchange.create_dataset('theta', data = theta)

if xrt_proj_img_enabled:
    fig1, axs1 = plt.subplots()

    im = axs1.imshow(proj_data_xrt[0, 0], cmap = 'jet', aspect = 'auto')
    axs1.axis('off')
    # axs1.tick_params(axis = 'both', which = 'major', labelsize = 14)
    axs1.set_title(r'XRT', fontsize = 16)
    # axs1.set_ylabel(r'$\theta$ (degrees)')
    # axs1.set_xlabel(r'Scan position index')

    txt = axs1.text(0.02, 0.02, r'$\theta = 0^{\circ}$', transform = axs1.transAxes, color = 'white', fontsize = 14)

    frames = []

    for theta_idx in range(200):
        im.set_data(proj_data_xrt[0, theta_idx])
        txt.set_text(r'$\theta = {0}^{{\circ}}$'.format(theta[theta_idx]))

        fig1.canvas.draw()
        
        frame = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]
        
        frames.append(frame)

    plt.close(fig1)

    gif_filename = os.path.join(dir_path, f'simulated_proj_data_xrt_64_64_64.gif')

    iio2.mimsave(gif_filename, frames, fps = 10)

if xrt_sino_enabled:
    fig, axs = plt.subplots()

    vmin = proj_data_xrt.min()
    vmax = proj_data_xrt.max()

    im = axs.imshow(proj_data_xrt[0, :, 0], vmin = vmin, vmax = vmax, cmap = 'jet', origin = 'lower', aspect = 'auto', extent = [-0.5, 63.5, theta.min() - dtheta/2, theta.max() + dtheta/2])
    # axs2.axis('off')
    axs.tick_params(axis = 'both', which = 'major', labelsize = 14)
    axs.set_title(r'XRT', fontsize = 16)
    axs.set_ylabel(r'$\theta$ (\textdegree{})', fontsize = 16)
    axs.set_xlabel(r'Scan position index', fontsize = 16)

    txt = axs.text(0.02, 0.02, r'Slice 0', transform = axs.transAxes, color = 'white', fontsize = 14)

    frames = []

    for slice_idx in range(64):
        im.set_data(proj_data_xrt[0, :, slice_idx])
        txt.set_text(r'Slice {0}'.format(slice_idx))

        fig.canvas.draw()
        
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
        frames.append(frame)

    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'simulated_proj_data_xrt_64_64_64.gif')

    iio2.mimsave(gif_filename, frames, fps = 10)

if xrf_proj_img_enabled:
    fig, axs = plt.subplots(2, 2)

    im1 = axs[0, 0].imshow(proj_data_xrf[0, 0], vmax = proj_data_xrf.max(), cmap = 'jet')
    im2 = axs[0, 1].imshow(proj_data_xrf[1, 0], vmax = proj_data_xrf.max(), cmap = 'jet')
    im3 = axs[1, 0].imshow(proj_data_xrf[2, 0], vmax = proj_data_xrf.max(), cmap = 'jet')
    im4 = axs[1, 1].imshow(proj_data_xrf[3, 0], vmax = proj_data_xrf.max(), cmap = 'jet')

    txt = axs[0, 0].text(0.02, 0.02, r'$\theta = 0^{\circ}$', transform = axs[0, 0].transAxes, color = 'white', fontsize = 14)
    
    for i, ax in enumerate(fig.axes):
        ax.axis('off')
        ax.set_title(elements_xrf[i], fontsize = 16)
    
    frames = []

    for theta_idx in range(200):
        im1.set_data(proj_data_xrf[0, theta_idx])
        im2.set_data(proj_data_xrf[1, theta_idx])
        im3.set_data(proj_data_xrf[2, theta_idx])
        im4.set_data(proj_data_xrf[3, theta_idx])

        txt.set_text(r'$\theta = {0}^{{\circ}}$'.format(theta[theta_idx]))

        fig.canvas.draw()
        
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
        frames.append(frame)

    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'simulated_proj_data_xrf_no_probe_att_no_selfab_64_64_64.gif')

    iio2.mimsave(gif_filename, frames, fps = 10)

if xrf_sino_enabled:
    fig, axs = plt.subplots(2, 2)

    vmin = proj_data_xrf.min()
    vmax = proj_data_xrf.max()

    im1 = axs[0, 0].imshow(proj_data_xrf[0, :, 0], vmin = vmin, vmax = vmax, cmap = 'jet', origin = 'lower', aspect = 'auto', extent = [-0.5, 63.5, theta.min() - dtheta/2, theta.max() + dtheta/2])
    im2 = axs[0, 1].imshow(proj_data_xrf[1, :, 0], vmin = vmin, vmax = vmax, cmap = 'jet', origin = 'lower', aspect = 'auto', extent = [-0.5, 63.5, theta.min() - dtheta/2, theta.max() + dtheta/2])
    im3 = axs[1, 0].imshow(proj_data_xrf[2, :, 0], vmin = vmin, vmax = vmax, cmap = 'jet', origin = 'lower', aspect = 'auto', extent = [-0.5, 63.5, theta.min() - dtheta/2, theta.max() + dtheta/2])
    im4 = axs[1, 1].imshow(proj_data_xrf[3, :, 0], vmin = vmin, vmax = vmax, cmap = 'jet', origin = 'lower', aspect = 'auto', extent = [-0.5, 63.5, theta.min() - dtheta/2, theta.max() + dtheta/2])

    for i, ax in enumerate(fig.axes):
        ax.set_title(elements_xrf[i], fontsize = 16)
        ax.set_xlabel(r'Scan position index', fontsize = 16)
        ax.set_ylabel(r'$\theta$ (\textdegree{})', fontsize = 16)

    txt = axs[0, 0].text(0.02, 0.02, r'Slice 0', transform = axs[0, 0].transAxes, color = 'white', fontsize = 14)
    
    frames = []

    for slice_idx in range(64):
        im1.set_data(proj_data_xrf[0, :, slice_idx])
        im2.set_data(proj_data_xrf[1, :, slice_idx])
        im3.set_data(proj_data_xrf[2, :, slice_idx])
        im4.set_data(proj_data_xrf[3, :, slice_idx])

        txt.set_text(r'Slice {0}'.format(slice_idx))
        
        fig.canvas.draw()
        
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
        frames.append(frame)

    plt.close(fig)

    gif_filename = os.path.join(dir_path, f'simulated_sino_data_xrf_no_probe_att_no_selfab_64_64_64.gif')

    iio2.mimsave(gif_filename, frames, fps = 10)