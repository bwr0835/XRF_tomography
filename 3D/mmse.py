import numpy as np

from matplotlib import pyplot as plt

file_name = '/Users/bwr0835/Documents/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only/2_ide_realigned_data_02_12_2026_iter_reproj_cor_correction_only/model_change_mse_epoch.csv'

mmse_array = np.loadtxt(file_name, delimiter = ',')

plt.plot(mmse_array[:, 1])
plt.xlabel('Epoch')
plt.ylabel('MMSE')
plt.show()