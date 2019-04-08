import matplotlib.pyplot as plt
import numpy as np
import h5py

f, axes = plt.subplots(1, 3)

file_1 = h5py.File('../snapshots/prob_unet/setup_8/prediction_00000000.hdf', 'r')
file_2 = h5py.File('../snapshots/prob_unet/setup_8/prediction_00000001.hdf', 'r')
file_3 = h5py.File('../snapshots/prob_unet/setup_8/prediction_00000002.hdf', 'r')

pred_1 = np.sum(np.array(file_1['volumes']['pred_affs']), axis=0)
pred_2 = np.sum(np.array(file_2['volumes']['pred_affs']), axis=0)
pred_3 = np.sum(np.array(file_3['volumes']['pred_affs']), axis=0)

diff_1_2 = pred_1 - pred_2
diff_1_3 = pred_1 - pred_3
diff_2_3 = pred_2 - pred_3
axes[0].imshow(diff_1_2[22])
axes[1].imshow(diff_1_3[22])
axes[2].imshow(diff_2_3[22])
plt.show()