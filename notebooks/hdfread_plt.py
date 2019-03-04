import matplotlib.pyplot as plt
import numpy as np
import h5py

plt.rcParams['figure.figsize'] = [15, 5]

num_samples = 3

f, axes = plt.subplots(3, num_samples)
print "axes: ", axes.shape

# DATA

f1 = h5py.File('../snapshots/prob_unet/test_sample.hdf', 'r')
data = [f1['labels'], np.sum(f1['raw_affs'], axis=0), f1['raw_out']]
# print "data: ", data.keys()

for i, ax in enumerate(axes[0]):
    if i > 0: #labels
		ax.imshow(data[i][22], cmap="Greys_r")
    else:
    	ax.imshow(data[i][22])
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])

# PREDICTIONS


fp1 = h5py.File('../snapshots/prob_unet/prediction_00000000.hdf', 'r')
fp2 = h5py.File('../snapshots/prob_unet/prediction_00000001.hdf', 'r')
fp3 = h5py.File('../snapshots/prob_unet/prediction_00000002.hdf', 'r')
print "fp1: ", fp1.keys()
pred_affs = [np.sum(fp1['pred_affs'], axis=0), np.sum(fp2['pred_affs'], axis=0), np.sum(fp3['pred_affs'], axis=0)]
sample_zs = [fp1['sample_z'], fp2['sample_z'], fp3['sample_z']]
# print "pred: ", pred_affs
print "sample_zs: ", sample_zs[0][0]

axes[1][0].imshow(pred_affs[0][22], cmap="Greys_r")
axes[1][1].imshow(pred_affs[1][22], cmap="Greys_r")
axes[1][2].imshow(pred_affs[2][22], cmap="Greys_r")

axes[2][0].imshow(sample_zs[0][0])
axes[2][1].imshow(sample_zs[1][0])
axes[2][2].imshow(sample_zs[2][0])

# axes[2].get_yaxis().set_ticks([])

# for i, ax in enumerate(axes[1]):
# 	ax.imshow(pred[0][22], cmap="Greys_r")
# 	ax.get_xaxis().set_ticks([])


# DISPLAY

plt.show()
