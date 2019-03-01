import matplotlib.pyplot as plt
import numpy as np
import h5py

plt.rcParams['figure.figsize'] = [15, 5]

num_samples = 3

f = h5py.File('../snapshots/prob_unet/prediction.hdf', 'r')
volumes = f['volumes']

data = [volumes['labels'], volumes['raw'], np.sum(volumes['raw_affs'], axis=0)]
print "data: ", len(data)
pred = [np.sum(volumes['pred_affs_1'], axis=0),np.sum(volumes['pred_affs_2'], axis=0), np.sum(volumes['pred_affs_3'], axis=0)]
f, axes = plt.subplots(len(data) + 1, num_samples)
print "pred: ", axes.shape

for i in xrange(len(axes)):
    for j, ax in enumerate(axes[i]):
        if i == 0:
			ax.imshow(data[i][22])
        elif i < len(axes) - 1:
			ax.imshow(data[i][22], cmap="Greys_r")
        else:
        	ax.imshow(pred[j][22])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
plt.show()

def join(affs):
	return np.sum(affs, axis=0)
