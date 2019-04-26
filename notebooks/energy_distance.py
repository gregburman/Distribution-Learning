import numpy as np
import h5py
from scipy.stats import energy_distance

labels = []
affs = []

num = 3

with h5py.File('../snapshots/all_affinities.hdf', 'r') as file:
	volumes = file['volumes']
	labels.append(np.array(volumes['labels']))
	# affs.append(np.array(volumes['gt_affs']))

	print "labels shape: ", labels[0].shape
	# print "affs shape: ", affs[0].shape

	for i in range(num):
		labels.append(np.array(volumes['merged_labels_%i'%(i+1)]))
		# affs.append(np.array(volumes['merged_affs_%i'%(i+1)]))

for i in range(num+1):

	# hash values
	labels_hash = np.sum(labels[i])
	print "labels ", i, " hash: ", labels_hash

	#labels count
	num_labels = len(np.unique(labels[i]))
	print "labels ", i, " count: ", num_labels


# Energy Distance calc
# gt_affs_z = np.array(affs[0][0])
# gt_affs_y = np.array(affs[0][1])
# gt_affs_x = np.array(affs[0][2])

# for i in range(1, num+1):

# 	pred_affs_z = np.array(affs[i][0])
# 	pred_affs_y = np.array(affs[i][1])
# 	pred_affs_x = np.array(affs[i][2])

# 	ed_z = energy_distance(gt_affs_z.flatten(), pred_affs_z.flatten())
# 	ed_y = energy_distance(gt_affs_y.flatten(), pred_affs_y.flatten())
# 	ed_x = energy_distance(gt_affs_x.flatten(), pred_affs_x.flatten())
# 	ed = np.sqrt(np.power(ed_z, 2) + np.power(ed_y, 2) + np.power(ed_x, 2))
# 	print("merge_%i ed: "%i, ed)


