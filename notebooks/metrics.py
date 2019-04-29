import numpy as np
import h5py
from scipy.stats import energy_distance

gt_diff = 0.00876

threshold = 0.9

num_predictions = 3

print("pred v gt")
pred_affs = []
for i in range(num_predictions):
	with h5py.File('../snapshots/prob_unet/setup_4/prediction_0000000{}.hdf'.format(i), 'r') as file:
		gt_affs = np.array(file['volumes']['gt_affs'])
		gt_affs_count = np.count_nonzero(gt_affs == 0)

		pred_affs.append(np.array(file['volumes']['pred_affs']))

print "gt_affs_count: ", gt_affs_count
# pred_1 v pred_2
p1_count = np.count_nonzero(pred_affs[0] < threshold)
p2_count = np.count_nonzero(pred_affs[1] < threshold)
p3_count = np.count_nonzero(pred_affs[2] < threshold)

pred_diff_12 = (p1_count- p2_count)/float(p1_count)
pred_diff_13 = (p1_count- p3_count)/float(p1_count)
pred_diff_23 = (p2_count- p3_count)/float(p2_count)

print "p1_count: ", p1_count
print "p2_count: ", p2_count
print "p3_count: ", p3_count


# diff = (pred_affs_count- gt_affs_count)/float(pred_affs_count)
print "pred_diff_12: ", pred_diff_12
print "pred_diff_13: ", pred_diff_13
print "pred_diff_23: ", pred_diff_23

print "ratio 1_2: ", pred_diff_12/gt_diff
print "ratio 1_3: ", pred_diff_13/gt_diff
print "ratio 2_3: ", pred_diff_23/gt_diff

