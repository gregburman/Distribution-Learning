from __future__ import print_function
import sys
sys.path.append('../')

from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt

from nodes import SequentialProvider

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score

logging.basicConfig(level=logging.INFO)

setup_name = sys.argv[1]

data_dir = "../snapshots/prob_unet/" + setup_name

def compute_scores(d, iterations):

	samples = ["prediction_%08i_A"%d, "prediction_%08i_B"%d, "prediction_%08i_C"%d, "prediction_%08i_D"%d]
	labels_key = ArrayKey('LABELS')
	gt_affs_key = ArrayKey('GT_AFFINITIES')
	pred_affinities_key = ArrayKey('PREDICTED_AFFS')
	sample_z_key = ArrayKey("SAMPLE_Z")

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate((132, 132, 132)) * voxel_size
	output_shape = Coordinate((44, 44, 44)) * voxel_size
	sample_shape = Coordinate((1, 1, 6)) * voxel_size

	request = BatchRequest()
	# request.add(labels_key, output_shape)
	request.add(gt_affs_key, output_shape, )
	request.add(pred_affinities_key, input_shape)
	# request.add(sample_z_key, sample_shape)

	dataset_names = {
		gt_affs_key: 'volumes/gt_affs',
		pred_affinities_key: 'volumes/pred_affs'
	}

	array_specs = {
		gt_affs_key: ArraySpec(interpolatable=True),
		pred_affinities_key: ArraySpec(interpolatable=True)
	}

	pipeline = tuple(
		Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = dataset_names,
            array_specs = array_specs
        ) +
        Pad(gt_affs_key, None) +
        Pad(pred_affinities_key, None)
        # Pad(merged_labels_key[i], None) for i in range(num_merges) # don't know why this doesn't work
        for sample in samples
	)

	pipeline += SequentialProvider()

	# pipeline += Snapshot(
	# 		dataset_names={
	# 			gt_affs_key: 'volumes/gt_affs',
	# 			pred_affinities_key: 'volumes/pred_affs',
	# 			# sample_z_key: 'volumes/sample_z',
	# 		},
	# 		output_filename='test_scores.hdf',
	# 		every=1,
	# 		dataset_dtypes={
	# 			gt_affs_key: np.float32,
	# 			pred_affinities_key: np.float32,
	# 			sample_z_key: np.float32,
	# 		})



	# print("Calculating Scores...")
	with build(pipeline) as p:
		aris = []
		pred_affs = []
		for i in range(iterations):
			req = p.request_batch(request)

			if i == 0:
				gt_affs = np.array(req[gt_affs_key].data)
			pred_affs.append(threshold(__crop_center(np.array(req[pred_affinities_key].data), (44, 44, 44))))

		# print(np.sum(gt_affs))
		# print(np.sum(pred_affs[0]))
		# print(np.sum(pred_affs[1]))
		ari_A = adjusted_rand_score(gt_affs.flatten(), pred_affs[0].flatten())
		ari_B = adjusted_rand_score(gt_affs.flatten(), pred_affs[1].flatten())
		ari_C = adjusted_rand_score(gt_affs.flatten(), pred_affs[2].flatten())
		ari_D = adjusted_rand_score(gt_affs.flatten(), pred_affs[3].flatten())
		ari_YS = (ari_A + ari_B + ari_C + ari_D)/4
		d_YS = 1 - ari_YS
		# print("gt_ari_avg: ", ari_YS)

		ari_AB = adjusted_rand_score(pred_affs[0].flatten(), pred_affs[1].flatten())
		ari_AC = adjusted_rand_score(pred_affs[0].flatten(), pred_affs[2].flatten())
		ari_AD = adjusted_rand_score(pred_affs[0].flatten(), pred_affs[3].flatten())
		ari_BC = adjusted_rand_score(pred_affs[1].flatten(), pred_affs[2].flatten())
		ari_BD = adjusted_rand_score(pred_affs[1].flatten(), pred_affs[3].flatten())
		ari_CD = adjusted_rand_score(pred_affs[2].flatten(), pred_affs[3].flatten())
		ari_SS = (ari_AB + ari_AC + ari_AD + ari_BC + ari_BD + ari_CD)/6
		d_SS = 1 - ari_SS
		GED = 2*d_YS - d_SS
		# print("pred_ari_avg: ", ari_SS)

		# print("GED: ", 2*d_YS - d_SS)
		return (ari_YS, d_SS, GED)

	# with open("ari/" + setup_name + ".txt", "wb") as fp:   #Pickling
	# 	pickle.dump(aris, fp)
	# print("Score calculation finished")

def __crop_center(img, crop):
	z, y,x = img[0].shape
	startz = z//2-(crop[0]//2)
	starty = y//2-(crop[1]//2)
	startx = x//2-(crop[2]//2)   
	cropped_img = np.zeros(([3] + list(crop)))
	# print(cropped_img.shape)

	for i in range(3):
		cropped_img[i] = img[i][startz:startz+crop[0], starty:starty+crop[1],startx:startx+crop[2]]
	return cropped_img

def threshold(img):
	return np.where(img > 0.5, 1, 0)

# def variation_of_information(X, Y):
# 	n = float(sum([len(x) for x in X]))
# 	sigma = 0.0
# 	for x in X:
# 		p = len(x) / n
# 		for y in Y:
# 			q = len(y) / n
# 			r = len(set(x) & set(y)) / n
# 			if r > 0.0:
# 				sigma += r * (log(r / p, 2) + log(r / q, 2))
# 		return abs(sigma)

if __name__ == "__main__":
	ari_YS = []
	d_SS = []
	GED = []
	for i in range(200):
		if i % 10 == 0:
			print('iteration: ', i)
		results = compute_scores(i, 4)
		ari_YS.append(results[0])
		d_SS.append(results[1])
		GED.append(results[2])

	data = {"ari_YS": ari_YS, "d_SS": d_SS, "GED": GED}
	with open("results/" + setup_name + ".txt", "wb") as fp:   #Pickling
		pickle.dump(data, fp)
	print("Score calculation finished")