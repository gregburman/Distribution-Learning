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

logging.basicConfig(level=logging.INFO)

data_dir = "../snapshots/prob_unet/setup_24b"
samples = ["prediction_%08i"%i for i in range(500)]

def compute_scores(iterations):

	labels_key = ArrayKey('LABELS')
	gt_affs_key = ArrayKey('GT_AFFINITIES')
	pred_affinities_key = ArrayKey('PREDICTED_AFFS')
	sample_z_key = ArrayKey("SAMPLE_Z")

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate((132, 132, 132)) * voxel_size
	output_shape = Coordinate((44, 44, 44)) * voxel_size
	sample_shape = Coordinate((1, 1, 6)) * voxel_size

	print ("input_size: ", input_shape)
	print ("output_size: ", output_shape)

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



	print("Calculating Scores...")
	with build(pipeline) as p:
		aris = []
		for i in range(iterations):
			if i % 10 == 0:
				print("iteration: ", i)
			req = p.request_batch(request)
			gt_affs = np.array(req[gt_affs_key].data)
			pred_affs = threshold(__crop_center(np.array(req[pred_affinities_key].data), (44,44,44)))

			aris.append(adjusted_rand_score(gt_affs.flatten(), pred_affs.flatten()))

		# print ("aris: ", aris)
		aris = np.array(aris)
		maximum = np.max(aris)

		minimum = np.min(aris)
		std = np.std(aris)
		mean = np.mean(aris)
		upper_std = mean + std
		lower_std = mean - std
		print ("maximum: ", maximum)
		print ("minimum: ", minimum)
		print ("mean: ", mean)
		print ("std: ", std)
		print ("upper_std: ", upper_std)
		print ("lower_std: ", lower_std)

	with open("ari/24b.txt", "wb") as fp:   #Pickling
		pickle.dump(aris, fp)
	print("Score calculation finished")

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



if __name__ == "__main__":
	compute_scores(iterations=int(sys.argv[1]))