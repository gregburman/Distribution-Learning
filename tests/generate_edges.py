from __future__ import print_function
import sys
sys.path.append('../')

from gunpowder import *
from nodes import ToyNeuronSegmentationGenerator
from nodes import AddJoinedAffinities
from nodes import AddRealism
from nodes import mergeLabels
import matplotlib.pyplot as plt
import numpy as np
import logging

import time

# logging.getLogger('gp.AddAffinities').setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

def generate_data(num_batches):

	labels_key = ArrayKey('GT_LABELS')
	affinities_pos_key= ArrayKey('AFFINITIES_POS')
	affinities_neg_key= ArrayKey('AFFINITIES_NEG')
	joined_affinities_pos_key= ArrayKey('JOINED_AFFINITIES_POS')
	joined_affinities_neg_key= ArrayKey('JOINED_AFFINITIES_NEG')
	raw_key = ArrayKey('RAW')
	gt_affs_mask = ArrayKey('GT_AFFINITIES_MASK')
	merged_labels_key = ArrayKey('MERGED_LABELS')

	voxel_size = Coordinate((1, 1, 1))
	input_size = Coordinate((132,132,132)) * voxel_size
	output_size = Coordinate((44,44,44)) * voxel_size

	print ("input_size: ", input_size)
	print ("output_size: ", output_size)

	request = BatchRequest()
	request.add(labels_key, input_size)
	request.add(affinities_pos_key, input_size)
	request.add(affinities_neg_key, input_size)
	# request.add(gt_affs_mask, input_size)
	request.add(joined_affinities_pos_key, input_size)
	request.add(joined_affinities_neg_key, input_size)
	request.add(merged_labels_key, input_size)
	# request.add(raw_key, input_size)
	# request.add(output_affinities_key, output_size)

	offset = Coordinate((input_size[i]-output_size[i])/2 for i in range(len(input_size)))
	crop_roi = Roi(offset, output_size)
	# print("crop_roi: ", crop_roi)

	# print ("input_affinities_key: ", input_affinities_key)

	pipeline = (
		ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			n_objects=15,
			points_per_skeleton=5,
			smoothness=3,
			noise_strength = 1,
			interpolation="random",
			seed=0) + 
		AddAffinities(
			affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
			labels=labels_key,
			affinities=affinities_neg_key) +
		AddAffinities(
			affinity_neighborhood=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
			labels=labels_key,
			affinities=affinities_pos_key) +
		# GrowBoundary(labels_key, steps=1, only_xy=True) +
		# AddAffinities(
		# 	affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
		# 	labels=labels_key,
		# 	affinities=input_affinities_key) +
		# AddAffinities(
		# 	affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
		# 	labels=labels_key,
		# 	affinities=output_affinities_key) +
		# Crop(
		# 	key=labels_key,
		# 	roi=crop_roi) +
		AddJoinedAffinities(
			input_affinities=affinities_pos_key,
			joined_affinities=joined_affinities_pos_key) +
		AddJoinedAffinities(
			input_affinities=affinities_neg_key,
			joined_affinities=joined_affinities_neg_key) +
		mergeLabels(
			n_objects = 15,
			labels = labels_key,
			joined_pos_affinities = joined_affinities_pos_key,
			joined_neg_affinities = joined_affinities_neg_key,
			merged_labels = merged_labels_key) + 
		#  AddRealism(
		#  	joined_affinities=joined_affinities_key,
		#  	raw=raw_key,
		#  	sp=0.25,
		#  	sigma=1,
		#  	contrast=0.7) +
		#  # PreCache(
		# 	# cache_size=32,
		# 	# num_workers=8) +
		 Snapshot(
		 	dataset_names={
				labels_key: 'volumes/labels',
				# joined_affinities_pos_key: 'volumes/affinities_neg',
				# joined_affinities_neg_key: 'volumes/affinities_pos',
				# joined_affinities_key: 'volumes/joined_affs',
				merged_labels_key: 'volumes/merged_labels',
				# raw_key: 'volumes/raw',
				# gt_affs_mask: 'volumes/affs_mask'
		 	},
		 	output_filename="test_edges.hdf",
		 	every=1,
		 	dataset_dtypes={
		 		labels_key: np.uint16,
		 		merged_labels_key: np.uint16,
		 		raw_key: np.float32,
			})
		 # PrintProfilingStats(every=8)
		)

	hashes = []
	with build(pipeline) as p:
		for i in range(num_batches):
			req = p.request_batch(request)
			label_hash = np.sum(req[labels_key].data)
			print ("data batch generated:", i, ", label_hash:", label_hash)
			if label_hash in hashes:
				print ("DUPLICATE")
				# break
			else:
				hashes.append(label_hash)

			# print ("labels: ", req[labels_key].data.dtype)
			# print ("affinities: ", req[affinities_key].data.dtype)
			# # print ("affinities_joined: ", req[joined_affinities_key].data.dtype)
			# print ("raw: ", req[raw_key].data.dtype)
			# print ("gt_affs_mask: ", req[gt_affs_mask].data.dtype)

			# plt.imshow(req[labels_key].data[0])
			# plt.show()

if __name__ == "__main__":
	print("Generating data...")
	generate_data(num_batches=int(sys.argv[1]))
	print ("Data generation test finished.")