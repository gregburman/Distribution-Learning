from __future__ import print_function
import sys
sys.path.append('../')

from gunpowder import *
from nodes import ToyNeuronSegmentationGenerator
from nodes import AddJoinedAffinities
from nodes import AddRealism
from nodes import MergeLabels
import matplotlib.pyplot as plt
import numpy as np
import logging

import time

# logging.getLogger('gp.AddAffinities').setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

def generate_data(num_batches):

	neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
	neighborhood_opp = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# define array-keys
	labels_key = ArrayKey('LABELS')
	
	raw_affs_key = ArrayKey('RAW_AFFINITIES')
	raw_joined_affs_key = ArrayKey('RAW_JOINED_AFFINITIES')
	raw_key = ArrayKey('RAW')

	affs_key = ArrayKey('AFFINITIES')
	affs_opp_key = ArrayKey('AFFINITIES_OPP')
	joined_affs_key = ArrayKey('JOINED_AFFINITIES')
	joined_affs_opp_key = ArrayKey('JOINED_AFFINITIES_OPP')
	merged_labels_key = ArrayKey('MERGED_LABELS')

	gt_affs_key = ArrayKey('GT_AFFINITIES')
	gt_affs_in_key = ArrayKey('GT_AFFINITIES_IN')
	gt_affs_mask_key = ArrayKey('GT_AFFINITIES_MASK')

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate((132,132,132)) * voxel_size
	input_affs_shape = Coordinate([i + 1 for i in (132,132,132)]) * voxel_size
	output_shape = Coordinate((44,44,44)) * voxel_size
	output_affs_shape = Coordinate([i + 1 for i in (44,44,44)]) * voxel_size

	print ("input_shape: ", input_shape)
	print ("output_shape: ", output_shape)

	request = BatchRequest()
	request.add(labels_key, output_shape)

	request.add(raw_key, input_shape)
	request.add(raw_affs_key, input_shape)
	request.add(raw_joined_affs_key, input_shape)

	request.add(affs_key, input_affs_shape)
	request.add(affs_opp_key, input_affs_shape)
	request.add(joined_affs_key, input_affs_shape)
	request.add(joined_affs_opp_key, input_affs_shape)
	request.add(merged_labels_key, output_shape)

	request.add(gt_affs_key, output_shape)
	request.add(gt_affs_in_key, input_shape)
	request.add(gt_affs_mask_key, output_shape)

	# offset = Coordinate((input_size[i]-output_size[i])/2 for i in range(len(input_size)))
	# crop_roi = Roi(offset, output_size)
	# print("crop_roi: ", crop_roi)

	# print ("input_affinities_key: ", input_affinities_key)

	pipeline = ()
	# print ("iteration: ", iteration)
	pipeline += ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			n_objects=15,
			points_per_skeleton=8,
			smoothness=3,
			noise_strength=1,
			interpolation="linear")

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=labels_key,
			affinities=raw_affs_key)

	pipeline += AddJoinedAffinities(
			input_affinities=raw_affs_key,
			joined_affinities=raw_joined_affs_key)

	pipeline += AddRealism(
			joined_affinities = raw_joined_affs_key,
			raw = raw_key,
			sp=0.25,
			sigma=1,
			contrast=0.7)

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=labels_key,
			affinities=affs_key)

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood_opp,
			labels=labels_key,
			affinities=affs_opp_key)

	pipeline += AddJoinedAffinities(
			input_affinities=affs_key,
			joined_affinities=joined_affs_key)

	pipeline += AddJoinedAffinities(
			input_affinities=affs_opp_key,
			joined_affinities=joined_affs_opp_key)

	pipeline += MergeLabels(
			labels = labels_key,
			joined_affinities = joined_affs_key,
			joined_affinities_opp = joined_affs_opp_key,
			merged_labels = merged_labels_key,
			every = 1) 

	# # pipeline += GrowBoundary(merged_labels_key, steps=1, only_xy=True)

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=merged_labels_key,
			affinities=gt_affs_in_key)

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=merged_labels_key,
			affinities=gt_affs_key,
			affinities_mask=gt_affs_mask_key)



	# pipeline += Snapshot(
	# 		dataset_names={
	# 			labels_key: 'volumes/labels',
	# 			merged_labels_key: 'volumes/merged_labels',
	# 			raw_key: 'volumes/raw',
	# 			raw_affs_key: 'volumes/raw_affs',
	# 			gt_affs_key: 'volumes/gt_affs',
	# 			gt_affs_in_key: 'volumes/gt_affs_in'
	# 		},
	# 		output_filename='test_edges.hdf',
	# 		every=1,
	# 		dataset_dtypes={
	# 			merged_labels_key: np.uint64,
	# 			labels_key: np.uint64,
	# 			raw_key: np.float32
			# })

	pipeline += PrintProfilingStats(every=10)

	hashes = []
	with build(pipeline) as p:
		for i in range(num_batches):
			# print("iteration: ", i)
			req = p.request_batch(request)
			# merged_labels = req[merged_labels_key].data
			# unique = np.unique(merged_labels)
			# print (len(unique))

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