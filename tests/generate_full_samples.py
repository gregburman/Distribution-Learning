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

def generate_full_samples(num_batches):

	neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
	neighborhood_opp = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

	# define array-keys
	labels_key = ArrayKey('LABELS')
	gt_affs_key = ArrayKey('RAW_AFFINITIES')
	gt_joined_affs_key = ArrayKey('RAW_JOINED_AFFINITIES')
	raw_key = ArrayKey('RAW')

	merged_labels_key = []
	merged_affs_key = []
	
	affs_neg_key = ArrayKey('AFFINITIES')
	affs_pos_key = ArrayKey('AFFINITIES_OPP')
	joined_affs_neg_key = ArrayKey('JOINED_AFFINITIES')
	joined_affs_pos_key = ArrayKey('JOINED_AFFINITIES_OPP')

	num_merges = 3
	for i in range(num_merges):
		merged_labels_key.append(ArrayKey('MERGED_LABELS_%i'%(i+1)))
		merged_affs_key.append(ArrayKey('MERGED_AFFINITIES_IN_%i'%(i+1)))

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate((132,132,132)) * voxel_size
	input_affs_shape = Coordinate([i + 1 for i in (132,132,132)]) * voxel_size
	output_shape = Coordinate((44,44,44)) * voxel_size
	output_affs_shape = Coordinate([i + 1 for i in (44,44,44)]) * voxel_size

	print ("input_shape: ", input_shape)
	print ("output_shape: ", output_shape)

	request = BatchRequest()
	request.add(labels_key, input_shape)
	
	# request.add(gt_affs_key, input_shape)
	# request.add(gt_joined_affs_key, input_shape)
	# request.add(raw_key, input_shape)


	request.add(affs_neg_key, input_affs_shape)
	request.add(affs_pos_key, input_affs_shape)
	request.add(joined_affs_neg_key, input_affs_shape)
	request.add(joined_affs_pos_key, input_affs_shape)

	for i in range(num_merges):
		request.add(merged_labels_key[i], input_shape)
		request.add(merged_affs_key[i], input_shape)

	pipeline = ()

	pipeline += ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			n_objects=50,
			points_per_skeleton=8,
			smoothness=3,
			noise_strength=1,
			interpolation="linear")

	# pipeline += AddAffinities(
	# 		affinity_neighborhood=neighborhood,
	# 		labels=labels_key,
	# 		affinities=gt_affs_key)

	# pipeline += AddJoinedAffinities(
	# 		input_affinities=gt_affs_key,
	# 		joined_affinities=gt_joined_affs_key)

	# pipeline += AddRealism(
	# 		joined_affinities = gt_joined_affs_key,
	# 		raw = raw_key,
	# 		sp=0.25,
	# 		sigma=1,
	# 		contrast=0.7)

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=labels_key,
			affinities=affs_neg_key)

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood_opp,
			labels=labels_key,
			affinities=affs_pos_key)

	pipeline += AddJoinedAffinities(
			input_affinities=affs_neg_key,
			joined_affinities=joined_affs_neg_key)

	pipeline += AddJoinedAffinities(
			input_affinities=affs_pos_key,
			joined_affinities=joined_affs_pos_key)

	for i in range(num_merges):

		pipeline += MergeLabels(
				labels = labels_key,
				joined_affinities = joined_affs_neg_key,
				joined_affinities_opp = joined_affs_pos_key,
				merged_labels = merged_labels_key[i],
				cropped_roi = output_shape,
				every = 1) 

		pipeline += AddAffinities(
				affinity_neighborhood=neighborhood,
				labels=merged_labels_key[i],
				affinities=merged_affs_key[i])

	pipeline += PreCache(
		cache_size=32,
		num_workers=8)

	dataset_names = {
		labels_key: 'volumes/labels',
	}

	dataset_dtypes = {
		labels_key: np.uint64,
	}

	for i in range(num_merges):
		dataset_names[merged_labels_key[i]] = 'volumes/merged_labels_%i'%(i+1)
		dataset_dtypes[merged_labels_key[i]] = np.uint64

	pipeline += Snapshot(
		dataset_names=dataset_names,
		output_filename='gt_1_merge_3_cropped/batch_{id}.hdf',
		every=1,
		dataset_dtypes=dataset_dtypes)

	# pipeline += PrintProfilingStats(every=10)


	hashes = []
	with build(pipeline) as p:
		for i in range(num_batches):
			print("\nDATA POINT: ", i)
			req = p.request_batch(request)
			# print ("labels shape: ", req[labels_key].shape)
			label_hash = np.sum(req[labels_key].data)
			print ("label_hash:", label_hash)
			if label_hash in hashes:
				print ("DUPLICATE")
				# break
			else:
				hashes.append(label_hash)

if __name__ == "__main__":
	print("Generating data...")
	generate_full_samples(num_batches=int(sys.argv[1]))
	print ("Data generation test finished.")