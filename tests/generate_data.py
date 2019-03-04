from __future__ import print_function
import sys
sys.path.append('../')

from gunpowder import *
from nodes import ToyNeuronSegmentationGenerator
from nodes import AddJoinedAffinities
from nodes import AddRealism
import matplotlib.pyplot as plt
import numpy as np
import logging

import time

# logging.getLogger('gp.AddAffinities').setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

def generate_data(num_batches):

	labels_key = ArrayKey('GT_LABELS')
	input_affinities_key= ArrayKey('AFFINITIES_IN')
	output_affinities_key= ArrayKey('AFFINITIES_OUT')
	raw_affinities_key= ArrayKey('AFFINITIES_RAW')
	joined_affinities_in_key= ArrayKey('JOINED_AFFINITIES_IN')
	joined_affinities_out_key= ArrayKey('JOINED_AFFINITIES_OUT')
	raw_in_key = ArrayKey('RAW_IN')
	raw_out_key = ArrayKey('RAW_OUT')

	voxel_size = Coordinate((1, 1, 1))
	input_size = Coordinate((132,132,132)) * voxel_size
	output_size = Coordinate((44,44,44)) * voxel_size

	print ("input_size: ", input_size)
	print ("output_size: ", output_size)

	request = BatchRequest()
	request.add(labels_key, output_size)
	# request.add(input_affinities_key, input_size)
	request.add(joined_affinities_in_key, output_size)
	request.add(joined_affinities_out_key, output_size)
	request.add(raw_affinities_key, output_size)
	request.add(raw_in_key, input_size)
	request.add(raw_out_key, output_size)
	# request.add(output_affinities_key, output_size)

	# offset = Coordinate((input_size[i]-output_size[i])/2 for i in range(len(input_size)))
	# crop_roi = Roi(offset, output_size)
	# print("crop_roi: ", crop_roi)

	# print ("input_affinities_key: ", input_affinities_key)

	pipeline = (
		ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			n_objects=20,
			points_per_skeleton=8,
			smoothness=3,
			noise_strength = 100,
			interpolation="random",
			seed = 6) + 
		AddAffinities(
			affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
			labels=labels_key,
			affinities=raw_affinities_key) +
		# GrowBoundary(labels_key, steps=1, only_xy=True) +
		# AddAffinities(
		# 	affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
		# 	labels=labels_key,
		# 	affinities=input_affinities_key) +
		# AddAffinities(
		# 	affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
		# 	labels=labels_key,
		# 	affinities=output_affinities_key) +
		# # Crop(
		# # 	key=input_affinities_key,
		# # 	roi=crop_roi) 
		AddJoinedAffinities(
			input_affinities=raw_affinities_key,
			joined_affinities=joined_affinities_in_key) +
		AddJoinedAffinities(
			input_affinities=raw_affinities_key,
			joined_affinities=joined_affinities_out_key) +
		 AddRealism(
		 	joined_affinities=joined_affinities_in_key,
		 	raw=raw_in_key,
		 	sp=0.25,
		 	sigma=1,
		 	contrast=0.7) +
		 AddRealism(
		 	joined_affinities=joined_affinities_out_key,
		 	raw=raw_out_key,
		 	sp=0.25,
		 	sigma=1,
		 	contrast=0.7) +
		 # PreCache(
			# cache_size=28,
			# num_workers=7) +
		 Snapshot(
		 	dataset_names={
		 		# raw_key: 'volumes/raw',
				labels_key: 'labels',
				# input_affinities_key: 'volumes/affinities_in',
				# joined_affinities_key: 'volumes/affinities_joined',
				# output_affinities_key: 'volumes/affinities_out',
				raw_affinities_key: 'raw_affs',
				raw_in_key: 'raw_in',
				raw_out_key: 'raw_out'
		 	},
		 	output_dir= "../snapshots/prob_unet/",
		 	output_filename="test_sample.hdf",
		 	every=1,
		 	dataset_dtypes={
		 		labels_key: np.uint16,
		 		raw_in_key: np.float32,
		 		raw_out_key: np.float32
			})

		 # PrintProfilingStats(every=1)
		)

	hashes = []
	with build(pipeline) as p:
		for i in range(num_batches):
			req = p.request_batch(request)
			label_hash = np.sum(req[labels_key].data)
			# print ("data batch generated:", i, ", label_hash:", label_hash)
			if label_hash in hashes:
				print ("DUPLICATE")
				# break
			else:
				hashes.append(label_hash)
			# print ("labels shape: ", req[labels_key].data.shape)
			# print ("affinities_in: ", req[input_affinities_key].data.shape)
			# print ("affinities_out: ", req[output_affinities_key].data.shape)
			# print ("affinities_joined: ", req[joined_affinities_key].data.shape)
			# print ("raw: ", req[raw_key].data.shape)
			# print ("raw: ", req[raw_key].data)

			# plt.imshow(req[raw_key].data[8], cmap="Greys_r")
			# plt.show()

if __name__ == "__main__":
	print("Generating data...")
	generate_data(num_batches=int(sys.argv[1]))
	print ("Data generation test finished.")