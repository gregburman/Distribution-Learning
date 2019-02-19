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

# logging.getLogger('gp.AddAffinities').setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

def generate_data(num_batches):

	labels_key = ArrayKey('GT_LABELS')
	input_affinities_key= ArrayKey('AFFINITIES_IN')
	output_affinities_key= ArrayKey('AFFINITIES_OUT')
	raw_affinities_key= ArrayKey('AFFINITIES_RAW')
	joined_affinities_key= ArrayKey('JOINED_AFFINITIES')
	raw_key = ArrayKey('RAW')

	voxel_size = Coordinate((1, 1, 1))
	input_size = Coordinate((132,132,132)) * voxel_size
	output_size = Coordinate((44,44,44)) * voxel_size

	print ("input_size: ", input_size)
	print ("output_size: ", output_size)

	request = BatchRequest()
	request.add(labels_key, input_size)
	request.add(input_affinities_key, input_size)
	request.add(joined_affinities_key, input_size)
	request.add(raw_affinities_key, input_size)
	request.add(raw_key, input_size)
	request.add(output_affinities_key, output_size)

	# offset = Coordinate((input_size[i]-output_size[i])/2 for i in range(len(input_size)))
	# crop_roi = Roi(offset, output_size)
	# print("crop_roi: ", crop_roi)

	# print ("input_affinities_key: ", input_affinities_key)

	pipeline = (
		ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			n_objects=50,
			points_per_skeleton=8,
			smoothness=3,
			interpolation="random") +
		AddAffinities(
			affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
			labels=labels_key,
			affinities=raw_affinities_key) +
		GrowBoundary(labels_key, steps=1, only_xy=True) +
		AddAffinities(
			affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
			labels=labels_key,
			affinities=input_affinities_key) +
		AddAffinities(
			affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
			labels=labels_key,
			affinities=output_affinities_key) +
		# Crop(
		# 	key=input_affinities_key,
		# 	roi=crop_roi) 
		AddJoinedAffinities(
			input_affinities=raw_affinities_key,
			joined_affinities=joined_affinities_key) +
		 AddRealism(
		 	joined_affinities=joined_affinities_key,
		 	raw=raw_key,
		 	sp=0.4,
		 	sigma=1) +
		 Snapshot(
		 	dataset_names={
		 	# 	raw_key: 'volumes/raw',
				labels_key: 'volumes/labels',
				# input_affinities_key: 'volumes/affinities_in',
				# joined_affinities_key: 'volumes/affinities_joined',
				# output_affinities_key: 'volumes/affinities_out',
				raw_affinities_key: 'volumes/affinities_raw',
				raw_key: 'volumes/raw'
		 	},
		 	output_filename="data_{id}.hdf",
		 	every=1,
		 	dataset_dtypes={
		 		raw_key: np.float32,
				labels_key: np.uint64
			}) +
		 PrintProfilingStats(every=1)
		)

	hashes = []
	with build(pipeline) as p:
		for i in range(num_batches):
			req = p.request_batch(request)
			# label_hash = np.sum(req[labels_key].data)
			# print ("data batch generated:", i, ", label_hash:", label_hash)
			# if label_hash in hashes:
			# 	print ("DUPLICATE")
			# else:
			# 	hashes.append(label_hash)
			# break
			# print ("labels shape: ", req[labels_key].data.shape)
			# print ("affinities_in: ", req[input_affinities_key].data.shape)
			# print ("affinities_out: ", req[output_affinities_key].data.shape)
			# print ("affinities_joined: ", req[joined_affinities_key].data.shape)
			# print ("raw: ", req[raw_key].data.shape)
			# # print ("raw: ", req[raw_key].data)

			# plt.imshow(req[raw_key].data[8], cmap="Greys_r")
			# plt.show()

if __name__ == "__main__":
	print("Generating data...")
	generate_data(num_batches=int(sys.argv[1]))
	print ("Data generation test finished.")