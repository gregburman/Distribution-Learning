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
	input_affinities_key= ArrayKey('AFFINITIES')
	joined_affinities_key= ArrayKey('JOINED_AFFINITIES')
	raw_key = ArrayKey('RAW')

	voxel_size = Coordinate((1, 1, 1))
	label_size = Coordinate((200,200,200)) * voxel_size
	aff_size = Coordinate((200,200,200)) * voxel_size
	raw_size = Coordinate((200,200,200)) * voxel_size

	print ("label_size: ", label_size)
	print ("aff_size: ", aff_size)
	print ("raw_size: ", raw_size)

	request = BatchRequest()
	request.add(labels_key, label_size)
	request.add(input_affinities_key, aff_size)
	request.add(joined_affinities_key, aff_size)
	request.add(raw_key, raw_size)

	pipeline = (
		ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			n_objects=50,
			points_per_skeleton=5,
			smoothness=2,
			interpolation="linear") +
		AddAffinities(
			affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
			labels=labels_key,
			affinities=input_affinities_key) +
		AddJoinedAffinities(
			input_affinities=input_affinities_key,
			joined_affinities=joined_affinities_key) +
		 AddRealism(
		 	joined_affinities=joined_affinities_key,
		 	raw=raw_key,
		 	sp=0.25,
		 	sigma=1) +
		 Snapshot(
		 	dataset_names={
		 		raw_key: 'volumes/raw',
				labels_key: 'volumes/labels',
				joined_affinities_key: 'volumes/affinities'
		 	},
		 	output_filename="data.hdf",
		 	every=1,
		 	dataset_dtypes={
		 		raw_key: np.uint64,
				labels_key: np.uint64
			}) +
		 PrintProfilingStats(every=1)
		)

	with build(pipeline) as p:
		for i in range(num_batches):
			req = p.request_batch(request)

	print ("Data Generation Test finished.")

if __name__ == "__main__":
	generate_data(num_batches=int(sys.argv[1]))