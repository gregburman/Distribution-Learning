from __future__ import print_function
import sys
sys.path.append('../')
from nodes import ToyNeuronSegmentationGenerator
from gunpowder import *
import numpy as np
import logging

# logging.getLogger('gunpowder.add_affinities').setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

shape = np.array ([50, 50, 50])  # z, y, x

def generate_affinities(num_batches):

	labels_key = ArrayKey('GT_LABELS')
	affinities_key= ArrayKey('AFFINITIES')

	voxel_size = Coordinate((1, 1, 1))
	input_size = Coordinate(s for s in shape) * voxel_size
	aff_size = Coordinate((20,20,20)) * voxel_size

	print ("input_size: ", input_size)
	# print ("aff_size: ", aff_size)

	request = BatchRequest()
	request.add(labels_key, input_size)
	request.add(affinities_key, aff_size)

	# request[affinities_key].roi = Roi((1,1,1), aff_size)

	pipeline = (
		ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			shape=input_size,
			n_objects=50,
			points_per_skeleton=5,
			smoothness=2,
			interpolation="linear") +
		AddAffinities(
			affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
			labels=labels_key,
			affinities=affinities_key)
		# RandomLocation()
		 # Snapshot(
		 # 	dataset_names={
		 # 		raw_key: 'volumes/raw',
			# 	labels_key: 'volumes/labels',
			# 	joined_affinities_key: 'volumes/affinities'
		 # 	},
		 # 	output_filename="tests/isotropic.hdf",
		 # 	every=1,
		 # 	dataset_dtypes={
		 # 		raw_key: np.uint64,
			# 	labels_key: np.uint64
			# }) +
		 # PrintProfilingStats(every=1)
		)

	with build(pipeline) as p:
		for i in range(num_batches):
			req = p.request_batch(request)
			# print ("request: ", req[labels_key].data)

	print ("Data Generation Test finished.")

if __name__ == "__main__":
	generate_affinities(num_batches=int(sys.argv[1]))