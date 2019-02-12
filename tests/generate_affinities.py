from __future__ import print_function
import sys
sys.path.append('../')
from nodes import ToyNeuronSegmentationGenerator
from gunpowder import *
import numpy as np
import logging

# logging.getLogger('gunpowder.add_affinities').setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

# shape = np.array ([50, 50, 50])  # z, y, x

def generate_affinities(num_batches):

	labels_key = ArrayKey('GT_LABELS')
	input_affinities_key= ArrayKey('AFFINITIES_IN')
	output_affinities_key= ArrayKey('AFFINITIES_OUT')

	voxel_size = Coordinate((1, 1, 1))
	input_affinities_size = Coordinate((132,132,132)) * voxel_size
	output_affinities_size = Coordinate((44,44,44)) * voxel_size

	print ("input_affinities_size: ", input_affinities_size)
	print ("output_affinities_size: ", output_affinities_size)

	request = BatchRequest()
	request.add(input_affinities_key, input_affinities_size)
	request.add(output_affinities_key, output_affinities_size)

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
		# AddAffinities(
		# 	affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
		# 	labels=labels_key,
		# 	affinities=output_affinities_key) +
		 # Snapshot(
		 # 	dataset_names={
			# 	labels_key: 'volumes/labels',
			# 	input_affinities_key: 'volumes/affinities_in',
			# 	output_affinities_key: 'volumes/affinities_out'
		 # 	},
		 # 	output_filename="affinities.hdf",
		 # 	every=1,
		 # 	dataset_dtypes={
			# 	labels_key: np.uint64
			# }) +
		 PrintProfilingStats(every=1)
		)

	with build(pipeline) as p:
		for i in range(num_batches):
			req = p.request_batch(request)
			print ("affs: dtype", req[input_affinities_key].data.dtype)

if __name__ == "__main__":
	print ("Generating affinities...")
	generate_affinities(num_batches=int(sys.argv[1]))
	print ("Affinities generation test finished.")