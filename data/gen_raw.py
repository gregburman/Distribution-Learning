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
import os

import time

# logging.getLogger('gp.AddAffinities').setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

data_dir = "../data/snapshots/results/data_gen"

def generate_data(num_batches):

	labels_key = ArrayKey('LABELS')
	gt_affs_key= ArrayKey('GT_AFFINITIES')
	joined_affs_key= ArrayKey('JOINED_AFFINITIES')
	raw_key1 = ArrayKey('RAW1')
	raw_key2 = ArrayKey('RAW2')
	raw_key3 = ArrayKey('RAW3')

	voxel_size = Coordinate((1, 1, 1))
	input_size = Coordinate((133,133,133)) * voxel_size
	affs_size = Coordinate((131,131,131)) * voxel_size
	output_size = Coordinate((44,44,44)) * voxel_size

	print ("input_size: ", input_size)
	print ("output_size: ", output_size)

	request = BatchRequest()
	request.add(labels_key, input_size)
	request.add(gt_affs_key, affs_size)
	request.add(joined_affs_key, affs_size)
	request.add(raw_key1, affs_size)

	pipeline = (
		Hdf5Source(
            os.path.join(data_dir, 'seg_standard.hdf'),
            datasets = {labels_key: "volumes/labels"},
            array_specs = {labels_key: ArraySpec(interpolatable=False)}
        ) +
        Pad(labels_key, None) +
		AddAffinities(
			affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
			labels=labels_key,
			affinities=gt_affs_key) +
		AddJoinedAffinities(
			input_affinities=gt_affs_key,
			joined_affinities=joined_affs_key) +
		 AddRealism(
		 	joined_affinities=joined_affs_key,
		 	raw=raw_key1,
		 	sp=0.25,
		 	sigma=1,
		 	contrast=0.7) +
		 Snapshot(
		 	dataset_names={
				raw_key1: 'volumes/raw',
				# gt_affs_key: 'volumes/gt_affs',
				# joined_affs_key: 'volumes/joined_affs',
				# raw_key1: 'volumes/raw1',
				# raw_key2: 'volumes/raw2',
				# raw_key3: 'volumes/raw3',
		 	},
		 	output_filename="results/data_gen/raw_synth/contrast_07.hdf",
		 	every=1,
		 	dataset_dtypes={
		 		# labels_key: np.uint64,
		 		raw_key1: np.float32,
		 		# raw_key2: np.float32,
		 		# raw_key3: np.float32,
		 		# gt_affs_key: np.float32,
		 		# joined_affs_key: np.float32
			})
		)

	hashes = []
	with build(pipeline) as p:
		for i in range(num_batches):
			print("\nDATA POINT: ", i)
			req = p.request_batch(request)
			labels = req[labels_key].data
			hashes = np.sum(labels)
			print(hashes)

if __name__ == "__main__":
	print("Generating data...")
	t0 = time.time()
	generate_data(num_batches=int(sys.argv[1]))
	print("time: ", time.time() - t0)
	print ("Data generation test finished.")