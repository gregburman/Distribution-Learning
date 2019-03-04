from __future__ import print_function
import sys
sys.path.append('../')

from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os

from nodes import ToyNeuronSegmentationGenerator
from nodes import AddJoinedAffinities
from nodes import AddRealism

logging.basicConfig(level=logging.INFO)

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'predict_net.json'), 'r') as f:
	config = json.load(f)

print ("net config: ", config)

def predict(checkpoint, iterations):

	labels_key = ArrayKey('GT_LABELS')
	joined_affinities_key = ArrayKey('GT_JOINED_AFFINITIES')
	raw_affinities_key = ArrayKey('RAW_AFFINITIES_KEY')
	raw_key = ArrayKey('RAW')
	pred_affinities_key = ArrayKey('PREDICTED_AFFS')
	sample_z_key = ArrayKey("SAMPLE_Z")

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate(config['input_shape']) * voxel_size
	output_shape = Coordinate(config['output_shape']) * voxel_size
	sample_shape = Coordinate((1, 1, 6)) * voxel_size

	print ("input_size: ", input_shape)
	print ("output_size: ", output_shape)

	request = BatchRequest()
	# request.add(labels_key, input_shape)
	# # request.add(joined_affinities_key, input_shape)
	# request.add(raw_affinities_key, input_shape)
	request.add(raw_key, input_shape)
	request.add(pred_affinities_key, output_shape)
	request.add(sample_z_key, sample_shape)

	pipeline = (
		Hdf5Source(
			filename="snapshots/prob_unet/test_sample.hdf",
			datasets = {
				# labels_key: 'labels',
				# raw_affinities_key: 'raw_affs',
				raw_key: 'raw_in'
			}) +
		# Pad(raw_key, size=None) +
		# Crop(raw_key, read_roi) +
		# Normalize(raw_key) +
		# IntensityScaleShift(raw_key, 2,-1) +
		Predict(
			checkpoint = os.path.join(setup_dir, 'train_net_checkpoint_%d' % checkpoint),
			inputs={
				config['raw']: raw_key
			},
			outputs={
				config['pred_affs']: pred_affinities_key,
				config['sample_z']: sample_z_key
			},
			graph=os.path.join(setup_dir, 'predict_net.meta')
		) +
		# IntensityScaleShift(
		# 	array=raw_key,
		# 	scale=0.5,
		# 	shift=0.5) +
		Snapshot(
			dataset_names={
				# labels_key: 'volumes/labels',
				# raw_affinities_key: 'volumes/raw_affs',
				# raw_key: 'volumes/raw',
				pred_affinities_key: 'pred_affs',
				sample_z_key: 'sample_z',
			},
			output_filename='prob_unet/prediction_{id}.hdf',
			every=1,
			dataset_dtypes={
				# raw_key: np.float32,
				pred_affinities_key: np.float32,
				sample_z_key: np.float32,
				# labels_key: np.uint64
			})
		# PrintProfilingStats(every=20)
	)

	print("Starting prediction...")
	with build(pipeline) as p:
		for i in range(iterations):
			p.request_batch(request)
	print("Prediction finished")

if __name__ == "__main__":
	predict(iterations=int(sys.argv[1]), checkpoint=int(sys.argv[2]))