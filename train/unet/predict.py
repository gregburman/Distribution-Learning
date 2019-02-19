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

with open(os.path.join(setup_dir, 'train_net.json'), 'r') as f:
	config = json.load(f)

def predict(iteration):

	labels_key = ArrayKey('GT_LABELS')
	joined_affinities_key = ArrayKey('GT_JOINED_AFFINITIES')
	raw_affinities_key = ArrayKey('RAW_AFFINITIES_KEY')
	raw_key = ArrayKey('RAW')
	pred_affinities_key = ArrayKey('PREDICTED_AFFS')

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate(config['input_shape']) * voxel_size
	output_shape = Coordinate(config['output_shape']) * voxel_size

	print ("input_size: ", input_shape)
	print ("output_size: ", output_shape)

	request = BatchRequest()
	# request.add(labels_key, input_shape) # TODO: why does adding this request cause a duplication of generations?
	request.add(joined_affinities_key, input_shape)
	request.add(raw_affinities_key, input_shape)
	request.add(raw_key, input_shape)
	request.add(pred_affinities_key, output_shape)

	pipeline = (
		ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			n_objects=7,
			points_per_skeleton=8,
			smoothness=3,
			interpolation="random") +
		AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=labels_key,
			affinities=raw_affinities_key) +
		AddJoinedAffinities(
			input_affinities=raw_affinities_key,
			joined_affinities=joined_affinities_key) +
		AddRealism(
			joined_affinities = joined_affinities_key,
			raw = raw_key,
			sp=0.65,
			sigma=1) +
		# Pad(raw_key, size=None) +
		# Crop(raw_key, read_roi) +
		# Normalize(raw_key) +
		IntensityScaleShift(raw_key, 2,-1) +
		Predict(
			checkpoint = os.path.join(setup_dir, 'train_net_checkpoint_%d' % iteration),
			inputs={
				config['raw']: raw_key
			},
			outputs={
				config['pred_affs']: pred_affinities_key
			},
			# graph=os.path.join(setup_dir, 'predict_net.meta')
		) +
		IntensityScaleShift(
			array=raw_key,
			scale=0.5,
			shift=0.5) +
		Snapshot(
			dataset_names={
				labels_key: 'volumes/labels/labels',
				raw_affinities_key: 'volumes/raw_affs',
				raw_key: 'volumes/raw',
				pred_affinities_key: 'volumes/pred_affs'
			},
			output_filename='unet/prediction.hdf',
			every=1,
			dataset_dtypes={
				raw_key: np.float32,
				pred_affinities_key: np.float32,
				labels_key: np.uint64
			}) + 
		PrintProfilingStats(every=20)
	)

	print("Starting prediction...")
	with build(pipeline) as p:
		p.request_batch(request)
	print("Prediction finished")

if __name__ == "__main__":
	predict(iteration=int(sys.argv[1]))