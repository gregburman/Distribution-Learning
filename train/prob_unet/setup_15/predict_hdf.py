from __future__ import print_function
import sys
sys.path.append('../../')

from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os

logging.basicConfig(level=logging.INFO)

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

setup_name = sys.argv[1]
setup_dir = 'train/prob_unet/' + setup_name + '/'

with open(setup_dir + 'predict_config.json', 'r') as f:
	config = json.load(f)

print ("net config: ", config)

def predict(checkpoint, iterations):

	print ("checkpoint: ", checkpoint)

	labels_key = ArrayKey('LABELS')
	affinities_key = ArrayKey('AFFINITIES')
	raw_key = ArrayKey('RAW')
	pred_affinities_key = ArrayKey('PREDICTED_AFFS')
	sample_z_key = ArrayKey("SAMPLE_Z")
	broadcast_key = ArrayKey("BROADCAST")
	# pred_logits_key = ArrayKey("PRED_LOGITS")
	sample_out_key = ArrayKey("SAMPLE_OUT")
	debug_key = ArrayKey("DEBUG")

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate(config['input_shape']) * voxel_size
	output_shape = Coordinate(config['output_shape']) * voxel_size
	sample_shape = Coordinate((1, 1, 6)) * voxel_size
	debug_shape = Coordinate((1, 1, 5)) * voxel_size

	print ("input_size: ", input_shape)
	print ("output_size: ", output_shape)

	request = BatchRequest()
	request.add(labels_key, output_shape)
	request.add(affinities_key, input_shape)
	request.add(raw_key, input_shape)
	request.add(pred_affinities_key, output_shape)
	request.add(broadcast_key, output_shape)
	request.add(sample_z_key, sample_shape)
	# request.add(pred_logits_key, output_shape)
	request.add(sample_out_key, sample_shape)
	request.add(debug_key, debug_shape)

	pipeline = (
		Hdf5Source(
			filename="snapshots/prob_unet/test_sample.hdf",
			datasets = {
				labels_key: 'volumes/labels',
				affinities_key: 'volumes/gt_affs',
				raw_key: 'volumes/raw'
			}) +
		# Pad(raw_key, size=None) +
		# Crop(raw_key, read_roi) +
		#Normalize(raw_key) +
		IntensityScaleShift(raw_key, 2,-1) +
		Predict(
			checkpoint = os.path.join(setup_dir, 'train_net_checkpoint_%d' % checkpoint),
			inputs={
				config['raw']: raw_key
			},
			outputs={
				config['pred_affs']: pred_affinities_key,
				config['broadcast']: broadcast_key,
				config['sample_z']: sample_z_key,
				# config['pred_logits']: pred_logits_key,
				config['sample_out']: sample_out_key,
				config['debug']: debug_key
			},
			graph=os.path.join(setup_dir, 'predict_net.meta')
		) +
		IntensityScaleShift(
			array=raw_key,
			scale=0.5,
			shift=0.5) +
		Snapshot(
			dataset_names={
				labels_key: 'volumes/labels',
				affinities_key: 'volumes/gt_affs',
				raw_key: 'volumes/raw',
				pred_affinities_key: 'volumes/pred_affs',
				broadcast_key: 'volumes/broadcast',
				sample_z_key: 'volumes/sample_z',
				# pred_logits_key: 'volumes/pred_logits',
				# sample_out_key: 'volumes/sample_out'
			},
			output_filename='prob_unet/' + setup_name + '/prediction_{id}.hdf',
			every=1,
			dataset_dtypes={
				labels_key: np.uint16,
				raw_key: np.float32,
				pred_affinities_key: np.float32,
				broadcast_key: np.float32,
				sample_z_key: np.float32,
				# pred_logits_key: np.float32
			})
		# PrintProfilingStats(every=20)
	)

	print("Starting prediction...")
	with build(pipeline) as p:
		for i in range(iterations):
			req = p.request_batch(request)
			sample_z = req[sample_z_key].data
			broadcast_sample = req[broadcast_key].data
			sample_out = req[sample_out_key].data
			print("sample_out:", sample_out)
			debug = req[debug_key].data
			print("debug", debug)

			print("sample_z: ", sample_z)
			print("Z - 0")
			print(np.unique(broadcast_sample[0, 0, :, :, :]))
			print("Z - 1")
			print(np.unique(broadcast_sample[0, 1, :, :, :]))
			print("Z - 2")
			print(np.unique(broadcast_sample[0, 2, :, :, :]))
			print("Z - 3")
			print(np.unique(broadcast_sample[0, 3, :, :, :]))
			print("Z - 4")
			print(np.unique(broadcast_sample[0, 4, :, :, :]))
			print("Z - 5")
			print(np.unique(broadcast_sample[0, 5, :, :, :]))
	print("Prediction finished")

if __name__ == "__main__":
	predict(checkpoint=int(sys.argv[2]), iterations=int(sys.argv[3]))