from __future__ import print_function
import sys
sys.path.append('../')

from gunpowder import *
from gunpowder.tensorflow import *
from nodes import ToyNeuronSegmentationGenerator
from nodes import AddJoinedAffinities
from nodes import AddRealism
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import json

import time

# logging.getLogger('gp.AddAffinities').setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

with open(os.path.join('tests/train_net.json'), 'r') as f:
	config = json.load(f)

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

def generate_data(num_batches):

	labels_key = ArrayKey('GT_LABELS')
	affinities_key= ArrayKey('AFFINITIES')
	joined_affinities_key= ArrayKey('JOINED_AFFINITIES')
	raw_key = ArrayKey('RAW')
	gt_affs_mask = ArrayKey('GT_AFFINITIES_MASK')
	pred_affinities_key = ArrayKey('PREDICTED_AFFS')
	debug_key = ArrayKey("DEBUG")
	gt_affs_in_key = ArrayKey('GT_AFFINITIES_IN')

	voxel_size = Coordinate((1, 1, 1))
	input_size = Coordinate((132,132,132)) * voxel_size
	output_shape = Coordinate((44,44,44)) * voxel_size
	debug_shape = Coordinate((1, 1, 1)) * voxel_size
	# output_size = Coordinate((44,44,44)) * voxel_size

	print ("input_size: ", input_size)
	# print ("output_size: ", output_size)

	request = BatchRequest()
	request.add(raw_key, input_size)
	request.add(pred_affinities_key, output_shape)
	request.add(debug_key, debug_shape)
	request.add(joined_affinities_key, input_size)
	request.add(raw_key, input_size)
	request.add(gt_affs_in_key, input_size)
	# request.add(output_affinities_key, output_size)

	# offset = Coordinate((input_size[i]-output_size[i])/2 for i in range(len(input_size)))
	# crop_roi = Roi(offset, output_size)
	# print("crop_roi: ", crop_roi)

	# print ("input_affinities_key: ", input_affinities_key)

	# seeds = [i for in range(10)]

	pipeline = ()

	pipeline += ToyNeuronSegmentationGenerator(
		array_key=labels_key,
		n_objects=50,
		points_per_skeleton=8,
		smoothness=3,
		noise_strength = 1,
		interpolation="linear") 

	pipeline +=  AddAffinities(
		affinity_neighborhood=neighborhood,
		labels=labels_key,
		affinities=gt_affs_in_key)

	pipeline +=  AddJoinedAffinities(
		input_affinities=gt_affs_in_key,
		joined_affinities=joined_affinities_key)

	pipeline +=  AddRealism(
		joined_affinities = joined_affinities_key,
		raw = raw_key,
		sp=0.25,
		sigma=1,
		contrast=0.7)

	predict = Predict(
		checkpoint = os.path.join('tests/train_net_checkpoint_1'),
		inputs={
			config['raw']: raw_key
		},
		outputs={
			config['pred_affs']: pred_affinities_key,
			config['debug']: debug_key,
		},
		# graph=os.path.join(setup_dir, 'predict_net.meta')
	)
	pipeline += predict

	hashes = []
	with build(pipeline) as p:
		for i in range(num_batches):
			print("\nDATA POINT: ", i)
			req = p.request_batch(request)
			with predict.session.as_default():
				d = predict.graph.get_tensor_by_name('debug:0')
				print(d.eval())

			# print ("labels: ", req[labels_key].data.dtype)
			# print ("affinities: ", req[affinities_key].data.dtype)
			# # print ("affinities_joined: ", req[joined_affinities_key].data.dtype)
			# print ("raw: ", req[raw_key].data.dtype)
			# print ("gt_affs_mask: ", req[gt_affs_mask].data.dtype)

			# plt.imshow(req[labels_key].data[0])
			# plt.show()

if __name__ == "__main__":
	print("Generating data...")
	t0 = time.time()
	generate_data(num_batches=int(sys.argv[1]))
	print("time: ", time.time() - t0)
	print ("Data generation test finished.")