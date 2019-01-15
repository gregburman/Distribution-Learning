from __future__ import print_function
import sys
sys.path.append('../')
from gunpowder import *
from gunpowder.tensorflow import *
import malis
import os
import math
import json
import tensorflow as tf
import numpy as np
import logging

from nodes import ToyNeuronSegmentationGenerator
from nodes import AddJoinedAffinities
from nodes import AddRealism


# logging.basicConfig(level=logging.DEBUG)

shape = np.array ([200, 200, 200])  # z, y, x
neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

def train_until(max_iteration):

	if tf.train.latest_checkpoint('.'):
		trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
	else:
		trained_until = 0
	if trained_until >= max_iteration:
		return

	with open('train/train_net.json', 'r') as f:
		config = json.load(f)

	# define array-keys
	raw = ArrayKey('RAW')
	labels = ArrayKey('GT_LABELS')
	input_affinities = ArrayKey('GT_AFFINITIES')
	joined_affinities = ArrayKey('GT_JOINED_AFFINITIES')
	input_affinities_scale = ArrayKey('GT_AFFINITIES_SCALE')
	pred_affinities = ArrayKey('PREDICTED_AFFS')
	pred_affinities_gradient = ArrayKey('AFFS_GRADIENT')

	voxel_size = Coordinate((1, 1, 1))
	input_size = Coordinate(s+1 for s in config['input_shape']) * voxel_size
	aff_size = Coordinate(config['input_shape']) * voxel_size
	output_size = Coordinate(config['output_shape']) * voxel_size

	print ("input_size: ", input_size)
	print("aff_size",  aff_size)
	print ("output_size: ", output_size)

	# define requests
	request = BatchRequest()
	request.add(raw, aff_size)
	request.add(labels, input_size)
	request.add(input_affinities, output_size)
	request.add(joined_affinities, output_size)
	request.add(input_affinities_scale, output_size)
	request.add(pred_affinities, output_size)

	offset = Coordinate((input_size[i]-output_size[i])/2 for i in range(len(input_size)))
	crop_roi = Roi(offset, output_size)
	print("crop_roi: ", crop_roi)

	train_pipeline = (
		ToyNeuronSegmentationGenerator(
			shape=shape,
			n_objects=50,
			points_per_skeleton=5,
			smoothness=2,
			interpolation="linear") + 
		# ElasticAugment(
		# 	control_point_spacing=[4,40,40],
		# 	jitter_sigma=[0,2,2],
		# 	rotation_interval=[0,math.pi/2.0],
		# 	prob_slip=0.05,
		# 	prob_shift=0.05,
		# 	max_misalign=10,
		# 	subsample=8) +
		# SimpleAugment(transpose_only=[1, 2]) +
		# IntensityAugment(labels, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
		# GrowBoundary(labels, labels_mask, steps=1, only_xy=True) +
        AddAffinities(
            neighborhood,
            labels=labels,
            affinities=input_affinities) +
		AddJoinedAffinities(
			input_affinities=input_affinities,
			joined_affinities=joined_affinities) +
		AddRealism(
			affinities = joined_affinities,
			realistic_data = raw,
			sp=0.25,
			sigma=1) +
		BalanceLabels(
			input_affinities,
			input_affinities_scale) +
		DefectAugment(
			raw,
			prob_missing=0.03,
			prob_low_contrast=0.01,
			contrast_scale=0.5,
			axis=0) +
		IntensityScaleShift(raw, 2,-1) +
		PreCache(
			cache_size=40,
			num_workers=10) +
		Crop(
			key=joined_affinities,
			roi=crop_roi) +
		Train(
			'train/train_net',
			optimizer=config['optimizer'],
			loss=config['loss'],
			inputs={
				config['raw']: raw,
				config['gt_affs']: input_affinities,
				config['affs_loss_weights']: input_affinities_scale
			},
			outputs={
				config['affs']: pred_affinities
			},
			gradients={
				config['affs']: pred_affinities_gradient
			},
			summary=config['summary'],
			log_dir='log',
			save_every=10000) +
		IntensityScaleShift(raw, 0.5, 0.5) +
		Snapshot({
				raw: 'volumes/raw',
				labels: 'volumes/labels/labels',
				joined_affinities: 'volumes/joined_affinities',
				pred_affinities: 'volumes/pred_affinities',
				# gt_mask: 'volumes/labels/gt_mask',
				# labels_mask: 'volumes/labels/mask',
				pred_affinities_gradient: 'volumes/pred_affinities_gradient'
			},
			dataset_dtypes={
				labels: np.uint64
			},
			every=100,
			output_filename='batch_{iteration}.hdf') + 
		PrintProfilingStats(every=10)
	)

	print("Starting training...")
	with build(train_pipeline) as b:
		for i in range(max_iteration - trained_until):
			req = b.request_batch(request)
			print ("iteration: ", i)
	print("Training finished")


if __name__ == "__main__":
	iteration = int(sys.argv[1])
	train_until(iteration)