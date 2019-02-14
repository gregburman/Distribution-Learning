from __future__ import print_function
import sys
sys.path.append('../')

from gunpowder import *
from gunpowder.tensorflow import *
import os
import math
import json
import tensorflow as tf
import numpy as np
import logging

from nodes import ToyNeuronSegmentationGenerator
from nodes import AddJoinedAffinities
from nodes import AddRealism

logging.basicConfig(level=logging.INFO)

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

def train(iterations):

	if tf.train.latest_checkpoint('.'):
		trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
	else:
		trained_until = 0
	if trained_until >= iterations:
		return

	with open('train/train_net.json', 'r') as f:
		config = json.load(f)

	# define array-keys
	
	labels_key = ArrayKey('GT_LABELS')
	input_affinities_key = ArrayKey('GT_AFFINITIES_IN')
	output_affinities_key = ArrayKey('GT_AFFINITIES_OUT')
	joined_affinities_key = ArrayKey('GT_JOINED_AFFINITIES')
	raw_affinities_key = ArrayKey('RAW_AFFINITIES_KEY')
	raw_key = ArrayKey('RAW')
	input_affinities_scale_key = ArrayKey('GT_AFFINITIES_SCALE')
	pred_affinities_key = ArrayKey('PREDICTED_AFFS')
	pred_affinities_gradient_key = ArrayKey('AFFS_GRADIENT')

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate(config['input_shape']) * voxel_size
	output_shape = Coordinate(config['output_shape']) * voxel_size

	print ("input_shape: ", input_shape)
	print ("output_shape: ", output_shape)

	# define requests
	request = BatchRequest()
	# request.add(labels_key, input_shape) # TODO: why does adding this request cause a duplication of generations?
	request.add(input_affinities_key, input_shape)
	request.add(joined_affinities_key, input_shape)
	request.add(raw_affinities_key, input_shape)
	request.add(raw_key, input_shape)
	request.add(output_affinities_key, output_shape)
	request.add(input_affinities_scale_key, output_shape)
	request.add(pred_affinities_key, output_shape)

	# offset = Coordinate((input_shape[i]-output_shape[i])/2 for i in range(len(input_shape)))
	# crop_roi = Roi(offset, output_shape)
	# print("crop_roi: ", crop_roi)

	pipeline = (
		ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			n_objects=50,
			points_per_skeleton=8,
			smoothness=3,
			interpolation="random") + 
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
		AddAffinities(
            affinity_neighborhood=neighborhood,
            labels=labels_key,
            affinities=raw_affinities_key) +
		GrowBoundary(labels_key, steps=1, only_xy=True) +
        AddAffinities(
            affinity_neighborhood=neighborhood,
            labels=labels_key,
            affinities=input_affinities_key) +
        AddAffinities(
            affinity_neighborhood=neighborhood,
            labels=labels_key,
            affinities=output_affinities_key) +
		AddJoinedAffinities(
			input_affinities=raw_affinities_key,
			joined_affinities=joined_affinities_key) +
		AddRealism(
			joined_affinities = joined_affinities_key,
			raw = raw_key,
			sp=0.65,
			sigma=1) +
		BalanceLabels(
			labels=output_affinities_key,
			scales=input_affinities_scale_key) +
		DefectAugment(
			intensities=raw_key,
			prob_missing=0.03,
			prob_low_contrast=0.01,
			contrast_scale=0.5,
			axis=0) +
		IntensityScaleShift(raw_key, 2,-1) +
		PreCache(
			cache_size=28,
			num_workers=7) +
		# Crop(
			# key=output_affinities_key,
			# roi=crop_roi) +
		Train(
			graph='train/train_net',
			optimizer=config['optimizer'],
			loss=config['loss'],
			inputs={
				config['raw']: raw_key,
				config['gt_affs_in']: input_affinities_key,
				config['gt_affs_out']: output_affinities_key,
				config['pred_affs_loss_weights']: input_affinities_scale_key
			},
			outputs={
				config['pred_affs']: pred_affinities_key
			},
			gradients={
				config['pred_affs']: pred_affinities_gradient_key
			},
			summary=config['summary'],
			log_dir='log',
			save_every=20) +
		IntensityScaleShift(
			array=raw_key,
			scale=0.5,
			shift=0.5) +
		Snapshot(
			dataset_names={
				labels_key: 'volumes/labels/labels',
				input_affinities_key: 'volumes/input_affs',
				raw_key: 'volumes/raw',
				pred_affinities_key: 'volumes/pred_affs',
				output_affinities_key: 'volumes/output_affs'
			},
			output_filename='train/batch_{iteration}.hdf',
			every=1,
			dataset_dtypes={
				raw_key: np.float32,
				labels_key: np.uint64
			}) + 
		PrintProfilingStats(every=20)
	)

	print("Starting training...")
	with build(pipeline) as p:
		for i in range(iterations - trained_until):
			req = p.request_batch(request)
			# print ("labels: ", req[labels_key].data.shape)
			# print ("affinities_in: ", req[input_affinities_key].data.shape)
			# print ("affinities_out: ", req[output_affinities_key].data.shape)
			# print ("affinities_joined: ", req[joined_affinities_key].data.shape)
			# print ("raw: ", req[raw_key].data.shape)
			# print ("affinities_in_scale: ", req[input_affinities_scale_key].data.shape)
	print("Training finished")

if __name__ == "__main__":
	train(iterations=int(sys.argv[1]))