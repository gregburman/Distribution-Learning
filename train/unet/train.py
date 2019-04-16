from __future__ import print_function
import sys
sys.path.append('../../')

from gunpowder import *
from gunpowder.tensorflow import *
import os
import math
import json
import tensorflow as tf
import numpy as np
import logging
import malis

from nodes import ToyNeuronSegmentationGenerator
from nodes import AddJoinedAffinities
from nodes import AddRealism

logging.basicConfig(level=logging.INFO)
neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

with open('train/unet/train_net.json', 'r') as f:
		config = json.load(f)

def train(iterations):
	tf.reset_default_graph()
	if tf.train.latest_checkpoint('.'):
		trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
	else:
		trained_until = 0
	if trained_until >= iterations:
		return

	# define array-keys
	labels_key = ArrayKey('GT_LABELS')
	gt_affs_in_key = ArrayKey('GT_AFFINITIES_IN')
	gt_affs_out_key = ArrayKey('GT_AFFINITIES_OUT')
	joined_affinities_key = ArrayKey('GT_JOINED_AFFINITIES')
	raw_key = ArrayKey('RAW')
	input_affinities_scale_key = ArrayKey('GT_AFFINITIES_SCALE')
	pred_affinities_key = ArrayKey('PREDICTED_AFFS')
	pred_affinities_gradient_key = ArrayKey('AFFS_GRADIENT')
	gt_affs_mask = ArrayKey('GT_AFFINITIES_MASK')
	debug_key = ArrayKey("DEBUG")

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate(config['input_shape']) * voxel_size
	output_shape = Coordinate(config['output_shape']) * voxel_size
	debug_shape = Coordinate((1, 1, 5)) * voxel_size

	print ("input_shape: ", input_shape)
	print ("output_shape: ", output_shape)

	# define requests
	request = BatchRequest()
	request.add(labels_key, output_shape)
	request.add(gt_affs_in_key, input_shape)
	request.add(joined_affinities_key, input_shape)
	request.add(raw_key, input_shape)
	request.add(gt_affs_out_key, output_shape)
	request.add(input_affinities_scale_key, output_shape)
	request.add(pred_affinities_key, output_shape)
	request.add(gt_affs_mask, output_shape)
	request.add(debug_key, debug_shape)

	offset = Coordinate((input_shape[i]-output_shape[i])/2 for i in range(len(input_shape)))
	crop_roi = Roi(offset, output_shape)
	# print("crop_roi: ", crop_roi)

	pipeline = ()
	pipeline += ToyNeuronSegmentationGenerator(
		array_key=labels_key,
		n_objects=50,
		points_per_skeleton=8,
		smoothness=3,
		noise_strength = 1,
		interpolation="linear") 
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
	pipeline +=  AddAffinities(
		affinity_neighborhood=neighborhood,
		labels=labels_key,
		affinities=gt_affs_in_key)
	pipeline +=  GrowBoundary(labels_key, steps=1, only_xy=True)
	pipeline +=  AddAffinities(
		affinity_neighborhood=neighborhood,
		labels=labels_key,
		affinities=gt_affs_out_key,
		affinities_mask=gt_affs_mask)
	pipeline +=  AddJoinedAffinities(
		input_affinities=gt_affs_in_key,
		joined_affinities=joined_affinities_key)
	pipeline +=  AddRealism(
		joined_affinities = joined_affinities_key,
		raw = raw_key,
		sp=0.25,
		sigma=1,
		contrast=0.7)
	pipeline +=  BalanceLabels(
		labels=gt_affs_out_key,
		scales=input_affinities_scale_key)
	pipeline +=  DefectAugment(
		intensities=raw_key,
		prob_missing=0.03,
		prob_low_contrast=0.01,
		contrast_scale=0.5,
		axis=0)
	pipeline +=  IntensityScaleShift(raw_key, 2,-1)
	pipeline +=  PreCache(
		cache_size=32,
		num_workers=8)
	pipeline +=  Crop(
		key=labels_key,
		roi=crop_roi)
	pipeline +=  RenumberConnectedComponents(labels=labels_key)
	train = Train(
		graph='train/unet/train_net',
		# optimizer=config['optimizer'],
		optimizer = add_malis_loss,
		# loss=config['loss'],
		loss=None,
		inputs={
			config['raw']: raw_key,
			"gt_seg:0": labels_key,
			"gt_affs_mask:0": gt_affs_mask,
			config['gt_affs']: gt_affs_out_key,
			config['pred_affs_loss_weights']: input_affinities_scale_key,
		},
		outputs={
			config['pred_affs']: pred_affinities_key,
			config['debug']: debug_key,
		},
		gradients={
			config['pred_affs']: pred_affinities_gradient_key
		},
		summary="malis_loss:0",
		log_dir='log/unet',
		save_every=1)
	pipeline += train
	pipeline +=  IntensityScaleShift(
		array=raw_key,
		scale=0.5,
		shift=0.5)
	# Snapshot(
	# 	dataset_names={
	# 		labels_key: 'volumes/labels',
	# 		raw_key: 'volumes/raw',
	# 		gt_affs_out_key: 'volumes/gt_affs',
	# 		pred_affinities_key: 'volumes/pred_affs'
	# 	},
	# 	output_filename='unet/train/batch_{iteration}.hdf',
	# 	every=100,
	# 	dataset_dtypes={
	# 		raw_key: np.float32,
	# 		labels_key: np.uint64
	# 	}) + 
	pipeline +=  PrintProfilingStats(every=8)

	print("Starting training... COOL BEANS")
	with build(pipeline) as p:
		for i in range(iterations - trained_until):
			req = p.request_batch(request)
			pred_affs = req[pred_affinities_key].data
			debug = req[debug_key].data
			print("debug", debug)
			print('pred_affs', pred_affs)
			print("name of pred_adds: ", req[pred_affinities_key])
			# print("train session: ", train.session)
			# print ("all vars: ", [n.name for n in tf.get_default_graph().as_graph_def().node])
			# graph_def = tf.graph_util.convert_variables_to_constants(train.session, tf.get_default_graph().as_graph_def(), ["pred_affs:0".split(':')[0]])
			# print ("labels: ", req[labels_key].data.shape)
			# print ("affinities_out: ", req[gt_affs_out_key].data.shape)
			# print ("affinities_joined: ", req[joined_affinities_key].data.shape)
			# print ("raw: ", req[raw_key].data.shape)
			# print ("affinities_in_scale: ", req[input_affinities_scale_key].data.shape)
	print("Training finished")

def add_malis_loss(graph):
	pred_affs = graph.get_tensor_by_name(config['pred_affs'])
	gt_affs = graph.get_tensor_by_name(config['gt_affs'])
	gt_seg = tf.placeholder(tf.int32, shape=(44, 44, 44), name='gt_seg')
	gt_affs_mask = tf.placeholder(tf.int32, shape=(3, 44, 44, 44), name='gt_affs_mask')
	
	mlo = malis.malis_loss_op(pred_affs, 
		gt_affs, 
		gt_seg,
		neighborhood,
		gt_affs_mask)

	# loss = mlo + beta config['kl_loss']
	summary = tf.summary.scalar('malis_loss', mlo)
	# tf.summary.scalar('kl_loss', kl_loss)
	opt = tf.train.AdamOptimizer(
		learning_rate=0.5e-4,
		beta1=0.95,
		beta2=0.999,
		epsilon=1e-8,
		name='malis_optimizer')

	# summary = tf.summary.merge_all()
	# print(summary)
	# opt = tf.train.AdamOptimizer()
	optimizer = opt.minimize(mlo)
	return (mlo, optimizer)

if __name__ == "__main__":
	train(iterations=int(sys.argv[1]))