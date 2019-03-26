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

setup_name = sys.argv[1]
setup_dir = 'train/prob_unet/' + setup_name + '/'
with open(setup_dir + 'train_config.json', 'r') as f:
	config = json.load(f)

beta = 1
phase_switch = 1000
neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

def train(iterations):

	# tf.reset_default_graph()
	if tf.train.latest_checkpoint('.'):
		trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
	else:
		trained_until = 0
	if trained_until >= iterations:
		return

	if trained_until < phase_switch and iterations > phase_switch:
		train(phase_switch)

	phase = 'euclid' if iterations <= phase_switch else 'malis'
	print("Training in phase %s until %i"%(phase, iterations))
		
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

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate(config['input_shape']) * voxel_size
	output_shape = Coordinate(config['output_shape']) * voxel_size

	print ("input_shape: ", input_shape)
	print ("output_shape: ", output_shape)

	# define requests
	request = BatchRequest()
	request.add(labels_key, output_shape)
	request.add(gt_affs_in_key, input_shape)
	request.add(joined_affinities_key, input_shape)
	request.add(raw_key, input_shape)
	request.add(gt_affs_out_key, output_shape)
	request.add(pred_affinities_key, output_shape)
	request.add(gt_affs_mask, output_shape)
	# if phase == 'euclid':
	request.add(input_affinities_scale_key, output_shape)

	offset = Coordinate((input_shape[i]-output_shape[i])/2 for i in range(len(input_shape)))
	crop_roi = Roi(offset, output_shape)
	# print("crop_roi: ", crop_roi)

	pipeline = ()

	pipeline += ToyNeuronSegmentationGenerator(
			array_key=labels_key,
			n_objects=50,
			points_per_skeleton=8,
			smoothness=3,
			noise_strength=1,
			interpolation="linear")

	pipeline +=AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=labels_key,
			affinities=gt_affs_in_key)

	pipeline +=	GrowBoundary(labels_key, steps=1, only_xy=True)

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=labels_key,
			affinities=gt_affs_out_key,
			affinities_mask=gt_affs_mask)

	pipeline +=	AddJoinedAffinities(
			input_affinities=gt_affs_in_key,
			joined_affinities=joined_affinities_key)

	pipeline += AddRealism(
			joined_affinities = joined_affinities_key,
			raw = raw_key,
			sp=0.65,
			sigma=1,
			contrast=0.7)

	# if phase == 'euclid':
	pipeline += BalanceLabels(
			labels=gt_affs_out_key,
			scales=input_affinities_scale_key)

	pipeline += DefectAugment(
			intensities=raw_key,
			prob_missing=0.03,
			prob_low_contrast=0.01,
			contrast_scale=0.5,
			axis=0)

	pipeline += IntensityScaleShift(raw_key, 2,-1)

	if phase == 'malis':
		pipeline += Crop(
			key=labels_key,
			roi=crop_roi)
		pipeline += RenumberConnectedComponents(labels=labels_key)

	pipeline += PreCache(
			cache_size=32,
			num_workers=8)

	train_inputs = {
		config['raw']: raw_key,
		config['gt_affs_in']: gt_affs_in_key,
		config['gt_affs_out']: gt_affs_out_key,
		config['pred_affs_loss_weights']: input_affinities_scale_key
	}

	if phase == 'euclid':
		train_loss = config['loss']
		train_optimizer = config['optimizer']
		train_summary = config['summary']
		# train_inputs[config['pred_affs_loss_weights']] = input_affinities_scale_key
	else:
		train_loss = None
		train_optimizer = add_malis_loss
		train_inputs['gt_seg:0'] = labels_key
		train_inputs['gt_affs_mask:0'] = gt_affs_mask
		train_summary = 'Merge/MergeSummary:0'

	pipeline += Train(
			graph=setup_dir + 'train_net',
			optimizer=train_optimizer,
			loss=train_loss,
			inputs=train_inputs,
			outputs={
				config['pred_affs']: pred_affinities_key
			},
			gradients={
				config['pred_affs']: pred_affinities_gradient_key
			},
			summary=train_summary,
			log_dir='log/prob_unet/' + setup_name,
			save_every=2000)

	pipeline += IntensityScaleShift(
			array=raw_key,
			scale=0.5,
			shift=0.5)

	pipeline += Snapshot(
			dataset_names={
				labels_key: 'volumes/labels',
				raw_key: 'volumes/raw',
				pred_affinities_key: 'volumes/pred_affs',
				gt_affs_out_key: 'volumes/output_affs'
			},
			output_filename='prob_unet/' + setup_name + '/batch_{iteration}.hdf',
			every=4000,
			dataset_dtypes={
				labels_key: np.uint64,
				raw_key: np.float32
			})

	pipeline += PrintProfilingStats(every=20)

	print("Starting training...")
	with build(pipeline) as p:
		for i in range(iterations - trained_until):
			req = p.request_batch(request)
			# print ("labels: ", req[labels_key].data.shape)
			# print ("affinities_in: ", req[gt_affs_in_key].data.shape)
			# print ("affinities_out: ", req[gt_affs_out_key].data.shape)
			# print ("affinities_joined: ", req[joined_affinities_key].data.shape)
			# print ("raw: ", req[raw_key].data.shape)
			# print ("affinities_in_scale: ", req[input_affinities_scale_key].data.shape)
	print("Training finished")

def add_malis_loss(graph):
	pred_affs = graph.get_tensor_by_name(config['pred_affs'])
	gt_affs = graph.get_tensor_by_name(config['gt_affs_out'])
	gt_seg = tf.placeholder(tf.int32, shape=config['output_shape'], name='gt_seg')
	gt_affs_mask = tf.placeholder(tf.int32, shape=[3] + config['output_shape'], name='gt_affs_mask')

	prior = graph.get_tensor_by_name(config['prior'])
	posterior = graph.get_tensor_by_name(config['posterior'])

	p = z(prior)
	q = z(posterior)

	mlo = malis.malis_loss_op(pred_affs, 
		gt_affs, 
		gt_seg,
		neighborhood,
		gt_affs_mask)

	kl = tf.distributions.kl_divergence(p, q)
	kl = tf.reshape(kl, [], name="kl_loss")

	loss = mlo + beta * kl
	tf.summary.scalar('malis_loss', mlo)
	tf.summary.scalar('kl_loss', kl)
	opt = tf.train.AdamOptimizer(
		learning_rate=0.5e-4,
		beta1=0.95,
		beta2=0.999,
		epsilon=1e-8,
		name='malis_optimizer')

	summary = tf.summary.merge_all()
	# print(summary)
	# opt = tf.train.AdamOptimizer()
	optimizer = opt.minimize(loss)
	return (loss, optimizer)

def z(fmaps):
	mean = fmaps[:, :config["latent_dims"]]
	log_sigma = fmaps[:, config["latent_dims"]:]
	return tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_sigma))

if __name__ == "__main__":
	train(iterations=int(sys.argv[2]))