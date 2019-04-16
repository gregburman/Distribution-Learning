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
from nodes import MergeLabels
from nodes import PickRandomLabel

logging.basicConfig(level=logging.INFO)

data_dir = "data/gt_1_merge_3_cropped"
samples = ["batch_%08i"%i for i in range(2000)]

setup_name = sys.argv[1]
setup_dir = 'train/prob_unet/' + setup_name + '/'

with open(setup_dir + 'train_config.json', 'r') as f:
	config = json.load(f)

beta = 1e-10
phase_switch = 2000
neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
neighborhood_opp = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

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
	labels_key = ArrayKey('LABELS')	
	raw_affs_key = ArrayKey('RAW_AFFINITIES')
	raw_joined_affs_key = ArrayKey('RAW_JOINED_AFFINITIES')
	raw_key = ArrayKey('RAW')
	
	merged_labels_keys = []
	# merged_affs_keys = []
	picked_labels_key = ArrayKey('PICKED_RANDOM_LABEL')

	affs_neg_key = ArrayKey('AFFINITIES')
	affs_pos_key = ArrayKey('AFFINITIES_OPP')
	joined_affs_neg_key = ArrayKey('JOINED_AFFINITIES')
	joined_affs_pos_key = ArrayKey('JOINED_AFFINITIES_OPP')

	num_merges = 3
	for i in range(num_merges):
		merged_labels_keys.append(ArrayKey('MERGED_LABELS_%i'%(i+1)))


	gt_affs_out_key = ArrayKey('GT_AFFINITIES')
	gt_affs_in_key = ArrayKey('GT_AFFINITIES_IN')
	gt_affs_mask_key = ArrayKey('GT_AFFINITIES_MASK')
	gt_affs_scale_key = ArrayKey('GT_AFFINITIES_SCALE')
	
	pred_affs_key = ArrayKey('PRED_AFFS')
	pred_affs_gradient_key = ArrayKey('PRED_AFFS_GRADIENT')

	sample_z_key = ArrayKey("SAMPLE_Z")
	broadcast_key = ArrayKey("BROADCAST")
	sample_out_key = ArrayKey("SAMPLE_OUT")
	debug_key = ArrayKey("DEBUG")


	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate(config['input_shape']) * voxel_size
	input_affs_shape = Coordinate([i + 1 for i in config['input_shape']]) * voxel_size
	output_shape = Coordinate(config['output_shape']) * voxel_size
	output_affs_shape = Coordinate([i + 1 for i in config['output_shape']]) * voxel_size
	sample_shape = Coordinate((1, 1, 6)) * voxel_size
	debug_shape = Coordinate((1, 1, 5)) * voxel_size

	print ("input_shape: ", input_shape)
	print ("input_affs_shape: ", input_affs_shape)
	print ("output_shape: ", output_shape)
	print ("output_affs_shape: ", output_affs_shape)

	request = BatchRequest()
	request.add(labels_key, input_shape)

	request.add(raw_affs_key, input_shape)
	request.add(raw_joined_affs_key, input_shape)
	request.add(raw_key, input_shape)

	for i in range(num_merges): 
		request.add(merged_labels_keys[i], input_shape)
	request.add(picked_labels_key, output_shape)		

	request.add(gt_affs_out_key, output_shape)
	request.add(gt_affs_in_key, input_shape)
	request.add(gt_affs_mask_key, output_shape)
	request.add(gt_affs_scale_key, output_shape)

	request.add(pred_affs_key, output_shape)
	request.add(pred_affs_gradient_key, output_shape)

	request.add(broadcast_key, output_shape)
	request.add(sample_z_key, sample_shape)
	request.add(sample_out_key, sample_shape)
	request.add(debug_key, debug_shape)

	# offset = Coordinate((input_shape[i]-output_shape[i])/2 for i in range(len(input_shape)))
	# crop_roi = Roi(offset, output_shape)
	# print("crop_roi: ", crop_roi)

	dataset_names = {
		labels_key: 'volumes/labels',
	}

	array_specs = {
		labels_key: ArraySpec(interpolatable=False)
	}

	for i in range(num_merges):
		dataset_names[merged_labels_keys[i]] = 'volumes/merged_labels_%i'%(i+1)
		array_specs[merged_labels_keys[i]] = ArraySpec(interpolatable=False)

	pipeline = tuple(
		Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = dataset_names,
            array_specs = array_specs
        ) +
        Pad(labels_key, None) +
        Pad(merged_labels_keys[0], None) +
        Pad(merged_labels_keys[1], None) +
        Pad(merged_labels_keys[2], None)
        # Pad(merged_labels_key[i], None) for i in range(num_merges) # don't know why this doesn't work
        for sample in samples
		)

	pipeline += RandomProvider()

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=labels_key,
			affinities=raw_affs_key)

	pipeline += AddJoinedAffinities(
			input_affinities=raw_affs_key,
			joined_affinities=raw_joined_affs_key)

	pipeline += AddRealism(
			joined_affinities = raw_joined_affs_key,
			raw = raw_key,
			sp=0.25,
			sigma=1,
			contrast=0.7)

	if phase == "euclid":

		pipeline += PickRandomLabel(
				input_labels = [labels_key]+ merged_labels_keys,
				output_label=picked_labels_key,
				probabilities=[1, 0, 0, 0])

	else: 

		pipeline += PickRandomLabel(
				input_labels = [labels_key] + merged_labels_keys,
				output_label=picked_labels_key,
				probabilities=[0.5, 0.5, 0, 0])

		pipeline += RenumberConnectedComponents(
			labels=picked_labels_key)		

	pipeline += GrowBoundary(picked_labels_key, steps=1, only_xy=True)

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=picked_labels_key,
			affinities=gt_affs_in_key)

	pipeline += AddAffinities(
			affinity_neighborhood=neighborhood,
			labels=picked_labels_key,
			affinities=gt_affs_out_key,
			affinities_mask=gt_affs_mask_key)

	# if phase == 'euclid':
	pipeline += BalanceLabels(
			labels=gt_affs_out_key,
			scales=gt_affs_scale_key)

	pipeline += DefectAugment(
			intensities=raw_key,
			prob_missing=0.03,
			prob_low_contrast=0.01,
			contrast_scale=0.5,
			axis=0)

	pipeline += IntensityScaleShift(raw_key, 2,-1)

	pipeline += PreCache(
			cache_size=8,
			num_workers=4)

	train_inputs = {
		config['raw']: raw_key,
		config['gt_affs_in']: gt_affs_in_key,
		config['gt_affs_out']: gt_affs_out_key,
		config['pred_affs_loss_weights']: gt_affs_scale_key
	}

	if phase == 'euclid':
		train_loss = config['loss']
		train_optimizer = config['optimizer']
		train_summary = config['summary']
		# train_inputs[config['pred_affs_loss_weights']] = input_affinities_scale_key
	else:
		train_loss = None
		train_optimizer = add_malis_loss
		train_inputs['gt_seg:0'] = picked_labels_key
		train_inputs['gt_affs_mask:0'] = gt_affs_mask_key
		train_summary = 'Merge/MergeSummary:0'

	pipeline += Train(
			graph=setup_dir + 'train_net',
			optimizer=train_optimizer,
			loss=train_loss,
			inputs=train_inputs,
			outputs={
				config['pred_affs']: pred_affs_key,
				config['broadcast']: broadcast_key,
				config['sample_z']: sample_z_key,
				config['sample_out']: sample_out_key,
				config['debug']: debug_key
			},
			gradients={
				config['pred_affs']: pred_affs_gradient_key
			},
			summary=train_summary,
			log_dir='log/prob_unet/' + setup_name,
			save_every=1)

	pipeline += IntensityScaleShift(
			array=raw_key,
			scale=0.5,
			shift=0.5)

	pipeline += Snapshot(
			dataset_names={
				labels_key: 'volumes/labels',
				picked_labels_key: 'volumes/merged_labels',
				raw_affs_key: 'volumes/raw_affs',
				raw_key: 'volumes/raw',
				pred_affs_key: 'volumes/pred_affs',
				gt_affs_out_key: 'volumes/gt_affs_out',
				gt_affs_in_key: 'volumes/gt_affs_in'
			},
			output_filename='prob_unet/' + setup_name + '/batch_{iteration}.hdf',
			every=1000,
			dataset_dtypes={
				labels_key: np.uint64,
				picked_labels_key: np.uint64,
				raw_key: np.float32
			})

	pipeline += PrintProfilingStats(every=20)

	print("Starting training...")
	with build(pipeline) as p:
		for i in range(iterations - trained_until):
			req = p.request_batch(request)
			sample_z = req[sample_z_key].data
			broadcast_sample = req[broadcast_key].data
			sample_out = req[sample_out_key].data
			print("sample_out:", sample_out)
			debug = req[debug_key].data
			print("debug", debug)

			print("sample_out: ", sample_z)
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
