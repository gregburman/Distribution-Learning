from __future__ import print_function
import sys
sys.path.append('../../')

import tensorflow as tf
import json
import mala

from models.unet import UNet

def create_network(input_shape, name):

	tf.reset_default_graph()

	raw = tf.placeholder(tf.float32, shape=input_shape)
	raw_batched = tf.reshape(raw, (1, 1) + input_shape)

	unet = UNet(
		fmaps_in = raw_batched,
		num_layers = 3,
		base_channels = 12,
		channel_inc_factor = 3,
		resample_factors = [[2,2,2], [2,2,2], [2,2,2]],
		padding_type = "valid",
		num_conv_passes = 2,
		down_kernel_size = [3, 3, 3],
		up_kernel_size = [3, 3, 3],
		activation_type = "relu",
		downsample_type = "max_pool",
		upsample_type = "conv_transpose",
		voxel_size = (1, 1, 1))
	unet.build()

	affs_batched, _ = mala.networks.conv_pass(
		fmaps_in=unet.get_fmaps(),
		kernel_sizes=[1],
		num_fmaps=3,
		activation='sigmoid',
		name='affs')

	output_shape_batched = affs_batched.get_shape().as_list()
	output_shape = output_shape_batched[1:] # strip the batch dimension

	pred_affs = tf.reshape(affs_batched, output_shape)

	gt_affs_out = tf.placeholder(tf.float32, shape=output_shape)
	pred_affs_loss_weights = tf.placeholder(tf.float32, shape=output_shape)
	
	# loss = tf.losses.mean_squared_error(
	# 	gt_affs_out,
	# 	pred_affs,
	# 	pred_affs_loss_weights)
	loss = tf.losses.mean_squared_error(gt_affs_out, pred_affs, pred_affs_loss_weights)
	# loss = tf.losses.sigmoid_cross_entropy(gt_affs_out, pred_affs, pred_affs_loss_weights)

	summary = tf.summary.scalar('mse_loss', loss)

	# opt = tf.train.AdamOptimizer(
	# 	learning_rate=0.5e-4,
	# 	beta1=0.95,
	# 	beta2=0.999,
	# 	epsilon=1e-8)
	opt = tf.train.AdamOptimizer()
	optimizer = opt.minimize(loss)

	print ("output before: ", output_shape)
	output_shape = output_shape[1:]
	print("input shape : %s"%(input_shape,))
	print("output shape: %s"%(output_shape,))

	tf.train.export_meta_graph(filename=name + '.meta')

	config = {
		'raw': raw.name,
		'pred_affs': pred_affs.name,
		'gt_affs_out': gt_affs_out.name,
		'pred_affs_loss_weights': pred_affs_loss_weights.name,
		'loss': loss.name,
		'optimizer': optimizer.name,
		'input_shape': input_shape,
		'output_shape': output_shape,
		'summary': summary.name
	}
	with open(name + '.json', 'w') as f:
		json.dump(config, f)

if __name__ == "__main__":
	create_network((132, 132, 132), 'train_net')


	# unet, _, _ = mala.networks.unet(
	# 	fmaps_in=raw_batched,
	# 	num_fmaps=12,
	# 	fmap_inc_factors=3,
	# 	downsample_factors=[[3,3,3],[2,2,2],[2,2,2]])

	# unet = prob_unet.unet(
	# 	fmaps_in=raw_batched,
	# 	num_layers=3,
	# 	base_num_fmaps=12,
	# 	fmap_inc_factor=3,
	# 	downsample_factors=[[2,2,2], [2,2,2], [2,2,2]])