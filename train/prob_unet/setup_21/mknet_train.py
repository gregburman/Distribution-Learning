from __future__ import print_function
import sys
sys.path.append('../../')

import tensorflow as tf
import json

from models.unet import UNet
from models.encoder import Encoder
from models.f_comb import FComb

def create_network(input_shape, setup_dir):

	print ("MKNET: PROB-UNET TRAIN")
	print("")
	tf.reset_default_graph()

	raw = tf.placeholder(tf.float32, shape=input_shape) # for gp
	raw_batched = tf.reshape(raw, (1, 1) + input_shape) # for tf
	gt_affs_in = tf.placeholder(tf.float32, shape = (3,) + input_shape)
	gt_affs_in_batched = tf.reshape(gt_affs_in, (1, 3) + input_shape)

	print ("raw_batched: ", raw_batched.shape)
	print ("gt_affs_in_batched: ", gt_affs_in_batched.shape)
	print ("")

	with tf.variable_scope("debug") as dbscope:
		debug_batched = tf.constant([[[1,2,3,4,5]]])
		# debug_batched = tf.reshape(debug, (1,1,5))
		print('DEBUG:', debug_batched.name)

	with tf.variable_scope("unet") as vs1:
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
			activation_type = tf.nn.relu,
			downsample_type = "max_pool",
			upsample_type = "conv_transpose",
			voxel_size = (1, 1, 1))
		unet.build()
		print("")

	with tf.variable_scope("prior") as vs2:
		prior = Encoder(
			fmaps_in = raw_batched,
			affmaps_in = None,
			num_layers = 3,
			latent_dims = 6,
			base_channels = 12,
			channel_inc_factor = 3,
			downsample_factors = [[2,2,2], [2,2,2], [2,2,2]],
			padding_type = "valid",
			num_conv_passes = 2,
			down_kernel_size = [3, 3, 3],
			activation_type = tf.nn.relu,
			downsample_type = "max_pool",
			voxel_size = (1, 1, 1),
			name = "prior")
		prior.build()
		print ("")

	with tf.variable_scope("posterior") as vs3:
		posterior = Encoder(
			fmaps_in = raw_batched,
			affmaps_in = gt_affs_in_batched,
			num_layers = 3,
			latent_dims = 6,
			base_channels = 12,
			channel_inc_factor = 3,
			downsample_factors = [[2,2,2], [2,2,2], [2,2,2]],
			padding_type = "valid",
			num_conv_passes = 2,
			down_kernel_size = [3, 3, 3],
			activation_type = tf.nn.relu,
			downsample_type = "max_pool",
			voxel_size = (1, 1, 1),
			name = "posterior")
		posterior.build()
		print ("")

		sample_z = posterior.sample()
		sample_z_batched = tf.reshape(sample_z, (1, 1, 6))

	with tf.variable_scope("f_comb") as vs4:
		f_comb = FComb(
			fmaps_in = unet.get_fmaps(),
			sample_in = sample_z,
			num_1x1_convs = 3,
			num_channels = 12,
			padding_type = 'valid',
			activation_type = tf.nn.relu,
			voxel_size = (1, 1, 1))
		f_comb.build()
		print ("")

		broadcast_sample = f_comb.broadcast_sample
		sample_out = f_comb.sample_out
		print("sample_out_name: ", f_comb.get_fmaps().name)
		sample_out_batched = tf.reshape(sample_out, (1, 1, 6)) 

	with tf.variable_scope("affs") as vs5:
		pred_logits = tf.layers.conv3d(
			inputs=f_comb.get_fmaps(),
			filters=3, 
			kernel_size=1,
			padding='valid',
			data_format="channels_first",
			activation=None,
			name="affs")
		print ("")

	pred_affs = tf.sigmoid(pred_logits)

	output_shape_batched = pred_logits.get_shape().as_list()
	output_shape = output_shape_batched[1:] # strip the batch dimension

	pred_logits = tf.squeeze(pred_logits, axis=[0])
	pred_affs = tf.squeeze(pred_affs, axis=[0])

	gt_affs_out = tf.placeholder(tf.float32, shape=output_shape)
	pred_affs_loss_weights = tf.placeholder(tf.float32, shape=output_shape)

	print ("gt_affs_out: ", gt_affs_out.shape)
	print ("pred_logits: ", pred_logits.shape)
	print ("pred_affs: ", pred_affs.shape)
	print ("")

	mse_loss = tf.losses.mean_squared_error(
		gt_affs_out,
		pred_affs,
		pred_affs_loss_weights)

	# sce_loss = tf.losses.sigmoid_cross_entropy(
	# 	multi_class_labels = gt_affs_out,
	# 	logits = pred_logits,
	# 	weights = pred_affs_loss_weights)

	summary = tf.summary.scalar('mse_loss', mse_loss)
	# summary = tf.summary.merge_all()

	# opt = tf.train.AdamOptimizer(
	# 	learning_rate=1e-6,
	# 	beta1=0.95,
	# 	beta2=0.999,
	# 	epsilon=1e-8)
	opt = tf.train.AdamOptimizer()
	optimizer = opt.minimize(mse_loss)

	output_shape = output_shape[1:]
	print("input shape : %s" % (input_shape,))
	print("output shape: %s" % (output_shape,))

	tf.train.export_meta_graph(filename= setup_dir + "train_net.meta")

	config = {
		'raw': raw.name,
		'pred_affs': pred_affs.name,
		'gt_affs_in': gt_affs_in.name,
		'gt_affs_out': gt_affs_out.name,
		'pred_affs_loss_weights': pred_affs_loss_weights.name,
		'loss': mse_loss.name,
		'optimizer': optimizer.name,
		'input_shape': input_shape,
		'output_shape': output_shape,
		'prior': prior.get_fmaps().name,
		'posterior': posterior.get_fmaps().name,
		'latent_dims': 6,
		'summary': summary.name,
		'broadcast': broadcast_sample.name,
		'sample_z': sample_z_batched.name,
		'pred_logits': pred_logits.name,
		'sample_out': sample_out_batched.name,
		'debug': debug_batched.name
	}
	with open(setup_dir + 'train_config.json', 'w') as f:
		json.dump(config, f)

if __name__ == "__main__":

	setup_name = sys.argv[1]
	setup_dir = "train/prob_unet/" + setup_name + "/"
	print (setup_dir)
	create_network((132, 132, 132), setup_dir)
