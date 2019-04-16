from __future__ import print_function
import sys
sys.path.append('../../')

import tensorflow as tf
import json
import mala
import malis

from models.unet import UNet

def create_network(input_shape, name):

	print ("MKNET: UNET TRAIN")
	print("")
	tf.reset_default_graph()

	raw = tf.placeholder(tf.float32, shape=input_shape)
	raw_batched = tf.reshape(raw, (1, 1) + input_shape)

	# with tf.variable_scope("debug") as dbscope:
	debug = tf.constant([[[1,2,3,4,5]]])
	print('DEBUG:', debug.name)

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

	pred_logits = tf.layers.conv3d(
		inputs=unet.get_fmaps(),
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

	pred_logits = tf.squeeze(pred_logits, axis=[0], name="pred_logits")
	pred_affs = tf.squeeze(pred_affs, axis=[0], name="pred_affs")

	gt_affs = tf.placeholder(tf.float32, shape=output_shape, name="gt_affs")
	pred_affs_loss_weights = tf.placeholder(tf.float32, shape=output_shape, name="pred_affs_loss_weights")

	# neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
	# gt_seg = tf.placeholder(tf.int64, shape=output_shape[1:], name='gt_seg')
	# print ("gt_seg: ", gt_seg)
	# gt_affs_mask = tf.placeholder(tf.int64, shape=output_shape, name='gt_affs_mask')
	
	print ("pred_affs: ", pred_affs)
	print ("gt_affs: ", gt_affs)
	# print ("pred_logits: ", pred_logits.shape)
	# print ("gt_seg: ", gt_seg)
	# print ("gt_affs_mask: ", gt_affs_mask)
	print ("")

	# loss = tf.losses.mean_squared_error(
	# 	gt_affs,
	# 	pred_affs,
	# 	pred_affs_loss_weights)

	# loss = malis.malis_loss_op(
	# 	pred_affs, 
	# 	gt_affs, 
	# 	gt_seg,
	# 	neighborhood,
	# 	gt_affs_mask)
	# print ("gt_seg: ", gt_seg)
	# print ("loss: ", loss)


	# loss = tf.losses.sigmoid_cross_entropy(
	# 	multi_class_labels = gt_affs,
	# 	logits = pred_logits,
	# 	weights = pred_affs_loss_weights)

	# summary = tf.summary.scalar('mse_loss', loss)
	# summary = tf.summary.scalar('sce', loss)

	# opt = tf.train.AdamOptimizer(
	# 	learning_rate=0.5e-4,
	# 	beta1=0.95,
	# 	beta2=0.999,
	# 	epsilon=1e-8)
	# opt = tf.train.AdamOptimizer()
	# optimizer = opt.minimize(loss)

	output_shape = output_shape[1:]
	print("input shape : %s" % (input_shape,))
	print("output shape: %s" % (output_shape,))

	tf.train.export_meta_graph(filename=name + '.meta')

	config = {
		'raw': raw.name,
		'pred_affs': pred_affs.name,
		'gt_affs': gt_affs.name,
		# 'gt_seg': gt_seg.name,
		# 'gt_affs_mask': gt_affs_mask.name,
		'pred_affs_loss_weights': pred_affs_loss_weights.name,
		# 'loss': loss.name,
		# 'optimizer': optimizer.name,
		'input_shape': input_shape,
		'output_shape': output_shape,
		'debug': debug.name
		# 'summary': summary.name
	}
	with open(name + '.json', 'w') as f:
		json.dump(config, f)

if __name__ == "__main__":
	create_network((132, 132, 132), 'train/unet/train_net')