from __future__ import print_function
import tensorflow as tf

def downsample(
	fmaps_in,
	downsample_type,
	downsample_factors,
	padding_type,
	voxel_size,
	name):

	# with tf.variable_scope(name) as vs:

	if downsample_type == "max_pool":
		fmaps_out = tf.layers.max_pooling3d(
			inputs=fmaps_in,
			pool_size=downsample_factors,
			strides=downsample_factors,
			padding=padding_type,
			data_format='channels_first',
			name=name)
	else:
		raise Exception('unsupported downsample type')

	return fmaps_out

def upsample(
	fmaps_in,
	num_channels,
	upsample_type,
	upsample_factors,
	activation_type,
	padding_type,
	voxel_size,
	name):

	# with tf.variable_scope(name) as vs:

	if upsample_type == "conv_transpose":
		fmaps_out = tf.layers.conv3d_transpose(
			inputs = fmaps_in,
			filters = num_channels,
			kernel_size = upsample_factors,
			strides = upsample_factors,
			padding = padding_type,
			data_format = 'channels_first',
			activation = activation_type,
			name = name)
	else:
		raise Exception('unsupported upsample type')

	return fmaps_out

def crop(fmaps_in, shape):

	in_shape = fmaps_in.get_shape().as_list()

	offset = [
		0, # batch
		0, # channel
		(in_shape[2] - shape[2])//2, # z
		(in_shape[3] - shape[3])//2, # y
		(in_shape[4] - shape[4])//2, # x
	]
	size = [
		in_shape[0],
		in_shape[1],
		shape[2],
		shape[3],
		shape[4],
	]

	fmaps_out = tf.slice(fmaps_in, offset, size)

	return fmaps_out