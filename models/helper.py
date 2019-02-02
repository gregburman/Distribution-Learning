from __future__ import print_function
import tensorflow as tf

def downsample(
	fmaps_in,
	downsample_type,
	downsample_factors,
	padding_type,
	voxel_size,
	name):

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