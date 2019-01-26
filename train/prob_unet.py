# assume tensorflow 1.3

import tensorflow as tf
from  tensorflow_probability import distributions as tfd


def unet(
	fmaps_in,
	num_layers,
	base_num_fmaps,
	fmap_inc_factor,
	downsample_factors,
	padding='valid',
	num_conv_passes=2,
	kernel_size_down=[3,3,3],
	kernel_size_up=[3,3,3],
	activation='relu',
	downsample_type="max_pool",
	upsample_type="conv_transpose",
	fov=(1, 1, 1),
	voxel_size=(1, 1, 1)):

	print "encode"
	fmaps = fmaps_in
	num_fmaps = base_num_fmaps
	across = []
	print fmaps.shape
	for layer in range(num_layers):
		for conv_pass in range(num_conv_passes):
			fmaps = tf.layers.conv3d(
				inputs=fmaps,
				filters=num_fmaps,
				kernel_size=kernel_size_down[layer],
				padding=padding,
				data_format="channels_first",
				activation=activation,
				name="enc_conv_pass_%i_%i"%(layer, conv_pass))
		across.append(fmaps)

		fmaps = downsample(
			fmaps_in=fmaps,
			downsample_type=downsample_type,
			downsample_factors=downsample_factors[layer],
			padding=padding,
			voxel_size=voxel_size,
			name="enc_downsample_%i"%layer)

		print fmaps.shape
		num_fmaps *= fmap_inc_factor

		if layer == num_layers-1:
			for conv_pass in range(num_conv_passes):
				fmaps = tf.layers.conv3d(
					inputs=fmaps,
					filters=num_fmaps,
					kernel_size=kernel_size_down[layer],
					padding=padding,
					data_format="channels_first",
					activation=activation,
					name="enc_conv_pass_bottom_%i"%conv_pass)

	print "embedded: ", fmaps.shape

	print "decode"
	for layer in reversed(range(num_layers)):
		num_fmaps /= fmap_inc_factor
		fmaps = upsample(
			fmaps_in=fmaps,
			num_fmaps=num_fmaps,
			upsample_type=upsample_type,
			upsample_factors=downsample_factors[layer],
			activation=activation,
			padding=padding,
			voxel_size=voxel_size,
			name="dec_upsample_%i"%layer)

		cropped = crop(across[layer], fmaps.get_shape().as_list())
		fmaps = tf.concat([cropped, fmaps], 1)

		for conv_pass in range(num_conv_passes):
			fmaps = tf.layers.conv3d(
				inputs=fmaps,
				filters=num_fmaps,
				kernel_size=kernel_size_up[layer],
				padding=padding,
				data_format="channels_first",
				activation=activation,
				name="dec_conv_pass_%i_%i"%(layer, conv_pass))
		print fmaps.shape

	return fmaps


def downsample(
	fmaps_in,
	downsample_type,
	downsample_factors,
	padding,
	voxel_size,
	name):

	if downsample_type == "max_pool":
		fmaps_out = tf.layers.max_pooling3d(
			inputs=fmaps_in,
			pool_size=downsample_factors,
			strides=downsample_factors,
			padding=padding,
			data_format='channels_first',
			name=name)
	else:
		raise Exception('unsupported downsample type')

	return fmaps_out


def upsample(
	fmaps_in,
	num_fmaps,
	upsample_type,
	upsample_factors,
	activation,
	padding,
	voxel_size,
	name):

	if upsample_type == "conv_transpose":
		fmaps_out = tf.layers.conv3d_transpose(
			inputs=fmaps_in,
			filters=num_fmaps,
			kernel_size=upsample_factors,
			strides=upsample_factors,
			padding=padding,
			data_format='channels_first',
			activation=activation,
			name=name)
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


if __name__ == "__main__":

	# prior = enc + trans + sampling
	# posterior = enc + transf + sampling
	# unet = unet (own enc + dec)

	raw = tf.placeholder(tf.float32, (1,1,196,196,196))
	gt = tf.placeholder(tf.float32, (1,3, 68, 68, 68))

	unet = unet(
		fmaps_in=raw,
		num_layers=3,
		base_num_fmaps=12,
		fmap_inc_factor=3,
		downsample_factors=[[3,3,3], [2,2,2], [2,2,2]])