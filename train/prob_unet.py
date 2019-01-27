# assume tensorflow 1.3

import tensorflow as tf
from  tensorflow_probability import distributions as tfd

def prob_unet(
	fmaps_in,
	affmaps_in,
	num_layers,
	latent_dim,
	base_num_fmaps,
	fmap_inc_factor,
	downsample_factors,
	num_classes,
	num_1x1_convs,
	padding='valid',
	num_conv_passes=2,
	kernel_size_down=[3,3,3],
	kernel_size_up=[3,3,3],
	activation='relu',
	downsample_type="max_pool",
	upsample_type="conv_transpose",
	fov=(1, 1, 1),
	voxel_size=(1, 1, 1),
	name="prob_unet"):

	_unet = unet(
		fmaps_in=fmaps_in,
		num_layers=num_layers,
		base_num_fmaps=base_num_fmaps,
		fmap_inc_factor=fmap_inc_factor,
		downsample_factors=downsample_factors)

	print ""

	_prior = prior(
		fmaps_in=fmaps_in,
		num_layers=num_layers,
		latent_dim=latent_dim,
		base_num_fmaps=base_num_fmaps,
		fmap_inc_factor=fmap_inc_factor,
		downsample_factors=downsample_factors)

	print ""

	_posterior = posterior(
		fmaps_in=fmaps_in,
		affmaps_in=affmaps_in,
		num_layers=num_layers,
		latent_dim=latent_dim,
		base_num_fmaps=base_num_fmaps,
		fmap_inc_factor=fmap_inc_factor,
		downsample_factors=downsample_factors)

	print ""

	_f_comb = f_comb(
		fmaps_in=fmaps_in,
		num_fmaps=base_num_fmaps,
		num_classes=num_classes,
		num_1x1_convs=num_1x1_convs,
		activation='relu',
		name='f_comb')

	return _unet, _prior, _posterior, _f_comb

def prior(
	fmaps_in,
	num_layers,
	latent_dim,
	base_num_fmaps,
	fmap_inc_factor,
	downsample_factors,
	padding='valid',
	num_conv_passes=2,
	kernel_size_down=[3,3,3],
	activation='relu',
	downsample_type="max_pool",
	fov=(1, 1, 1),
	voxel_size=(1, 1, 1),
	name="prior"):

	print "PRIOR"
	spacial_axes = [2,3,4]

	encoding =  encoder(fmaps_in, num_layers, base_num_fmaps, fmap_inc_factor,\
		downsample_factors, padding, num_conv_passes, kernel_size_down,\
		activation, downsample_type, fov, voxel_size, name)
	encoding = tf.reduce_mean(encoding, axis=spacial_axes, keepdims=True)

	mu_log_sigma = tf.layers.conv3d(
		inputs=encoding,
		filters=latent_dim*2,
		kernel_size=1,
		padding=padding,
		data_format="channels_first",
		activation=activation,
		name="prior_conv")

	mu_log_sigma = tf.squeeze(mu_log_sigma, axis=spacial_axes)
	mu = mu_log_sigma[:, :latent_dim]
	log_sigma = mu_log_sigma[:, latent_dim:]
	sigma =tf.exp(log_sigma)

	fout = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
	print "output: ", fout.event_shape

	return fout

def posterior(
	fmaps_in,
	affmaps_in,
	num_layers,
	latent_dim,
	base_num_fmaps,
	fmap_inc_factor,
	downsample_factors,
	padding='valid',
	num_conv_passes=2,
	kernel_size_down=[3,3,3],
	activation='relu',
	downsample_type="max_pool",
	fov=(1, 1, 1),
	voxel_size=(1, 1, 1),
	name="posterior"):

	print "POSTERIOR"
	channel_axis = 1
	spacial_axes = [2,3,4]
	affmaps_in = tf.cast(affmaps_in, tf.float32)
	fmaps_in = tf.concat([fmaps_in, affmaps_in], axis=channel_axis)

	encoding =  encoder(fmaps_in, num_layers, base_num_fmaps, fmap_inc_factor,\
		downsample_factors, padding, num_conv_passes, kernel_size_down,\
		activation, downsample_type, fov, voxel_size, name)
	encoding = tf.reduce_mean(encoding, axis=spacial_axes, keepdims=True)

	mu_log_sigma = tf.layers.conv3d(
		inputs=encoding,
		filters=latent_dim*2,
		kernel_size=1,
		padding=padding,
		data_format="channels_first",
		activation=activation,
		name="posterior_conv")

	mu_log_sigma = tf.squeeze(mu_log_sigma, axis=spacial_axes)
	mu = mu_log_sigma[:, :latent_dim]
	log_sigma = mu_log_sigma[:, latent_dim:]
	sigma =tf.exp(log_sigma)

	fout = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
	print "output: ", fout.event_shape

	return fout

def encoder(
	fmaps_in,
	num_layers,
	base_num_fmaps,
	fmap_inc_factor,
	downsample_factors,
	padding='valid',
	num_conv_passes=2,
	kernel_size_down=[3,3,3],
	activation='relu',
	downsample_type="max_pool",
	fov=(1, 1, 1),
	voxel_size=(1, 1, 1),
	name="encoder"):

	fmaps = fmaps_in
	num_fmaps = base_num_fmaps
	print "input: ", fmaps.shape
	for layer in range(num_layers):
		for conv_pass in range(num_conv_passes):
			fmaps = tf.layers.conv3d(
				inputs=fmaps,
				filters=num_fmaps,
				kernel_size=kernel_size_down[layer],
				padding=padding,
				data_format="channels_first",
				activation=activation,
				name="%s_enc_conv_pass_%i_%i"%(name, layer, conv_pass))

		fmaps = downsample(
			fmaps_in=fmaps,
			downsample_type=downsample_type,
			downsample_factors=downsample_factors[layer],
			padding=padding,
			voxel_size=voxel_size,
			name="%s_enc_downsample_%i"%(name, layer))

		print "layer ", layer, ": ", fmaps.shape
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
					name="%s_enc_conv_pass_bottom_%i"%(name, conv_pass))

	print "bottom: ", fmaps.shape
	return fmaps

def f_comb(
	fmaps_in,
	num_fmaps,
	num_classes,
	num_1x1_convs,
	padding='valid',
	activation='relu',
	name='f_comb'):

	print "F_COMB"
	fmaps = fmaps_in
	print "input: ", fmaps.shape
	for conv_pass in range(num_1x1_convs):
		fmaps = tf.layers.conv3d(
			inputs=fmaps,
			filters=num_fmaps,
			kernel_size=1,
			padding=padding,
			data_format="channels_first",
			activation=activation,
			name="%s_conv_pass_%i"%(name, conv_pass))

	print "output: ", fmaps.shape
	return fmaps


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
	voxel_size=(1, 1, 1),
	name="unet"):

	print "UNET"
	fmaps = fmaps_in
	num_fmaps = base_num_fmaps
	across = []
	print "input: ", fmaps.shape
	for layer in range(num_layers):
		for conv_pass in range(num_conv_passes):
			fmaps = tf.layers.conv3d(
				inputs=fmaps,
				filters=num_fmaps,
				kernel_size=kernel_size_down[layer],
				padding=padding,
				data_format="channels_first",
				activation=activation,
				name="%s_enc_conv_pass_%i_%i"%(name, layer, conv_pass))
		across.append(fmaps)

		fmaps = downsample(
			fmaps_in=fmaps,
			downsample_type=downsample_type,
			downsample_factors=downsample_factors[layer],
			padding=padding,
			voxel_size=voxel_size,
			name="%s_enc_downsample_%i"%(name, layer))

		print "layer ", (layer + 1), ": ", fmaps.shape
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
					name="%s_enc_conv_pass_bottom_%i"%(name, conv_pass))

	print "bottom ", fmaps.shape

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
			name="%s_dec_upsample_%i"%(name, layer))

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
				name="%s_dec_conv_pass_%i_%i"%(name, layer, conv_pass))
		print "layer ", (layer + 1), ": ", fmaps.shape

	print "last layer: ", fmaps.shape

	fmaps = tf.layers.conv3d(
		inputs=fmaps,
		filters=3, #flatten?
		kernel_size=1,
		padding=padding,
		data_format="channels_first",
		activation='sigmoid',
		name="%s_output_conv_pass_%i_%i"%(name, layer, conv_pass))
	print "output: ", fmaps.shape
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
	gt = tf.placeholder(tf.float32, (1,1, 196, 196, 196)) # flatten affmaps?

	unet, prior, posterior, f_comb = prob_unet(
		fmaps_in=raw,
		affmaps_in=gt,
		num_layers=3,
		latent_dim=6,
		base_num_fmaps=12,
		fmap_inc_factor=3,
		downsample_factors=[[3,3,3], [2,2,2], [2,2,2]],
		num_classes=4,
		num_1x1_convs=3)
