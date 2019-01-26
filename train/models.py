import tensorflow as tf
from  tensorflow_probability import distributions as tfd

def conv_pass(
		fmaps_in,
		kernel_sizes,
		num_fmaps,
		activation='relu',
		name='conv_pass',
		fov=(1, 1, 1),
		voxel_size=(1, 1, 1)):

	
	fmaps = fmaps_in
	if activation is not None:
		activation = getattr(tf.nn, activation)

	for i, kernel_size in enumerate(kernel_sizes):
		if isinstance(kernel_size, int):
			kernel_size = [kernel_size]*len(voxel_size)

		fov = tuple(f + (k - 1)*vs for f, k, vs in zip(fov, kernel_size, voxel_size))
		fmaps = tf.layers.conv3d(
			inputs=fmaps,
			filters=num_fmaps,
			kernel_size=kernel_size,
			padding='valid',
			data_format='channels_first',
			activation=activation,
			name=name + '_%i'%i)

	return fmaps, fov

def downsample(
		fmaps_in,
		factors,
		name='down',
		voxel_size=(1, 1, 1)):
	
	voxel_size = tuple(vs*fac for vs, fac in zip(voxel_size, factors))
	fmaps = tf.layers.max_pooling3d(
		inputs=fmaps_in,
		pool_size=factors,
		strides=factors,
		padding='valid',
		data_format='channels_first',
		name=name)

	return fmaps, voxel_size

def upsample(
		fmaps_in,
		factors,
		num_fmaps,
		activation='relu',
		name='up',
		voxel_size=(1, 1, 1)):

	voxel_size = tuple(vs/fac for vs, fac in zip(voxel_size, factors))
	if activation is not None:
		activation = getattr(tf.nn, activation)

	fmaps = tf.layers.conv3d_transpose(
		fmaps_in,
		filters=num_fmaps,
		kernel_size=factors,
		strides=factors,
		padding='valid',
		data_format='channels_first',
		activation=activation,
		name=name)

	return fmaps, voxel_size

def crop_zyx(fmaps_in, shape):

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
	
	fmaps = tf.slice(fmaps_in, offset, size)

	return fmaps

def encoder(
		fmaps_in,
		num_fmaps,
		fmap_inc_factors,
		downsample_factors	,
		kernel_size_down=None,
		kernel_size_up=None,
		activation='relu',
		layer=0,
		fov=(1, 1, 1),
		voxel_size=(1, 1, 1)):

	prefix = "    "*layer
	print(prefix + "Creating Prior layer %i"%layer)
	print(prefix + "f_in: " + str(fmaps_in.shape))
	
	if isinstance(fmap_inc_factors, int):
		fmap_inc_factors = [fmap_inc_factors]*len(downsample_factors)

	if kernel_size_down is None:
		kernel_size_down = [ [3, 3, 3] ]*(len(downsample_factors) + 1)

	assert (
		len(fmap_inc_factors) ==
		len(downsample_factors) ==
		len(kernel_size_down) - 1)

	# convolve
	f_out, fov = conv_pass(
		fmaps_in=fmaps_in,
		kernel_sizes=kernel_size_down[layer],
		num_fmaps=num_fmaps,
		activation=activation,
		name='prior_layer_%i'%layer,
		fov=fov,
		voxel_size=voxel_size)

	# last layer does not recurse
	last_layer = (layer == len(downsample_factors))
	if last_layer:
		print(prefix + "last layer")
		print(prefix + "f_out: " + str(f_out.shape))
		return f_out, fov, voxel_size

	# downsample
	g_in, voxel_size = downsample(
		fmaps_in=f_out,
		factors=downsample_factors[layer],
		name='prior_down_%i_to_%i'%(layer, layer + 1),
		voxel_size=voxel_size)

	# recursive encoder
	g_out, fov, voxel_size = encoder(
		fmaps_in=g_in,
		num_fmaps=num_fmaps*fmap_inc_factors[layer],
		fmap_inc_factors=fmap_inc_factors,
		downsample_factors=downsample_factors,
		kernel_size_down=kernel_size_down,
		kernel_size_up=kernel_size_up,
		activation=activation,
		layer=layer+1,
		fov=fov,
		voxel_size=voxel_size)
	
	return g_out, fov, voxel_size

def unet(
		fmaps_in,
		num_fmaps,
		fmap_inc_factors,
		downsample_factors,
		kernel_size_down=None,
		kernel_size_up=None,
		activation='relu',
		layer=0,
		fov=(1, 1, 1),
		voxel_size=(1, 1, 1)):

	prefix = "    "*layer
	print(prefix + "Creating U-Net layer %i"%layer)
	print(prefix + "f_in: " + str(fmaps_in.shape))

	if isinstance(fmap_inc_factors, int):
		fmap_inc_factors = [fmap_inc_factors]*len(downsample_factors)

	# by default, create 2 3x3x3 convolutions per layer
	if kernel_size_down is None:
		kernel_size_down = [ [3, 3] ]*(len(downsample_factors) + 1)
	if kernel_size_up is None:
		kernel_size_up = [ [3, 3] ]*(len(downsample_factors) + 1)

	assert (
		len(fmap_inc_factors) ==
		len(downsample_factors) ==
		len(kernel_size_down) - 1 ==
		len(kernel_size_up) - 1)

	# convolve
	f_left, fov = conv_pass(
		fmaps_in,
		kernel_sizes=kernel_size_down[layer],
		num_fmaps=num_fmaps,
		activation=activation,
		name='unet_layer_%i_left'%layer,
		fov=fov,
		voxel_size=voxel_size)

	# last layer does not recurse
	bottom_layer = (layer == len(downsample_factors))
	if bottom_layer:
		print(prefix + "bottom layer")
		print(prefix + "f_out: " + str(f_left.shape))
		return f_left, fov, voxel_size

	# downsample
	g_in, voxel_size = downsample(
		f_left,
		downsample_factors[layer],
		'unet_down_%i_to_%i'%(layer, layer + 1),
		voxel_size=voxel_size)

	# recursive U-net
	g_out, fov, voxel_size = unet(
		g_in,
		num_fmaps=num_fmaps*fmap_inc_factors[layer],
		fmap_inc_factors=fmap_inc_factors,
		downsample_factors=downsample_factors,
		kernel_size_down=kernel_size_down,
		kernel_size_up=kernel_size_up,
		activation=activation,
		layer=layer+1,
		fov=fov,
		voxel_size=voxel_size)

	print(prefix + "g_out: " + str(g_out.shape))

	# upsample
	g_out_upsampled, voxel_size = upsample(
		g_out,
		downsample_factors[layer],
		num_fmaps,
		activation=activation,
		name='unet_up_%i_to_%i'%(layer + 1, layer),
		voxel_size=voxel_size)

	print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

	# copy-crop
	f_left_cropped = crop_zyx(f_left, g_out_upsampled.get_shape().as_list())

	print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

	# concatenate along channel dimension
	f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

	print(prefix + "f_right: " + str(f_right.shape))

	# convolve
	f_out, fov = conv_pass(
		f_right,
		kernel_sizes=kernel_size_up[layer],
		num_fmaps=num_fmaps,
		name='unet_layer_%i_right'%layer,
		fov=fov,
		voxel_size=voxel_size)

	print(prefix + "f_out: " + str(f_out.shape))

	return f_out, fov, voxel_size

def enc(
	fmaps_in,
	num_fmaps,
	fmap_inc_factors,
	downsample_factors	,
	latent_dim,
	kernel_size_down=None,
	kernel_size_up=None,
	activation='relu',
	voxel_size=(1, 1, 1)):

	for i in layers:
		for j in conv_passes
			conv
		downsample



def prior(
	fmaps_in,
	num_fmaps,
	fmap_inc_factors,
	downsample_factors	,
	latent_dim,
	kernel_size_down=None,
	kernel_size_up=None,
	activation='relu',
	voxel_size=(1, 1, 1)):

	enc, _, _ = encoder(
		fmaps_in=fmaps_in,
		num_fmaps=num_fmaps,
		fmap_inc_factors=fmap_inc_factors,
		downsample_factors=downsample_factors,
		kernel_size_down=kernel_size_down,
		kernel_size_up=kernel_size_up,
		activation=activation,
		layer=0,
		fov=(1, 1, 1),
		voxel_size=(1, 1, 1))

	r_mean = tf.reduce_mean(enc, axis=[2,3,4], keepdims=True)

	latents = tf.layers.conv3d(
		inputs=r_mean,
		filters=latent_dim*2,
		kernel_size=(1, 1, 1),
		padding='valid',
		data_format='channels_first',
		activation=activation,
		name="gauss")

	mu_log_sigma = tf.squeeze(latents, axis=[2,3,4])
	mu = mu_log_sigma[:, :latent_dim]
	log_sigma = mu_log_sigma[:, latent_dim:]

	print "enc: ", enc.get_shape().as_list()
	print "r_mean: ", r_mean.get_shape().as_list()
	print "latents: ", latents.get_shape().as_list()
	print "mu_log_sigma: ", mu_log_sigma.get_shape().as_list()
	print "mu: ", mu.get_shape().as_list()
	print "log_sigma: ", log_sigma.get_shape().as_list()

	return tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))


if __name__ == "__main__":

	# prior = enc + trans + sampling
	# posterior = enc + transf + sampling
	# unet = unet (own enc + dec)

	raw = tf.placeholder(tf.float32, (1,1,196,196,196))
	gt = tf.placeholder(tf.float32, (1,3, 68, 68, 68))

	num_fmaps = 12
	fmap_inc_factors = 3
	downsample_factors = [[3,3,3],[2,2,2],[2,2,2]]
	latent_dim = 6

	# unet, _, _ = unet(
	# fmaps_in=raw,
	# num_fmaps=num_fmaps,
	# fmap_inc_factors=fmap_inc_factors,
	# downsample_factors=downsample_factors)

	# prior = encoder(
	# 	fmaps_in=raw,
	# 	num_fmaps=num_fmaps,
	# 	fmap_inc_factors=fmap_inc_factors,
	# 	downsample_factors=downsample_factors,
	# 	latent_dim=6)

	posterior = encoder(
		fmaps_in=raw,
		num_fmaps=num_fmaps,
		fmap_inc_factors=fmap_inc_factors,
		downsample_factors=downsample_factors)