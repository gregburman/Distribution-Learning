from __future__ import print_function
import tensorflow as tf
from  tensorflow_probability import distributions as tfd
import helper

class Encoder():

	def __init__ (self,
		fmaps_in,
		affmaps_in,
		num_layers,
		latent_dims,
		base_channels,
		channel_inc_factor,
		downsample_factors,
		padding_type,
		num_conv_passes,
		down_kernel_size,
		activation_type,
		downsample_type,
		voxel_size,
		name):

		self.fmaps_in = fmaps_in
		self.affmaps_in = affmaps_in
		self.num_layers = num_layers
		self.latent_dims = latent_dims
		self.base_channels = base_channels
		self.channel_inc_factor = channel_inc_factor
		self.downsample_factors = downsample_factors
		self.padding_type = padding_type
		self.num_conv_passes = num_conv_passes
		self.down_kernel_size = down_kernel_size
		self.activation_type = activation_type
		self.downsample_type = downsample_type
		self.voxel_size = voxel_size
		self.name = name

		self.fmaps = None

	def build(self):
		print ("BUILD:", self.name)

		channel_axis = 1
		spacial_axes = [2,3,4]
		fmaps = self.fmaps_in
		num_channels = self.base_channels

		if self.affmaps_in is not None:
			self.affmaps_in = tf.cast(self.affmaps_in, tf.float32)
			fmaps = tf.concat([fmaps, self.affmaps_in], axis=channel_axis)

		print ("fmaps_in : ", fmaps.shape)
		for layer in range(self.num_layers):
			for conv_pass in range(self.num_conv_passes):
				fmaps = tf.layers.conv3d(
					inputs = fmaps,
					filters = num_channels,
					kernel_size = self.down_kernel_size[layer],
					padding = self.padding_type,
					data_format = "channels_first",
					activation = self.activation_type,
					name = "%s_layer_%i_conv_pass_%i" % (self.name, layer, conv_pass))

			fmaps = helper.downsample(
				fmaps_in = fmaps,
				downsample_type = self.downsample_type,
				downsample_factors = self.downsample_factors[layer],
				padding_type = self.padding_type,
				voxel_size = self.voxel_size,
				name = "%s_downsample_layer_%i" % (self.name, layer))

			print ("layer ", (layer + 1), ": ", fmaps.shape)
			num_channels *= self.channel_inc_factor

			if layer == self.num_layers - 1:
				for conv_pass in range(self.num_conv_passes):
					fmaps = tf.layers.conv3d(
						inputs = fmaps,
						filters = num_channels,
						kernel_size = self.down_kernel_size[layer],
						padding = self.padding_type,
						data_format = "channels_first",
						activation = self.activation_type,
						name = "%s_bottom_conv_pass_%i" % (self.name, conv_pass))

		print ("bottom   : ", fmaps.shape)

		encoding = tf.reduce_mean(fmaps, axis=spacial_axes, keepdims=True)

		print ("encoding : ", encoding.shape)

		mu_log_sigma = tf.layers.conv3d(
			inputs = encoding,
			filters = self.latent_dims * 2,
			kernel_size = 1,
			padding = self.padding_type,
			data_format = "channels_first",
			activation = self.activation_type,
			name = "%s_1x1_conv" % (self.name))

		mu_log_sigma = tf.squeeze(mu_log_sigma, axis=spacial_axes)
		mean = mu_log_sigma[:, :self.latent_dims]
		log_sigma = mu_log_sigma[:, self.latent_dims:]

		f_out = tfd.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_sigma))
		print ("output   : ", f_out.event_shape)
		self.fmaps = f_out

	def get_fmaps(self):
		return self.fmaps


if __name__ == "__main__":

	raw = tf.placeholder(tf.float32, (1,1,196,196,196))
	gt = tf.placeholder(tf.float32, (1,3, 196, 196, 196))

	prior = Encoder(
		fmaps_in = raw,
		affmaps_in = None,
		num_layers = 3,
		latent_dims = 6,
		base_channels = 12,
		channel_inc_factor = 3,
		downsample_factors = [[3,3,3], [2,2,2], [2,2,2]],
		padding_type = "valid",
		num_conv_passes = 2,
		down_kernel_size = [3, 3, 3],
		activation_type = "relu",
		downsample_type = "max_pool",
		voxel_size = (1, 1, 1),
		name = "prior")
	prior.build()
	fmaps = prior.get_fmaps()
	print ("fmaps: ", fmaps.event_shape)

	print ("")

	posterior = Encoder(
		fmaps_in = raw,
		affmaps_in = gt,
		num_layers = 3,
		latent_dims = 6,
		base_channels = 12,
		channel_inc_factor = 3,
		downsample_factors = [[3,3,3], [2,2,2], [2,2,2]],
		padding_type = "valid",
		num_conv_passes = 2,
		down_kernel_size = [3, 3, 3],
		activation_type = "relu",
		downsample_type = "max_pool",
		voxel_size = (1, 1, 1),
		name = "posterior")
	posterior.build()
	fmaps = posterior.get_fmaps()
	print ("fmaps: ", fmaps.event_shape)