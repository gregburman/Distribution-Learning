from __future__ import print_function
import tensorflow as tf
from  tensorflow_probability import distributions as tfd
import helper


class ProbUNet():

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
		self.resample_factors = resample_factors
		self.padding_type = padding_type
		self.num_conv_passes = num_conv_passes
		self.down_kernel_size = down_kernel_size
		self.up_kernel_size = up_kernel_size
		self.activation_type = activation_type
		self.downsample_type = downsample_type
		self.upsample_type = upsample_type
		self.voxel_size = voxel_size
		self.name = name

		self.unet = None
		self.prior = None
		self.posterior = None
		self.f_comb = None
		self.fmaps = None

	def build(self):
		print ("BUILD:", self.name)

		self.unet.build()
		self.prior.build()
		self.posterior.build()

		self.fmaps = 

	def get_fmaps(self):
		return self.fmaps


	if __name__ == "__main__":

		raw = tf.placeholder(tf.float32, (1,1,196,196,196))
		gt = tf.placeholder(tf.float32, (1,3, 196, 196, 196))

		prob_unet = ProbUNet(
			)