from __future__ import print_function
import tensorflow as tf
from  tensorflow_probability import distributions as tfd
import helper
from unet import UNet
from encoder import Encoder


class ProbUNet():

	# def __init__ (self,
	# 	fmaps_in,
	# 	affmaps_in,
	# 	num_layers,
	# 	latent_dims,
	# 	base_channels,
	# 	channel_inc_factor,
	# 	resample_factors,
	# 	padding_type,
	# 	num_conv_passes,
	# 	down_kernel_size,
	# 	up_kernel_size,
	# 	activation_type,
	# 	downsample_type,
	# 	upsample_type,
	# 	voxel_size):

	# 	self.fmaps_in = fmaps_in
	# 	self.affmaps_in = affmaps_in
	# 	self.num_layers = num_layers
	# 	self.latent_dims = latent_dims
	# 	self.base_channels = base_channels
	# 	self.channel_inc_factor = channel_inc_factor
	# 	self.resample_factors = resample_factors
	# 	self.padding_type = padding_type
	# 	self.num_conv_passes = num_conv_passes
	# 	self.down_kernel_size = down_kernel_size
	# 	self.up_kernel_size = up_kernel_size
	# 	self.activation_type = activation_type
	# 	self.downsample_type = downsample_type
	# 	self.upsample_type = upsample_type
	# 	self.voxel_size = voxel_size
	# 	self.name = "prob_unet"

	# 	self.unet = UNet(self.fmaps_in, self.num_layers, self.base_channels, self.channel_inc_factor,
	# 		self.resample_factors, self.padding_type, self.num_conv_passes, self.down_kernel_size,
	# 		self.up_kernel_size, self.activation_type, self.downsample_type, self.upsample_type, self.voxel_size)

	# 	self.prior = Encoder(self.fmaps_in, None, self.num_layers, self.latent_dims,
	# 		self.base_channels, self.channel_inc_factor, self.resample_factors, self.padding_type,
	# 		self.num_conv_passes, self.down_kernel_size, self.activation_type, self.downsample_type,
	# 		self.voxel_size, "prior")

	# 	self.posterior = Encoder(self.fmaps_in, self.affmaps_in, self.num_layers, self.latent_dims,
	# 		self.base_channels, self.channel_inc_factor, self.resample_factors, self.padding_type,
	# 		self.num_conv_passes, self.down_kernel_size, self.activation_type, self.downsample_type,
	# 		self.voxel_size, "posterior")

	# 	# get sample

	# 	self.f_comb = 	FComb(self.unet.get_fmaps(), sample, self.num_classes, self.num_1x1_convs, ,
	# 	padding_type = 'valid',
	# 	activation_type = 'relu',
	# 	voxel_size = (1, 1, 1))

	# todo: figure out how to "overload" __init__
	def __init__(self,
		unet,
		prior,
		posterior,
		f_comb):

		self.unet = unet
		self.prior = prior
		self.posterior = posterior
		self.f_comb = f_comb
		self.f_comb.set_sample(self.posterior.sample())
		self.name = "prob_unet"

		self.fmaps = None

	def build(self):
		print ("BUILD:", self.name)
		print("")

		#build component networks
		self.unet.build()
		print("")
		self.prior.build()
		print("")
		self.posterior.build()
		print("")
		self.f_comb.build()
		print("")

	def get_fmaps(self):
		return self.fmaps


if __name__ == "__main__":

	raw = tf.placeholder(tf.float32, (1,1,196,196,196))
	gt = tf.placeholder(tf.float32, (1,3, 196, 196, 196))

	prob_unet = ProbUNet(
		fmaps_in = raw,
		affmaps_in = gt,
		num_layers = 3,
		latent_dims = 6,
		base_channels = 12,
		channel_inc_factor = 3,
		resample_factors = [[3,3,3], [2,2,2], [2,2,2]],
		padding_type = "valid",
		num_conv_passes = 2,
		down_kernel_size = [3, 3, 3],
		up_kernel_size = [3, 3, 3],
		activation_type = "relu",
		downsample_type =  "max_pool",
		upsample_type =  "conv_transpose",
		voxel_size =  (1, 1, 1))
	prob_unet.build()