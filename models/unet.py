from __future__ import print_function
import tensorflow as tf
import helper


class UNet():

	def __init__ (self,
		fmaps_in,
		num_layers,
		base_channels,
		channel_inc_factor,
		resample_factors,
		padding_type,
		num_conv_passes,
		down_kernel_size,
		up_kernel_size,
		activation_type,
		downsample_type,
		upsample_type,
		voxel_size):

		self.fmaps_in = fmaps_in
		self.num_layers = num_layers
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
		self.name = "unet"

		self.fmaps = None

	def build(self):
		print ("BUILD: ", self.name)

		fmaps = self.fmaps_in
		num_channels = self.base_channels
		across = []

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
					name = "%s_down_layer_%i_conv_pass_%i" % (self.name, layer, conv_pass))
			across.append(fmaps)

			fmaps = helper.downsample(
				fmaps_in = fmaps,
				downsample_type = self.downsample_type,
				downsample_factors = self.resample_factors[layer],
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

		for layer in reversed(range(self.num_layers)):
			num_channels /= self.channel_inc_factor
			fmaps = helper.upsample(
				fmaps_in = fmaps,
				num_channels = num_channels,
				upsample_type = self.upsample_type,
				upsample_factors = self.resample_factors[layer],
				activation_type = self.activation_type,
				padding_type = self.padding_type,
				voxel_size = self.voxel_size,
				name = "%s_upsample_layer_%i" % (self.name, layer))

			cropped = helper.crop(across[layer], fmaps.get_shape().as_list())
			fmaps = tf.concat([cropped, fmaps], 1)

			for conv_pass in range(self.num_conv_passes):
				fmaps = tf.layers.conv3d(
					inputs = fmaps,
					filters = num_channels,
					kernel_size = self.up_kernel_size[layer],
					padding = self.padding_type,
					data_format = "channels_first",
					activation = self.activation_type,
					name = "%s_up_layer_%i_conv_pass_%i" % (self.name, layer, conv_pass))
			print ("layer ", (layer + 1), ": ", fmaps.shape)

		print ("fmaps_out: ", fmaps.shape)
		
		self.fmaps = fmaps

	def get_fmaps(self):
		return self.fmaps


if __name__ == "__main__":

	raw = tf.placeholder(tf.float32, (1,1,132,132,132))

	unet = UNet(
		fmaps_in = raw,
		num_layers = 3,
		base_channels = 12,
		channel_inc_factor = 3,
		resample_factors = [[2,2,2], [2,2,2], [2,2,2]],
		padding_type = "valid",
		num_conv_passes = 2,
		down_kernel_size = [3, 3, 3],
		up_kernel_size = [3, 3, 3],
		activation_type = "relu",
		downsample_type = "max_pool",
		upsample_type = "conv_transpose",
		voxel_size = (1, 1, 1))
	unet.build()
	fmaps = unet.get_fmaps()
	print ("fmaps: ", fmaps.shape)
