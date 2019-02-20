from __future__ import print_function
import tensorflow as tf
import helper


class FComb():

	def __init__ (self,
		fmaps_in,
		sample_in,
		num_1x1_convs,
		num_channels,
		padding_type,
		activation_type,
		voxel_size):

		self.fmaps_in = fmaps_in
		self.sample_in = sample_in
		self.num_1x1_convs = num_1x1_convs
		self.num_channels = num_channels
		self.padding_type = padding_type
		self.activation_type = activation_type
		self.voxel_size = voxel_size
		self.name = "f_comb"

		self.fmaps = None

	def build(self):
		print ("BUILD: ", self.name)

		channel_axis = 1
		spatial_axis = [2,3,4]
		sample = self.sample_in
		print ("fmaps_in: ", self.fmaps_in.shape)

		# broadcast
		shape = self.fmaps_in.get_shape()
		spatial_shape = [shape[axis] for axis in spatial_axis]
		multiples = [1] + spatial_shape
		multiples.insert(channel_axis, 1)

		if len(sample.get_shape()) == 2:
			sample = tf.expand_dims(sample, axis=2)
			sample = tf.expand_dims(sample, axis=2)
			sample = tf.expand_dims(sample, axis=2)

		print ("sample: ", sample.shape)

		broadcast_sample = tf.tile(sample, multiples)
		fmaps = tf.concat([self.fmaps_in, broadcast_sample], axis=channel_axis)

		print ("broadcast_sample: ", broadcast_sample.shape)
		print ("fmaps concat: ", fmaps.shape)

		for conv_pass in range(self.num_1x1_convs):
			fmaps = tf.layers.conv3d(
				inputs = fmaps,
				filters = self.num_channels,
				kernel_size = 1,
				padding = self.padding_type,
				data_format = "channels_first",
				activation = self.activation_type,
				name = "%s_conv_pass_%i"%(self.name, conv_pass))

		print ("output: ", fmaps.shape)
		self.fmaps = fmaps

	def get_fmaps(self):
		return self.fmaps

if __name__ == "__main__":

	raw = tf.placeholder(tf.float32, (1,12,68,68,68))
	sample = tf.placeholder(tf.float32, (1,6))

	f_comb = FComb(
		fmaps_in = raw,
		sample_in = sample,
		num_1x1_convs = 3,
		num_channels = 12,
		padding_type = 'valid',
		activation_type = 'relu',
		voxel_size = (1, 1, 1))
	f_comb.build()
	fmaps = f_comb.get_fmaps()
	print ("fmaps: ", fmaps.shape)