from __future__ import print_function
import sys
sys.path.append('../')

import tensorflow as tf
from  tensorflow_probability import distributions as tfd
import json

from models.unet import UNet
from models.encoder import Encoder
from models.f_comb import FComb


def create_network(input_shape, name):

	tf.reset_default_graph()

	raw = tf.placeholder(tf.float32, shape=input_shape, name="raw") # for gp
	raw_batched = tf.reshape(raw, (1, 1) + input_shape, name="raw_batched") # for tf

	# print ("raw_batched: ", raw_batched.shape)
	# print ("")

	unet = UNet(
		fmaps_in = raw_batched,
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
	print("")

	prior = Encoder(
		fmaps_in = raw_batched,
		affmaps_in = None,
		num_layers = 3,
		latent_dims = 6,
		base_channels = 12,
		channel_inc_factor = 3,
		downsample_factors = [[2,2,2], [2,2,2], [2,2,2]],
		padding_type = "valid",
		num_conv_passes = 2,
		down_kernel_size = [3, 3, 3],
		activation_type = "relu",
		downsample_type = "max_pool",
		voxel_size = (1, 1, 1),
		name = "prior")
	prior.build()
	print ("")

	f_comb = FComb(
		fmaps_in = unet.get_fmaps(),
		sample_in = prior.sample(),
		num_classes = 3,
		num_1x1_convs = 3,
		num_channels = 12,
		padding_type = 'valid',
		activation_type = 'relu',
		voxel_size = (1, 1, 1))
	f_comb.build()
	print ("")

	affs_batched = tf.layers.conv3d(
		inputs=f_comb.get_fmaps(),
		filters=3, 
		kernel_size=1,
		padding='valid',
		data_format="channels_first",
		activation='sigmoid',
		name="affs")
	print ("")

	output_shape_batched = affs_batched.get_shape().as_list()
	print ("output_shape_batched: ", output_shape_batched)
	output_shape = output_shape_batched[1:] # strip the batch dimension

	pred_affs = tf.reshape(affs_batched, output_shape, name="pred_affs")

	output_shape = output_shape[1:]
	print("input shape : %s"%(input_shape,))
	print("output shape: %s"%(output_shape,))

	tf.train.export_meta_graph(filename=name + '.meta')

	config = {
		'raw': raw.name,
		'pred_affs': pred_affs.name,
		'input_shape': input_shape,
		'output_shape': output_shape,
	}
	with open(name + '.json', 'w') as f:
		json.dump(config, f)

if __name__ == "__main__":
	create_network((132, 132, 132), 'predict_net')