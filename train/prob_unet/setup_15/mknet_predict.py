from __future__ import print_function
import sys
sys.path.append('../../../')

import tensorflow as tf
# from  tensorflow_probability import distributions as tfd
import json

from models.unet import UNet
from models.encoder import Encoder
from models.f_comb import FComb

import numpy as np


def create_network(input_shape, setup_dir):

	print ("MKNET: PROB-UNET SAMPLE")
	print("")
	tf.reset_default_graph()

	raw = tf.placeholder(tf.float32, shape=input_shape, name="raw") # for gp
	raw_batched = tf.reshape(raw, (1, 1) + input_shape, name="raw_batched") # for tf

	print ("raw_batched: ", raw_batched.shape)
	print ("")

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
		activation_type = tf.nn.relu,
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
		activation_type = tf.nn.relu,
		downsample_type = "max_pool",
		voxel_size = (1, 1, 1),
		name = "prior")
	prior.build()
	print ("")


	z = prior.sample()*100

	f_comb = FComb(
		fmaps_in = unet.get_fmaps(),
		sample_in = z,
		# sample_in = tf.reshape(tf.constant([5,5,5,-5,-5,-5], dtype=np.float32), (1,6)),
		num_1x1_convs = 3,
		num_channels = 12,
		padding_type = 'valid',
		activation_type = tf.nn.relu,
		voxel_size = (1, 1, 1))
	f_comb.build()
	print ("")

	pred_logits = tf.layers.conv3d(
		inputs=f_comb.get_fmaps(),
		filters=3, 
		kernel_size=1,
		padding='valid',
		data_format="channels_first",
		activation=None,
		name="affs")
	print ("")

	broadcast_sample = f_comb.out

	pred_affs = tf.sigmoid(pred_logits)

	output_shape_batched = pred_logits.get_shape().as_list()
	output_shape = output_shape_batched[1:] # strip the batch dimension

	pred_logits = tf.squeeze(pred_logits, axis=[0], name="pred_logits")
	pred_affs = tf.squeeze(pred_affs, axis=[0], name="pred_affs")
	
	# sample_z = tf.squeeze(prior.sample(), axis=[0], name="sample_z")
	# sample_z = prior.sample()
	# sample_z_batched = tf.reshape(sample_z, (1, 1, 6), name="sample_z") # for tf
	# print("sample_z", sample_z_batched.shape)

	print ("pred_logits: ", pred_logits.shape)
	print ("pred_affs: ", pred_affs.shape)

	output_shape = output_shape[1:]
	print("input shape : %s"%(input_shape,))
	print("output shape: %s"%(output_shape,))

	tf.train.export_meta_graph(filename=setup_dir + 'predict_net.meta')

	config = {
		'raw': raw.name,
		'pred_affs': pred_affs.name,
		'input_shape': input_shape,
		'output_shape': output_shape,
		'broadcast': broadcast_sample.name
		# 'sample_z': sample_z_batched.name
	}
	with open(setup_dir + 'predict_config.json', 'w') as f:
		json.dump(config, f)

def z(fmaps, latent_dims):
	mean = fmaps[:, :latent_dims]
	log_sigma = fmaps[:, latent_dims:]
	return tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_sigma))

if __name__ == "__main__":

	setup_name = sys.argv[1]
	setup_dir = "train/prob_unet/" + setup_name + "/"
	print (setup_dir)
	create_network((132, 132, 132), setup_dir)