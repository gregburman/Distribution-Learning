from __future__ import print_function
import sys
sys.path.append('../')

import tensorflow as tf
from  tensorflow_probability import distributions as tfd
import json

from models.unet import UNet
from models.encoder import Encoder
from models.f_comb import FComb


def create_network(input_shape, output_shape, name):

	# beta = 1 stuck at loss=0.693147
	# beta = 0.75 stuck at loss=0.693147
	beta = 0.1

	tf.reset_default_graph()

	raw = tf.placeholder(tf.float32, shape=input_shape, name="raw") # for gp
	raw_batched = tf.reshape(raw, (1, 1) + input_shape, name="raw_batched") # for tf
	gt_affs_in = tf.placeholder(tf.float32, shape = (3,) + input_shape, name="gt_affs_in")
	gt_affs_in_batched = tf.reshape(gt_affs_in, (1, 3) + input_shape, name="gt_affs_in_batched")

	print ("raw_batched: ", raw_batched.shape)
	print ("gt_affs_in_batched: "), gt_affs_in_batched.shape
	print ("")

	# unet, prior, posterior, f_comb = prob_unet.prob_unet(
	# 	fmaps_in=raw_batched,
	# 	affmaps_in=gt_affs_in_batched,
	# 	num_layers=3,
	# 	num_classes=3,
	# 	latent_dim=6,
	# 	base_num_fmaps=12,
	# 	fmap_inc_factor=3,
	# 	downsample_factors=[[2,2,2], [2,2,2], [2,2,2]],
	# 	num_1x1_convs=3)

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
		downsample_factors = [[3,3,3], [2,2,2], [2,2,2]],
		padding_type = "valid",
		num_conv_passes = 2,
		down_kernel_size = [3, 3, 3],
		activation_type = "relu",
		downsample_type = "max_pool",
		voxel_size = (1, 1, 1),
		name = "prior")
	prior.build()
	print ("")

	posterior = Encoder(
		fmaps_in = raw_batched,
		affmaps_in = gt_affs_in_batched,
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
	print ("")

	f_comb = FComb(
		fmaps_in = unet.get_fmaps(),
		sample_in = posterior.sample(),
		num_classes = 3,
		num_1x1_convs = 3,
		num_channels = 12,
		padding_type = 'valid',
		activation_type = 'relu',
		voxel_size = (1, 1, 1))
	f_comb.build()

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
	print ("output_shape_batched: "), output_shape_batched
	output_shape = output_shape_batched[1:] # strip the batch dimension

	pred_affs = tf.reshape(affs_batched, output_shape, name="pred_affs")
	gt_affs_out = tf.placeholder(tf.float32, shape=output_shape, name="gt_affs_out")
	pred_affs_loss_weights = tf.placeholder(tf.float32, shape=output_shape, name="pred_affs_loss_weights")

	sample_p = prior.get_fmaps()
	sample_q = posterior.get_fmaps()

	kl_loss = tf.distributions.kl_divergence(sample_p, sample_q)
	ce_loss = tf.losses.sigmoid_cross_entropy(gt_affs_out, pred_affs, pred_affs_loss_weights)
	# mse_loss = tf.losses.mean_squared_error(	gt_affs_out, 	pred_affs, pred_affs_loss_weights)
	loss = ce_loss + beta * kl_loss
	# summary = tf.summary.scalar('loss', loss)

	# kl_summary = tf.summary.scalar('kl_loss', kl_loss)
	# ce_summary = tf.summary.scalar('ce_loss', ce_loss)
	# summary = tf.summary.scalar(['loss'], loss)

	# opt = tf.train.AdamOptimizer(
	# 	learning_rate=1e-6,
	# 	beta1=0.95,
	# 	beta2=0.999,
	# 	epsilon=1e-8)
	opt = tf.train.AdamOptimizer()
	optimizer = opt.minimize(loss)

	output_shape = output_shape[1:]
	print("input shape : %s"%(input_shape,))
	print("output shape: %s"%(output_shape,))

	tf.train.export_meta_graph(filename=name + '.meta')

	config = {
		'raw': raw.name,
		'pred_affs': pred_affs.name,
		'gt_affs_in': gt_affs_in.name,
		'gt_affs_out': gt_affs_out.name,
		'pred_affs_loss_weights': pred_affs_loss_weights.name,
		'loss': loss.name,
		'optimizer': optimizer.name,
		'input_shape': input_shape,
		'output_shape': output_shape,
		# 'summary': summary.name,
	}
	with open(name + '.json', 'w') as f:
		json.dump(config, f)

# def kl(mean, log_sigma, batch_size, free_bits=0.0):
#     kl_div = tf.reduce_sum(tf.maximum(free_bits, 0.5 * (tf.square(mean) + tf.exp(2 * log_sigma) - 2 * log_sigma - 1)))
#     kl_div /= float(batch_size)
#     return kl_div

# def kl(p, q):
# 	z_q = q.sample()
# 	log_q = q.log_prob(z_q)
# 	log_p = p.log_prob(z_q)
# 	return log_q - log_p

if __name__ == "__main__":
	create_network((132, 132, 132), (44, 44, 44), 'train_net') # shape -1 
