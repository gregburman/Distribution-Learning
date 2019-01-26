import tensorflow as tf
from  tensorflow_probability import distributions as tfd

def unet(
	input,
	layers,
	base_num_fmaps,
	fmap_inc_factor,
	num_conv_passes,
	downsample_factors,
	upsample_factors,
	kernel_size_down=[3,3,3],
	kernel_size_up=[3,3,3],
	activation='relu',
	fov=(1, 1, 1),
	voxel_size=(1, 1, 1)):

	# encode
	for l in layers:
		for c in num_conv_passes:
			# fout = conv pass (fin)
		# fout = downsample (fin)

	# decode
	for l in layers:
		for c in num_conv_passes:
			# fout = conv pass (fin)
		# fout = upsample (fin)