import gunpowder as gp
from BlobGenerator import BlobGenerator
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

	size = 500
	
	gt_labels = gp.ArrayKey('LABELS')
	gt_affs= gp.ArrayKey('AFFINITIES')
	# gt_labels_mask = gp.ArrayKey('LABELS_MASK')
	# gt_affs_mask = gp.ArrayKey('AFFINITIES_MASK')
	# (self, size, n_objects, points_per_skeleton, smoothness, interpolation):
	
	blobgenerator = BlobGenerator(size, 20, 5, 2, "linear")
	addaffinities = gp.AddAffinities(
            [[-1, 0, 0], [0, -1, 0]],
            gt_labels,
            gt_affs,
            # unlabelled=gt_labels_mask,
            # affinities_mask=gt_affs_mask
	) 
	
	pipeline = blobgenerator + addaffinities

	voxel_size = gp.Coordinate((1, 1, 1))
	input_size = gp.Coordinate((size,size, 1))*voxel_size
	output_size = gp.Coordinate((size-1, size-1, 1))*voxel_size

	request = gp.BatchRequest()
	request.add(gt_labels, input_size)
	request.add(gt_affs, output_size)
	# request.add(gt_labels_mask, output_size)
	# request.add(gt_affs_mask, output_size)

	with gp.build(pipeline) as p:
		for i in range(1):
			req = p.request_batch(request)
			labels = req.arrays[gt_labels].data
			labels = labels.reshape((size,size))
			print 'channels: ', req.arrays[gt_affs].data.shape
			affs1 = req.arrays[gt_affs].data
			affs1 = req.arrays[gt_affs].data[0]
			affs1 = affs1.reshape((size-1,size-1))
			affs2 = req.arrays[gt_affs].data[1]
			affs2 = affs2.reshape((size-1,size-1))
			# print affs1

			f1 = plt.figure(1)
			plt.imshow(labels, cmap='nipy_spectral')
			f2 = plt.figure(2)
			plt.imshow(affs1, alpha=0.5)
			# f3 = plt.figure(3)
			plt.imshow(affs2, alpha=0.5)
			plt.show()