import gunpowder as gp
from BlobGenerator import BlobGenerator
import matplotlib.pyplot as plt

if __name__ == "__main__":
	
	# blobs = gp.ArrayKey('BLOBS')
	gt_labels = gp.ArrayKey('LABELS')
	gt_affs= gp.ArrayKey('AFFINITIES')

	# gt_labels_mask = gp.ArrayKey('LABELS_MASK')
	# gt_affs_mask = gp.ArrayKey('AFFINITIES_MASK')

	blobgenerator = BlobGenerator(101, 127, 2)
	addaffinities = gp.AddAffinities(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            gt_labels,
            gt_affs,
            # unlabelled=gt_labels_mask,
            # affinities_mask=gt_affs_mask
	) 
	
	pipeline = blobgenerator + addaffinities

	print 'addaffinities: ', addaffinities

	voxel_size = gp.Coordinate((1, 1, 1))
	input_size = gp.Coordinate((101,101, 1))*voxel_size
	output_size = gp.Coordinate((50, 50, 1))*voxel_size

	request = gp.BatchRequest()
	# request.add(blobs, input_size)
	request.add(gt_labels, input_size)
	request.add(gt_affs, output_size)
	# request.add(gt_labels_mask, output_size)
	# request.add(gt_affs_mask, output_size)
	print 'request: ', request

	with gp.build(pipeline) as p:
		for i in range(1):
			p.request_batch(request)
			# test = x.arrays[arrKey].data
			# f1 = plt.figure(1)
			# plt.imshow(test, cmap='nipy_spectral')

	# plt.show()