from __future__ import print_function
import sys
sys.path.append('../')

from gunpowder import *
import numpy as np
import os

data_dir = "../data/datasets/gt_1_merge_3"
neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

# samples = ["batch_%08i"%i for i in range(1)]


def generate_affinities(iteration):

	samples = ["batch_%08i"%iteration]

	labels_key = ArrayKey('GT_LABELS')
	gt_affs_key = ArrayKey('GT_AFFINITIES')

	merged_labels_keys = []
	merged_affs_keys = []

	num_merges = 3
	for i in range(num_merges):
		merged_labels_keys.append(ArrayKey('MERGED_LABELS_%i'%(i+1)))
		merged_affs_keys.append(ArrayKey('MERGED_AFFINITIES_%i'%(i+1)))

	voxel_size = Coordinate((1, 1, 1))
	input_shape = Coordinate((132,132,132)) * voxel_size
	output_shape = Coordinate((44,44,44)) * voxel_size

	# print ("input_shape: ", input_shape)
	# print ("output_shape: ", output_shape)

	request = BatchRequest()
	request.add(labels_key, output_shape)
	request.add(gt_affs_key, output_shape)

	for i in range(num_merges): 
		request.add(merged_labels_keys[i], output_shape)
		request.add(merged_affs_keys[i], output_shape)


	read_dataset_names = {
		labels_key: 'volumes/labels',
	}

	read_array_specs = {
		labels_key: ArraySpec(interpolatable=False)
	}


	for i in range(num_merges):
		read_dataset_names[merged_labels_keys[i]] = 'volumes/merged_labels_%i'%(i+1)
		read_array_specs[merged_labels_keys[i]] = ArraySpec(interpolatable=False)
		# read_array_specs[merged_affs_keys[i]] = ArraySpec(interpolatable=True)

	pipeline = tuple(
		Hdf5Source(
			os.path.join(data_dir, sample + '.hdf'),
			datasets = read_dataset_names,
			array_specs = read_array_specs
		) +
		Pad(labels_key, None) +
		Pad(merged_labels_keys[0], None) +
		Pad(merged_labels_keys[1], None) +
		Pad(merged_labels_keys[2], None)
		for sample in samples
	)

	# pipeline += PreCache(
	# 	cache_size=32,
	# 	num_workers=8)

	pipeline += RenumberConnectedComponents(
			labels=labels_key)

	pipeline += AddAffinities(
		affinity_neighborhood=neighborhood,
		labels=labels_key,
		affinities=gt_affs_key)



	for i in range(num_merges): 

		# pipeline += RenumberConnectedComponents(
		# 	labels=merged_labels_keys[i])

		pipeline += AddAffinities(
				affinity_neighborhood=neighborhood,
				labels=merged_labels_keys[i],
				affinities=merged_affs_keys[i])



	write_dataset_names = {
		labels_key: 'volumes/labels',
		gt_affs_key: 'volumes/gt_affs'
	}

	write_dataset_dtypes = {
		labels_key: np.uint16,
		gt_affs_key: np.float32,
	}

	for i in range(num_merges):
		write_dataset_names[merged_labels_keys[i]] = 'volumes/merged_labels_%i'%(i+1)
		write_dataset_dtypes[merged_labels_keys[i]] = np.uint16
		write_dataset_names[merged_affs_keys[i]] = 'volumes/merged_affs_%i'%(i+1)
		write_dataset_dtypes[merged_affs_keys[i]] = np.float32

	# pipeline += Snapshot(
	# 	dataset_names=write_dataset_names,
	# 	output_filename='roi_constraint.hdf',
	# 	every=1,
	# 	dataset_dtypes=write_dataset_dtypes)

	with build(pipeline) as p:
		num_labels_same = 0
		num_labels_diff = 0
		num_affs_same = 0
		num_affs_diff = 0

		req = p.request_batch(request)
		labels = np.array(req[labels_key].data)
		gt_affs = np.array(req[gt_affs_key].data)
		num_labels = len(np.unique(labels))
		# print("num labels (unmerged): ", num_labels)
		gt_affs_count = np.count_nonzero(gt_affs == 0)
		# print("gt_affs: ", gt_affs_count)

		for i in range(num_merges):
			merged_labels = np.array(req[merged_labels_keys[i]].data)
			merged_affs = np.array(req[merged_affs_keys[i]].data)
			num_merged_labels = len(np.unique(merged_labels))
			merged_affs_count = np.count_nonzero(merged_affs == 0)
			# print("merged_affs_count: ", merged_affs_count)
			if num_merged_labels == num_labels:
				num_labels_same += 1
				# print("SAME")
			else: # merge occurs
				num_labels_diff +=1

			if merged_affs_count == gt_affs_count:
				num_affs_same += 1
			else:
				num_affs_diff += 1

				
				# merged_affs_count = np.count_nonzero(merged_affs == 0)
				# diff = gt_affs_count - merged_affs_count
				# diff_norm.append(diff/float(gt_affs_count))
			# print("diff: ", diff)
			# print("merged: ", merged_affs_count)
				# print("gt_affs: ", gt_affs_count)
			# print("diff_norm: ", diff_norm)
			# print("num labels (merged)", num_merged_labels)
		# print("foo")
	return (num_labels_same, num_labels_diff, num_affs_same, num_affs_diff)

if __name__ == "__main__":
	print ("Generating affinities...")
	num_labels_same = 0
	num_labels_diff = 0
	num_affs_same = 0
	num_affs_diff = 0

	print ("running...")
	merges_per_point_labels = {0:0, 1:0, 2:0, 3:0}
	merges_per_point_affs = {0:0, 1:0, 2:0, 3:0}
	for i in range(int(sys.argv[1])):
		print("iteration: ", i)
		results = generate_affinities(iteration=i)
		num_labels_same += results[0]
		num_labels_diff += results[1]
		num_affs_same += results[2]
		num_affs_diff += results[3]
		merges_per_point_labels[results[1]]+=results[1]
		merges_per_point_affs[results[3]]+=results[3]
		# for _, r in enumerate(results[2]):
		# 	diff_norm += r
	print("total labels num_same: ", num_labels_same)
	print("total labels num_diff: ", num_labels_diff)
	print("total affs num_same: ", num_affs_same)
	print("total affs num_diff: ", num_affs_diff)

	print("merges_per_point_labels:", merges_per_point_labels)
	print("merges_per_point_affs:", merges_per_point_affs)

	print ("Affinities generation test finished.")