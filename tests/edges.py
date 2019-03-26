import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
import time

if __name__ == "__main__":

	SIZE = 132
	f, axes = plt.subplots(3, 2)

	file = h5py.File('snapshots/edges.hdf', 'r')
	volumes = file['volumes']
	labels = np.array(volumes['labels'])
	affs_pos = np.array(volumes['affinities_pos'], dtype=np.int32)
	affs_neg = np.array(volumes['affinities_neg'], dtype=np.int32)	
	# exit()

	ts = 0
	iters = 1
	indices = np.indices((SIZE, SIZE, SIZE))
	print("indices:", indices.shape)
	for t in range(iters):
		
		t0 = time.time()
		L = t+1 # randomly chosen label
		F = [0]*50 # list of all labels, from 1-n_objects
		
		mask_full = np.where(labels == L, L, 0)
		mask_edge = np.where(((affs_neg == 0) | (affs_pos == 0)) & (labels == L), L, 0)

		# exit()
		
		mask_indices = indices[:, mask_edge == L]
		print("mask_indices: ", mask_indices)
		r = np.random.randint(1, mask_indices.shape[1])
		mask_index = mask_indices[:, r]
		print("r: ", r)
		print("mask_index: ", mask_index)

		n = 0
		for i in range(-1, 2):
				for j in range(-1, 2):
					for k in range(-1, 2):
						try:
							label = labels[mask_index[0] + i, mask_index[1] +j, mask_index[2] + k]
							if label != L:
								n = label
								break
						except Exception as e:
							print(e)
							pass
					else:
						continue
					break
				else:
					continue
				break

		# print ("n: ", n)

		# t1 = time.time()
		# for z, y, x in np.nditer([m for m in mask_indices]):
		# 	for i in range(-1, 2):
		# 		for j in range(-1, 2):
		# 			for k in range(-1, 2):
		# 				try:
		# 					label = labels[z + i][y + j][x + k]-1
		# 					if label != L:
		# 						F[label] += 1
		# 						break
		# 				except Exception as e:
		# 					# print(e)
		# 					pass
		# print("loop time: ", time.time() - t1)
		# print ("F: ", F)

		# 2D slice[0] for debugging
		# mask_full = np.where(labels[0] == L, L, 0)
		# print("mask_full: ", np.count_nonzero(mask_full))
		# indices = np.indices((SIZE, SIZE))
		# mask_indices = indices[:, mask_full == L]
		# print(mask_indices.shape)

		
		# for y, x in np.nditer([m for m in mask_indices]):
		# 	for j in range(-1, 2):
		# 		for k in range(-1, 2):
		# 			try:
		# 				label = labels[0][y + j][x + k]-1
		# 				F[label] += 1
		# 			except Exception as e:
		# 				pass
		
		# print ("F: ", F)


		#randomly pick a neighbour (eg. L = 1)
		# neighbours = {i+1: n for i, n in enumerate(F) if n != 0 and (i+1) != L}
		# print ("neighbours: ", neighbours)
		# n = random.choice(list(neighbours.keys()))
		# print("randomly chosen neighbour:", n)

		merged = np.where(labels == n, L, labels)

		ts += time.time() - t0
		print("iter time: ", (time.time() - t0))
	print ("avg time: ", ts/iters)

	# print(shifted[1])
	axes[0][0].imshow(labels[0])
	axes[0][1].imshow(mask_full[0])
	axes[1][0].imshow(affs_neg[0])
	axes[1][1].imshow(affs_pos[0])
	axes[2][0].imshow(mask_edge[0])
	axes[2][1].imshow(merged[0])
	plt.show()