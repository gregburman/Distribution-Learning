import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_filter import BatchFilter

class mergeLabels(BatchFilter):

	def __init__(self, n_objects, labels, joined_pos_affinities, joined_neg_affinities, merged_labels):
		self.n_objects = n_objects
		self.labels = labels
		self.joined_pos_affinities = joined_pos_affinities
		self.joined_neg_affinities = joined_neg_affinities
		self.merged_labels = merged_labels

	def setup(self):
		spec = self.spec[self.labels].copy()
		self.provides(self.merged_labels, spec)

	def prepare(self, request):
		request[self.labels].roi = request[self.merged_labels].roi.copy()


	def process(self, batch, request):
		labels = batch[self.labels].data.copy()
		joined_pos_affinities = batch[self.joined_pos_affinities].data.copy()
		joined_neg_affinities = batch[self.joined_neg_affinities].data.copy()


		random_label = np.random.randint(1, self.n_objects + 1)
		mask_edge = np.where(((joined_pos_affinities == 0) | (joined_neg_affinities == 0)) & (labels == random_label), random_label, 0)

		indices = np.indices(labels.shape)
		mask_indices = indices[:, mask_edge == random_label]
		random_point = np.random.randint(1, mask_indices.shape[1]) # not sure about this
		r_index = mask_indices[:, random_point]

		random_neighbour = 0
		for i in range(-1, 2):
				for j in range(-1, 2):
					for k in range(-1, 2):
						try:
							label = labels[r_index[0] + i, r_index[1] +j, r_index[2] + k]
							if label != random_label:
								random_neighbour = label
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

		print ("random_label: ", random_label)
		print ("random_neighbour: ", random_neighbour)
		merged = np.where(labels == random_neighbour, random_label, labels)

		spec = self.spec[self.merged_labels].copy()
		spec.roi = request[self.merged_labels].roi
		batch.arrays[self.merged_labels] = gp.Array(merged, spec)

		roi = request[self.labels].roi
		batch.arrays[self.labels] = batch.arrays[self.labels].crop(roi)

