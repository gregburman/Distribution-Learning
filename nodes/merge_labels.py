import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_filter import BatchFilter

class MergeLabels(BatchFilter):

	def __init__(self, labels, joined_affinities, joined_affinities_opp, merged_labels, every = 1):
		self.labels = labels
		self.joined_affinities = joined_affinities
		self.joined_affinities_opp = joined_affinities_opp
		self.merged_labels = merged_labels
		self.every = every
		self.n = 0


	def setup(self):
		spec = self.spec[self.labels].copy()
		self.provides(self.merged_labels, spec)


	def prepare(self, request):
		self.do_merge = self.n % self.every == 0
		if self.labels not in request:
			request[self.labels] = request[self.merged_labels].copy()
		request[self.labels].roi = request[self.merged_labels].roi.copy()


	def process(self, batch, request):
		
		labels = batch.arrays[self.labels].data.copy()
		if self.do_merge:
			joined_affinities = np.array(batch.arrays[self.joined_affinities].data.copy(), np.int16)
			joined_affinities_opp = np.array(batch.arrays[self.joined_affinities_opp].data.copy(), np.int16)
			fully_joined = np.logical_and(joined_affinities == 1, joined_affinities_opp == 1)

			available_labels = np.unique(labels)
			random_label = np.random.choice(available_labels)

			# print("available_labels: ", available_labels.shape)			

			# print("random_label: ", random_label)
			# print("joined_affinities: ", joined_affinities.shape)
			# print("joined_affinities_opp: ", joined_affinities_opp.shape)

			# fully_joined = fully_joined[:-1, :-1, :-1]
			mask_edge = np.where((fully_joined == False) & (labels == random_label), random_label, 0)

			indices = np.indices(labels.shape)
			mask_indices = indices[:, mask_edge == random_label]
			random_point = np.random.randint(1, mask_indices.shape[1])
			r_index = mask_indices[:, random_point]

			random_neighbour = 0
			for i in range(-1, 2):
					for j in range(-1, 2):
						for k in range(-1, 2):
							try:
								label = labels[r_index[0] + i, r_index[1] +j, r_index[2] + k]
								if label != random_label:
									random_neighbour = label
									# print("random_neighbour: ", random_neighbour)
									break
							except Exception as e:
								# print(e)
								pass
						else:
							continue
						break
					else:
						continue
					break

			merged = np.where(labels == random_neighbour, random_label, labels)
			# print("remaining_labels: ", np.unique(merged).shape)	
			spec = self.spec[self.merged_labels].copy()
			spec.roi = request[self.merged_labels].roi
			batch.arrays[self.merged_labels] = gp.Array(merged, spec)

			roi = request[self.labels].roi
			batch.arrays[self.labels] = batch.arrays[self.labels].crop(roi)
		else:
			spec = self.spec[self.merged_labels].copy()
			spec.roi = request[self.merged_labels].roi
			batch.arrays[self.merged_labels] = gp.Array(labels, spec)

			roi = request[self.labels].roi
			batch.arrays[self.labels] = batch.arrays[self.labels].crop(roi)
		self.n += 1
