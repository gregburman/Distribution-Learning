import gunpowder as gp
from toy_neuron_segmentation_generator import ToyNeuronSegmentationGenerator
from add_joined_affinities import AddJoinedAffinities
from add_realism import AddRealism
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
	
	labels = gp.ArrayKey('LABELS')
	affinities= gp.ArrayKey('AFFINITIES')
	joined_affinities= gp.ArrayKey('JOINED_AFFINITIES')
	realistic_data = gp.ArrayKey('REALISTIC_DATA')

	ds1 = {labels: "labels"}
	ds2 = {joined_affinities: "affmap"}
	ds3 = {realistic_data: "realism"}
	
	shape = np.array ([10, 1000, 1000])  # z, y, x
	create_labels = ToyNeuronSegmentationGenerator(shape, 0, 30, 8, 3, "linear")
	add_affinities = gp.AddAffinities([[0, 0, -1], [0, -1, 0]], labels, affinities) 
	add_joined_affinities = AddJoinedAffinities(affinities, joined_affinities)
	add_realism = AddRealism(joined_affinities, realistic_data, 0.25, 1)
	defect_augment = gp.DefectAugment(realistic_data, prob_missing=0.0, prob_deform=0.7, axis=2)
	intensity_augment = gp.IntensityAugment(realistic_data, 0.9, 1.3, -0.1, 0.3)
	
	snapshot_labels = gp.Snapshot(ds1, "labels")
	snapshot_affinities = gp.Snapshot(ds2, "affmaps")
	snapshot_final = gp.Snapshot(ds3, "final")

	pipeline = create_labels + add_affinities + add_joined_affinities + add_realism
	# pipeline = labels + snapshot_labels + affinities + joined_affinities + snapshot_affinities + realism + snapshot_realism

	voxel_size = gp.Coordinate((1, 1, 1))
	input_size = gp.Coordinate((1, shape[1], shape[2])) * voxel_size
	output_size = gp.Coordinate((1, shape[1] - 1, shape[2] - 1)) * voxel_size

	request = gp.BatchRequest()
	request.add(labels, input_size)
	request.add(affinities, output_size)
	request.add(joined_affinities, output_size)
	request.add(realistic_data, output_size)

	with gp.build(pipeline) as p:
		for i in range(1):
			req = p.request_batch(request)

			labels = req.arrays[labels].data
			labels = labels.reshape((shape[1],shape[2]))
			affinities = req.arrays[joined_affinities].data
			affinities = affinities.reshape((shape[1]-1,shape[2]-1))
			realistic_data = req.arrays[realistic_data].data
			realistic_data = realistic_data.reshape((shape[1]-1,shape[2]-1))

			# f1 = plt.figure(1)
			# plt.imshow(labels)
			# f2 = plt.figure(2)
			# plt.imshow(affinities, alpha=0.5)
			f3 = plt.figure(3)
			# plt.imshow(realistic_data, cmap="Greys_r")
			plt.imshow(realistic_data, cmap="Greys_r", vmin=0, vmax=1)
			
			plt.show()