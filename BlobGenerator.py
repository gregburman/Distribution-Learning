import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_provider import BatchProvider
from scipy.ndimage.filters import gaussian_filter
import skimage.measure as measure
import skelerator as sk
import matplotlib.pyplot as plt

class BlobGenerator(BatchProvider):

	def __init__(self, size, n_objects, points_per_skeleton, smoothness, interpolation):
		self.size = size
		self.n_objects = n_objects
		self.points_per_skeleton = points_per_skeleton
		self.smoothness = smoothness
		self.interpolation = interpolation

		self.ones = np.ones((size), dtype = np.uint16)
		self.zeros = np.zeros((size), dtype = np.uint8)

	def setup(self):

		self.provides(
			gp.ArrayKey('LABELS'),
			gp.ArraySpec(
				roi=gp.Roi((0, 0, 0), (self.size, self.size, 1)),
				voxel_size=(1, 1, 1)))

	def provide(self, request):
		batch = gp.Batch()        

		for (array_key, request_spec) in request.array_specs.items():

			array_spec = self.spec[array_key].copy()
			array_spec.roi = request_spec.roi

			# input_canvas= np.random.randint(0, 255, (self.size, self.size))
			# filtered_canvas = gaussian_filter(input_canvas, 1)
			# thresholded_canvas = np.where(filtered_canvas > 127, self.ones, self.zeros)
			# labeled_canvas, num_labels = measure.label(thresholded_canvas, connectivity = 1, return_num = True)
			# labeled_canvas = labeled_canvas[:, :, np.newaxis] + 1
			# print "labeled_canvas: ", np.shape(labeled_canvas)

			shape = np.array([self.size, self.size])
			# shape = np.array([self.size, self.size, self.size])
			data = sk.create_segmentation(shape, self.n_objects, self.points_per_skeleton, self.interpolation , self.smoothness)
			segmentation = data["segmentation"]
			segmentation = segmentation[:, :, np.newaxis] + 1
			print "segmentation: ", np.shape(segmentation)
			# plt.imshow(segmentation)
			# plt.show()
			batch.arrays[array_key] = gp.Array(segmentation, array_spec)

		return batch

	# shape = np.array([100, 100])
 #    n_objects = 20
 #    points_per_skeleton = 5
 #    smoothness = 2
 #    write_to = "./test_segmentation.h5"
 #    interpolation = "linear"
 #    data = create_segmentation(shape, n_objects, points_per_skeleton, interpolation , smoothness, write_to=write_to)