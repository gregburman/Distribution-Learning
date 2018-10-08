import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_provider import BatchProvider
from scipy.ndimage.filters import gaussian_filter
import skimage.measure as measure

class BlobGenerator(BatchProvider):

	def __init__(self, size, threshold, sigma):
		self.size = size
		self.threshold = threshold
		self.sigma = sigma

		self.ones = np.ones((size), dtype = np.uint16)
		self.zeros = np.zeros((size), dtype = np.uint8)

	def setup(self):
		print ('BlobGenerator setup() called')

		# self.provides(
		#    gp.ArrayKey('BLOBS'),
		# 	gp.ArraySpec(
		# 		roi=gp.Roi((0, 0), (self.size, self.size)),
		# 		voxel_size=(1, 1)))

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
			print array_spec

			input_canvas= np.random.randint(0, 255, (self.size, self.size))
			filtered_canvas = gaussian_filter(input_canvas, self.sigma)
			thresholded_canvas = np.where(filtered_canvas > self.threshold, self.ones, self.zeros)
			labeled_canvas, num_labels = measure.label(thresholded_canvas, connectivity = 1, return_num = True)
			labeled_canvas = labeled_canvas[:, :, np.newaxis]
			print 'labeled_canvas: ', labeled_canvas.shape

			batch.arrays[array_key] = gp.Array(labeled_canvas, array_spec)

		return batch