import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_provider import BatchProvider
from scipy.ndimage.filters import gaussian_filter
import skimage.measure as measure
# from skelerator import Tree, Skeleton
from skelerator.forest import create_segmentation
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
from mahotas import cwatershed
import traceback
import sys

class ToyNeuronSegmentationGenerator(BatchProvider):
	"""
	Creates a toy segmentation containing skeletons.

	Args:

	shape: Size of the desired dataset
	
	n_objects: The number of skeleton/neurons to generate in the given volume

	points_per_skeleton: The number of potential branch points that are sampled per skeleton. 
						 Higher numbers lead to more complex shapes.

	interpolation: Method of interpolation between two sample points. Can be either linear or
				   random (constrained random walk).

	smoothness: Controls the smoothness of the initial noise map used to generate object boundaries.
	"""

	def __init__(self, array_key, shape, n_objects, points_per_skeleton, smoothness, interpolation):
		assert len(shape) == 3

		self.array_key = array_key
		self.shape = shape
		# self.pos = pos
		self.n_objects = n_objects
		self.points_per_skeleton = points_per_skeleton
		self.smoothness = smoothness
		self.interpolation = interpolation

	def setup(self):
		print "setup"
		# if self.pos == -1:
		# 	roi_size = tuple(_ for _ in self.shape)
		# else:
		# 	roi_size = (1, self.shape[1], self.shape[2])

		roi_size = tuple(_ for _ in self.shape)
		# print "self.array: ", self.array
		# print "roi: ", gp.Roi((0, 0, 0), roi_size)
		self.provides(
			self.array_key,
			gp.ArraySpec(
				roi=gp.Roi((0, 0, 0), roi_size),
				voxel_size=(1, 1, 1)))

	def provide(self, request):
		# print "provide"
		batch = gp.Batch()        

		for (array_key, request_spec) in request.array_specs.items():
			# print request

			# print "request_spec roi: ", request_spec.roi
			
			array_spec = self.spec[array_key].copy()
			# print "array_spec: ", array_spec
			array_spec.roi = request_spec.roi

			# shape = array_spec.roi.get_shape()/array_spec.voxel_size
			# print ("shape: ", shape)

			# if shape

			# print "array_spec.roi: ", array_spec.roi

			data = create_segmentation(
				self.shape,
				self.n_objects,
				self.points_per_skeleton,
				self.interpolation ,
				self.smoothness)
			segmentation = data["segmentation"]

			# if self.pos > -1:
			# 	segmentation = segmentation[np.newaxis, :, : ] + 1

			# roi = request[self.input_affinities].roi
			# batch.arrays[self.input_affinities] = batch.arrays[self.input_affinities].crop(roi)

			batch.arrays[array_key] = gp.Array(segmentation, array_spec)

		return batch