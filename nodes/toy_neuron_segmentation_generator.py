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

	def __init__(self, array_key, n_objects, points_per_skeleton, smoothness, interpolation):
		# assert len(shape) == 3

		self.array_key = array_key
		# self.shape = shape
		# self.pos = pos
		self.n_objects = n_objects
		self.points_per_skeleton = points_per_skeleton
		self.smoothness = smoothness
		self.interpolation = interpolation

	def setup(self):
		print "setup"

		# roi_size = tuple(_ for _ in self.shape)
		# print "self.array: ", self.array
		# print "roi: ", gp.Roi((0, 0, 0), roi_size)
		self.provides(
			self.array_key,
			gp.ArraySpec(
				# roi=gp.Roi((0, 0, 0), roi_size),
				None,
				voxel_size=(1, 1, 1)))

	def provide(self, request):
		print "provide"
		batch = gp.Batch()        

		for (array_key, request_spec) in request.array_specs.items():
			
			array_spec = self.spec[array_key].copy()
			array_spec.roi = request_spec.roi
			shape = array_spec.roi.get_shape()
			
			# enlarge
			lshape = list(shape)
			inc = [0]*len(shape)
			for i, s in enumerate(shape):
				if s % 2 != 0:
					inc[i] += 1
					lshape[i] += 1
			shape = gp.Coordinate(lshape)

			data = create_segmentation(
				shape=shape,
				n_objects=self.n_objects,
				points_per_skeleton=self.points_per_skeleton,
				interpolation=self.interpolation ,
				smoothness=self.smoothness)
			segmentation = data["segmentation"]

			# crop (more elegant & general way to do this?)
			segmentation = segmentation[:lshape[0] - inc[0], :lshape[1] - inc[1], :lshape[2] - inc[2]]
			# segmentation = segmentation[:lshape_out[i] - inc[i] for i in range(len(shape))]

			batch.arrays[array_key] = gp.Array(segmentation, array_spec)

		return batch