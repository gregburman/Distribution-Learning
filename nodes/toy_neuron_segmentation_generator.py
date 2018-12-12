import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_provider import BatchProvider
from scipy.ndimage.filters import gaussian_filter
import skimage.measure as measure
from skelerator import Tree, Skeleton
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

	def __init__(self, shape, pos, n_objects, points_per_skeleton, smoothness, interpolation):
		assert len(shape) == 3

		self.shape = shape
		self.pos = pos
		self.n_objects = n_objects
		self.points_per_skeleton = points_per_skeleton
		self.smoothness = smoothness
		self.interpolation = interpolation

	def setup(self):

		if self.pos == -1:
			roi_size = tuple(_ for _ in self.shape)
		else:
			roi_size = (1, self.shape[1], self.shape[2])
		
		self.provides(
			gp.ArrayKey('GT_LABELS'),
			gp.ArraySpec(
				roi=gp.Roi((0, 0, 0), roi_size),
				voxel_size=(1, 1, 1)))

	def provide(self, request):
		batch = gp.Batch()        

		for (array_key, request_spec) in request.array_specs.items():

			array_spec = self.spec[array_key].copy()
			array_spec.roi = request_spec.roi

			segmentation = self.create_segmentation(self.shape, self.pos, self.n_objects, self.points_per_skeleton, self.interpolation , self.smoothness)

			if self.pos > -1:
				segmentation = segmentation[np.newaxis, :, : ] + 1

			batch.arrays[array_key] = gp.Array(segmentation, array_spec)

		return batch

	def create_segmentation(self, shape, pos, n_objects, points_per_skeleton, interpolation, smoothness, seed=0):

		noise = np.abs(np.random.randn(*shape))
		smoothed_noise = gaussian_filter(noise, sigma=smoothness)
		
		# Sample one tree for each object and generate its skeleton:
		seeds = np.zeros(2*shape, dtype=int)

		for i in range(n_objects):
			"""
			We make the virtual volume twice as large to avoid border effects. To keep the density
			of points the same we also increase the number of points by a factor of 8 = 2**3. Such that
			on average we keep the same number of points per unit volume.
			"""
			points = np.stack([np.random.randint(0, 2*shape[2-dim], (2**3)*points_per_skeleton) for dim in range(3)], axis=1)
			tree = Tree(points)
			skeleton = Skeleton(tree, [1,1,1], "linear", generate_graph=False)
			seeds = skeleton.draw(seeds, np.array([0,0,0]), i + 1)

		# Cut the volume to original size.
		seeds = seeds[int(shape[0]/2):int(3*shape[0]/2), int(shape[1]/2):int(3*shape[1]/2), int(shape[2]/2):int(3*shape[2]/2)]

		"""
		We generate an artificial segmentation by first filtering skeleton points that are too close to
		each other via a non max supression to avoid artifacts. A distance transform of the skeletons
		plus smoothed noise 	is then used to calculate a watershed transformation with the skeletons
		as seeds resulting in the final segmentation.
		"""
		seeds[maximum_filter(seeds, size=4) != seeds] = 0
		seeds_dt = distance_transform_edt(seeds==0) + 5. * smoothed_noise
		segmentation = cwatershed(seeds_dt, seeds)
		# boundaries = find_boundaries(segmentation)
		# data = {"segmentation": segmentation, "skeletons": seeds, "raw": boundaries}

		if self.pos > -1:
			return segmentation[pos]
		else:
			return segmentation