import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_filter import BatchFilter
from skimage.util import random_noise
from skimage.filters import gaussian

class AddJoinedAffinities(BatchFilter):

	def __init__ (self, input_affinities, joined_affinities):
		self.input_affinities = input_affinities
		self.joined_affinities = joined_affinities

	def setup (self):
		spec = self.spec[self.input_affinities].copy()
		self.provides(self.joined_affinities, spec)

	def process (self, batch, request):
		input_affinities = batch.arrays[self.input_affinities].data
		affs1 = input_affinities[0]
		affs2 = input_affinities[1]
		joined_affinities = np.logical_and(affs1 == 1, affs2 == 1)

		spec = self.spec[self.joined_affinities].copy()
		batch.arrays[self.joined_affinities] = gp.Array(joined_affinities, spec)

