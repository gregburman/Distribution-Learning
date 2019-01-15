import logging
import numpy as np
import gunpowder as gp

from gunpowder.nodes.batch_filter import BatchFilter
from skimage.util import random_noise
from skimage.filters import gaussian

logger = logging.getLogger(__name__)


class AddJoinedAffinities(BatchFilter):

	def __init__ (self, input_affinities, joined_affinities):
		self.input_affinities = input_affinities
		self.joined_affinities = joined_affinities


	def setup (self):
		spec = self.spec[self.input_affinities].copy()
		self.provides(self.joined_affinities, spec)


	def prepare(self, request):
		request[self.input_affinities].roi = request[self.joined_affinities].roi.copy()


	def process (self, batch, request):
		joined_affinities_roi = request[self.joined_affinities].roi.copy()

		input_affinities = batch.arrays[self.input_affinities].data.copy()
		affs1 = input_affinities[0]
		affs2 = input_affinities[1]
		joined_affinities = np.logical_and(affs1 == 1, affs2 == 1)

		# crop to requested ROI
		joined_affinities = self.__crop_center(joined_affinities, joined_affinities_roi.get_shape())

		spec = self.spec[self.joined_affinities].copy()
		spec.roi = request[self.joined_affinities].roi.copy()
		batch.arrays[self.joined_affinities] = gp.Array(joined_affinities, spec)

		roi = request[self.input_affinities].roi
		batch.arrays[self.input_affinities] = batch.arrays[self.input_affinities].crop(roi)


	def __crop_center(self, img, crop):
		z, y,x = img.shape
		startz = z//2-(crop[0]//2)
		starty = y//2-(crop[1]//2)
		startx = x//2-(crop[2]//2)    
		return img[startz:startz+crop[0], starty:starty+crop[1],startx:startx+crop[2]]