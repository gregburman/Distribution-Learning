import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_filter import BatchFilter
from skimage.util import random_noise
from skimage.filters import gaussian

class AddRealism(BatchFilter):

	def __init__(self, joined_affinities, raw, sp, sigma):
		self.joined_affinities = joined_affinities
		self.raw = raw
		self.sp = sp
		self.sigma = sigma


	def setup(self):
		spec = self.spec[self.joined_affinities].copy()
		self.provides(self.raw, spec)


	def prepare(self, request):
		request[self.joined_affinities].roi = request[self.raw].roi.copy()


	def process(self, batch, request):
		joined_affinities = batch[self.joined_affinities].data.copy()

		raw = random_noise(joined_affinities, 's&p', amount=self.sp)
		raw = gaussian(raw,self.sigma)
		raw = raw*0.7

		spec = self.spec[self.raw].copy()
		spec.roi = request[self.raw].roi
		batch.arrays[self.raw] = gp.Array(raw, spec)

		roi = request[self.joined_affinities].roi
		batch.arrays[self.joined_affinities] = batch.arrays[self.joined_affinities].crop(roi)

