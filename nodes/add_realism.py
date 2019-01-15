import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_filter import BatchFilter
from skimage.util import random_noise
from skimage.filters import gaussian

class AddRealism(BatchFilter):

	def __init__(self, affinities, realistic_data, sp, sigma):
		self.affinities = affinities
		self.realistic_data = realistic_data
		self.sp = sp
		self.sigma = sigma


	def setup(self):
		spec = self.spec[self.affinities].copy()
		self.provides(self.realistic_data, spec)


	def prepare(self, request):
		request[self.affinities].roi = request[self.realistic_data].roi.copy()


	def process(self, batch, request):
		affinities = batch[self.affinities].data.copy()

		realistic_data = random_noise(affinities, 's&p', amount=self.sp)
		realistic_data = gaussian(realistic_data,self.sigma)
		realistic_data = realistic_data*0.7

		spec = self.spec[self.realistic_data].copy()
		spec.roi = request[self.realistic_data].roi
		batch.arrays[self.realistic_data] = gp.Array(realistic_data, spec)

		roi = request[self.affinities].roi
		batch.arrays[self.affinities] = batch.arrays[self.affinities].crop(roi)

