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

	def process(self, batch, request):
		affinities = batch[self.affinities].data

		# mean_map = np.ones(affinities.shape)*0.5
		# rn = random_noise(mean_map, 's&p', amount=self.sp)
		# gf = gaussian(rn,self.sigma)
		# final = affinities + gf

		realistic_data = random_noise(affinities, 's&p', amount=self.sp)
		realistic_data = gaussian(realistic_data,self.sigma)
		realistic_data = realistic_data*0.7

		spec = self.spec[self.realistic_data].copy()
		batch.arrays[self.realistic_data] = gp.Array(realistic_data, spec)


