import logging
import numpy as np
import gunpowder as gp
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class PickRandomLabel(BatchFilter):

	def __init__(self, input_labels, output_label):
		self.input_labels = input_labels
		self.output_label = output_label

	def setup(self):
		spec = self.spec[self.input_labels[0]].copy()
		self.provides(self.output_label, spec)


	def prepare(self, request):
		for i in self.input_labels:
			if i not in request:
				request[i] = request[self.output_label].copy()
			request[i].roi = request[self.output_label].roi.copy()


	def process(self, batch, request):

		random_pick = np.random.choice(self.input_labels)
		output_label = batch.arrays[random_pick].data.copy()

		# logger.info(random_pick)
	
		spec = self.spec[self.output_label].copy()
		spec.roi = request[self.output_label].roi.copy()
		batch.arrays[self.output_label] = gp.Array(output_label, spec)

		for i in self.input_labels:
			roi = request[i].roi
			batch.arrays[i] = batch.arrays[i].crop(roi)
